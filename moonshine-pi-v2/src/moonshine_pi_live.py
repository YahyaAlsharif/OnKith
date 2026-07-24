#!/usr/bin/env python3
"""
Final local true-live STT deployment for Raspberry Pi 5.

Validated deployment path:
USB microphone -> local PCM chunks -> Moonshine Medium Streaming -> transcript events.

The default configuration exactly matches the final anti-hallucination live test:
- Moonshine Medium Streaming (English, quantized on-device model)
- RNNoise disabled
- 100 ms microphone blocks
- 250 ms transcript updates
- VAD threshold 0.50
- VAD window 0.30 s
- VAD max segment 10 s
- max_tokens_per_second 6.5
- adaptive local energy gate with 400 ms pre-roll
- automatic endpoint after 0.85 s of post-speech silence
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import queue
import re
import signal
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "final_config.json"
STOP_REQUESTED = threading.Event()


def _signal_handler(signum: int, frame: Any) -> None:
    del signum, frame
    STOP_REQUESTED.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


@dataclass(frozen=True)
class AppConfig:
    model_language: str
    model_arch: str
    tested_moonshine_voice_version: str
    rnnoise_enabled: bool
    update_interval_seconds: float
    vad_threshold: float
    vad_window_duration_seconds: float
    vad_max_segment_duration_seconds: float
    max_tokens_per_second: float
    block_seconds: float
    calibration_seconds: float
    pre_roll_seconds: float
    start_confirm_seconds: float
    end_silence_seconds: float
    min_speech_seconds: float
    max_utterance_seconds: float
    wait_for_speech_timeout_seconds: float
    min_start_rms: float
    noise_multiplier: float
    continue_ratio: float
    queue_max_chunks: int
    save_debug_audio: bool
    debug_audio_directory: str

    @classmethod
    def from_json(cls, path: Path) -> "AppConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def validate(self) -> None:
        if self.model_language != "en":
            raise ValueError("This final configuration is validated only for English.")
        if self.model_arch != "medium_streaming":
            raise ValueError("The validated final model is medium_streaming.")
        if self.rnnoise_enabled:
            raise ValueError("RNNoise must remain disabled for the validated pipeline.")
        positive = {
            "update_interval_seconds": self.update_interval_seconds,
            "vad_window_duration_seconds": self.vad_window_duration_seconds,
            "vad_max_segment_duration_seconds": self.vad_max_segment_duration_seconds,
            "max_tokens_per_second": self.max_tokens_per_second,
            "block_seconds": self.block_seconds,
            "calibration_seconds": self.calibration_seconds,
            "pre_roll_seconds": self.pre_roll_seconds,
            "start_confirm_seconds": self.start_confirm_seconds,
            "end_silence_seconds": self.end_silence_seconds,
            "min_speech_seconds": self.min_speech_seconds,
            "max_utterance_seconds": self.max_utterance_seconds,
            "wait_for_speech_timeout_seconds": self.wait_for_speech_timeout_seconds,
            "min_start_rms": self.min_start_rms,
            "noise_multiplier": self.noise_multiplier,
            "continue_ratio": self.continue_ratio,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive finite number.")
        if not 0 <= self.vad_threshold <= 1:
            raise ValueError("vad_threshold must be between 0 and 1.")
        if self.queue_max_chunks < 10:
            raise ValueError("queue_max_chunks must be at least 10.")


def normalize_standard(text: str) -> str:
    """Case/punctuation-normalized WER, matching the live benchmark."""
    text = (text or "").lower().strip().replace("%", " percent ")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_SMALL_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _normalize_number_words(tokens: Sequence[str]) -> List[str]:
    """Normalize common English number phrases (0-999) into digit tokens."""
    output: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token not in _SMALL_NUMBERS and token not in _TENS and token != "hundred":
            output.append(token)
            i += 1
            continue

        j = i
        total = 0
        current = 0
        consumed = False
        while j < len(tokens):
            word = tokens[j]
            if word in _SMALL_NUMBERS:
                current += _SMALL_NUMBERS[word]
                consumed = True
            elif word in _TENS:
                current += _TENS[word]
                consumed = True
            elif word == "hundred" and current > 0:
                current *= 100
                consumed = True
            elif word == "and" and consumed:
                # Allow "one hundred and five".
                pass
            else:
                break
            j += 1
        if consumed:
            total += current
            output.append(str(total))
            i = j
        else:
            output.append(token)
            i += 1
    return output


def normalize_command(text: str) -> str:
    """Supplementary semantic normalization for command evaluation."""
    text = normalize_standard(text)
    text = text.replace("don't", "do not").replace("can't", "cannot")
    tokens = _normalize_number_words(text.split())
    return " ".join(tokens)


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.reference_words if self.reference_words else 0.0


def word_error_counts(reference: str, hypothesis: str, *, command_normalized: bool) -> ErrorCounts:
    normalizer = normalize_command if command_normalized else normalize_standard
    ref = normalizer(reference).split()
    hyp = normalizer(hypothesis).split()
    n, m = len(ref), len(hyp)

    # Each cell: (cost, substitutions, deletions, insertions).
    dp: List[List[Tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            c, s, d, ins = dp[i - 1][j - 1]
            substitute = (c + 1, s + 1, d, ins)
            c, s, d, ins = dp[i - 1][j]
            delete = (c + 1, s, d + 1, ins)
            c, s, d, ins = dp[i][j - 1]
            insert = (c + 1, s, d, ins + 1)
            dp[i][j] = min(
                (substitute, delete, insert),
                key=lambda item: (item[0], item[2] + item[3], item[1]),
            )

    _, substitutions, deletions, insertions = dp[n][m]
    return ErrorCounts(substitutions, deletions, insertions, n)


def rms_and_peak(chunk: Any) -> Tuple[float, float]:
    import numpy as np

    values = np.asarray(chunk, dtype=np.float32)
    if values.size == 0:
        return 0.0, 0.0
    rms = float(np.sqrt(np.mean(values * values) + 1e-12))
    peak = float(np.max(np.abs(values)))
    return rms, peak


def line_id_from_event(event: Any) -> int:
    line = event.line
    for name in ("line_id", "lineId", "id"):
        value = getattr(line, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return id(line)


class TranscriptCollector:
    """Created dynamically as a Moonshine listener subclass at runtime."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lines: "OrderedDict[int, str]" = OrderedDict()
        self.last_visible = ""

    def reset(self) -> None:
        with self.lock:
            self.lines.clear()
            self.last_visible = ""

    def update(self, event: Any, *, live_output: bool = True) -> None:
        key = line_id_from_event(event)
        text = (getattr(event.line, "text", "") or "").strip()
        with self.lock:
            self.lines[key] = text
            visible = " ".join(value for value in self.lines.values() if value).strip()
            changed = bool(visible and visible != self.last_visible)
            if changed:
                self.last_visible = visible
        if changed and live_output:
            print(f"\rPARTIAL: {visible[:180]:<180}", end="", flush=True)

    def final_text(self) -> str:
        with self.lock:
            return " ".join(value for value in self.lines.values() if value).strip()


@dataclass
class UtteranceResult:
    transcript: str
    model_audio_seconds: float
    decode_seconds: float
    finalization_seconds: float
    max_backlog_seconds: float
    overflow_count: int
    endpoint_reason: str

    @property
    def decode_rtf(self) -> float:
        return self.decode_seconds / self.model_audio_seconds if self.model_audio_seconds else math.nan

    @property
    def full_pipeline_rtf(self) -> float:
        if not self.model_audio_seconds:
            return math.nan
        return (self.decode_seconds + self.finalization_seconds) / self.model_audio_seconds


def list_input_devices() -> None:
    import sounddevice as sd

    print("Available microphone devices:")
    for idx, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) > 0:
            marker = " <DEFAULT>" if idx == sd.default.device[0] else ""
            print(
                f"[{idx}] {device['name']} | channels={int(device['max_input_channels'])} "
                f"| sample_rate={float(device['default_samplerate']):.0f}{marker}"
            )


def resolve_device(device: Optional[int]) -> Tuple[Optional[int], float, str]:
    import sounddevice as sd

    device_id = sd.default.device[0] if device is None else device
    info = sd.query_devices(device_id, "input")
    return device_id, float(info["default_samplerate"]), str(info["name"])


def calibrate_noise(config: AppConfig, device_id: Optional[int], sample_rate: float) -> float:
    import numpy as np
    import sounddevice as sd

    print(f"Calibrating room noise for {config.calibration_seconds:.1f} s. Stay silent...")
    block_size = max(1, int(round(sample_rate * config.block_seconds)))
    rms_values: List[float] = []

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if status:
            print(f"\nCalibration microphone status: {status}", file=sys.stderr)
        rms, _ = rms_and_peak(np.asarray(indata[:, 0], dtype=np.float32))
        rms_values.append(rms)

    with sd.InputStream(
        device=device_id,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        callback=callback,
    ):
        time.sleep(config.calibration_seconds)

    if not rms_values:
        raise RuntimeError("No microphone audio arrived during calibration.")

    noise_rms = float(np.percentile(rms_values, 90))
    threshold = max(config.min_start_rms, noise_rms * config.noise_multiplier)
    print(f"Noise RMS={noise_rms:.5f}; speech-start threshold={threshold:.5f}")
    return threshold


def create_transcriber(config: AppConfig) -> Tuple[Any, Any, TranscriptCollector]:
    from moonshine_voice import (
        ModelArch,
        Transcriber,
        TranscriptEventListener,
        get_model_for_language,
    )

    # Explicitly select Medium Streaming; do not rely only on default ordering.
    model_path, model_arch = get_model_for_language(
        config.model_language,
        ModelArch.MEDIUM_STREAMING,
    )
    if model_arch != ModelArch.MEDIUM_STREAMING:
        raise RuntimeError(f"Expected MEDIUM_STREAMING, received {model_arch!r}.")

    options: Dict[str, str] = {
        "vad_threshold": str(config.vad_threshold),
        "vad_window_duration": str(config.vad_window_duration_seconds),
        "vad_max_segment_duration": str(config.vad_max_segment_duration_seconds),
        "max_tokens_per_second": str(config.max_tokens_per_second),
    }
    if config.save_debug_audio:
        debug_path = (PROJECT_ROOT / config.debug_audio_directory).resolve()
        debug_path.mkdir(parents=True, exist_ok=True)
        options["save_input_wav_path"] = str(debug_path)

    transcriber = Transcriber(
        model_path=model_path,
        model_arch=model_arch,
        update_interval=config.update_interval_seconds,
        options=options,
    )
    collector = TranscriptCollector()

    class Listener(TranscriptEventListener):
        def on_line_started(self, event: Any) -> None:
            collector.update(event)

        def on_line_updated(self, event: Any) -> None:
            collector.update(event)

        def on_line_text_changed(self, event: Any) -> None:
            collector.update(event)

        def on_line_completed(self, event: Any) -> None:
            collector.update(event)

    listener = Listener()
    transcriber.add_listener(listener)
    return transcriber, listener, collector


def capture_utterance(
    config: AppConfig,
    transcriber: Any,
    collector: TranscriptCollector,
    device_id: Optional[int],
    sample_rate: float,
    start_rms_threshold: float,
    *,
    prompt_before_arm: bool,
) -> Optional[UtteranceResult]:
    import numpy as np
    import sounddevice as sd

    if prompt_before_arm:
        input("Press Enter to arm the microphone, then speak after READY...")

    audio_queue: "queue.Queue[Optional[Any]]" = queue.Queue(maxsize=config.queue_max_chunks)
    speech_started = threading.Event()
    endpoint_reached = threading.Event()
    worker_stop = threading.Event()
    worker_errors: List[BaseException] = []
    stats_lock = threading.Lock()

    decode_seconds = 0.0
    max_queue_chunks = 0
    overflow_count = 0
    fed_chunks = 0
    endpoint_reason = ""

    block_size = max(1, int(round(sample_rate * config.block_seconds)))
    pre_roll_count = max(1, int(round(config.pre_roll_seconds / config.block_seconds)))
    confirm_count = max(1, int(round(config.start_confirm_seconds / config.block_seconds)))
    pre_roll: "deque[Any]" = deque(maxlen=pre_roll_count)
    collector.reset()

    above_start = 0
    last_voice_clock: Optional[float] = None
    speech_start_clock: Optional[float] = None
    state_lock = threading.Lock()

    def enqueue(chunk: Any) -> None:
        nonlocal max_queue_chunks, fed_chunks, overflow_count
        try:
            audio_queue.put_nowait(chunk)
            with stats_lock:
                fed_chunks += 1
                max_queue_chunks = max(max_queue_chunks, audio_queue.qsize())
        except queue.Full:
            with stats_lock:
                overflow_count += 1

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        nonlocal above_start, last_voice_clock, speech_start_clock, endpoint_reason, overflow_count
        del frames, time_info
        if STOP_REQUESTED.is_set():
            endpoint_reached.set()
            return

        chunk = np.asarray(indata[:, 0], dtype=np.float32).copy()
        rms, peak = rms_and_peak(chunk)
        now = time.perf_counter()

        if status:
            with stats_lock:
                overflow_count += 1

        with state_lock:
            pre_roll.append(chunk)
            if not speech_started.is_set():
                if rms >= start_rms_threshold or peak >= start_rms_threshold * 4.0:
                    above_start += 1
                else:
                    above_start = 0
                if above_start >= confirm_count:
                    speech_started.set()
                    speech_start_clock = now
                    last_voice_clock = now
                    for buffered in list(pre_roll):
                        enqueue(buffered)
                    pre_roll.clear()
                    print("VOICE DETECTED")
                return

            enqueue(chunk)
            continue_threshold = start_rms_threshold * config.continue_ratio
            if rms >= continue_threshold or peak >= continue_threshold * 4.0:
                last_voice_clock = now

            active_seconds = now - (speech_start_clock or now)
            silence_seconds = now - (last_voice_clock or now)
            if active_seconds >= config.min_speech_seconds and silence_seconds >= config.end_silence_seconds:
                endpoint_reason = f"{silence_seconds:.2f}s post-speech silence"
                endpoint_reached.set()
            elif active_seconds >= config.max_utterance_seconds:
                endpoint_reason = "maximum utterance duration"
                endpoint_reached.set()

    def worker() -> None:
        nonlocal decode_seconds
        transcriber_started = False
        try:
            while not worker_stop.is_set() or not audio_queue.empty():
                try:
                    chunk = audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if chunk is None:
                    audio_queue.task_done()
                    break
                if not transcriber_started:
                    transcriber.start()
                    transcriber_started = True
                started = time.perf_counter()
                transcriber.add_audio(chunk, int(round(sample_rate)))
                elapsed = time.perf_counter() - started
                with stats_lock:
                    decode_seconds += elapsed
                audio_queue.task_done()
        except BaseException as exc:
            worker_errors.append(exc)
            endpoint_reached.set()

    worker_thread = threading.Thread(target=worker, name="moonshine-worker", daemon=True)
    worker_thread.start()

    print("READY — speak now.")
    with sd.InputStream(
        device=device_id,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        callback=callback,
    ):
        deadline = time.perf_counter() + config.wait_for_speech_timeout_seconds
        while not speech_started.is_set() and not STOP_REQUESTED.is_set():
            if time.perf_counter() >= deadline:
                endpoint_reason = "no speech detected"
                endpoint_reached.set()
                break
            time.sleep(0.02)
        while speech_started.is_set() and not endpoint_reached.is_set() and not STOP_REQUESTED.is_set():
            time.sleep(0.02)

    if not speech_started.is_set():
        worker_stop.set()
        try:
            audio_queue.put_nowait(None)
        except queue.Full:
            pass
        worker_thread.join(timeout=5)
        print("No speech detected; utterance was not transcribed.")
        return None

    drain_deadline = time.perf_counter() + max(15.0, config.max_utterance_seconds * 3.0)
    while audio_queue.unfinished_tasks:
        if worker_errors:
            break
        if time.perf_counter() >= drain_deadline:
            raise RuntimeError(
                f"Timed out draining {audio_queue.unfinished_tasks} queued audio chunks."
            )
        time.sleep(0.02)

    worker_stop.set()
    if not worker_errors:
        audio_queue.put(None)
    worker_thread.join(timeout=10)
    if worker_thread.is_alive():
        raise RuntimeError("Audio worker did not stop cleanly.")
    if worker_errors:
        raise RuntimeError(f"Moonshine worker failed: {worker_errors[0]}")

    finalize_started = time.perf_counter()
    transcriber.stop()
    finalization_seconds = time.perf_counter() - finalize_started
    time.sleep(0.05)
    print()  # end the partial-transcript line
    transcript = collector.final_text()

    with stats_lock:
        local_decode = decode_seconds
        local_max_queue = max_queue_chunks
        local_overflows = overflow_count
        local_fed_chunks = fed_chunks

    model_audio_seconds = local_fed_chunks * config.block_seconds
    return UtteranceResult(
        transcript=transcript,
        model_audio_seconds=model_audio_seconds,
        decode_seconds=local_decode,
        finalization_seconds=finalization_seconds,
        max_backlog_seconds=local_max_queue * config.block_seconds,
        overflow_count=local_overflows,
        endpoint_reason=endpoint_reason,
    )


def print_result(result: UtteranceResult) -> None:
    payload = {
        "type": "final_transcript",
        "text": result.transcript,
        "decode_rtf": round(result.decode_rtf, 3),
        "full_pipeline_rtf": round(result.full_pipeline_rtf, 3),
        "finalization_ms": round(result.finalization_seconds * 1000.0),
        "max_backlog_seconds": round(result.max_backlog_seconds, 2),
        "overflow_count": result.overflow_count,
        "endpoint_reason": result.endpoint_reason,
    }
    print("FINAL:", result.transcript or "[EMPTY]")
    print("METRICS:", json.dumps(payload, ensure_ascii=False))
    # Integration point for the next project stages:
    # privacy_gate.process(result.transcript)
    # intent_router.process(result.transcript)


def run_live(config: AppConfig, device: Optional[int], once: bool) -> int:
    device_id, sample_rate, device_name = resolve_device(device)
    print(f"Microphone: {device_name} ({sample_rate:.0f} Hz)")
    threshold = calibrate_noise(config, device_id, sample_rate)
    transcriber, listener, collector = create_transcriber(config)

    try:
        while not STOP_REQUESTED.is_set():
            result = capture_utterance(
                config,
                transcriber,
                collector,
                device_id,
                sample_rate,
                threshold,
                prompt_before_arm=False,
            )
            if result is not None:
                print_result(result)
            if once:
                break
            if not STOP_REQUESTED.is_set():
                print("\nRe-armed for the next utterance. Press Ctrl+C to stop.")
    finally:
        if hasattr(transcriber, "close"):
            transcriber.close()
    return 0


def load_labels(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def run_benchmark(config: AppConfig, device: Optional[int], labels_path: Path) -> int:
    labels = load_labels(labels_path)
    if not labels:
        raise ValueError(f"No labels found in {labels_path}.")
    device_id, sample_rate, device_name = resolve_device(device)
    print(f"Microphone: {device_name} ({sample_rate:.0f} Hz)")
    threshold = calibrate_noise(config, device_id, sample_rate)
    transcriber, listener, collector = create_transcriber(config)

    standard_total = ErrorCounts(0, 0, 0, 0)
    command_total = ErrorCounts(0, 0, 0, 0)

    def add_counts(a: ErrorCounts, b: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            a.substitutions + b.substitutions,
            a.deletions + b.deletions,
            a.insertions + b.insertions,
            a.reference_words + b.reference_words,
        )

    try:
        for index, reference in enumerate(labels, start=1):
            if STOP_REQUESTED.is_set():
                break
            print(f"\nLABEL {index}/{len(labels)}\n{reference}")
            result = capture_utterance(
                config,
                transcriber,
                collector,
                device_id,
                sample_rate,
                threshold,
                prompt_before_arm=True,
            )
            if result is None:
                continue
            print_result(result)
            standard = word_error_counts(reference, result.transcript, command_normalized=False)
            command = word_error_counts(reference, result.transcript, command_normalized=True)
            standard_total = add_counts(standard_total, standard)
            command_total = add_counts(command_total, command)
            print(f"Standard WER: {standard.wer * 100:.2f}%")
            print(f"Command-normalized WER: {command.wer * 100:.2f}%")
            print(
                f"Cumulative: standard={standard_total.wer * 100:.2f}% | "
                f"command-normalized={command_total.wer * 100:.2f}%"
            )
    finally:
        if hasattr(transcriber, "close"):
            transcriber.close()
    return 0


def verify_package_version(config: AppConfig) -> None:
    try:
        installed = importlib.metadata.version("moonshine-voice")
    except importlib.metadata.PackageNotFoundError:
        return
    if installed != config.tested_moonshine_voice_version:
        print(
            f"WARNING: measured results used moonshine-voice "
            f"{config.tested_moonshine_voice_version}, but {installed} is installed. "
            "Re-run the benchmark after any library upgrade.",
            file=sys.stderr,
        )


def self_test(config: AppConfig) -> int:
    config.validate()
    assert normalize_standard("Hello, WORLD!") == "hello world"
    assert normalize_command("twenty two percent") == "22 percent"
    assert normalize_command("Don't send ten messages.") == "do not send 10 messages"
    exact = word_error_counts("turn on the light", "Turn on the light.", command_normalized=False)
    assert exact.wer == 0.0
    number_equivalent = word_error_counts(
        "set it to twenty two degrees",
        "set it to 22 degrees",
        command_normalized=True,
    )
    assert number_equivalent.wer == 0.0
    one_error = word_error_counts("turn on light", "turn off light", command_normalized=False)
    assert one_error.errors == 1 and one_error.reference_words == 3
    print("SELF-TEST PASSED")
    print(json.dumps(config.__dict__, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moonshine Medium true-live STT for Raspberry Pi.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=int, default=None, help="sounddevice input index")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    subparsers = parser.add_subparsers(dest="mode")
    live = subparsers.add_parser("live", help="continuous local live transcription")
    live.add_argument("--once", action="store_true", help="stop after one utterance")
    bench = subparsers.add_parser("benchmark", help="prompted WER benchmark")
    bench.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmark_labels.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AppConfig.from_json(args.config)
    config.validate()

    if args.print_config:
        print(json.dumps(config.__dict__, indent=2))
        return 0
    if args.self_test:
        return self_test(config)

    # Lazy audio imports allow --self-test and --print-config to run before installation.
    try:
        import numpy  # noqa: F401
        import sounddevice  # noqa: F401
        import moonshine_voice  # noqa: F401
    except ImportError as exc:
        print(f"Missing runtime dependency: {exc}. Run ./install.sh first.", file=sys.stderr)
        return 2

    verify_package_version(config)
    if args.list_devices:
        list_input_devices()
        return 0
    if args.mode == "benchmark":
        return run_benchmark(config, args.device, args.labels)
    return run_live(config, args.device, getattr(args, "once", False))


if __name__ == "__main__":
    raise SystemExit(main())

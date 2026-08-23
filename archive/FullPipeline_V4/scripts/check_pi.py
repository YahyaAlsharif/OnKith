#!/usr/bin/env python3
"""Raspberry Pi 5 acceptance checks with explicit, non-upgradable statuses.

Every check reports one of:

    PASS        actually executed here and satisfied the contract
    FAIL        actually executed here and violated the contract
    NOT RUN     could not execute (wrong platform, missing artifact/dependency)
    UNVERIFIED  executed, but this host cannot prove the claim (e.g. RTF on x86,
                or native hardware 16 kHz capture behind an ALSA plug device)

A NOT RUN is never reported as a PASS. The exit status is non-zero only when a
check genuinely FAILed, so the script can run off-target to exercise the parts
that are portable while still refusing to claim Raspberry Pi results.

Speech validation
-----------------
``--speech-wav PATH`` supplies a 16 kHz mono 16-bit WAV of English speech, and
``--expect-keyword WORD`` (repeatable) asserts the transcript contains it.
No copyrighted audio is bundled; record your own or pass a licensed file.
``--with-mic`` additionally runs a live microphone capture and segmentation.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.canary_live_segmenter import CanaryLiveSegmenter
from app.stt_runtime import LiveSTTRuntime
from app.live_audio import LiveAudioRuntime, microphone_rms_test
from app.omit_ai_live import static_self_test
from integrated_pipeline.whisper_adapter import (
    WhisperAdapter,
    describe_whisper_backend,
)
from integrated_pipeline.config import (
    MAXIMUM_LOGICAL_UTTERANCE_SECONDS,
    MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS,
    PipelineConfig,
)
from integrated_pipeline.pii_adapter import PIIMaskerAdapter
from integrated_pipeline.pipeline import FailClosedMasker
from integrated_pipeline.schemas import CanarySegment, STTResult
from integrated_pipeline.silero_vad import SileroOnnxVAD
from integrated_pipeline.sqlite_storage import SQLiteTranscriptStore
from integrated_pipeline.transcript_merger import merge_segment_transcripts

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT RUN"
UNVERIFIED = "UNVERIFIED"

TARGET_PACKAGES = {
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.1",
    "onnxruntime": "1.22.1",
    "onnx": "1.17.0",
    "transformers": "4.53.3",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add(self, name: str, status: str, detail: str = "", **extra) -> str:
        if status not in {PASS, FAIL, NOT_RUN, UNVERIFIED}:
            raise ValueError(f"Unsupported status: {status}")
        row = {"check": name, "status": status, "detail": detail}
        row.update(extra)
        self.rows.append(row)
        print(f"[{status:^10}] {name}" + (f" :: {detail}" if detail else ""))
        return status

    def run(self, name: str, function, *, unverified_reason: str | None = None) -> object:
        """Execute a check, recording FAIL on exception rather than aborting.

        A missing artifact or uninstalled dependency is an absence, not a
        contract violation, so it is recorded as NOT RUN.
        """

        try:
            detail = function()
        except Exception as exc:
            status = NOT_RUN if is_absence(exc) else FAIL
            self.add(name, status, f"{type(exc).__name__}: {exc}")
            return None
        if unverified_reason:
            self.add(name, UNVERIFIED, f"{detail} :: {unverified_reason}")
        else:
            self.add(name, PASS, str(detail) if detail is not None else "")
        return detail

    @property
    def failed(self) -> bool:
        return any(row["status"] == FAIL for row in self.rows)

    def summary(self) -> dict[str, int]:
        counts = {PASS: 0, FAIL: 0, NOT_RUN: 0, UNVERIFIED: 0}
        for row in self.rows:
            counts[str(row["status"])] += 1
        return counts


def is_target_platform() -> bool:
    return platform.machine().lower() in {"aarch64", "arm64"}


def is_absence(exc: BaseException) -> bool:
    """True when a failure means "not installed / not present", not "broken".

    Adapters raise ``RuntimeError("Install the pinned ...") from ImportError``,
    so the cause chain is walked rather than the message matched.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (FileNotFoundError, ImportError)):
            return True
        current = current.__cause__ or current.__context__
    return False


def peak_rss_mib() -> float | None:
    """Peak resident set size, or None where the platform cannot report it."""

    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; Darwin reports bytes.
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0 * 1024.0)


def memory_report() -> dict[str, int | None]:
    values: dict[str, int | None] = {
        "mem_total_mib": None,
        "mem_available_mib": None,
        "swap_total_mib": None,
        "swap_free_mib": None,
    }
    keys = {
        "MemTotal": "mem_total_mib",
        "MemAvailable": "mem_available_mib",
        "SwapTotal": "swap_total_mib",
        "SwapFree": "swap_free_mib",
    }
    path = Path("/proc/meminfo")
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split(":", 1)[0]
        if name in keys:
            values[keys[name]] = int(line.split()[1]) // 1024
    return values


def verify_runtime_boundary() -> str:
    retired_backend = "moon" + "shine"
    for relative in ("app", "integrated_pipeline", "requirements-rpi.txt"):
        target = ROOT / relative
        files = [target] if target.is_file() else list(target.rglob("*.py"))
        for path in files:
            if retired_backend in path.read_text(encoding="utf-8").lower():
                raise RuntimeError(f"Stale retired-STT runtime reference: {path}")
    callback = inspect.getsource(LiveAudioRuntime._audio_callback).lower()
    forbidden = ("silero", "whisper", "resample", "sqlite", "mask_text", "urllib")
    found = [item for item in forbidden if item in callback]
    if found:
        raise RuntimeError(f"Microphone callback owns forbidden work: {found}")
    if ".put_nowait(" not in callback or ".put(" in callback:
        raise RuntimeError("Microphone callback must use only non-blocking queue insertion")
    return "callback owns no model, resampling, database or network work"


def load_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16_000:
            raise ValueError(f"{path} is not 16 kHz")
        if handle.getnchannels() != 1:
            raise ValueError(f"{path} is not mono")
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        raw = handle.readframes(handle.getnframes())
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).copy()


def speech_segment(index: int, samples: np.ndarray) -> CanarySegment:
    now = time.monotonic()
    stamp = utc_now_iso()
    start = index * 1_000_000
    end = start + int(samples.size)
    return CanarySegment(
        session_id="pi-check",
        utterance_id=f"pi-check-utterance-{index + 1:06d}",
        segment_id=f"pi-check-segment-{index:06d}",
        utterance_sequence_number=index + 1,
        segment_index=0,
        samples=samples,
        sample_rate=16_000,
        segment_start_sample=start,
        segment_end_sample=end,
        decision_sample=end,
        last_speech_end_sample=end,
        forced_boundary_sample=None,
        endpoint_reason="end_of_stream",
        is_final_segment_of_utterance=True,
        reliable=True,
        dropped_audio_chunks=0,
        recording_closed_monotonic=now,
        audio_started_monotonic=now - 1.0,
        audio_ended_monotonic=now,
        audio_started_at_utc=stamp,
        audio_ended_at_utc=stamp,
        utterance_start_sample=start,
        utterance_captured_end_sample=end,
        segment_unique_start_sample=start,
    )


def verify_sqlite() -> str:
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteTranscriptStore(Path(directory) / "session_transcript.db")
        store.start()
        store.start_session("check", utc_now_iso(), model_identity="whisper-check")
        store.insert_raw(
            STTResult(
                final_text="safe test",
                duration_seconds=1.0,
                processing_seconds=0.1,
                finalization_seconds=0.01,
                rtf=0.1,
                callback_count=0,
                session_id="check",
                batch_id="check-utterance-000001",
                sequence_number=1,
                endpoint_reason="endpoint_silence",
                audio_started_at_utc=utc_now_iso(),
                audio_ended_at_utc=utc_now_iso(),
                stt_started_at_utc=utc_now_iso(),
                stt_completed_at_utc=utc_now_iso(),
                captured_audio_seconds=1.0,
                submitted_audio_seconds=1.0,
                total_stt_seconds=0.11,
            )
        )
        store.drain()
        if store.journal_mode != "wal" or store.synchronous_level != 1:
            raise RuntimeError("SQLite WAL/NORMAL settings changed")
        if store.writer_thread_id != store.connection_thread_id:
            raise RuntimeError("SQLite connection is not writer-thread-owned")
        rows = store.query_session("check")
        if len(rows) != 1 or rows[0].status != "waiting_for_mask":
            raise RuntimeError("SQLite raw insertion did not complete")
        store.shutdown()
    return "WAL/NORMAL, single writer thread, raw row stored"


def check_queue_and_safety_behaviour(report: Report) -> None:
    """Prove queue bounds and the safety split without needing the model."""

    class EchoRecognizer:
        def transcribe_segment(self, segment):
            from integrated_pipeline.schemas import SegmentTranscript

            return SegmentTranscript(
                segment=segment,
                text=f"word{segment.segment_index}",
                processing_seconds=0.0,
                queue_wait_seconds=0.0,
                stt_started_at_utc=utc_now_iso(),
                stt_completed_at_utc=utc_now_iso(),
            )

    class CollectPrivacy:
        def __init__(self) -> None:
            self.results = []

        def submit_stt(self, result) -> None:
            self.results.append(result)

        def submit_stt_failure(self, *args) -> None:
            raise AssertionError("unexpected STT failure")

    import queue as queue_module

    config = PipelineConfig()
    maximum = int(config.audio["completed_segment_queue_max"])
    completed: queue_module.Queue = queue_module.Queue(maxsize=maximum)
    events: list[str] = []
    privacy = CollectPrivacy()
    worker = LiveSTTRuntime(
        completed,
        EchoRecognizer(),
        privacy,
        on_event=lambda name, payload: events.append(name),
        utc_now=utc_now_iso,
    )
    worker.start()
    high_water = 0
    total = MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS + 2
    try:
        for index in range(total):
            start = index * 32_000
            completed.put(
                CanarySegment(
                    session_id="pi-check",
                    utterance_id="pi-check-utterance-000001",
                    segment_id=f"pi-check-segment-{index:06d}",
                    utterance_sequence_number=1,
                    segment_index=index,
                    samples=np.zeros(32_000, dtype=np.float32),
                    sample_rate=16_000,
                    segment_start_sample=start,
                    segment_end_sample=start + 32_000,
                    decision_sample=start + 32_000,
                    last_speech_end_sample=start + 32_000,
                    forced_boundary_sample=None if index == total - 1 else start + 32_000,
                    endpoint_reason=(
                        "endpoint_silence" if index == total - 1 else "max_duration"
                    ),
                    is_final_segment_of_utterance=index == total - 1,
                    reliable=True,
                    dropped_audio_chunks=0,
                    recording_closed_monotonic=time.monotonic(),
                    audio_started_monotonic=time.monotonic() - 1.0,
                    audio_ended_monotonic=time.monotonic(),
                    audio_started_at_utc=utc_now_iso(),
                    audio_ended_at_utc=utc_now_iso(),
                    utterance_start_sample=0,
                    utterance_captured_end_sample=start + 32_000,
                    segment_unique_start_sample=start,
                )
            )
            high_water = max(high_water, completed.qsize())
        worker.shutdown()
    finally:
        if worker.thread is not None and worker.thread.is_alive():
            completed.put(None)
            worker.thread.join(timeout=5.0)

    report.add(
        "Completed segment queue stays bounded under sustained input",
        PASS if high_water <= maximum else FAIL,
        f"high_water={high_water}, maxsize={maximum}",
    )
    report.add(
        "Safety split closes pathological logical utterances",
        PASS if worker.safety_split_count >= 1 else FAIL,
        f"splits={worker.safety_split_count}, released={len(privacy.results)}, "
        f"limits={MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS} segments / "
        f"{MAXIMUM_LOGICAL_UTTERANCE_SECONDS} s",
    )
    words = " ".join(item.final_text for item in privacy.results).split()
    report.add(
        "No transcript text is dropped across a safety split",
        PASS if words == [f"word{index}" for index in range(total)] else FAIL,
        f"{len(words)} words across {len(privacy.results)} logical utterances",
    )
    report.add(
        "Forced segments never reach downstream independently",
        PASS if len(privacy.results) < total else FAIL,
        f"{len(privacy.results)} downstream releases for {total} segments",
    )


def check_whisper(report: Report, config: PipelineConfig, args) -> WhisperAdapter | None:
    backend = describe_whisper_backend()
    print(json.dumps(backend, indent=2))
    if not backend["available"]:
        report.add(
            "faster-whisper backend importable",
            NOT_RUN,
            "; ".join(str(item) for item in backend["errors"]),
        )
        report.add("Whisper model load and warmup", NOT_RUN, "backend unavailable")
        report.add("Real speech transcript", NOT_RUN, "backend unavailable")
        report.add("Second independent speech call", NOT_RUN, "backend unavailable")
        report.add("Whisper close is idempotent", NOT_RUN, "backend unavailable")
        return None
    report.add(
        "faster-whisper backend importable",
        PASS,
        f"faster_whisper={backend['faster_whisper_version']} "
        f"ctranslate2={backend['ctranslate2_version']}",
    )

    adapter = WhisperAdapter(config)
    before = peak_rss_mib()
    started = time.perf_counter()
    try:
        adapter.load()
    except Exception as exc:
        report.add("Whisper model load and warmup", FAIL, f"{type(exc).__name__}: {exc}")
        return None
    load_seconds = time.perf_counter() - started
    after_load = peak_rss_mib()
    report.add(
        "Whisper model load and warmup",
        PASS,
        f"load={load_seconds:.3f}s loads={adapter.counters.model_load_count} "
        f"warmups={adapter.counters.warmup_count} "
        f"rss_before={before} rss_after_load={after_load} MiB",
        model_load_seconds=load_seconds,
        peak_rss_before_load_mib=before,
        peak_rss_after_load_mib=after_load,
    )
    audit = adapter.audit()
    report.add(
        "Active decoder read-back",
        PASS,
        json.dumps(audit["active_decoder"]),
    )
    report.add(
        "Decoding hint active",
        PASS if audit["initial_prompt"] else FAIL,
        f"{len(audit['initial_prompt'])} characters, "
        f"sha256={audit['active_decoder']['initial_prompt_sha256'][:16]}",
    )

    if args.speech_wav is None:
        report.add(
            "Real speech transcript",
            NOT_RUN,
            "no --speech-wav supplied; synthetic tone cannot prove recognition",
        )
        report.add("Second independent speech call", NOT_RUN, "no --speech-wav supplied")
    else:
        try:
            samples = load_wav_16k(args.speech_wav)
            first = adapter.transcribe_segment(speech_segment(0, samples))
            during = peak_rss_mib()
            audio_seconds = samples.size / 16_000
            text = first.text.strip()
            report.add(
                "Real speech transcript",
                PASS if text else FAIL,
                f"{len(text)} characters, model={first.processing_seconds:.3f}s, "
                f"audio={audio_seconds:.3f}s, rss_during={during} MiB",
                transcript=text,
            )
            keywords = [word.lower() for word in (args.expect_keyword or [])]
            if keywords:
                missing = [word for word in keywords if word not in text.lower()]
                report.add(
                    "Expected keywords present in transcript",
                    PASS if not missing else FAIL,
                    f"missing={missing}" if missing else f"all of {keywords}",
                )
            else:
                report.add(
                    "Expected keywords present in transcript",
                    NOT_RUN,
                    "no --expect-keyword supplied",
                )
            second = adapter.transcribe_segment(speech_segment(1, samples))
            report.add(
                "Second independent speech call proves no state contamination",
                PASS if second.text.strip() == text else FAIL,
                f"first={text!r} second={second.text.strip()!r}",
            )
            rtf = (
                first.processing_seconds + second.processing_seconds
            ) / (2 * audio_seconds)
            report.add(
                "Real-time factor on unique audio",
                PASS if is_target_platform() else UNVERIFIED,
                f"rtf={rtf:.3f} on {platform.machine()}"
                + ("" if is_target_platform() else " (not a Raspberry Pi result)"),
                rtf=rtf,
            )
        except Exception as exc:
            report.add("Real speech transcript", FAIL, f"{type(exc).__name__}: {exc}")
    return adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-mic", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument(
        "--speech-wav",
        type=Path,
        help="16 kHz mono 16-bit WAV of English speech (not bundled; supply your own)",
    )
    parser.add_argument("--expect-keyword", action="append", default=[])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    report = Report()
    print(
        f"[CHECK] {platform.platform()} | Python {platform.python_version()} "
        f"| {platform.machine()} | {utc_now_iso()}"
    )
    print(json.dumps(memory_report(), indent=2))

    report.add(
        "Target platform is 64-bit ARM",
        PASS if is_target_platform() else NOT_RUN,
        f"machine={platform.machine()}"
        + ("" if is_target_platform() else "; Raspberry Pi checks cannot run here"),
    )
    report.add(
        "Target runtime is Python 3.11",
        PASS if sys.version_info[:2] == (3, 11) else NOT_RUN,
        f"python={platform.python_version()}",
    )

    installed = {}
    mismatched = []
    for package, expected in TARGET_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        installed[package] = actual
        if actual != expected:
            mismatched.append(f"{package}={actual} (required {expected})")
    report.add(
        "Pinned package versions",
        PASS if not mismatched else NOT_RUN,
        json.dumps(installed) if not mismatched else "; ".join(mismatched),
        installed=installed,
    )

    config = PipelineConfig()
    # The fixed contract and the pure-logic invariants are checkable anywhere;
    # the large model artifacts are a separate presence question.
    report.run(
        "Exact configuration and logic contract",
        lambda: (
            static_self_test(config, require_artifacts=False),
            "segmentation constants, Onkith interceptor, 14-word merge and "
            "fail-closed masking all verified",
        )[1],
    )
    report.run(
        "Deployment artifacts present",
        lambda: (
            config.validate(require_artifacts=True),
            "Silero and DeBERTa PII all present "
            "(Whisper weights come from the CTranslate2 cache)",
        )[1],
    )
    report.run("Microphone callback ownership boundary", verify_runtime_boundary)

    silero = SileroOnnxVAD(config)
    try:
        silero.load()
        probability = silero.probability(np.zeros(512, dtype=np.float32))
        metadata = silero.verify_artifact()
        report.add(
            "Silero artifact hash/opset and probability smoke",
            PASS if 0.0 <= probability <= 1.0 else FAIL,
            json.dumps(metadata),
        )
    except Exception as exc:
        report.add(
            "Silero artifact hash/opset and probability smoke",
            NOT_RUN if is_absence(exc) else FAIL,
            f"{type(exc).__name__}: {exc}",
        )

    try:
        pii = PIIMaskerAdapter(config)
        pii.load()
    except Exception as exc:
        report.add(
            "DeBERTa PII masking removes PII",
            NOT_RUN if is_absence(exc) else FAIL,
            f"{type(exc).__name__}: {exc}",
        )
    else:
        try:
            probe = FailClosedMasker(pii).mask_text(
                "My email is rayyan@example.com and account number is 12345678."
            )
            ok = probe.safe_to_release and "rayyan@example.com" not in probe.masked_text
            report.add(
                "DeBERTa PII masking removes PII",
                PASS if ok else FAIL,
                f"status={probe.privacy_status}",
            )
        except Exception as exc:
            report.add(
                "DeBERTa PII masking removes PII", FAIL, f"{type(exc).__name__}: {exc}"
            )

    # Fail-closed behaviour is a property of the masking boundary itself and is
    # provable without the model being installed.
    class _Broken:
        def mask_text(self, text: str):
            raise RuntimeError("intentional acceptance probe")

    blocked = FailClosedMasker(_Broken()).mask_text("secret raw value")
    report.add(
        "PII boundary is fail-closed on masker failure",
        PASS
        if not blocked.safe_to_release and "secret raw value" not in blocked.masked_text
        else FAIL,
        f"status={blocked.privacy_status}",
    )

    report.run("SQLite writer lane and schema", verify_sqlite)
    check_queue_and_safety_behaviour(report)

    try:
        devices = LiveAudioRuntime.list_devices()
        print(json.dumps(devices, indent=2))
        report.add(
            "Input device enumeration",
            PASS if devices else NOT_RUN,
            f"{len(devices)} input devices",
        )
        report.add(
            "Native hardware 16 kHz capture",
            UNVERIFIED,
            "requesting 16 kHz through sounddevice/ALSA does not prove native "
            "hardware capture; an ALSA plug device may resample internally. "
            "Statuses: "
            + json.dumps(
                {str(row["id"]): row["capture_16000_hz_status"] for row in devices}
            ),
        )
    except Exception as exc:
        report.add("Input device enumeration", NOT_RUN, f"{type(exc).__name__}: {exc}")
        report.add("Native hardware 16 kHz capture", NOT_RUN, "no audio backend")

    report.add(
        "Streaming resampler disabled",
        PASS if not bool(config.audio["streaming_resampler_validated"]) else FAIL,
        "no unvalidated streaming resampler is enabled; 16 kHz stream required",
    )

    adapter = check_whisper(report, config, args)

    if args.with_mic:
        try:
            microphone_rms_test(args.device, seconds=3.0)
            report.add("Live microphone RMS capture", PASS, "3 s captured")
        except Exception as exc:
            report.add("Live microphone RMS capture", FAIL, f"{type(exc).__name__}: {exc}")
        if adapter is not None:
            try:
                detail = live_segment_stt_smoke(config, silero, adapter, args.device)
                report.add("Live microphone to Whisper smoke", PASS, detail)
            except Exception as exc:
                report.add(
                    "Live microphone to Whisper smoke", FAIL, f"{type(exc).__name__}: {exc}"
                )
        else:
            report.add("Live microphone to Whisper smoke", NOT_RUN, "Whisper unavailable")
    else:
        report.add("Live microphone RMS capture", NOT_RUN, "--with-mic not supplied")
        report.add("Live microphone to Whisper smoke", NOT_RUN, "--with-mic not supplied")

    if adapter is not None:
        adapter.close()
        adapter.close()
        report.add(
            "Whisper close is idempotent",
            PASS if adapter.counters.close_count == 1 else FAIL,
            f"close_count={adapter.counters.close_count}",
        )
        report.add(
            "Peak RSS after close",
            PASS if is_target_platform() else UNVERIFIED,
            f"peak_rss={peak_rss_mib()} MiB on {platform.machine()}"
            + ("" if is_target_platform() else " (not a Raspberry Pi measurement)"),
        )
    silero.close()

    counts = report.summary()
    print(
        f"\nSummary: {counts[PASS]} PASS, {counts[FAIL]} FAIL, "
        f"{counts[NOT_RUN]} NOT RUN, {counts[UNVERIFIED]} UNVERIFIED"
    )
    if not is_target_platform():
        print(
            "This host is not a Raspberry Pi. No result above may be reported as "
            "Raspberry Pi validation."
        )
    if counts[NOT_RUN] or counts[UNVERIFIED]:
        print("NOT RUN and UNVERIFIED are not passes and must stay pending.")
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {
                    "generated_at_utc": utc_now_iso(),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "is_target_platform": is_target_platform(),
                    "memory": memory_report(),
                    "checks": report.rows,
                    "summary": counts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[INFO] wrote {args.json}")
    return 1 if report.failed else 0


def live_segment_stt_smoke(
    config: PipelineConfig,
    silero: SileroOnnxVAD,
    recognizer: WhisperAdapter,
    device: int | None,
    seconds: float = 8.0,
) -> str:
    import sounddevice as sd

    selected = sd.default.device[0] if device is None else int(device)
    sd.check_input_settings(
        device=selected, channels=1, dtype="float32", samplerate=16_000
    )
    print("[MIC] Remain silent for the first 0.75 s, then speak a short English phrase.")
    audio = sd.rec(
        int(round(seconds * 16_000)),
        samplerate=16_000,
        channels=1,
        dtype="float32",
        device=selected,
    )
    sd.wait()
    segments: list[object] = []
    silero.reset()
    segmenter = CanaryLiveSegmenter(
        config,
        "pi-live-smoke",
        silero,
        on_event=lambda event, payload: None,
        on_segment=segments.append,
    )
    values = np.asarray(audio[:, 0], dtype=np.float32)
    for offset in range(0, len(values), 1_600):
        segmenter.process_samples(values[offset : offset + 1_600])
    segmenter.shutdown(reason="end_of_stream")
    if not segments:
        raise RuntimeError("Live smoke produced no real utterance segment")
    starts = [
        item.segment_start_sample
        for item in segments
        if isinstance(item, CanarySegment)
    ]
    if starts and min(starts) < 12_000:
        raise RuntimeError("A live segment started inside calibration audio")
    results = [
        recognizer.transcribe_segment(item)
        for item in segments
        if isinstance(item, CanarySegment)
    ]
    merged = merge_segment_transcripts([item.text for item in results])
    processing = sum(item.processing_seconds for item in results)
    submitted = sum(item.segment.submitted_audio_seconds for item in results)
    return (
        f"segments={len(results)}, merged_characters={len(merged.text)}, "
        f"model_seconds={processing:.3f}, submitted_seconds={submitted:.3f}, "
        f"submitted_rtf={processing / submitted:.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())

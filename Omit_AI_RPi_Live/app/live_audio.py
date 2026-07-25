"""Continuous microphone capture and utterance segmentation."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.schemas import STTResult


@dataclass(frozen=True)
class AudioChunk:
    samples: Any
    sample_rate: int
    captured_at: float
    rms: float
    peak: float
    status: str = ""

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    @property
    def end_time(self) -> float:
        return self.captured_at + self.duration


def rms_and_peak(samples: Any) -> tuple[float, float]:
    import numpy as np

    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0
    return (
        float(np.sqrt(np.mean(values * values) + 1e-12)),
        float(np.max(np.abs(values))),
    )


class LiveSegmentationStateMachine:
    """Model-independent calibration, pre-roll, and endpoint state."""

    def __init__(
        self,
        config: PipelineConfig,
        moonshine: Any,
        *,
        on_event: Callable[[str, dict[str, Any]], None],
        on_final: Callable[[STTResult], None],
        on_partial: Callable[[str], None] | None = None,
    ):
        self.settings = config.audio
        self.moonshine = moonshine
        self.on_event = on_event
        self.on_final = on_final
        self.on_partial = on_partial
        self.state = "calibrating"
        self.calibration_rms: list[float] = []
        self.calibration_duration = 0.0
        self.start_threshold: float | None = None
        self.pre_roll: deque[AudioChunk] = deque()
        self.pre_roll_duration = 0.0
        self.start_candidate_duration = 0.0
        self.active_session: Any | None = None
        self.utterance_started_at: float | None = None
        self.last_voice_at: float | None = None
        self.unreliable = False
        self.pending_unreliable = False
        self.final_count = 0

    def _append_pre_roll(self, chunk: AudioChunk) -> None:
        self.pre_roll.append(chunk)
        self.pre_roll_duration += chunk.duration
        target = float(self.settings["pre_roll_seconds"])
        while self.pre_roll and self.pre_roll_duration - self.pre_roll[0].duration >= target:
            removed = self.pre_roll.popleft()
            self.pre_roll_duration -= removed.duration

    def _finish_calibration(self) -> None:
        import numpy as np

        if not self.calibration_rms:
            raise RuntimeError("No microphone chunks arrived during calibration")
        noise_rms = float(np.percentile(self.calibration_rms, 90))
        self.start_threshold = max(
            float(self.settings["minimum_start_rms"]),
            noise_rms * float(self.settings["noise_multiplier"]),
        )
        self.state = "idle"
        self.on_event(
            "calibration_complete",
            {"noise_rms": noise_rms, "start_threshold": self.start_threshold},
        )

    def _start_utterance(self, chunk: AudioChunk) -> None:
        assert self.start_threshold is not None
        self.on_event("speech_started", {"rms": chunk.rms})
        self.active_session = self.moonshine.open_utterance(self.on_partial)
        queued = list(self.pre_roll)
        self.pre_roll.clear()
        self.pre_roll_duration = 0.0
        self.utterance_started_at = (
            queued[0].captured_at if queued else chunk.captured_at
        )
        self.last_voice_at = chunk.end_time
        self.unreliable = self.pending_unreliable
        self.pending_unreliable = False
        for item in queued:
            self.active_session.add_audio(item.samples, item.sample_rate)
        self.state = "speaking"
        self.start_candidate_duration = 0.0

    def _finalize(self, reason: str) -> None:
        session = self.active_session
        if session is None:
            return
        self.state = "finalizing"
        self.on_event("speech_endpoint", {"reason": reason})
        try:
            result = session.finalize(reliable=not self.unreliable)
        except Exception as exc:
            self.on_event(
                "utterance_error",
                {"error": f"{type(exc).__name__}: {exc}"},
            )
        else:
            self.final_count += 1
            self.on_event(
                "stt_finalized",
                {
                    "duration_seconds": result.duration_seconds,
                    "processing_seconds": result.processing_seconds,
                    "finalization_seconds": result.finalization_seconds,
                    "rtf": result.rtf,
                    "callback_count": result.callback_count,
                    "reliable": result.reliable,
                },
            )
            self.on_final(result)
        finally:
            self.active_session = None
            self.utterance_started_at = None
            self.last_voice_at = None
            self.unreliable = False
            self.state = "idle"

    def process(self, chunk: AudioChunk) -> None:
        if self.state == "calibrating":
            self.calibration_rms.append(chunk.rms)
            self.calibration_duration += chunk.duration
            if self.calibration_duration >= float(
                self.settings["calibration_seconds"]
            ):
                self._finish_calibration()
            return

        if self.state == "idle":
            self._append_pre_roll(chunk)
            assert self.start_threshold is not None
            if chunk.rms >= self.start_threshold:
                self.start_candidate_duration += chunk.duration
            else:
                self.start_candidate_duration = 0.0
            if self.start_candidate_duration >= float(
                self.settings["start_confirm_seconds"]
            ):
                self._start_utterance(chunk)
            return

        if self.state != "speaking" or self.active_session is None:
            return
        self.active_session.add_audio(chunk.samples, chunk.sample_rate)
        assert self.start_threshold is not None
        assert self.utterance_started_at is not None
        assert self.last_voice_at is not None
        continue_threshold = self.start_threshold * float(
            self.settings["continue_ratio"]
        )
        if chunk.rms >= continue_threshold:
            self.last_voice_at = chunk.end_time
        utterance_duration = chunk.end_time - self.utterance_started_at
        quiet_duration = chunk.end_time - self.last_voice_at
        if utterance_duration >= float(self.settings["maximum_utterance_seconds"]):
            self._finalize("maximum_utterance")
        elif (
            utterance_duration >= float(self.settings["minimum_speech_seconds"])
            and quiet_duration >= float(self.settings["endpoint_silence_seconds"])
        ):
            self._finalize("endpoint_silence")

    def mark_overflow(self) -> None:
        self.pending_unreliable = True
        if self.state in {"speaking", "finalizing"}:
            self.unreliable = True

    def shutdown(self) -> None:
        if self.active_session is not None:
            self.unreliable = True
            self._finalize("shutdown")


class LiveAudioRuntime:
    """Open one InputStream and move every expensive task to a worker."""

    def __init__(
        self,
        config: PipelineConfig,
        moonshine: Any,
        *,
        device: int | None,
        once: bool,
        on_event: Callable[[str, dict[str, Any]], None],
        on_final: Callable[[STTResult], None],
        on_partial: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.moonshine = moonshine
        self.device = device
        self.once = once
        self.on_event = on_event
        self.on_final = on_final
        self.on_partial = on_partial
        self.clock = clock
        self.audio_queue: queue.Queue[AudioChunk] = queue.Queue(
            maxsize=int(config.audio["queue_max_chunks"])
        )
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self._overflow_lock = threading.Lock()
        self._overflow_count = 0
        self._reported_overflow_count = 0
        self.device_id: int | None = None
        self.sample_rate = 0
        self.device_name = ""
        self.state_machine: LiveSegmentationStateMachine | None = None

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        import sounddevice as sd

        rows = []
        default_input = sd.default.device[0]
        for index, info in enumerate(sd.query_devices()):
            if int(info["max_input_channels"]) <= 0:
                continue
            rows.append(
                {
                    "id": index,
                    "name": str(info["name"]),
                    "channels": int(info["max_input_channels"]),
                    "sample_rate": int(round(float(info["default_samplerate"]))),
                    "default": index == default_input,
                }
            )
        return rows

    def _resolve_device(self) -> None:
        import sounddevice as sd

        selected = sd.default.device[0] if self.device is None else self.device
        if selected is None or int(selected) < 0:
            raise RuntimeError(
                "No default microphone exists. Use --list-devices and --device ID."
            )
        try:
            info = sd.query_devices(selected, "input")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot select microphone device {selected}: {exc}"
            ) from exc
        self.device_id = int(selected)
        self.sample_rate = int(round(float(info["default_samplerate"])))
        self.device_name = str(info["name"])

    def _audio_callback(
        self, indata: Any, frames: int, time_info: Any, status: Any
    ) -> None:
        del frames, time_info
        import numpy as np

        samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
        rms, peak = rms_and_peak(samples)
        chunk = AudioChunk(
            samples=samples,
            sample_rate=self.sample_rate,
            captured_at=self.clock(),
            rms=rms,
            peak=peak,
            status=str(status) if status else "",
        )
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            with self._overflow_lock:
                self._overflow_count += 1

    def _report_overflows(self) -> None:
        with self._overflow_lock:
            count = self._overflow_count
        if count <= self._reported_overflow_count:
            return
        new_count = count - self._reported_overflow_count
        self._reported_overflow_count = count
        self.moonshine.note_queue_overflow(new_count)
        assert self.state_machine is not None
        self.state_machine.mark_overflow()
        self.on_event(
            "queue_overflow",
            {
                "new_dropped_chunks": new_count,
                "total_dropped_chunks": count,
                "utterance_reliable": False,
            },
        )

    def _worker_loop(self) -> None:
        assert self.state_machine is not None
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            self._report_overflows()
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if chunk.status:
                    self.state_machine.mark_overflow()
                    self.on_event("microphone_status", {"status": chunk.status})
                self.state_machine.process(chunk)
                if self.once and self.state_machine.final_count >= 1:
                    self.stop_event.set()
            except Exception as exc:
                if self.state_machine.active_session is not None:
                    self.state_machine.active_session.abort()
                    self.state_machine.active_session = None
                    self.state_machine.state = "idle"
                self.on_event(
                    "utterance_error",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
            finally:
                self.audio_queue.task_done()

    def run(self) -> None:
        import sounddevice as sd

        self._resolve_device()
        block_size = max(
            1,
            int(
                round(
                    self.sample_rate
                    * float(self.config.moonshine["chunk_seconds"])
                )
            ),
        )
        self.state_machine = LiveSegmentationStateMachine(
            self.config,
            self.moonshine,
            on_event=self.on_event,
            on_final=self.on_final,
            on_partial=self.on_partial,
        )
        self.worker = threading.Thread(
            target=self._worker_loop,
            name="omit-audio-worker",
            daemon=True,
        )
        self.worker.start()
        self.on_event(
            "microphone_open",
            {
                "device_id": self.device_id,
                "name": self.device_name,
                "sample_rate": self.sample_rate,
            },
        )
        self.on_event(
            "calibration_started",
            {"seconds": float(self.config.audio["calibration_seconds"])},
        )
        try:
            with sd.InputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=self._audio_callback,
            ):
                while not self.stop_event.wait(0.1):
                    pass
        finally:
            self.stop_event.set()
            if self.worker is not None:
                self.worker.join()
            if self.state_machine is not None:
                self.state_machine.shutdown()

    def stop(self) -> None:
        self.stop_event.set()


def microphone_rms_test(device: int | None, seconds: float = 3.0) -> None:
    """Print live levels without saving audio."""

    import numpy as np
    import sounddevice as sd

    selected = sd.default.device[0] if device is None else int(device)
    info = sd.query_devices(selected, "input")
    sample_rate = int(round(float(info["default_samplerate"])))
    block_size = max(1, int(round(sample_rate * 0.1)))
    levels: queue.Queue[tuple[float, float]] = queue.Queue()

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
        levels.put_nowait(rms_and_peak(samples))
        if status:
            levels.put_nowait((float("nan"), float("nan")))

    print(
        f"Microphone {selected}: {info['name']} at {sample_rate} Hz "
        f"for {seconds:.1f} seconds"
    )
    deadline = time.monotonic() + seconds
    with sd.InputStream(
        device=selected,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        callback=callback,
    ):
        while time.monotonic() < deadline:
            try:
                rms, peak = levels.get(timeout=0.3)
            except queue.Empty:
                continue
            print(f"RMS={rms:.6f} peak={peak:.6f}")

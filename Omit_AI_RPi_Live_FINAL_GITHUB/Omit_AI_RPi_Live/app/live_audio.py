"""Continuous microphone capture and model-free recording segmentation."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.schemas import RecordedUtterance


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    """Calibrate, detect endpoints, and emit completed recordings only."""

    def __init__(
        self,
        config: PipelineConfig,
        session_id: str,
        *,
        on_event: Callable[[str, dict[str, Any]], None],
        on_recording: Callable[[RecordedUtterance], None],
        clock: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], str] = utc_now_iso,
    ):
        self.settings = config.audio
        self.session_id = session_id
        self.on_event = on_event
        self.on_recording = on_recording
        self.clock = clock
        self.utc_now = utc_now
        self.state = "calibrating"
        self.calibration_rms: list[float] = []
        self.calibration_duration = 0.0
        self.start_threshold: float | None = None
        self.pre_roll: deque[AudioChunk] = deque()
        self.pre_roll_duration = 0.0
        self.start_candidate_duration = 0.0
        self.active_chunks: list[AudioChunk] = []
        self.active_sample_rate: int | None = None
        self.active_frames = 0
        self.active_new_frames = 0
        self.last_voice_frame = 0
        self.active_unreliable = False
        self.active_dropped_chunks = 0
        self.pending_dropped_chunks = 0
        self.next_overlap_with_previous = False
        self.final_count = 0
        self._sequence_number = 0

    @property
    def has_active_recording(self) -> bool:
        return self.state == "speaking" and bool(self.active_chunks)

    def _append_pre_roll(self, chunk: AudioChunk) -> None:
        self.pre_roll.append(chunk)
        self.pre_roll_duration += chunk.duration
        target = float(self.settings["pre_roll_seconds"])
        while (
            self.pre_roll
            and self.pre_roll_duration - self.pre_roll[0].duration >= target
        ):
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

    def _start_recording(self, trigger: AudioChunk) -> None:
        assert self.start_threshold is not None
        queued = list(self.pre_roll)
        self.pre_roll.clear()
        self.pre_roll_duration = 0.0
        if not queued:
            queued = [trigger]
        sample_rates = {int(item.sample_rate) for item in queued}
        if len(sample_rates) != 1:
            raise ValueError("Microphone sample rate changed before speech start")
        self.active_chunks = queued
        self.active_sample_rate = sample_rates.pop()
        self.active_frames = sum(len(item.samples) for item in queued)
        self.active_new_frames = self.active_frames
        continue_threshold = self.start_threshold * float(
            self.settings["continue_ratio"]
        )
        voice_frames = 0
        cumulative = 0
        for item in queued:
            cumulative += len(item.samples)
            if item.rms >= continue_threshold:
                voice_frames = cumulative
        self.last_voice_frame = voice_frames or self.active_frames
        self.active_unreliable = self.pending_dropped_chunks > 0
        self.active_dropped_chunks = self.pending_dropped_chunks
        self.pending_dropped_chunks = 0
        self.next_overlap_with_previous = False
        self.state = "speaking"
        self.start_candidate_duration = 0.0
        self.on_event(
            "recording_started",
            {
                "rms": trigger.rms,
                "next_sequence_number": self._sequence_number + 1,
            },
        )

    def _append_active(self, chunk: AudioChunk) -> None:
        assert self.active_sample_rate is not None
        if int(chunk.sample_rate) != self.active_sample_rate:
            raise ValueError("Microphone sample rate changed during a recording")
        self.active_chunks.append(chunk)
        self.active_frames += len(chunk.samples)
        self.active_new_frames += len(chunk.samples)
        assert self.start_threshold is not None
        continue_threshold = self.start_threshold * float(
            self.settings["continue_ratio"]
        )
        if chunk.rms >= continue_threshold:
            self.last_voice_frame = self.active_frames

    def _allocate_identity(self) -> tuple[int, str]:
        self._sequence_number += 1
        sequence = self._sequence_number
        return sequence, f"{self.session_id}-batch-{sequence:06d}"

    def _close_recording(self, reason: str) -> RecordedUtterance | None:
        import numpy as np

        if not self.active_chunks or self.active_sample_rate is None:
            return None
        sample_rate = self.active_sample_rate
        captured_frames = self.active_frames
        captured_seconds = captured_frames / float(sample_rate)
        submitted_frames = captured_frames
        if reason == "endpoint_silence":
            tail_frames = int(
                round(
                    float(self.settings["trailing_silence_context_seconds"])
                    * sample_rate
                )
            )
            submitted_frames = min(
                captured_frames, self.last_voice_frame + tail_frames
            )
        samples = np.concatenate(
            [
                np.asarray(item.samples, dtype=np.float32).reshape(-1)
                for item in self.active_chunks
            ]
        )
        samples = np.ascontiguousarray(samples[:submitted_frames], dtype=np.float32)
        audio_started = self.active_chunks[0].captured_at
        audio_ended = self.active_chunks[-1].end_time
        closed_monotonic = max(self.clock(), audio_ended)
        ended_utc = self.utc_now()
        try:
            ended_datetime = datetime.fromisoformat(
                ended_utc.replace("Z", "+00:00")
            )
            started_utc = (
                ended_datetime - timedelta(seconds=captured_seconds)
            ).isoformat().replace("+00:00", "Z")
        except ValueError:
            started_utc = ended_utc
        sequence, batch_id = self._allocate_identity()
        recording = RecordedUtterance(
            session_id=self.session_id,
            batch_id=batch_id,
            sequence_number=sequence,
            samples=samples,
            sample_rate=sample_rate,
            audio_started_monotonic=audio_started,
            audio_ended_monotonic=audio_ended,
            audio_started_at_utc=started_utc,
            audio_ended_at_utc=ended_utc,
            captured_at_utc=ended_utc,
            recording_closed_monotonic=closed_monotonic,
            endpoint_reason=reason,
            reliable=not self.active_unreliable,
            dropped_audio_chunks=self.active_dropped_chunks,
            captured_audio_seconds=captured_seconds,
            overlap_with_previous=self.next_overlap_with_previous,
        )
        self.final_count += 1
        self.on_event(
            "segment_closed",
            {
                "session_id": self.session_id,
                "batch_id": batch_id,
                "sequence_number": sequence,
                "reason": reason,
                "captured_audio_seconds": captured_seconds,
                "submitted_audio_seconds": recording.submitted_audio_seconds,
                "reliable": recording.reliable,
                "dropped_audio_chunks": recording.dropped_audio_chunks,
            },
        )
        return recording

    def _reset_to_idle(self) -> None:
        self.active_chunks = []
        self.active_sample_rate = None
        self.active_frames = 0
        self.active_new_frames = 0
        self.last_voice_frame = 0
        self.active_unreliable = False
        self.active_dropped_chunks = 0
        self.next_overlap_with_previous = False
        self.state = "idle"

    def _continue_after_forced_split(self, recording: RecordedUtterance) -> None:
        overlap_seconds = float(self.settings["forced_split_overlap_seconds"])
        overlap_frames = min(
            len(recording.samples),
            int(round(overlap_seconds * recording.sample_rate)),
        )
        overlap_samples = recording.samples[-overlap_frames:] if overlap_frames else []
        overlap_started = (
            recording.audio_ended_monotonic
            - overlap_frames / float(recording.sample_rate)
        )
        rms, peak = rms_and_peak(overlap_samples)
        overlap_chunk = AudioChunk(
            samples=overlap_samples,
            sample_rate=recording.sample_rate,
            captured_at=overlap_started,
            rms=rms,
            peak=peak,
        )
        self.active_chunks = [overlap_chunk] if overlap_frames else []
        self.active_sample_rate = recording.sample_rate
        self.active_frames = overlap_frames
        self.active_new_frames = 0
        self.last_voice_frame = overlap_frames
        self.active_unreliable = not recording.reliable
        self.active_dropped_chunks = recording.dropped_audio_chunks
        self.next_overlap_with_previous = overlap_frames > 0
        self.state = "speaking"
        self.on_event(
            "forced_split_continuation",
            {
                "after_batch_id": recording.batch_id,
                "overlap_seconds": overlap_frames
                / float(recording.sample_rate),
            },
        )

    def _emit_and_transition(self, reason: str) -> None:
        recording = self._close_recording(reason)
        if recording is None:
            return
        if reason == "maximum_utterance":
            self._continue_after_forced_split(recording)
        else:
            self._reset_to_idle()
        self.on_recording(recording)

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
                self._start_recording(chunk)
            return

        if self.state != "speaking":
            return
        self._append_active(chunk)
        assert self.active_sample_rate is not None
        active_duration = self.active_frames / float(self.active_sample_rate)
        quiet_duration = (
            self.active_frames - self.last_voice_frame
        ) / float(self.active_sample_rate)
        if active_duration + 1e-9 >= float(
            self.settings["maximum_utterance_seconds"]
        ):
            self._emit_and_transition("maximum_utterance")
        elif (
            active_duration + 1e-9
            >= float(self.settings["minimum_speech_seconds"])
            and quiet_duration + 1e-9
            >= float(self.settings["endpoint_silence_seconds"])
        ):
            self._emit_and_transition("endpoint_silence")

    def mark_overflow(self, count: int = 1) -> None:
        count = max(0, int(count))
        if self.state == "speaking":
            self.active_unreliable = True
            self.active_dropped_chunks += count
        else:
            self.pending_dropped_chunks += count

    def shutdown(self) -> None:
        if self.has_active_recording and (
            not self.next_overlap_with_previous or self.active_new_frames > 0
        ):
            self.active_unreliable = True
            self._emit_and_transition("shutdown")


class LiveAudioRuntime:
    """Capture and segmentation lanes with no model or database calls."""

    def __init__(
        self,
        config: PipelineConfig,
        session_id: str,
        *,
        device: int | None,
        once: bool,
        on_event: Callable[[str, dict[str, Any]], None],
        on_level: Callable[[float], None] = lambda rms: None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.session_id = session_id
        self.device = device
        self.once = once
        self.on_event = on_event
        self.on_level = on_level
        self.clock = clock
        self.audio_chunk_queue: queue.Queue[AudioChunk] = queue.Queue(
            maxsize=int(config.audio["queue_max_chunks"])
        )
        # Compatibility name for older diagnostics.
        self.audio_queue = self.audio_chunk_queue
        self.completed_recording_queue: queue.Queue[
            RecordedUtterance | None
        ] = queue.Queue(
            maxsize=int(config.audio["completed_recording_queue_max"])
        )
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self._overflow_lock = threading.Lock()
        self._overflow_count = 0
        self._reported_overflow_count = 0
        self._pending_recordings: deque[RecordedUtterance] = deque()
        self.completed_queue_overload_count = 0
        self.audio_queue_high_water_mark = 0
        self.completed_queue_high_water_mark = 0
        self.callback_count = 0
        self.callback_total_seconds = 0.0
        self.callback_max_seconds = 0.0
        self.worker_error: str | None = None
        self.device_id: int | None = None
        self.sample_rate = 0
        self.device_name = ""
        self.state_machine = LiveSegmentationStateMachine(
            config,
            session_id,
            on_event=on_event,
            on_recording=self._accept_recording,
            clock=clock,
        )

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

    def prepare_device(self) -> None:
        if self.sample_rate:
            return
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
        callback_started = time.perf_counter()
        try:
            del frames, time_info
            import numpy as np

            samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
            rms, peak = rms_and_peak(samples)
            self.on_level(rms)
            chunk = AudioChunk(
                samples=samples,
                sample_rate=self.sample_rate,
                captured_at=self.clock(),
                rms=rms,
                peak=peak,
                status=str(status) if status else "",
            )
            try:
                self.audio_chunk_queue.put_nowait(chunk)
                self.audio_queue_high_water_mark = max(
                    self.audio_queue_high_water_mark,
                    self.audio_chunk_queue.qsize(),
                )
            except queue.Full:
                with self._overflow_lock:
                    self._overflow_count += 1
        finally:
            elapsed = time.perf_counter() - callback_started
            self.callback_count += 1
            self.callback_total_seconds += elapsed
            self.callback_max_seconds = max(self.callback_max_seconds, elapsed)

    def _report_overflows(self) -> None:
        with self._overflow_lock:
            count = self._overflow_count
        if count <= self._reported_overflow_count:
            return
        new_count = count - self._reported_overflow_count
        self._reported_overflow_count = count
        self.state_machine.mark_overflow(new_count)
        self.on_event(
            "queue_overflow",
            {
                "new_dropped_chunks": new_count,
                "total_dropped_chunks": count,
                "recording_reliable": False,
            },
        )

    def _accept_recording(self, recording: RecordedUtterance) -> None:
        if self._pending_recordings:
            self._pending_recordings.append(recording)
            return
        try:
            self.completed_recording_queue.put_nowait(recording)
            self.completed_queue_high_water_mark = max(
                self.completed_queue_high_water_mark,
                self.completed_recording_queue.qsize(),
            )
        except queue.Full:
            self._pending_recordings.append(recording)
            self.completed_queue_overload_count += 1
            self.on_event(
                "completed_queue_overload",
                {
                    "batch_id": recording.batch_id,
                    "pending_recordings": len(self._pending_recordings),
                },
            )
        if self.once and self.state_machine.final_count >= 1:
            self.stop_event.set()

    def _flush_pending_nonblocking(self) -> None:
        while self._pending_recordings:
            try:
                self.completed_recording_queue.put_nowait(
                    self._pending_recordings[0]
                )
            except queue.Full:
                return
            self._pending_recordings.popleft()
            self.completed_queue_high_water_mark = max(
                self.completed_queue_high_water_mark,
                self.completed_recording_queue.qsize(),
            )

    def _flush_pending_blocking(self) -> None:
        while self._pending_recordings:
            try:
                self.completed_recording_queue.put(
                    self._pending_recordings[0], timeout=0.1
                )
            except queue.Full:
                continue
            self._pending_recordings.popleft()
            self.completed_queue_high_water_mark = max(
                self.completed_queue_high_water_mark,
                self.completed_recording_queue.qsize(),
            )

    def _worker_loop(self) -> None:
        try:
            while not self.stop_event.is_set() or not self.audio_chunk_queue.empty():
                self._report_overflows()
                self._flush_pending_nonblocking()
                try:
                    chunk = self.audio_chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if chunk.status:
                        self.state_machine.mark_overflow()
                        self.on_event(
                            "microphone_status", {"status": chunk.status}
                        )
                    self.state_machine.process(chunk)
                except Exception as exc:
                    self.worker_error = f"{type(exc).__name__}: {exc}"
                    self.on_event(
                        "segmentation_error", {"error": self.worker_error}
                    )
                    self.stop_event.set()
                finally:
                    self.audio_chunk_queue.task_done()
            self._report_overflows()
            self.state_machine.shutdown()
            self._flush_pending_blocking()
        except Exception as exc:
            self.worker_error = f"{type(exc).__name__}: {exc}"
            self.on_event("segmentation_error", {"error": self.worker_error})
            self.stop_event.set()

    def run(self) -> None:
        import sounddevice as sd

        self.prepare_device()
        block_size = max(
            1,
            int(
                round(
                    self.sample_rate
                    * float(self.config.moonshine["chunk_seconds"])
                )
            ),
        )
        self.worker = threading.Thread(
            target=self._worker_loop,
            name="omit-segmentation-worker",
            daemon=False,
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
            # InputStream has closed before the worker is asked to finish.
            self.stop_event.set()
            if self.worker is not None:
                self.worker.join()
            if self.worker_error is not None:
                raise RuntimeError(self.worker_error)

    def stop(self) -> None:
        self.stop_event.set()

    @property
    def overflow_count(self) -> int:
        with self._overflow_lock:
            return self._overflow_count

    def audit(self) -> dict[str, Any]:
        return {
            "callback_count": self.callback_count,
            "callback_average_seconds": (
                self.callback_total_seconds / self.callback_count
                if self.callback_count
                else 0.0
            ),
            "callback_max_seconds": self.callback_max_seconds,
            "audio_queue_high_water_mark": self.audio_queue_high_water_mark,
            "completed_queue_high_water_mark": (
                self.completed_queue_high_water_mark
            ),
            "completed_queue_overload_count": (
                self.completed_queue_overload_count
            ),
            "queue_overflow_count": self.overflow_count,
            "pending_recordings": len(self._pending_recordings),
        }


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

"""Direct-16-kHz microphone callback and model-free segmentation worker."""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from integrated_pipeline.config import INPUT_CHUNK_SAMPLES, SAMPLE_RATE, PipelineConfig
from integrated_pipeline.schemas import CanarySegment, CanaryUtteranceEnd
from integrated_pipeline.silero_vad import SileroOnnxVAD

from .canary_live_segmenter import CanaryLiveSegmenter


@dataclass(frozen=True)
class AudioChunk:
    samples: Any
    sample_rate: int
    captured_at: float
    rms: float
    peak: float
    status: str = ""


def rms_and_peak(samples: Any) -> tuple[float, float]:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0
    return (
        float(np.sqrt(np.mean(values * values, dtype=np.float64) + 1e-18)),
        float(np.max(np.abs(values))),
    )


def reference_resample_poly(samples: Any, source_rate: int) -> np.ndarray:
    """Whole-array scipy reference for finite replay/tests, never per live chunk."""

    values = np.ascontiguousarray(
        np.asarray(samples, dtype=np.float32).reshape(-1), dtype=np.float32
    )
    if int(source_rate) == SAMPLE_RATE:
        return values.copy()
    if int(source_rate) <= 0:
        raise ValueError("Source sample rate must be positive")
    from scipy.signal import resample_poly

    divisor = math.gcd(SAMPLE_RATE, int(source_rate))
    output = resample_poly(
        values,
        SAMPLE_RATE // divisor,
        int(source_rate) // divisor,
    )
    return np.ascontiguousarray(output, dtype=np.float32)


class LiveAudioRuntime:
    """Keep the callback minimal and run Silero/segmentation on one worker."""

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
        vad: Any | None = None,
    ) -> None:
        self.config = config
        self.session_id = session_id
        self.device = device
        self.once = once
        self.on_event = on_event
        self.on_level = on_level
        self.clock = clock
        self.vad = vad if vad is not None else SileroOnnxVAD(config)
        self.audio_chunk_queue: queue.Queue[AudioChunk] = queue.Queue(
            maxsize=int(config.audio["queue_max_chunks"])
        )
        self.audio_queue = self.audio_chunk_queue
        self.completed_segment_queue: queue.Queue[
            CanarySegment | CanaryUtteranceEnd | None
        ] = queue.Queue(
            maxsize=int(config.audio["completed_segment_queue_max"])
        )
        # Compatibility alias for callers migrating from the old recording queue.
        self.completed_recording_queue = self.completed_segment_queue
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self._overflow_lock = threading.Lock()
        self._overflow_count = 0
        self._reported_overflow_count = 0
        self.audio_queue_high_water_mark = 0
        self.completed_queue_high_water_mark = 0
        self.callback_count = 0
        self.callback_total_seconds = 0.0
        self.callback_max_seconds = 0.0
        self.worker_error: str | None = None
        self.device_id: int | None = None
        self.sample_rate = SAMPLE_RATE
        self.device_name = ""
        self.device_default_sample_rate: int | None = None
        # Software can only ever confirm that a 16 kHz stream was *requested*
        # and accepted. Native hardware capture stays unverified until it is
        # confirmed on the physical device.
        self.capture_16000_hz_status = "not_opened"
        self.segmenter = CanaryLiveSegmenter(
            config,
            session_id,
            self.vad,
            on_event=on_event,
            on_segment=self._accept_segment,
            clock=clock,
        )
        self.state_machine = self.segmenter

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        import sounddevice as sd

        rows = []
        default_input = sd.default.device[0]
        for index, info in enumerate(sd.query_devices()):
            if int(info["max_input_channels"]) <= 0:
                continue
            try:
                sd.check_input_settings(
                    device=index,
                    channels=1,
                    dtype="float32",
                    samplerate=SAMPLE_RATE,
                )
                supports_direct = True
            except Exception:
                supports_direct = False
            rows.append(
                {
                    "id": index,
                    "name": str(info["name"]),
                    "channels": int(info["max_input_channels"]),
                    "default_sample_rate": int(round(float(info["default_samplerate"]))),
                    # The API accepted a 16 kHz request; that is not proof of
                    # native hardware capture at 16 kHz.
                    "accepts_16000_hz_request": supports_direct,
                    "capture_16000_hz_status": (
                        LiveAudioRuntime.classify_16k_capture(info)
                        if supports_direct
                        else "unavailable"
                    ),
                    "default": index == default_input,
                }
            )
        return rows

    @staticmethod
    def classify_16k_capture(info: Any) -> str:
        """Report how a 16 kHz stream is being obtained, without overclaiming.

        Accepting a 16 kHz request through sounddevice/ALSA does not prove the
        hardware samples at 16 kHz: an ALSA ``plug`` device resamples silently.
        The only signal available in software is the device's own default rate
        and its name, so this returns a status, never a guarantee:

        ``native_16000_hz_reported``
            The device reports 16 kHz as its own default rate. Still only a
            report from the driver, not measured hardware behaviour.
        ``alsa_plug_conversion_likely``
            The device is an ALSA plug/dmix wrapper, which resamples.
        ``requested_16000_hz_unverified``
            The stream opened at 16 kHz but the device's default rate differs,
            so a conversion layer may exist. This is the honest default.
        """

        try:
            default_rate = int(round(float(info["default_samplerate"])))
        except (KeyError, TypeError, ValueError):
            return "requested_16000_hz_unverified"
        name = str(info.get("name", "")).lower()
        if any(token in name for token in ("plug", "dmix", "dsnoop", "default")):
            return "alsa_plug_conversion_likely"
        if default_rate == SAMPLE_RATE:
            return "native_16000_hz_reported"
        return "requested_16000_hz_unverified"

    def prepare_device(self) -> None:
        if self.device_id is not None:
            return
        import sounddevice as sd

        selected = sd.default.device[0] if self.device is None else self.device
        if selected is None or int(selected) < 0:
            raise RuntimeError(
                "No default microphone exists. Use --list-devices and --device ID."
            )
        try:
            info = sd.query_devices(selected, "input")
            sd.check_input_settings(
                device=selected,
                channels=1,
                dtype="float32",
                samplerate=SAMPLE_RATE,
            )
        except Exception as exc:
            raise RuntimeError(
                "The selected microphone cannot provide a 16,000 Hz mono float32 "
                "stream. This build has no parity-validated streaming resampler; select "
                f"a 16-kHz-capable device. Device {selected}: {exc}"
            ) from exc
        self.device_id = int(selected)
        self.device_name = str(info["name"])
        self.device_default_sample_rate = int(
            round(float(info["default_samplerate"]))
        )
        self.capture_16000_hz_status = self.classify_16k_capture(info)

    def _audio_callback(
        self, indata: Any, frames: int, time_info: Any, status: Any
    ) -> None:
        callback_started = time.perf_counter()
        try:
            del time_info
            samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
            rms, peak = rms_and_peak(samples)
            self.on_level(rms)
            chunk = AudioChunk(
                samples=samples,
                sample_rate=SAMPLE_RATE,
                captured_at=self.clock() - int(frames) / SAMPLE_RATE,
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
            total = self._overflow_count
        if total <= self._reported_overflow_count:
            return
        new_count = total - self._reported_overflow_count
        self._reported_overflow_count = total
        self.segmenter.mark_overflow(new_count)
        self.on_event(
            "queue_overflow",
            {
                "new_dropped_chunks": new_count,
                "total_dropped_chunks": total,
                "recording_reliable": False,
            },
        )

    def _accept_segment(
        self, segment: CanarySegment | CanaryUtteranceEnd
    ) -> None:
        # This worker may apply backpressure; the real-time callback never blocks.
        self.completed_segment_queue.put(segment)
        self.completed_queue_high_water_mark = max(
            self.completed_queue_high_water_mark,
            self.completed_segment_queue.qsize(),
        )
        if self.once and (
            isinstance(segment, CanaryUtteranceEnd)
            or segment.is_final_segment_of_utterance
        ):
            self.stop_event.set()

    def _worker_loop(self) -> None:
        try:
            while not self.stop_event.is_set() or not self.audio_chunk_queue.empty():
                self._report_overflows()
                try:
                    chunk = self.audio_chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if chunk.sample_rate != SAMPLE_RATE:
                        raise RuntimeError("Live capture changed away from direct 16 kHz")
                    if chunk.status:
                        self.segmenter.mark_overflow()
                        self.on_event("microphone_status", {"status": chunk.status})
                    self.segmenter.process_samples(
                        chunk.samples, captured_at=chunk.captured_at
                    )
                finally:
                    self.audio_chunk_queue.task_done()
            self._report_overflows()
            self.segmenter.shutdown(reason="shutdown")
        except Exception as exc:
            self.worker_error = f"{type(exc).__name__}: {exc}"
            self.on_event("segmentation_error", {"error": self.worker_error})
            self.stop_event.set()

    def run(self) -> None:
        import sounddevice as sd

        self.prepare_device()
        self.vad.load()
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
                "sample_rate": SAMPLE_RATE,
                "requested_16000_hz": True,
                "device_default_sample_rate": self.device_default_sample_rate,
                "capture_16000_hz_status": self.capture_16000_hz_status,
                "native_hardware_16000_hz_verified": False,
            },
        )
        self.on_event(
            "calibration_started",
            {"seconds": 0.75, "samples": 12_000, "instruction": "remain silent"},
        )
        try:
            with sd.InputStream(
                device=self.device_id,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=INPUT_CHUNK_SAMPLES,
                callback=self._audio_callback,
            ):
                while not self.stop_event.wait(0.1):
                    pass
        finally:
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
            "completed_queue_high_water_mark": self.completed_queue_high_water_mark,
            "queue_overflow_count": self.overflow_count,
            "pending_recordings": 0,
            "sample_rate": SAMPLE_RATE,
            "requested_16000_hz": True,
            "device_default_sample_rate": self.device_default_sample_rate,
            "capture_16000_hz_status": self.capture_16000_hz_status,
            "native_hardware_16000_hz_verified": False,
            "segmenter": self.segmenter.audit(),
        }


def microphone_rms_test(device: int | None, seconds: float = 3.0) -> None:
    """Print direct-16-kHz levels without saving audio."""

    import sounddevice as sd

    selected = sd.default.device[0] if device is None else int(device)
    info = sd.query_devices(selected, "input")
    try:
        sd.check_input_settings(
            device=selected,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Microphone {selected} does not support required direct 16 kHz capture: {exc}"
        ) from exc
    levels: queue.Queue[tuple[float, float]] = queue.Queue()

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        levels.put_nowait(rms_and_peak(np.asarray(indata[:, 0], dtype=np.float32)))
        if status:
            levels.put_nowait((float("nan"), float("nan")))

    print(f"Microphone {selected}: {info['name']} at 16000 Hz for {seconds:.1f} seconds")
    deadline = time.monotonic() + seconds
    with sd.InputStream(
        device=selected,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=INPUT_CHUNK_SAMPLES,
        callback=callback,
    ):
        while time.monotonic() < deadline:
            try:
                rms, peak = levels.get(timeout=0.25)
            except queue.Empty:
                continue
            print(f"rms={rms:.6f} peak={peak:.6f}")

from __future__ import annotations

import inspect
import sys
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from app.live_audio import LiveAudioRuntime
from integrated_pipeline.config import PipelineConfig


class FakeVAD:
    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def probability(self, frame):
        self.calls += 1
        return 0.0

    def reset(self):
        self.resets += 1


class CallbackOwnershipTests(unittest.TestCase):
    def runtime(self):
        vad = FakeVAD()
        result = LiveAudioRuntime(
            PipelineConfig(),
            "session",
            device=None,
            once=False,
            on_event=lambda event, payload: None,
            vad=vad,
            clock=lambda: 10.0,
        )
        return result, vad

    def test_callback_only_copies_levels_and_puts_nonblocking(self) -> None:
        runtime, vad = self.runtime()
        runtime._audio_callback(
            np.ones((1_600, 1), dtype=np.float32), 1_600, None, None
        )
        self.assertEqual(vad.calls, 0)
        chunk = runtime.audio_chunk_queue.get_nowait()
        self.assertEqual(chunk.sample_rate, 16_000)
        runtime.audio_chunk_queue.task_done()
        source = inspect.getsource(LiveAudioRuntime._audio_callback).lower()
        for forbidden in (
            "probability(",
            "transcribe",
            "resample_poly",
            "sqlite",
            "mask_text",
            "urllib",
            ".put(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("put_nowait", source)

    def test_full_queue_drops_without_blocking(self) -> None:
        runtime, _ = self.runtime()
        for _ in range(runtime.audio_chunk_queue.maxsize):
            runtime.audio_chunk_queue.put_nowait(object())
        started = time.perf_counter()
        runtime._audio_callback(
            np.ones((1_600, 1), dtype=np.float32), 1_600, None, None
        )
        self.assertLess(time.perf_counter() - started, 0.1)
        self.assertEqual(runtime.overflow_count, 1)

    def test_unsupported_direct_16000_device_fails_clearly(self) -> None:
        runtime, _ = self.runtime()
        fake = types.SimpleNamespace(
            default=types.SimpleNamespace(device=(2, -1)),
            query_devices=lambda device, kind: {
                "name": "48k-only",
                "default_samplerate": 48_000,
            },
            check_input_settings=lambda **kwargs: (_ for _ in ()).throw(
                ValueError("invalid sample rate")
            ),
        )
        with patch.dict(sys.modules, {"sounddevice": fake}):
            with self.assertRaisesRegex(RuntimeError, "no parity-validated streaming resampler"):
                runtime.prepare_device()


class CaptureHonestyTests(unittest.TestCase):
    """Requesting 16 kHz must never be reported as verified native capture."""

    def test_classification_of_capture_paths(self) -> None:
        self.assertEqual(
            LiveAudioRuntime.classify_16k_capture(
                {"name": "USB Mic", "default_samplerate": 16_000}
            ),
            "native_16000_hz_reported",
        )
        self.assertEqual(
            LiveAudioRuntime.classify_16k_capture(
                {"name": "plughw:1,0", "default_samplerate": 16_000}
            ),
            "alsa_plug_conversion_likely",
        )
        self.assertEqual(
            LiveAudioRuntime.classify_16k_capture(
                {"name": "USB Mic", "default_samplerate": 48_000}
            ),
            "requested_16000_hz_unverified",
        )
        self.assertEqual(
            LiveAudioRuntime.classify_16k_capture({"name": "broken"}),
            "requested_16000_hz_unverified",
        )

    def test_audit_never_claims_verified_native_hardware_capture(self) -> None:
        runtime = LiveAudioRuntime(
            PipelineConfig(),
            "session",
            device=None,
            once=False,
            on_event=lambda event, payload: None,
            vad=FakeVAD(),
        )
        audit = runtime.audit()
        self.assertTrue(audit["requested_16000_hz"])
        self.assertFalse(audit["native_hardware_16000_hz_verified"])
        self.assertEqual(audit["capture_16000_hz_status"], "not_opened")
        self.assertNotIn("direct_16000_hz", audit)

    def test_prepare_device_records_conversion_status(self) -> None:
        runtime = LiveAudioRuntime(
            PipelineConfig(),
            "session",
            device=None,
            once=False,
            on_event=lambda event, payload: None,
            vad=FakeVAD(),
        )
        fake = types.SimpleNamespace(
            default=types.SimpleNamespace(device=(3, -1)),
            query_devices=lambda device, kind: {
                "name": "plughw:2,0",
                "default_samplerate": 48_000,
            },
            check_input_settings=lambda **kwargs: None,
        )
        with patch.dict(sys.modules, {"sounddevice": fake}):
            runtime.prepare_device()
        # The stream opened at 16 kHz, but this is an ALSA plug device.
        self.assertEqual(
            runtime.capture_16000_hz_status, "alsa_plug_conversion_likely"
        )
        self.assertEqual(runtime.device_default_sample_rate, 48_000)
        self.assertFalse(runtime.audit()["native_hardware_16000_hz_verified"])


if __name__ == "__main__":
    unittest.main()

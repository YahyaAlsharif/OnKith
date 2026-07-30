from __future__ import annotations

import inspect
import unittest

from app.live_audio import LiveSegmentationStateMachine
from integrated_pipeline.config import PipelineConfig

from tests.helpers import chunk


class SegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PipelineConfig()
        self.events: list[tuple[str, dict]] = []
        self.recordings = []
        self.machine = LiveSegmentationStateMachine(
            self.config,
            "session-segmentation",
            on_event=lambda name, data: self.events.append((name, data)),
            on_recording=self.recordings.append,
            clock=lambda: 100.0,
            utc_now=lambda: "2026-07-29T10:00:00Z",
        )
        self.index = 0
        for _ in range(20):
            self.feed(rms=0.001)

    def feed(self, *, rms: float, value: float = 0.0) -> None:
        self.machine.process(
            chunk(self.index, rms=rms, value=value)
        )
        self.index += 1

    def start_speech(self) -> None:
        self.feed(rms=0.001, value=0.01)
        self.feed(rms=0.001, value=0.02)
        self.feed(rms=0.02, value=0.03)
        self.feed(rms=0.02, value=0.04)

    def test_segmentation_has_no_moonshine_dependency(self) -> None:
        source = inspect.getsource(LiveSegmentationStateMachine).lower()
        self.assertNotIn("moonshine", source)
        self.assertNotIn("add_audio", source)
        self.assertNotIn("transcribe", source)

    def test_shutdown_during_calibration_creates_no_phantom_recording(self):
        recordings = []
        machine = LiveSegmentationStateMachine(
            self.config,
            "calibration-session",
            on_event=lambda *args: None,
            on_recording=recordings.append,
        )
        machine.process(chunk(0, rms=0.001))
        machine.shutdown()
        self.assertEqual(recordings, [])

    def test_pre_roll_is_included_once_and_silence_endpoint_is_exact(self) -> None:
        self.start_speech()
        self.feed(rms=0.02, value=0.05)
        for _ in range(24):
            self.feed(rms=0.001)
        self.assertEqual(self.recordings, [])
        self.feed(rms=0.001)
        self.assertEqual(len(self.recordings), 1)
        result = self.recordings[0]
        self.assertEqual(result.endpoint_reason, "endpoint_silence")
        self.assertAlmostEqual(result.captured_audio_seconds, 3.0)
        self.assertAlmostEqual(result.submitted_audio_seconds, 0.85)
        # Four distinct 100 ms pre-roll blocks occur once at the front.
        self.assertAlmostEqual(float(result.samples[0]), 0.01)
        self.assertAlmostEqual(float(result.samples[10]), 0.02)
        self.assertAlmostEqual(float(result.samples[20]), 0.03)
        self.assertAlmostEqual(float(result.samples[30]), 0.04)

    def test_continuous_speech_forces_maximum_split_and_continues(self) -> None:
        self.start_speech()
        while not self.recordings:
            self.feed(rms=0.02, value=0.2)
        first = self.recordings[0]
        self.assertEqual(first.endpoint_reason, "maximum_utterance")
        self.assertAlmostEqual(first.captured_audio_seconds, 8.5)
        self.assertEqual(self.machine.state, "speaking")
        self.assertAlmostEqual(
            self.machine.active_frames / self.machine.active_sample_rate,
            0.35,
        )
        self.feed(rms=0.02, value=0.3)
        self.machine.shutdown()
        self.assertEqual(len(self.recordings), 2)
        second = self.recordings[1]
        self.assertEqual(second.endpoint_reason, "shutdown")
        self.assertTrue(second.overlap_with_previous)
        self.assertGreater(second.submitted_audio_seconds, 0.35)

    def test_shutdown_preserves_pending_active_recording(self) -> None:
        self.start_speech()
        self.feed(rms=0.02, value=0.2)
        self.machine.shutdown()
        self.assertEqual(len(self.recordings), 1)
        self.assertEqual(self.recordings[0].endpoint_reason, "shutdown")

    def test_overflow_marks_recording_unreliable_and_counts_chunks(self) -> None:
        self.start_speech()
        self.machine.mark_overflow(3)
        self.feed(rms=0.02, value=0.2)
        self.machine.shutdown()
        result = self.recordings[0]
        self.assertFalse(result.reliable)
        self.assertEqual(result.dropped_audio_chunks, 3)

    def test_recording_samples_are_contiguous_read_only_float32(self) -> None:
        self.start_speech()
        self.machine.shutdown()
        samples = self.recordings[0].samples
        self.assertEqual(str(samples.dtype), "float32")
        self.assertTrue(samples.flags.c_contiguous)
        self.assertFalse(samples.flags.writeable)


if __name__ == "__main__":
    unittest.main()

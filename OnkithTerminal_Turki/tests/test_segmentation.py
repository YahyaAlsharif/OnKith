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

    def calibrated_with(self, noise_rms: float) -> LiveSegmentationStateMachine:
        events: list[tuple[str, dict]] = []
        machine = LiveSegmentationStateMachine(
            self.config,
            "session-calibration",
            on_event=lambda name, data: events.append((name, data)),
            on_recording=lambda recording: None,
            clock=lambda: 100.0,
            utc_now=lambda: "2026-07-29T10:00:00Z",
        )
        for index in range(25):
            machine.process(chunk(index, rms=noise_rms))
        machine.calibration_events = [
            data for name, data in events if name == "calibration_complete"
        ]
        return machine

    def test_a_noisy_calibration_does_not_leave_the_device_deaf(self) -> None:
        """Seen in the wild: threshold 0.48969, against 0.06 when quiet.

        The threshold was the noise floor times four with no ceiling, so
        two loud seconds during calibration set a bar no voice reaches.
        The device then printed "listening continuously" and heard nobody
        for the rest of the session, with nothing on screen to say why.
        """

        machine = self.calibrated_with(noise_rms=0.1224)
        self.assertAlmostEqual(machine.start_threshold, 0.15, places=6)
        reported = machine.calibration_events[0]
        self.assertTrue(reported["capped"])
        # What it would have been is kept, because that number is the
        # evidence that the room was the problem.
        self.assertGreater(reported["measured_threshold"], 0.48)

    def test_a_quiet_calibration_is_left_exactly_as_it_was(self) -> None:
        machine = self.calibrated_with(noise_rms=0.016)
        self.assertAlmostEqual(machine.start_threshold, 0.064, places=6)
        self.assertFalse(machine.calibration_events[0]["capped"])

    def test_silence_still_floors_at_the_minimum(self) -> None:
        machine = self.calibrated_with(noise_rms=0.0)
        self.assertAlmostEqual(machine.start_threshold, 0.006, places=6)
        self.assertFalse(machine.calibration_events[0]["capped"])

    def test_the_ceiling_is_above_every_threshold_that_ever_worked(self) -> None:
        # 0.0646 and 0.0571 were the two quiet calibrations measured on
        # this machine, and both have to survive the cap untouched.
        ceiling = float(self.config.audio["maximum_start_rms"])
        for worked in (0.0646, 0.0571):
            with self.subTest(threshold=worked):
                self.assertLess(worked, ceiling)

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


class OwnVoiceTests(unittest.TestCase):
    """The device answered itself, out loud, in front of the person.

    Its reply came back through the microphone, was transcribed as a new
    utterance and was answered, and nothing here stopped it. The loop
    ended because the language model said the two of them were repeating
    each other.
    """

    def setUp(self) -> None:
        self.config = PipelineConfig()
        self.events: list[tuple[str, dict]] = []
        self.recordings = []
        self.now = 100.0
        self.speaking = False
        self.machine = LiveSegmentationStateMachine(
            self.config,
            "session-own-voice",
            on_event=lambda name, data: self.events.append((name, data)),
            on_recording=self.recordings.append,
            clock=lambda: self.now,
            utc_now=lambda: "2026-08-09T10:00:00Z",
            speaking=lambda: self.speaking,
        )
        self.index = 0
        for _ in range(20):
            self.feed(rms=0.001)

    def feed(self, *, rms: float, value: float = 0.0) -> None:
        self.machine.process(chunk(self.index, rms=rms, value=value))
        self.index += 1

    def talk(self) -> None:
        for value in (0.01, 0.02, 0.03, 0.04):
            self.feed(rms=0.02, value=value)

    def test_the_device_does_not_record_its_own_reply(self) -> None:
        self.speaking = True
        self.talk()
        for _ in range(30):
            self.feed(rms=0.02, value=0.2)
        self.machine.shutdown()
        self.assertEqual(self.recordings, [])
        self.assertEqual(self.machine.state, "idle")

    def test_the_gate_is_held_open_past_the_last_word(self) -> None:
        # Synthesis finishing is not the room going quiet. The end of it
        # is still in the air and in the input buffer, and without the
        # tail those chunks start a recording of the device's own voice.
        self.speaking = True
        self.feed(rms=0.02, value=0.2)
        self.speaking = False
        self.talk()
        self.assertEqual(self.machine.state, "idle")
        # Past the tail, the microphone is the person's again.
        self.now += float(self.config.audio["speech_tail_seconds"]) + 0.01
        self.talk()
        self.assertEqual(self.machine.state, "speaking")

    def test_talking_over_the_reply_drops_the_recording(self) -> None:
        # Half their voice and half ours. Finishing it is how the loop
        # starts, so it is dropped, and it says so rather than vanishing.
        self.talk()
        self.assertEqual(self.machine.state, "speaking")
        self.speaking = True
        self.feed(rms=0.02, value=0.2)
        self.assertEqual(self.machine.state, "idle")
        self.assertEqual(self.recordings, [])
        self.assertIn(
            ("recording_discarded", {"reason": "device_was_speaking"}),
            self.events,
        )

    def test_the_reply_does_not_lead_the_next_utterance(self) -> None:
        # The pre-roll keeps 0.4s back to catch a word that began before
        # the level crossed. Left alone it puts the tail of the device's
        # own reply at the front of what the person says next.
        self.speaking = True
        for _ in range(8):
            self.feed(rms=0.02, value=0.9)
        self.speaking = False
        self.now += float(self.config.audio["speech_tail_seconds"]) + 0.01
        self.talk()
        self.machine.shutdown()
        self.assertEqual(len(self.recordings), 1)
        loudest = max(abs(float(sample)) for sample in self.recordings[0].samples)
        self.assertLess(loudest, 0.9)

    def test_nothing_is_gated_when_the_device_is_silent(self) -> None:
        self.talk()
        self.machine.shutdown()
        self.assertEqual(len(self.recordings), 1)


if __name__ == "__main__":
    unittest.main()

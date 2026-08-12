"""Capture survives a bad frame of audio.

One chunk the segmenter could not handle used to end listening for the rest
of the session, with no way back but killing the process and starting it
again. A device meant to sit in a room all day cannot go deaf over a single
frame.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.live_audio import (
    MAXIMUM_CONSECUTIVE_SEGMENTATION_ERRORS,
    AudioChunk,
    LiveAudioRuntime,
)
from integrated_pipeline.config import PipelineConfig


class Flaky:
    """A segmenter that fails on the chunks it is told to."""

    def __init__(self, fail_on: set[int], state: str = "speaking") -> None:
        self.fail_on = fail_on
        self.state = state
        self.received = 0
        self.handled = 0
        self.resets = 0

    def mark_overflow(self, count: int = 1) -> None:
        pass

    def process(self, chunk: AudioChunk) -> None:
        self.received += 1
        if self.received in self.fail_on:
            raise ValueError(f"bad chunk {self.received}")
        self.handled += 1

    def _reset_to_idle(self) -> None:
        self.resets += 1
        self.state = "idle"

    def shutdown(self) -> None:
        pass


class CaptureRecoveryTest(unittest.TestCase):
    def run_chunks(self, segmenter: Flaky, count: int) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        runtime = LiveAudioRuntime(
            PipelineConfig(),
            "test-session",
            device=None,
            once=False,
            on_event=lambda name, data: events.append((name, data)),
            on_level=lambda rms: None,
        )
        runtime.state_machine = segmenter
        chunk = AudioChunk(
            samples=[0.0] * 160,
            sample_rate=16000,
            captured_at=0.0,
            rms=0.0,
            peak=0.0,
        )
        for _ in range(count):
            runtime.audio_chunk_queue.put(chunk)
        worker = threading.Thread(target=runtime._worker_loop)
        worker.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and segmenter.received < count:
            if runtime.stop_event.is_set():
                break
            time.sleep(0.02)
        runtime.stop_event.set()
        worker.join(timeout=5)
        self.runtime = runtime
        return events

    def errors(self, events: list[tuple[str, dict]]) -> list[dict]:
        return [data for name, data in events if name == "segmentation_error"]

    def test_a_bad_chunk_does_not_end_the_session(self) -> None:
        segmenter = Flaky(fail_on={2, 5})
        events = self.run_chunks(segmenter, 10)
        self.assertEqual(segmenter.received, 10)
        self.assertEqual(segmenter.handled, 8)
        self.assertIsNone(self.runtime.worker_error)
        self.assertTrue(all(error["recovered"] for error in self.errors(events)))

    def test_the_screen_is_told_to_stop_showing_it_is_listening(self) -> None:
        """It had been told an utterance was open, and now there is not one.

        Without this the creature stays lit and the device looks like it is
        still hearing somebody who has long since stopped.
        """

        events = self.run_chunks(Flaky(fail_on={2}, state="speaking"), 4)
        closed = [data for name, data in events if name == "segment_closed"]
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["reason"], "segmentation_error")

    def test_nothing_is_announced_when_it_was_not_recording(self) -> None:
        events = self.run_chunks(Flaky(fail_on={2}, state="idle"), 4)
        self.assertEqual([name for name, _ in events if name == "segment_closed"], [])

    def test_a_segmenter_that_never_recovers_does_give_up(self) -> None:
        """Carrying on forever would be a device that looks alive and is not.

        A run of failures with no success between them is not a bad frame,
        it is a segmenter that will not come back.
        """

        limit = MAXIMUM_CONSECUTIVE_SEGMENTATION_ERRORS
        segmenter = Flaky(fail_on=set(range(1, limit + 1)))
        events = self.run_chunks(segmenter, limit + 3)
        self.assertTrue(self.runtime.stop_event.is_set())
        self.assertIsNotNone(self.runtime.worker_error)
        self.assertFalse(self.errors(events)[-1]["recovered"])

    def test_the_run_of_failures_has_to_be_unbroken(self) -> None:
        # Four failures with good chunks between them is a noisy microphone,
        # not a broken segmenter, and it must not add up to giving up.
        limit = MAXIMUM_CONSECUTIVE_SEGMENTATION_ERRORS
        spaced = {index * 2 for index in range(1, limit + 1)}
        segmenter = Flaky(fail_on=spaced)
        self.run_chunks(segmenter, max(spaced) + 2)
        self.assertIsNone(self.runtime.worker_error)


if __name__ == "__main__":
    unittest.main()

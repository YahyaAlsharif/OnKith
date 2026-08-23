"""Bounded retention for continuous speech, and the safety-split policy."""

from __future__ import annotations

import gc
import queue
import unittest
import weakref

import numpy as np

from app.stt_runtime import LiveSTTRuntime, continuation_batch_id
from integrated_pipeline.config import (
    MAXIMUM_LOGICAL_UTTERANCE_SECONDS,
    MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS,
)
from integrated_pipeline.schemas import CanaryUtteranceEnd, SegmentSummary
from tests.helpers import canary_segment, segment_transcript

SEGMENT_SAMPLES = 136_192
SUBMITTED_SAMPLES = SEGMENT_SAMPLES + 11_264

# Real forced splits submit 8.512 s of unique audio each, so the 120 s span
# bound is reached after 15 segments - before the 16-segment bound.
FORCED_SEGMENTS_UNTIL_SPLIT = 15
# Short segments let the segment-count bound be the one that trips.
SHORT_SAMPLES = 2 * 16_000


class CountingCanary:
    """Decode without ever holding a reference to the segment or its audio."""

    def __init__(self) -> None:
        self.calls = 0

    def transcribe_segment(self, segment):
        self.calls += 1
        return segment_transcript(segment, f"word{segment.segment_index}")


class RecordingPrivacy:
    def __init__(self) -> None:
        self.results = []
        self.failures = []

    def submit_stt(self, result) -> None:
        self.results.append(result)

    def submit_stt_failure(self, *args) -> None:
        self.failures.append(args)


class FakeMusic:
    def __init__(self) -> None:
        self.plays = []
        self.closes = 0

    def submit_play(self, text: str) -> None:
        self.plays.append(text)

    def submit_close(self) -> None:
        self.closes += 1


def forced_segment(index: int, *, final: bool = False, **kwargs):
    """A segment shaped like a real forced split: 8.512 s unique + 0.704 s overlap."""

    unique_start = index * SEGMENT_SAMPLES
    return canary_segment(
        index,
        final=final,
        start=max(0, unique_start - 5_632),
        end=unique_start + SEGMENT_SAMPLES + 5_632,
        **kwargs,
    )


def short_segment(index: int, *, final: bool = False, **kwargs):
    """A 2 s segment, so 16 of them stay far inside the 120 s span bound."""

    start = index * SHORT_SAMPLES
    return canary_segment(
        index, final=final, start=start, end=start + SHORT_SAMPLES, **kwargs
    )


class WorkerTestCase(unittest.TestCase):
    """Always stop the non-daemon STT worker, even when a test fails."""

    def build_runtime(self, privacy, canary=None, music=None):
        completed: queue.Queue = queue.Queue(maxsize=4)
        events: list[tuple[str, dict]] = []
        worker = LiveSTTRuntime(
            completed,
            canary or CountingCanary(),
            privacy,
            on_event=lambda event, payload: events.append((event, payload)),
            music_worker=music,
            utc_now=lambda: "2026-08-05T10:00:03Z",
        )
        worker.start()
        self.addCleanup(self._force_stop, completed, worker)
        return completed, worker, events

    @staticmethod
    def _force_stop(completed: queue.Queue, worker: LiveSTTRuntime) -> None:
        if worker.thread is None or not worker.thread.is_alive():
            return
        completed.put(None)
        worker.thread.join(timeout=5.0)

    def feed(self, completed: queue.Queue, segments) -> None:
        for segment in segments:
            completed.put(segment)
            completed.join()


class AudioReleaseTests(WorkerTestCase):
    def test_segment_audio_is_released_after_inference(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, _ = self.build_runtime(privacy)
        first = forced_segment(0)
        audio_ref = weakref.ref(first.samples)
        segment_ref = weakref.ref(first)
        completed.put(first)
        completed.join()
        del first
        gc.collect()
        # The accumulator is still open, but neither the segment nor its audio
        # array survives once the inference for it has completed.
        self.assertIsNone(segment_ref(), "CanarySegment outlived its inference")
        self.assertIsNone(audio_ref(), "Segment audio outlived its inference")
        completed.put(forced_segment(1, final=True))
        worker.shutdown()
        self.assertEqual(len(privacy.results), 1)

    def test_accumulator_holds_only_scalar_summaries(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, _ = self.build_runtime(privacy)
        self.feed(completed, [forced_segment(0)])
        held = worker._accumulator.summaries
        self.assertEqual(len(held), 1)
        self.assertIsInstance(held[0], SegmentSummary)
        for value in vars(held[0]).values():
            self.assertNotIsInstance(value, np.ndarray)
        completed.put(forced_segment(1, final=True))
        worker.shutdown()

    def test_retained_memory_is_metadata_not_audio_across_many_segments(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, _ = self.build_runtime(privacy)
        # One below the safety limit: the most a logical utterance ever holds.
        count = FORCED_SEGMENTS_UNTIL_SPLIT - 1
        self.feed(completed, [forced_segment(index) for index in range(count)])
        summaries = worker._accumulator.summaries
        self.assertEqual(len(summaries), count)
        retained = sum(
            sum(
                len(value.encode("utf-8")) if isinstance(value, str) else 8
                for value in vars(item).values()
            )
            for item in summaries
        )
        audio_bytes_if_retained = count * SUBMITTED_SAMPLES * 4
        # Retention is O(metadata/text): kilobytes, not the megabytes the audio
        # for the same span would have cost.
        self.assertLess(retained, 8_192)
        self.assertGreater(audio_bytes_if_retained, 8_000_000)
        completed.put(forced_segment(count, final=True))
        worker.shutdown()
        self.assertEqual(len(privacy.results), 1)

    def test_failure_path_allocates_no_audio_buffer(self) -> None:
        class FailingCanary:
            def transcribe_segment(self, segment):
                raise RuntimeError("model inference failed")

        privacy = RecordingPrivacy()
        completed, worker, _ = self.build_runtime(privacy, FailingCanary())
        segment = forced_segment(0, final=True)
        expected_seconds = (
            segment.segment_end_sample - segment.segment_start_sample
        ) / 16_000
        completed.put(segment)
        worker.shutdown()
        self.assertEqual(len(privacy.failures), 1)
        recording = privacy.failures[0][0]
        self.assertEqual(len(recording.samples), 0)
        # The exact submitted duration is still reported from scalars.
        self.assertAlmostEqual(recording.submitted_audio_seconds, expected_seconds)


class SafetySplitTests(WorkerTestCase):
    def test_segment_count_bound_closes_the_logical_utterance(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, events = self.build_runtime(privacy)
        self.feed(
            completed,
            [
                short_segment(index)
                for index in range(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS)
            ],
        )
        # The bound closed and released part one without any final segment.
        self.assertEqual(len(privacy.results), 1)
        released = privacy.results[0]
        self.assertEqual(released.segment_count, MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS)
        self.assertEqual(released.endpoint_reason, "utterance_safety_split")
        payloads = [p for n, p in events if n == "utterance_safety_split"]
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["limit"], "maximum_segments")
        self.assertEqual(payloads[0]["next_part_index"], 2)
        self.assertLess(
            payloads[0]["submitted_span_seconds"], MAXIMUM_LOGICAL_UTTERANCE_SECONDS
        )
        # Later audio continues as a new logical utterance and is still merged.
        completed.put(
            short_segment(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS, final=True)
        )
        worker.shutdown()
        self.assertEqual(len(privacy.results), 2)
        self.assertEqual(worker.safety_split_count, 1)

    def test_duration_bound_trips_first_for_real_forced_splits(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, events = self.build_runtime(privacy)
        self.feed(
            completed,
            [
                forced_segment(index)
                for index in range(FORCED_SEGMENTS_UNTIL_SPLIT)
            ],
        )
        payloads = [p for n, p in events if n == "utterance_safety_split"]
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["limit"], "maximum_duration")
        self.assertEqual(payloads[0]["segment_count"], FORCED_SEGMENTS_UNTIL_SPLIT)
        self.assertLess(
            payloads[0]["segment_count"], MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS
        )
        self.assertGreaterEqual(
            payloads[0]["submitted_span_seconds"], MAXIMUM_LOGICAL_UTTERANCE_SECONDS
        )
        completed.put(
            forced_segment(FORCED_SEGMENTS_UNTIL_SPLIT, final=True)
        )
        worker.shutdown()
        self.assertEqual(len(privacy.results), 2)

    def test_no_text_is_dropped_across_a_safety_split(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, _ = self.build_runtime(privacy)
        total = MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS + 2
        self.feed(
            completed,
            [
                short_segment(index, final=index == total - 1)
                for index in range(total)
            ],
        )
        worker.shutdown()
        self.assertEqual(len(privacy.results), 2)
        words = " ".join(item.final_text for item in privacy.results).split()
        self.assertEqual(words, [f"word{index}" for index in range(total)])

    def test_continuation_identity_is_unique_and_disjoint(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, _ = self.build_runtime(privacy)
        total = MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS + 2
        self.feed(
            completed,
            [
                short_segment(index, final=index == total - 1)
                for index in range(total)
            ],
        )
        worker.shutdown()
        batch_ids = [item.batch_id for item in privacy.results]
        sequences = [item.sequence_number for item in privacy.results]
        self.assertEqual(len(set(batch_ids)), len(batch_ids))
        self.assertEqual(len(set(sequences)), len(sequences))
        # Part one keeps the untouched acoustic identity.
        self.assertEqual(batch_ids[0], "session-test-utterance-000001")
        self.assertEqual(batch_ids[1], "session-test-utterance-000001-part-000002")
        # Sequence numbers are chronological, not a reserved high range.
        self.assertEqual(sequences, [1, 2])
        # The split relationship lives in dedicated metadata instead.
        for index, result in enumerate(privacy.results, start=1):
            self.assertEqual(
                result.logical_utterance_id, "session-test-utterance-000001"
            )
            self.assertEqual(result.continuation_part_index, index)

    def test_continuation_batch_id_derivation(self) -> None:
        self.assertEqual(continuation_batch_id("u", 1), "u")
        self.assertEqual(continuation_batch_id("u", 2), "u-part-000002")
        self.assertEqual(continuation_batch_id("u", 11), "u-part-000011")

    def test_sequence_numbers_are_chronological_across_utterances(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, _ = self.build_runtime(privacy)
        # One utterance long enough to split, then two ordinary utterances.
        self.feed(
            completed,
            [
                short_segment(index)
                for index in range(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS)
            ],
        )
        self.feed(completed, [short_segment(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS,
                                            final=True)])
        self.feed(
            completed,
            [canary_segment(0, final=True, utterance=2, start=0, end=32_000)],
        )
        self.feed(
            completed,
            [canary_segment(0, final=True, utterance=3, start=0, end=32_000)],
        )
        worker.shutdown()
        sequences = [item.sequence_number for item in privacy.results]
        self.assertEqual(sequences, [1, 2, 3, 4])
        self.assertEqual(sequences, sorted(sequences))
        # Both parts of the split utterance precede the later independent ones.
        parts = [
            item.sequence_number
            for item in privacy.results
            if item.logical_utterance_id == "session-test-utterance-000001"
        ]
        later = [
            item.sequence_number
            for item in privacy.results
            if item.logical_utterance_id != "session-test-utterance-000001"
        ]
        self.assertEqual(parts, [1, 2])
        self.assertTrue(max(parts) < min(later))

    def test_normal_session_numbering_is_unchanged_without_splits(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, _ = self.build_runtime(privacy)
        for utterance in (1, 2, 3):
            self.feed(
                completed,
                [
                    canary_segment(
                        0, final=True, utterance=utterance, start=0, end=32_000
                    )
                ],
            )
        worker.shutdown()
        # Identical to the acoustic numbering, as before this change.
        self.assertEqual([item.sequence_number for item in privacy.results], [1, 2, 3])
        for item in privacy.results:
            self.assertEqual(item.continuation_part_index, 1)
            self.assertEqual(item.logical_utterance_id, item.batch_id)

    def test_normal_utterance_below_the_limits_is_unchanged(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, events = self.build_runtime(privacy)
        completed.put(canary_segment(0, final=False, start=0, end=141_824))
        completed.put(canary_segment(1, final=True, start=130_560, end=180_000))
        worker.shutdown()
        self.assertEqual(len(privacy.results), 1)
        result = privacy.results[0]
        # Identity, sequencing and endpoint reason all keep acoustic semantics.
        self.assertEqual(result.batch_id, "session-test-utterance-000001")
        self.assertEqual(result.sequence_number, 1)
        self.assertEqual(result.endpoint_reason, "endpoint_silence")
        self.assertEqual(result.final_text, "word0 word1")
        self.assertNotIn("utterance_safety_split", [name for name, _ in events])
        self.assertEqual(worker.safety_split_count, 0)

    def test_onkith_runs_once_per_safety_bounded_logical_utterance(self) -> None:
        privacy = RecordingPrivacy()
        music = FakeMusic()

        class CommandCanary:
            def transcribe_segment(self, segment):
                text = (
                    "Onkith play believer"
                    if segment.segment_index == 0
                    else f"word{segment.segment_index}"
                )
                return segment_transcript(segment, text)

        completed, worker, _ = self.build_runtime(privacy, CommandCanary(), music)
        self.feed(
            completed,
            [
                short_segment(index)
                for index in range(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS)
            ],
        )
        # Intercepted exactly once, on the merged text of the safety-bounded
        # logical utterance that contained the wake word.
        self.assertEqual(len(music.plays), 1)
        self.assertTrue(music.plays[0].startswith("play believer"))
        completed.put(
            short_segment(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS, final=True)
        )
        worker.shutdown()
        self.assertEqual(len(music.plays), 1)

    def test_reliability_metadata_survives_a_safety_split(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, _ = self.build_runtime(privacy)
        self.feed(
            completed,
            [
                short_segment(
                    index,
                    reliable=index != 3,
                    dropped=7 if index == 3 else 0,
                )
                for index in range(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS)
            ],
        )
        self.assertEqual(len(privacy.results), 1)
        self.assertFalse(privacy.results[0].reliable)
        self.assertEqual(privacy.results[0].dropped_audio_chunks, 7)
        completed.put(
            short_segment(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS, final=True)
        )
        worker.shutdown()
        # The continuation is judged on its own segments, not the earlier drop.
        self.assertTrue(privacy.results[1].reliable)

    def test_shutdown_right_after_a_safety_split_is_clean(self) -> None:
        privacy = RecordingPrivacy()
        completed, worker, events = self.build_runtime(privacy)
        self.feed(
            completed,
            [
                short_segment(index)
                for index in range(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS)
            ],
        )
        self.assertEqual(len(privacy.results), 1)
        reference = short_segment(0)
        completed.put(
            CanaryUtteranceEnd(
                session_id=reference.session_id,
                utterance_id=reference.utterance_id,
                utterance_sequence_number=reference.utterance_sequence_number,
                decision_sample=9_000_000,
                endpoint_reason="shutdown",
                reliable=False,
                dropped_audio_chunks=0,
                recording_closed_monotonic=reference.recording_closed_monotonic,
                audio_ended_monotonic=reference.audio_ended_monotonic,
                audio_ended_at_utc=reference.audio_ended_at_utc,
                utterance_start_sample=reference.utterance_start_sample,
            )
        )
        worker.shutdown()
        # Part one was already released; the empty continuation is not an
        # unterminated utterance and produces no extra row and no failure.
        self.assertEqual(len(privacy.results), 1)
        self.assertEqual(privacy.failures, [])
        self.assertIn(
            "utterance_end_after_safety_split", [name for name, _ in events]
        )


if __name__ == "__main__":
    unittest.main()

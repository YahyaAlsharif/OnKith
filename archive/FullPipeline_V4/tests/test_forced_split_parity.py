"""Release gate: online segmenter vs the literal reference implementation.

Every scenario runs the same audio and the same per-frame Silero probability
script through both implementations and asserts exact equality of segment
count, submitted start, unique start, end, forced boundary, decision sample,
endpoint reason, overlap size, final-utterance marker behaviour and the
reliable flag.
"""

from __future__ import annotations

import random
import unittest

import numpy as np

from app.canary_live_segmenter import CanaryLiveSegmenter
from integrated_pipeline.config import (
    CALIBRATION_SAMPLES,
    FORCED_OVERLAP_SAMPLES,
    MAXIMUM_UNIQUE_SEGMENT_SAMPLES,
    PipelineConfig,
    TRAILING_CONTEXT_SAMPLES,
    VAD_FRAME_SAMPLES,
)
from integrated_pipeline.schemas import CanarySegment, CanaryUtteranceEnd
from integrated_pipeline.transcript_merger import merge_segment_transcripts
from tests.reference_segmenter import ReferenceSegment, reference_run

ADJACENT_OVERLAP = FORCED_OVERLAP_SAMPLES + TRAILING_CONTEXT_SAMPLES


class ScriptedVAD:
    """Replay a frozen per-frame probability script, like a fixed Silero."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = list(probabilities)
        self.index = 0
        self.reset_count = 0
        self.frame_sizes: list[int] = []

    def probability(self, frame: np.ndarray) -> float:
        self.frame_sizes.append(int(np.asarray(frame).size))
        if self.index >= len(self.probabilities):
            raise AssertionError("Online run consumed more frames than the script")
        value = self.probabilities[self.index]
        self.index += 1
        return float(value)

    def reset(self) -> None:
        # A reset clears model state; it never rewinds the frame script.
        self.reset_count += 1


def speech_script(signal: np.ndarray, speaking: np.ndarray) -> list[float]:
    """One probability per operational frame, derived from a speech mask."""

    total = int(signal.size)
    values: list[float] = []
    start = CALIBRATION_SAMPLES
    while start < total:
        end = min(start + VAD_FRAME_SAMPLES, total)
        window = speaking[start:end]
        values.append(0.9 if window.size and bool(window.all()) else 0.1)
        start += VAD_FRAME_SAMPLES
    # One spare entry covers the zero-padded final partial frame at shutdown.
    return values + [0.1]


def online_run(
    signal: np.ndarray,
    probabilities: list[float],
    chunks: list[int],
    *,
    shutdown_reason: str | None = "end_of_stream",
) -> tuple[list[object], ScriptedVAD, CanaryLiveSegmenter]:
    outputs: list[object] = []
    vad = ScriptedVAD(probabilities)
    segmenter = CanaryLiveSegmenter(
        PipelineConfig(),
        "parity",
        vad,
        on_event=lambda event, payload: None,
        on_segment=outputs.append,
        clock=lambda: 1_000.0,
    )
    offset = 0
    for size in chunks:
        if offset >= len(signal):
            break
        segmenter.process_samples(signal[offset : offset + size], captured_at=0.0)
        offset += size
    if offset < len(signal):
        segmenter.process_samples(signal[offset:], captured_at=0.0)
    if shutdown_reason is not None:
        segmenter.shutdown(reason=shutdown_reason)
    return outputs, vad, segmenter


def as_reference(outputs: list[object]) -> list[ReferenceSegment]:
    """Project emitted objects onto the reference comparison shape."""

    rows: list[ReferenceSegment] = []
    for item in outputs:
        if isinstance(item, CanaryUtteranceEnd):
            rows.append(
                ReferenceSegment(
                    utterance_sequence_number=item.utterance_sequence_number,
                    segment_index=-1,
                    segment_start_sample=-1,
                    segment_unique_start_sample=-1,
                    segment_end_sample=item.decision_sample,
                    decision_sample=item.decision_sample,
                    forced_boundary_sample=None,
                    endpoint_reason=item.endpoint_reason,
                    is_final_segment_of_utterance=True,
                    is_control_marker=True,
                )
            )
            continue
        assert isinstance(item, CanarySegment)
        rows.append(
            ReferenceSegment(
                utterance_sequence_number=item.utterance_sequence_number,
                segment_index=item.segment_index,
                segment_start_sample=item.segment_start_sample,
                segment_unique_start_sample=int(item.segment_unique_start_sample),
                segment_end_sample=item.segment_end_sample,
                decision_sample=item.decision_sample,
                forced_boundary_sample=item.forced_boundary_sample,
                endpoint_reason=item.endpoint_reason,
                is_final_segment_of_utterance=item.is_final_segment_of_utterance,
            )
        )
    return rows


def normalize_markers(rows: list[ReferenceSegment]) -> list[ReferenceSegment]:
    """Control markers carry no interval; compare only their identity."""

    result = []
    for row in rows:
        if row.is_control_marker:
            result.append(
                ReferenceSegment(
                    utterance_sequence_number=row.utterance_sequence_number,
                    segment_index=-1,
                    segment_start_sample=-1,
                    segment_unique_start_sample=-1,
                    segment_end_sample=row.segment_end_sample,
                    decision_sample=row.decision_sample,
                    forced_boundary_sample=None,
                    endpoint_reason=row.endpoint_reason,
                    is_final_segment_of_utterance=True,
                    is_control_marker=True,
                )
            )
        else:
            result.append(row)
    return result


def build(*spans: tuple[int, bool]) -> tuple[np.ndarray, np.ndarray]:
    """Build (signal, speech-mask) from (sample_count, is_speech) spans."""

    pieces = []
    flags = []
    for count, speaking in spans:
        pieces.append(
            np.ones(count, dtype=np.float32)
            if speaking
            else np.zeros(count, dtype=np.float32)
        )
        flags.append(np.full(count, speaking, dtype=bool))
    return np.concatenate(pieces), np.concatenate(flags)


class ForcedSplitParityTests(unittest.TestCase):
    """Each case must agree exactly with the literal reference."""

    def assert_parity(
        self,
        signal: np.ndarray,
        speaking: np.ndarray,
        *,
        shutdown_reason: str | None = "end_of_stream",
        chunk_plans: list[list[int]] | None = None,
    ) -> list[object]:
        script = speech_script(signal, speaking)
        expected = normalize_markers(
            reference_run(signal, script, shutdown_reason=shutdown_reason)
        )
        plans = chunk_plans or [
            [len(signal)],
            [1_600] * ((len(signal) + 1_599) // 1_600),
            [512] * ((len(signal) + 511) // 512),
        ]
        rng = random.Random(20260805)
        plans = plans + [[rng.randint(1, 4_000) for _ in range(400)]]
        first_outputs: list[object] | None = None
        for plan in plans:
            outputs, _, _ = online_run(
                signal, script, plan, shutdown_reason=shutdown_reason
            )
            self.assertEqual(
                normalize_markers(as_reference(outputs)),
                expected,
                f"online/reference mismatch for chunk plan starting {plan[:3]}",
            )
            if first_outputs is None:
                first_outputs = outputs
        assert first_outputs is not None
        return first_outputs

    def test_speech_through_one_forced_split(self) -> None:
        # 153,600 speech samples span one 136,192 unique segment plus its
        # 5,632 trailing context, but fall short of a second boundary.
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (300 * 512, True),
            (100 * 512, False),
        )
        outputs = self.assert_parity(signal, speaking)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].endpoint_reason, "max_duration")
        self.assertEqual(outputs[1].endpoint_reason, "endpoint_silence")
        self.assertEqual(outputs[0].segment_start_sample, CALIBRATION_SAMPLES)
        self.assertEqual(
            outputs[0].forced_boundary_sample,
            CALIBRATION_SAMPLES + MAXIMUM_UNIQUE_SEGMENT_SAMPLES,
        )
        self.assertEqual(
            outputs[0].segment_end_sample - outputs[1].segment_start_sample,
            ADJACENT_OVERLAP,
        )
        self.assertTrue(outputs[1].is_final_segment_of_utterance)
        self.assertTrue(all(item.reliable for item in outputs))

    def test_speech_through_multiple_forced_splits(self) -> None:
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (600 * 512, True),
            (100 * 512, False),
        )
        outputs = self.assert_parity(signal, speaking)
        forced = [
            item for item in outputs if item.endpoint_reason == "max_duration"
        ]
        self.assertGreaterEqual(len(forced), 2)
        for previous, following in zip(outputs, outputs[1:]):
            self.assertEqual(
                previous.segment_end_sample - following.segment_start_sample,
                ADJACENT_OVERLAP,
            )
            self.assertEqual(
                following.segment_unique_start_sample,
                previous.forced_boundary_sample,
            )
        self.assertEqual(outputs[-1].endpoint_reason, "endpoint_silence")
        self.assertEqual(
            [item.segment_index for item in outputs], list(range(len(outputs)))
        )

    def test_silence_begins_inside_trailing_context_collection(self) -> None:
        # Speech stops after the forced boundary is pending but before the
        # 5,632 trailing samples have been collected.
        speech = MAXIMUM_UNIQUE_SEGMENT_SAMPLES + 2 * 512
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (speech, True),
            (120 * 512, False),
        )
        self.assert_parity(signal, speaking)

    def test_endpoint_silence_immediately_after_forced_split(self) -> None:
        speech = MAXIMUM_UNIQUE_SEGMENT_SAMPLES + TRAILING_CONTEXT_SAMPLES
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (speech, True),
            (120 * 512, False),
        )
        outputs = self.assert_parity(signal, speaking)
        # This is the exact case the old implementation got wrong: without a
        # zero reset the post-boundary look-back alone satisfied minimum
        # speech, producing an extra endpoint_silence segment.
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].endpoint_reason, "max_duration")
        self.assertEqual(outputs[1].endpoint_reason, "end_of_stream")

    def test_short_dropout_around_the_forced_boundary(self) -> None:
        before = MAXIMUM_UNIQUE_SEGMENT_SAMPLES - 2 * 512
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (before, True),
            (3 * 512, False),
            (60 * 512, True),
            (110 * 512, False),
        )
        self.assert_parity(signal, speaking)

    def test_shutdown_immediately_after_forced_split(self) -> None:
        speech = MAXIMUM_UNIQUE_SEGMENT_SAMPLES + TRAILING_CONTEXT_SAMPLES
        signal, speaking = build((CALIBRATION_SAMPLES, False), (speech, True))
        outputs = self.assert_parity(signal, speaking, shutdown_reason="shutdown")
        self.assertEqual(len(outputs), 2)
        self.assertIsInstance(outputs[0], CanarySegment)
        self.assertEqual(outputs[0].endpoint_reason, "max_duration")
        self.assertIsInstance(outputs[1], CanaryUtteranceEnd)

    def test_partial_final_512_frame(self) -> None:
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (20 * 512 + 199, True),
        )
        outputs = self.assert_parity(signal, speaking, shutdown_reason="shutdown")
        self.assertEqual(outputs[-1].endpoint_reason, "shutdown")
        self.assertFalse(outputs[-1].reliable)

    def test_dropped_audio_marks_the_active_utterance_unreliable(self) -> None:
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (60 * 512, True),
            (110 * 512, False),
        )
        script = speech_script(signal, speaking)
        outputs: list[object] = []
        segmenter = CanaryLiveSegmenter(
            PipelineConfig(),
            "parity",
            ScriptedVAD(script),
            on_event=lambda event, payload: None,
            on_segment=outputs.append,
        )
        half = CALIBRATION_SAMPLES + 30 * 512
        segmenter.process_samples(signal[:half])
        segmenter.mark_overflow(3)
        segmenter.process_samples(signal[half:])
        segmenter.shutdown(reason="end_of_stream")
        self.assertTrue(outputs)
        self.assertFalse(outputs[-1].reliable)
        self.assertEqual(outputs[-1].dropped_audio_chunks, 3)

    def test_silero_is_not_reset_at_an_ordinary_forced_split(self) -> None:
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (600 * 512, True),
            (110 * 512, False),
        )
        script = speech_script(signal, speaking)
        outputs, vad, _ = online_run(
            signal, script, [1_600] * 800, shutdown_reason=None
        )
        forced = [
            item for item in outputs if item.endpoint_reason == "max_duration"
        ]
        self.assertGreaterEqual(len(forced), 2)
        # One reset only, and only when the complete acoustic utterance ended.
        self.assertEqual(vad.reset_count, 1)

    def test_merged_transcript_ordering_follows_segment_order(self) -> None:
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (600 * 512, True),
            (100 * 512, False),
        )
        outputs = self.assert_parity(signal, speaking)
        texts = [f"word{item.segment_index}" for item in outputs]
        merged = merge_segment_transcripts(texts)
        self.assertEqual(merged.text, " ".join(texts))


class CalibrationBoundaryTests(unittest.TestCase):
    """No emitted segment may start before sample 12,000."""

    def assert_first_start_is_operational(
        self, signal: np.ndarray, speaking: np.ndarray, chunks: list[int]
    ) -> list[object]:
        script = speech_script(signal, speaking)
        outputs, _, _ = online_run(
            signal, script, chunks, shutdown_reason="end_of_stream"
        )
        segments = [item for item in outputs if isinstance(item, CanarySegment)]
        self.assertTrue(segments)
        for item in segments:
            self.assertGreaterEqual(item.segment_start_sample, CALIBRATION_SAMPLES)
        self.assertEqual(
            normalize_markers(as_reference(outputs)),
            normalize_markers(reference_run(signal, script)),
        )
        return segments

    def test_speech_immediately_after_calibration(self) -> None:
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (40 * 512, True),
            (110 * 512, False),
        )
        segments = self.assert_first_start_is_operational(
            signal, speaking, [1_600] * 200
        )
        self.assertEqual(segments[0].segment_start_sample, CALIBRATION_SAMPLES)
        # The submitted audio must contain no calibration prefix: calibration
        # is exact silence here, speech is exact ones.
        self.assertTrue(bool(np.all(segments[0].samples[:512] == 1.0)))

    def test_speech_within_less_than_one_pre_roll_after_calibration(self) -> None:
        signal, speaking = build(
            (CALIBRATION_SAMPLES + 4 * 512, False),
            (40 * 512, True),
            (110 * 512, False),
        )
        segments = self.assert_first_start_is_operational(
            signal, speaking, [1_600] * 200
        )
        self.assertEqual(segments[0].segment_start_sample, CALIBRATION_SAMPLES)

    def test_speech_after_more_than_one_pre_roll_window(self) -> None:
        lead = 20 * 512
        signal, speaking = build(
            (CALIBRATION_SAMPLES + lead, False),
            (40 * 512, True),
            (110 * 512, False),
        )
        segments = self.assert_first_start_is_operational(
            signal, speaking, [1_600] * 200
        )
        self.assertEqual(
            segments[0].segment_start_sample,
            CALIBRATION_SAMPLES + lead - 6_144,
        )
        self.assertGreater(segments[0].segment_start_sample, CALIBRATION_SAMPLES)

    def test_speech_in_the_same_callback_that_completes_calibration(self) -> None:
        # Calibration completes mid-chunk at sample 12,000; the remainder of
        # that same 0.1 s callback is already speech.
        signal, speaking = build(
            (CALIBRATION_SAMPLES, False),
            (40 * 512, True),
            (110 * 512, False),
        )
        segments = self.assert_first_start_is_operational(
            signal, speaking, [1_600] * 200
        )
        self.assertEqual(segments[0].segment_start_sample, CALIBRATION_SAMPLES)

    def test_first_segment_never_contains_calibration_audio(self) -> None:
        # Loud calibration noise would be audible if it leaked into Canary.
        rng = np.random.default_rng(20260805)
        noise = rng.normal(0.0, 0.2, CALIBRATION_SAMPLES).astype(np.float32)
        signal = np.concatenate((noise, np.ones(40 * 512, dtype=np.float32),
                                 np.zeros(110 * 512, dtype=np.float32)))
        speaking = np.concatenate(
            (
                np.zeros(CALIBRATION_SAMPLES, dtype=bool),
                np.ones(40 * 512, dtype=bool),
                np.zeros(110 * 512, dtype=bool),
            )
        )
        script = speech_script(signal, speaking)
        outputs, _, _ = online_run(signal, script, [1_600] * 200)
        segments = [item for item in outputs if isinstance(item, CanarySegment)]
        self.assertTrue(segments)
        self.assertGreaterEqual(segments[0].segment_start_sample, CALIBRATION_SAMPLES)
        self.assertNotIn(
            True,
            [bool(np.any(np.isclose(segments[0].samples, value))) for value in noise[:8]],
        )


if __name__ == "__main__":
    unittest.main()

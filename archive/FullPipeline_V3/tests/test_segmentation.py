from __future__ import annotations

import random
import unittest

import numpy as np

from app.canary_live_segmenter import CanaryLiveSegmenter, exact_frame_rms
from app.live_audio import reference_resample_poly
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.schemas import CanarySegment, CanaryUtteranceEnd


class EnergyVAD:
    def __init__(self, forced_probability: float | None = None) -> None:
        self.forced_probability = forced_probability
        self.calls: list[np.ndarray] = []
        self.reset_count = 0

    def probability(self, frame: np.ndarray) -> float:
        self.calls.append(frame.copy())
        if self.forced_probability is not None:
            return self.forced_probability
        return 1.0 if np.max(np.abs(frame)) > 0.5 else 0.0

    def reset(self) -> None:
        self.reset_count += 1


def run_signal(signal: np.ndarray, chunks: list[int]) -> tuple[list[object], EnergyVAD, CanaryLiveSegmenter]:
    segments: list[object] = []
    vad = EnergyVAD()
    segmenter = CanaryLiveSegmenter(
        PipelineConfig(),
        "session",
        vad,
        on_event=lambda event, payload: None,
        on_segment=segments.append,
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
    return segments, vad, segmenter


class CalibrationTests(unittest.TestCase):
    def test_exact_12000_sample_p95_and_frozen_gate(self) -> None:
        frames = [np.full(512, index / 1000, dtype=np.float32) for index in range(23)]
        calibration = np.concatenate(frames + [np.full(224, 0.023, dtype=np.float32)])
        vad = EnergyVAD()
        events: list[tuple[str, dict]] = []
        segmenter = CanaryLiveSegmenter(
            PipelineConfig(),
            "session",
            vad,
            on_event=lambda event, payload: events.append((event, payload)),
            on_segment=lambda segment: None,
        )
        segmenter.process_samples(calibration[:11_999])
        self.assertFalse(segmenter.calibrated)
        segmenter.process_samples(calibration[11_999:])
        expected_values = [exact_frame_rms(item) for item in frames]
        expected_values.append(exact_frame_rms(calibration[-224:]))
        expected = max(1e-5, float(np.percentile(expected_values, 95)) * 2.25)
        self.assertAlmostEqual(segmenter.energy_gate, expected, places=12)
        self.assertEqual(len(vad.calls), 0)
        frozen = segmenter.energy_gate
        segmenter.process_samples(np.ones(512, dtype=np.float32))
        self.assertEqual(segmenter.energy_gate, frozen)
        self.assertEqual(events[0][1]["final_partial_frame_samples"], 224)

    def test_gate_minimum(self) -> None:
        segments, _, segmenter = run_signal(np.zeros(12_000, dtype=np.float32), [12_000])
        self.assertEqual(segments, [])
        self.assertEqual(segmenter.energy_gate, 1e-5)


class IncrementalSegmentationTests(unittest.TestCase):
    @staticmethod
    def long_signal() -> np.ndarray:
        return np.concatenate(
            (
                np.zeros(12_000 + 8 * 512, dtype=np.float32),
                np.ones(300 * 512, dtype=np.float32),
                np.zeros(80 * 512, dtype=np.float32),
            )
        )

    @staticmethod
    def signature(segments: list[object]) -> list[tuple]:
        return [
            (
                item.segment_index,
                item.segment_start_sample,
                item.segment_end_sample,
                item.forced_boundary_sample,
                item.endpoint_reason,
                item.is_final_segment_of_utterance,
            )
            for item in segments
        ]

    def test_full_replay_1600_and_random_chunks_are_identical(self) -> None:
        signal = self.long_signal()
        reference, _, _ = run_signal(signal, [len(signal)])
        live, _, _ = run_signal(signal, [1_600] * ((len(signal) + 1_599) // 1_600))
        rng = random.Random(20260805)
        random_chunks = [rng.randint(1, 4_000) for _ in range(200)]
        varied, _, _ = run_signal(signal, random_chunks)
        expected = self.signature(reference)
        self.assertEqual(self.signature(live), expected)
        self.assertEqual(self.signature(varied), expected)
        self.assertEqual(len(reference), 2)
        self.assertEqual(reference[0].endpoint_reason, "max_duration")
        self.assertEqual(reference[1].endpoint_reason, "endpoint_silence")
        self.assertEqual(
            reference[0].segment_end_sample - reference[1].segment_start_sample,
            11_264,
        )
        self.assertEqual(
            reference[0].forced_boundary_sample - reference[0].segment_start_sample,
            136_192,
        )

    def test_silero_lifecycle_continues_across_forced_split(self) -> None:
        segments, vad, _ = run_signal(self.long_signal(), [1_600] * 200)
        self.assertEqual(len(segments), 2)
        self.assertEqual(vad.reset_count, 1)

    def test_high_energy_below_silero_threshold_never_activates(self) -> None:
        vad = EnergyVAD(forced_probability=0.49)
        segments: list[object] = []
        segmenter = CanaryLiveSegmenter(
            PipelineConfig(),
            "session",
            vad,
            on_event=lambda event, payload: None,
            on_segment=segments.append,
        )
        segmenter.process_samples(np.concatenate((np.zeros(12_000), np.ones(20_000))))
        self.assertFalse(segmenter.active)
        self.assertEqual(segments, [])

    def test_short_dropout_bridges_and_endpoint_is_39936_samples(self) -> None:
        signal = np.concatenate(
            (
                np.zeros(12_000, dtype=np.float32),
                np.ones(8 * 512, dtype=np.float32),
                np.zeros(3 * 512, dtype=np.float32),
                np.ones(8 * 512, dtype=np.float32),
                np.zeros(80 * 512, dtype=np.float32),
            )
        )
        segments, _, _ = run_signal(signal, [1_600] * 50)
        self.assertEqual(len(segments), 1)
        final = segments[0]
        self.assertEqual(
            final.decision_sample - final.last_speech_end_sample,
            39_936,
        )
        self.assertEqual(
            final.segment_end_sample - final.last_speech_end_sample,
            5_632,
        )

    def test_shutdown_pads_only_final_partial_frame(self) -> None:
        vad = EnergyVAD()
        segments: list[object] = []
        segmenter = CanaryLiveSegmenter(
            PipelineConfig(),
            "session",
            vad,
            on_event=lambda event, payload: None,
            on_segment=segments.append,
        )
        signal = np.concatenate((np.zeros(12_000), np.ones(6 * 512 + 123)))
        segmenter.process_samples(signal)
        calls_before = len(vad.calls)
        segmenter.shutdown()
        self.assertEqual(len(vad.calls), calls_before + 1)
        self.assertEqual(len(vad.calls[-1]), 512)
        self.assertEqual(segments[-1].endpoint_reason, "shutdown")
        self.assertFalse(segments[-1].reliable)

    def test_shutdown_after_forced_emit_uses_control_not_overlap_only_audio(self) -> None:
        vad = EnergyVAD()
        outputs: list[object] = []
        segmenter = CanaryLiveSegmenter(
            PipelineConfig(),
            "session",
            vad,
            on_event=lambda event, payload: None,
            on_segment=outputs.append,
        )
        # Speech starts in the first operational frame, so the clamped segment
        # start is exactly 12,000 and the unique segment runs 12,000..148,192.
        # The signal ends exactly when that forced segment's 5,632 trailing
        # samples have arrived (12,000 + 136,192 + 5,632 = 153,824).
        signal = np.concatenate(
            (np.zeros(12_000, dtype=np.float32), np.ones(277 * 512, dtype=np.float32))
        )
        segmenter.process_samples(signal)
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], CanarySegment)
        self.assertEqual(outputs[0].endpoint_reason, "max_duration")
        segmenter.shutdown()
        self.assertEqual(len(outputs), 2)
        self.assertIsInstance(outputs[1], CanaryUtteranceEnd)

    def test_reference_resample_poly_is_whole_array_exact(self) -> None:
        from scipy.signal import resample_poly

        source = np.linspace(-1, 1, 4_410, dtype=np.float32)
        expected = np.asarray(resample_poly(source, 160, 441), dtype=np.float32)
        actual = reference_resample_poly(source, 44_100)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()

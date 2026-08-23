"""Literal offline reference for the locked 16 kHz segmentation rules.

This module is deliberately independent of ``app.canary_live_segmenter``. It
implements the sample-count contract straight from the specification against a
complete signal, with no incremental audio store and no chunk boundaries. The
release gate compares it against the online state machine driven by identical
audio chunks and an identical per-frame Silero probability script; any
disagreement is a parity failure.

Rules implemented literally:

* callback chunking is irrelevant here - the whole signal is present;
* calibration consumes samples [0, 12000) as 512-sample frames plus one final
  224-sample partial frame, gate = max(1e-5, P95(frame RMS) * 2.25);
* operational framing starts at sample 12,000 and steps 512 samples;
* a frame is speech only when probability >= 0.50 AND frame RMS >= gate;
* pre-roll 6,144 / start-confirm 3,072 / minimum speech 3,072 /
  dropout bridge 2,048 / endpoint silence 39,936 /
  maximum unique segment 136,192 / forced overlap 5,632 /
  trailing context 5,632, all clamped so nothing starts before sample 12,000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from integrated_pipeline.config import (
    CALIBRATION_SAMPLES,
    DROPOUT_BRIDGE_SAMPLES,
    ENDPOINT_SILENCE_SAMPLES,
    FORCED_OVERLAP_SAMPLES,
    MAXIMUM_UNIQUE_SEGMENT_SAMPLES,
    MINIMUM_SPEECH_SAMPLES,
    OPERATIONAL_AUDIO_START_SAMPLE,
    PRE_ROLL_SAMPLES,
    START_CONFIRM_SAMPLES,
    TRAILING_CONTEXT_SAMPLES,
    VAD_FRAME_SAMPLES,
)


@dataclass(frozen=True)
class ReferenceSegment:
    """Every field the parity gate compares, in exact samples."""

    utterance_sequence_number: int
    segment_index: int
    segment_start_sample: int
    segment_unique_start_sample: int
    segment_end_sample: int
    decision_sample: int
    forced_boundary_sample: int | None
    endpoint_reason: str
    is_final_segment_of_utterance: bool
    is_control_marker: bool = False


def frame_rms(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(data**2, dtype=np.float64) + 1e-18))


def calibration_gate(signal: np.ndarray) -> float:
    """Reproduce the exact frozen startup gate."""

    values = []
    for start in range(0, CALIBRATION_SAMPLES, VAD_FRAME_SAMPLES):
        end = min(start + VAD_FRAME_SAMPLES, CALIBRATION_SAMPLES)
        values.append(frame_rms(signal[start:end]))
    return max(1e-5, float(np.percentile(values, 95)) * 2.25)


def reference_run(
    signal: np.ndarray,
    probabilities: Sequence[float],
    *,
    shutdown_reason: str | None = "end_of_stream",
) -> list[ReferenceSegment]:
    """Return the exact segment list the locked contract requires."""

    audio = np.asarray(signal, dtype=np.float32).reshape(-1)
    total = int(audio.size)
    if total < CALIBRATION_SAMPLES:
        return []
    gate = calibration_gate(audio)

    emitted: list[ReferenceSegment] = []
    utterance_sequence = 0
    segment_index = 0
    active = False
    segment_start: int | None = None
    unique_start: int | None = None
    total_voiced = 0
    candidate_start: int | None = None
    candidate_speech = 0
    raw_silence_start: int | None = None
    last_raw_speech_end: int | None = None
    last_effective_speech_end: int | None = None
    pending_boundary: int | None = None
    last_emitted_end: int | None = None

    def close_utterance() -> None:
        nonlocal active, segment_start, unique_start, total_voiced
        nonlocal candidate_start, candidate_speech, raw_silence_start
        nonlocal last_raw_speech_end, last_effective_speech_end
        nonlocal pending_boundary, segment_index, last_emitted_end
        active = False
        segment_start = None
        unique_start = None
        total_voiced = 0
        candidate_start = None
        candidate_speech = 0
        raw_silence_start = None
        last_raw_speech_end = None
        last_effective_speech_end = None
        pending_boundary = None
        segment_index = 0
        last_emitted_end = None

    frame_start = OPERATIONAL_AUDIO_START_SAMPLE
    frame_number = 0
    while True:
        if frame_start + VAD_FRAME_SAMPLES <= total:
            real_length = VAD_FRAME_SAMPLES
            window = audio[frame_start : frame_start + VAD_FRAME_SAMPLES]
        elif shutdown_reason is not None and frame_start < total:
            # The stream ends inside a frame: pad to 512 for the model but
            # score only the real samples.
            real_length = total - frame_start
            window = audio[frame_start:total]
        else:
            break
        frame_end = frame_start + real_length
        probability = float(probabilities[frame_number])
        frame_number += 1
        speech = probability >= 0.5 and frame_rms(window[:real_length]) >= gate

        if speech:
            if candidate_start is None:
                candidate_start = frame_start
                candidate_speech = 0
            candidate_speech += real_length
            raw_silence_start = None
            last_raw_speech_end = frame_end
            last_effective_speech_end = frame_end
            if not active and candidate_speech >= START_CONFIRM_SAMPLES:
                utterance_sequence += 1
                segment_index = 0
                segment_start = max(
                    OPERATIONAL_AUDIO_START_SAMPLE, candidate_start - PRE_ROLL_SAMPLES
                )
                unique_start = segment_start
                total_voiced = candidate_speech
                pending_boundary = None
                last_emitted_end = None
                active = True
            elif active:
                total_voiced += real_length
        elif not active:
            candidate_start = None
            candidate_speech = 0
        else:
            if raw_silence_start is None:
                raw_silence_start = frame_start
            if frame_end - raw_silence_start < DROPOUT_BRIDGE_SAMPLES:
                total_voiced += real_length
                last_effective_speech_end = frame_end
            else:
                raw_end = (
                    raw_silence_start
                    if last_raw_speech_end is None
                    else last_raw_speech_end
                )
                last_effective_speech_end = max(raw_end, raw_silence_start)

        if active:
            assert unique_start is not None and segment_start is not None
            if (
                pending_boundary is None
                and frame_end - unique_start >= MAXIMUM_UNIQUE_SEGMENT_SAMPLES
            ):
                pending_boundary = unique_start + MAXIMUM_UNIQUE_SEGMENT_SAMPLES
            if (
                pending_boundary is not None
                and frame_end >= pending_boundary + TRAILING_CONTEXT_SAMPLES
            ):
                boundary = pending_boundary
                end = boundary + TRAILING_CONTEXT_SAMPLES
                emitted.append(
                    ReferenceSegment(
                        utterance_sequence_number=utterance_sequence,
                        segment_index=segment_index,
                        segment_start_sample=segment_start,
                        segment_unique_start_sample=unique_start,
                        segment_end_sample=end,
                        decision_sample=frame_end,
                        forced_boundary_sample=boundary,
                        endpoint_reason="max_duration",
                        is_final_segment_of_utterance=False,
                    )
                )
                last_emitted_end = end
                segment_index += 1
                segment_start = max(
                    OPERATIONAL_AUDIO_START_SAMPLE, boundary - FORCED_OVERLAP_SAMPLES
                )
                unique_start = boundary
                pending_boundary = None
                # The forced split resets only voiced eligibility. Silero
                # state, the acoustic utterance, and the silence bookkeeping
                # all continue untouched.
                total_voiced = 0

        if active and raw_silence_start is not None:
            assert last_effective_speech_end is not None
            assert segment_start is not None and unique_start is not None
            silence = frame_end - last_effective_speech_end
            if silence >= ENDPOINT_SILENCE_SAMPLES and total_voiced >= (
                MINIMUM_SPEECH_SAMPLES
            ):
                end = min(total, last_effective_speech_end + TRAILING_CONTEXT_SAMPLES)
                emitted.append(
                    ReferenceSegment(
                        utterance_sequence_number=utterance_sequence,
                        segment_index=segment_index,
                        segment_start_sample=segment_start,
                        segment_unique_start_sample=unique_start,
                        segment_end_sample=end,
                        decision_sample=frame_end,
                        forced_boundary_sample=None,
                        endpoint_reason="endpoint_silence",
                        is_final_segment_of_utterance=True,
                    )
                )
                close_utterance()

        frame_start = frame_end
        if real_length != VAD_FRAME_SAMPLES:
            break

    if shutdown_reason is not None and active:
        assert segment_start is not None and unique_start is not None
        has_unsubmitted = last_emitted_end is None or total > last_emitted_end
        if total > segment_start and has_unsubmitted:
            emitted.append(
                ReferenceSegment(
                    utterance_sequence_number=utterance_sequence,
                    segment_index=segment_index,
                    segment_start_sample=segment_start,
                    segment_unique_start_sample=unique_start,
                    segment_end_sample=total,
                    decision_sample=total,
                    forced_boundary_sample=None,
                    endpoint_reason=shutdown_reason,
                    is_final_segment_of_utterance=True,
                )
            )
        elif last_emitted_end is not None:
            emitted.append(
                ReferenceSegment(
                    utterance_sequence_number=utterance_sequence,
                    segment_index=segment_index,
                    segment_start_sample=segment_start,
                    segment_unique_start_sample=unique_start,
                    segment_end_sample=total,
                    decision_sample=total,
                    forced_boundary_sample=None,
                    endpoint_reason=shutdown_reason,
                    is_final_segment_of_utterance=True,
                    is_control_marker=True,
                )
            )
    return emitted

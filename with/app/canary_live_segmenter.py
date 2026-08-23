"""Exact incremental 16 kHz Silero plus frozen-energy Canary segmenter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

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
    SAMPLE_RATE,
    START_CONFIRM_SAMPLES,
    TRAILING_CONTEXT_SAMPLES,
    VAD_FRAME_SAMPLES,
    VAD_NEGATIVE_THRESHOLD,
    VAD_SPEECH_THRESHOLD,
    PipelineConfig,
)
from integrated_pipeline.schemas import CanarySegment, CanaryUtteranceEnd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def exact_frame_rms(samples: Any) -> float:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values**2, dtype=np.float64) + 1e-18))


@dataclass(frozen=True)
class FrameDiagnostic:
    frame_start: int
    frame_end: int
    real_samples: int
    probability: float
    rms: float
    calibrated_speech: bool


class CanaryLiveSegmenter:
    """Turn an arbitrary live chunk stream into exact notebook-style segments."""

    def __init__(
        self,
        config: PipelineConfig,
        session_id: str,
        vad: Any,
        *,
        on_event: Callable[[str, dict[str, Any]], None],
        on_segment: Callable[[CanarySegment | CanaryUtteranceEnd], None],
        clock: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], str] = utc_now_iso,
        collect_frame_diagnostics: bool = False,
    ) -> None:
        config.validate()
        self.config = config
        self.session_id = session_id
        self.vad = vad
        self.on_event = on_event
        self.on_segment = on_segment
        self.clock = clock
        self.utc_now = utc_now
        self.collect_frame_diagnostics = collect_frame_diagnostics

        self.sample_position = 0
        self.next_frame_start = CALIBRATION_SAMPLES
        self.energy_gate: float | None = None
        self.noise_frame_rms_p95: float | None = None
        self._audio_store_start = 0
        self._audio_store = np.empty(0, dtype=np.float32)
        self._stream_origin_monotonic: float | None = None

        self.candidate_start: int | None = None
        self.candidate_speech_samples = 0
        self.active = False
        self.segment_start: int | None = None
        self.segment_unique_start: int | None = None
        self.total_voiced_samples = 0
        self.raw_silence_start: int | None = None
        self.last_raw_speech_end: int | None = None
        self.last_effective_speech_end: int | None = None
        self.pending_forced_boundary: int | None = None
        self.segment_index = 0
        self.utterance_id: str | None = None
        self.utterance_start_sample: int | None = None
        self._utterance_sequence = 0
        self._last_emitted_end: int | None = None

        self.pending_dropped_chunks = 0
        self.active_dropped_chunks = 0
        self.active_reliable = True
        self.false_start_count = 0
        self.segment_count = 0
        self.final_count = 0
        self.frame_count = 0
        self.frame_diagnostics: list[FrameDiagnostic] = []
        self._shutdown = False

    @property
    def calibrated(self) -> bool:
        return self.energy_gate is not None

    @property
    def has_active_utterance(self) -> bool:
        return self.active and self.utterance_id is not None

    @staticmethod
    def _operational_start(sample: int) -> int:
        """Clamp any candidate start to the post-calibration operational floor."""

        return max(OPERATIONAL_AUDIO_START_SAMPLE, int(sample))

    def _append_audio(self, samples: np.ndarray) -> None:
        if samples.size:
            self._audio_store = np.concatenate((self._audio_store, samples))
            self.sample_position += int(samples.size)

    def _slice(self, start: int, end: int) -> np.ndarray:
        if start < self._audio_store_start or end > self.sample_position or end <= start:
            raise RuntimeError(
                "Requested segment audio is outside the retained stream: "
                f"[{start}, {end}) retained=[{self._audio_store_start}, {self.sample_position})"
            )
        local_start = start - self._audio_store_start
        local_end = end - self._audio_store_start
        return np.ascontiguousarray(
            self._audio_store[local_start:local_end], dtype=np.float32
        )

    def _trim_store(self) -> None:
        # Every branch clamps to the operational lower bound: calibration audio
        # can never be needed again once the frozen gate exists, and no emitted
        # segment may start before it.
        if self.active and self.segment_start is not None:
            keep_from = self.segment_start
        elif self.candidate_start is not None:
            keep_from = self._operational_start(self.candidate_start - PRE_ROLL_SAMPLES)
        else:
            keep_from = self._operational_start(
                self.next_frame_start - PRE_ROLL_SAMPLES
            )
        if keep_from <= self._audio_store_start:
            return
        drop = keep_from - self._audio_store_start
        self._audio_store = np.ascontiguousarray(self._audio_store[drop:])
        self._audio_store_start = keep_from

    def _finish_calibration(self) -> None:
        calibration = self._slice(0, CALIBRATION_SAMPLES)
        values = []
        for start in range(0, CALIBRATION_SAMPLES, VAD_FRAME_SAMPLES):
            frame = calibration[start : min(start + VAD_FRAME_SAMPLES, CALIBRATION_SAMPLES)]
            values.append(exact_frame_rms(frame))
        self.noise_frame_rms_p95 = float(np.percentile(values, 95))
        self.energy_gate = max(1e-5, self.noise_frame_rms_p95 * 2.25)
        self.on_event(
            "calibration_complete",
            {
                "calibration_samples": CALIBRATION_SAMPLES,
                "calibration_frames": len(values),
                "final_partial_frame_samples": CALIBRATION_SAMPLES % VAD_FRAME_SAMPLES,
                "noise_frame_rms_p95": self.noise_frame_rms_p95,
                "energy_gate": self.energy_gate,
            },
        )

    def process_samples(self, samples: Any, *, captured_at: float | None = None) -> None:
        if self._shutdown:
            raise RuntimeError("Cannot process audio after segmenter shutdown")
        values = np.ascontiguousarray(
            np.asarray(samples, dtype=np.float32).reshape(-1), dtype=np.float32
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Live audio contains non-finite samples")
        if self._stream_origin_monotonic is None:
            self._stream_origin_monotonic = (
                self.clock() if captured_at is None else float(captured_at)
            )
        self._append_audio(values)
        if not self.calibrated and self.sample_position >= CALIBRATION_SAMPLES:
            self._finish_calibration()
        if not self.calibrated:
            return
        while self.next_frame_start + VAD_FRAME_SAMPLES <= self.sample_position:
            start = self.next_frame_start
            end = start + VAD_FRAME_SAMPLES
            self._process_frame(self._slice(start, end), start, VAD_FRAME_SAMPLES)
            self.next_frame_start = end
        self._trim_store()

    def _activate(self) -> None:
        assert self.candidate_start is not None
        self._utterance_sequence += 1
        self.utterance_id = (
            f"{self.session_id}-utterance-{self._utterance_sequence:06d}"
        )
        self.segment_index = 0
        # The pre-roll may never reach back into startup calibration audio.
        self.segment_start = self._operational_start(
            self.candidate_start - PRE_ROLL_SAMPLES
        )
        self.segment_unique_start = self.segment_start
        self.utterance_start_sample = self.segment_start
        self.total_voiced_samples = self.candidate_speech_samples
        self.pending_forced_boundary = None
        self._last_emitted_end = None
        self.active_dropped_chunks = self.pending_dropped_chunks
        self.pending_dropped_chunks = 0
        self.active_reliable = self.active_dropped_chunks == 0
        self.active = True
        self.on_event(
            "recording_started",
            {
                "utterance_id": self.utterance_id,
                "utterance_sequence_number": self._utterance_sequence,
                "candidate_start_sample": self.candidate_start,
                "segment_start_sample": self.segment_start,
                "energy_gate": self.energy_gate,
            },
        )

    def _process_frame(self, model_frame: np.ndarray, frame_start: int, real_length: int) -> None:
        if model_frame.size != VAD_FRAME_SAMPLES:
            raise ValueError("The model-facing Silero frame must contain exactly 512 samples")
        frame_end = frame_start + real_length
        probability = float(self.vad.probability(model_frame))
        rms = exact_frame_rms(model_frame[:real_length])
        threshold = VAD_NEGATIVE_THRESHOLD if self.active else VAD_SPEECH_THRESHOLD
        speech = probability >= threshold and rms >= float(self.energy_gate)

        if speech:
            if self.candidate_start is None:
                self.candidate_start = frame_start
                self.candidate_speech_samples = 0
            self.candidate_speech_samples += real_length
            self.raw_silence_start = None
            self.last_raw_speech_end = frame_end
            self.last_effective_speech_end = frame_end
            if not self.active and self.candidate_speech_samples >= START_CONFIRM_SAMPLES:
                self._activate()
            elif self.active:
                self.total_voiced_samples += real_length
        elif not self.active:
            if self.candidate_start is not None:
                self.false_start_count += 1
                self.on_event(
                    "speech_false_start",
                    {
                        "candidate_start_sample": self.candidate_start,
                        "candidate_speech_samples": self.candidate_speech_samples,
                    },
                )
            self.candidate_start = None
            self.candidate_speech_samples = 0
        else:
            if self.raw_silence_start is None:
                self.raw_silence_start = frame_start
            silence_len = frame_end - self.raw_silence_start
            if silence_len < DROPOUT_BRIDGE_SAMPLES:
                self.total_voiced_samples += real_length
                self.last_effective_speech_end = frame_end
            else:
                raw_end = (
                    self.raw_silence_start
                    if self.last_raw_speech_end is None
                    else self.last_raw_speech_end
                )
                self.last_effective_speech_end = max(raw_end, self.raw_silence_start)

        self.frame_count += 1
        if self.collect_frame_diagnostics:
            self.frame_diagnostics.append(
                FrameDiagnostic(
                    frame_start,
                    frame_end,
                    real_length,
                    probability,
                    rms,
                    speech,
                )
            )

        if self.active:
            self._check_forced_split(frame_end)
        if self.active:
            self._check_endpoint(frame_end)

    def _check_forced_split(self, frame_end: int) -> None:
        assert self.segment_unique_start is not None
        if (
            self.pending_forced_boundary is None
            and frame_end - self.segment_unique_start >= MAXIMUM_UNIQUE_SEGMENT_SAMPLES
        ):
            self.pending_forced_boundary = (
                self.segment_unique_start + MAXIMUM_UNIQUE_SEGMENT_SAMPLES
            )
        boundary = self.pending_forced_boundary
        if boundary is None:
            return
        desired_end = boundary + TRAILING_CONTEXT_SAMPLES
        if frame_end < desired_end:
            return
        self._emit_segment(
            segment_end=desired_end,
            decision_sample=frame_end,
            reason="max_duration",
            is_final=False,
            forced_boundary=boundary,
        )
        self.segment_index += 1
        self.segment_start = self._operational_start(boundary - FORCED_OVERLAP_SAMPLES)
        self.segment_unique_start = boundary
        self.pending_forced_boundary = None
        # Exact notebook forced-split transition. The acoustic utterance, the
        # Silero state, and the raw-silence/last-speech state all continue
        # unchanged, but the new unique segment starts its minimum-speech
        # eligibility from zero. Retained look-back audio, the previous
        # segment's trailing context, and post-boundary frames already
        # processed while waiting for that trailing context are all audio the
        # previous segment already owned; counting them again would let the
        # next segment satisfy MINIMUM_SPEECH_SAMPLES without any newly voiced
        # audio, which can turn a later end_of_stream boundary into an
        # endpoint_silence one and change the emitted segment count.
        self.total_voiced_samples = 0
        self.on_event(
            "forced_split_continuation",
            {
                "utterance_id": self.utterance_id,
                "next_segment_index": self.segment_index,
                "next_segment_start_sample": self.segment_start,
                "next_segment_unique_start_sample": boundary,
                "adjacent_submitted_overlap_samples": (
                    FORCED_OVERLAP_SAMPLES + TRAILING_CONTEXT_SAMPLES
                ),
            },
        )
        self._trim_store()

    def _check_endpoint(self, frame_end: int) -> None:
        if self.raw_silence_start is None or self.last_effective_speech_end is None:
            return
        silence = frame_end - self.last_effective_speech_end
        if silence < ENDPOINT_SILENCE_SAMPLES or self.total_voiced_samples < MINIMUM_SPEECH_SAMPLES:
            return
        assert self.segment_start is not None
        submitted_end = min(
            self.sample_position,
            self.last_effective_speech_end + TRAILING_CONTEXT_SAMPLES,
        )
        if submitted_end <= self.segment_start:
            raise RuntimeError("Endpoint would create an overlap-only final segment")
        self._emit_segment(
            segment_end=submitted_end,
            decision_sample=frame_end,
            reason="endpoint_silence",
            is_final=True,
            forced_boundary=None,
        )
        self._complete_utterance()

    def _sample_monotonic(self, sample: int) -> float:
        origin = self._stream_origin_monotonic
        return self.clock() if origin is None else origin + sample / SAMPLE_RATE

    def _emit_segment(
        self,
        *,
        segment_end: int,
        decision_sample: int,
        reason: str,
        is_final: bool,
        forced_boundary: int | None,
    ) -> CanarySegment:
        assert self.segment_start is not None
        assert self.segment_unique_start is not None
        assert self.utterance_start_sample is not None
        assert self.utterance_id is not None
        if segment_end > self.sample_position:
            raise RuntimeError("Cannot emit audio that has not arrived")
        samples = self._slice(self.segment_start, segment_end)
        ended_at_utc = self.utc_now()
        try:
            ended_dt = datetime.fromisoformat(ended_at_utc.replace("Z", "+00:00"))
            started_at_utc = (
                ended_dt
                - timedelta(
                    seconds=(decision_sample - self.utterance_start_sample) / SAMPLE_RATE
                )
            ).isoformat().replace("+00:00", "Z")
        except ValueError:
            started_at_utc = ended_at_utc
        segment = CanarySegment(
            session_id=self.session_id,
            utterance_id=self.utterance_id,
            segment_id=f"{self.utterance_id}-segment-{self.segment_index:06d}",
            utterance_sequence_number=self._utterance_sequence,
            segment_index=self.segment_index,
            samples=samples,
            sample_rate=SAMPLE_RATE,
            segment_start_sample=self.segment_start,
            segment_end_sample=segment_end,
            decision_sample=decision_sample,
            last_speech_end_sample=(
                self.last_effective_speech_end
                if self.last_effective_speech_end is not None
                else segment_end
            ),
            forced_boundary_sample=forced_boundary,
            endpoint_reason=reason,
            is_final_segment_of_utterance=is_final,
            reliable=self.active_reliable,
            dropped_audio_chunks=self.active_dropped_chunks,
            recording_closed_monotonic=max(
                self.clock(), self._sample_monotonic(decision_sample)
            ),
            audio_started_monotonic=self._sample_monotonic(self.utterance_start_sample),
            audio_ended_monotonic=self._sample_monotonic(decision_sample),
            audio_started_at_utc=started_at_utc,
            audio_ended_at_utc=ended_at_utc,
            utterance_start_sample=self.utterance_start_sample,
            utterance_captured_end_sample=decision_sample,
            segment_unique_start_sample=self.segment_unique_start,
        )
        self.segment_count += 1
        self._last_emitted_end = segment_end
        self.on_event(
            "segment_closed",
            {
                "utterance_id": segment.utterance_id,
                "segment_id": segment.segment_id,
                "segment_index": segment.segment_index,
                "reason": reason,
                "segment_start_sample": segment.segment_start_sample,
                "segment_end_sample": segment.segment_end_sample,
                "decision_sample": decision_sample,
                "forced_boundary_sample": forced_boundary,
                "is_final_segment_of_utterance": is_final,
                "submitted_audio_seconds": segment.submitted_audio_seconds,
                "reliable": segment.reliable,
                "dropped_audio_chunks": segment.dropped_audio_chunks,
            },
        )
        self.on_segment(segment)
        return segment

    def _complete_utterance(self) -> None:
        self.final_count += 1
        self.active = False
        self.candidate_start = None
        self.candidate_speech_samples = 0
        self.segment_start = None
        self.segment_unique_start = None
        self.total_voiced_samples = 0
        self.raw_silence_start = None
        self.last_raw_speech_end = None
        self.last_effective_speech_end = None
        self.pending_forced_boundary = None
        self.segment_index = 0
        self.utterance_id = None
        self.utterance_start_sample = None
        self._last_emitted_end = None
        self.active_dropped_chunks = 0
        self.active_reliable = True
        self.vad.reset()
        self._trim_store()

    def mark_overflow(self, count: int = 1) -> None:
        value = max(0, int(count))
        if self.active:
            self.active_dropped_chunks += value
            self.active_reliable = False
        else:
            self.pending_dropped_chunks += value

    def shutdown(self, *, reason: str = "shutdown") -> None:
        if self._shutdown:
            return
        if reason not in {"shutdown", "end_of_stream"}:
            raise ValueError("Shutdown reason must be shutdown or end_of_stream")
        if self.calibrated and self.next_frame_start < self.sample_position:
            start = self.next_frame_start
            real = self.sample_position - start
            frame = np.zeros(VAD_FRAME_SAMPLES, dtype=np.float32)
            frame[:real] = self._slice(start, self.sample_position)
            self._process_frame(frame, start, real)
            self.next_frame_start = self.sample_position
        if self.active:
            assert self.segment_start is not None
            assert self.segment_unique_start is not None
            has_unsubmitted_audio = (
                self._last_emitted_end is None
                or self.sample_position > self._last_emitted_end
            )
            if self.sample_position > self.segment_start and has_unsubmitted_audio:
                self.active_reliable = False
                self._emit_segment(
                    segment_end=self.sample_position,
                    decision_sample=self.sample_position,
                    reason=reason,
                    is_final=True,
                    forced_boundary=None,
                )
            elif self._last_emitted_end is not None:
                assert self.utterance_id is not None
                assert self.utterance_start_sample is not None
                marker = CanaryUtteranceEnd(
                    session_id=self.session_id,
                    utterance_id=self.utterance_id,
                    utterance_sequence_number=self._utterance_sequence,
                    decision_sample=self.sample_position,
                    endpoint_reason=reason,
                    reliable=False,
                    dropped_audio_chunks=self.active_dropped_chunks,
                    recording_closed_monotonic=max(
                        self.clock(), self._sample_monotonic(self.sample_position)
                    ),
                    audio_ended_monotonic=self._sample_monotonic(self.sample_position),
                    audio_ended_at_utc=self.utc_now(),
                    utterance_start_sample=self.utterance_start_sample,
                )
                self.on_event(
                    "utterance_end_without_new_segment",
                    {
                        "utterance_id": marker.utterance_id,
                        "reason": reason,
                        "decision_sample": marker.decision_sample,
                    },
                )
                self.on_segment(marker)
            self._complete_utterance()
        else:
            self.vad.reset()
        self._shutdown = True

    def audit(self) -> dict[str, Any]:
        return {
            "sample_position": self.sample_position,
            "next_frame_start": self.next_frame_start,
            "calibrated": self.calibrated,
            "noise_frame_rms_p95": self.noise_frame_rms_p95,
            "energy_gate": self.energy_gate,
            "frame_count": self.frame_count,
            "false_start_count": self.false_start_count,
            "segment_count": self.segment_count,
            "final_utterance_count": self.final_count,
            "active": self.active,
            "pending_dropped_chunks": self.pending_dropped_chunks,
        }

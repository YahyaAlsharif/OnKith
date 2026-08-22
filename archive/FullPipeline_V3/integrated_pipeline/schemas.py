"""Typed records shared by capture, Canary, masking, and persistence lanes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_ENDPOINT_REASONS = {
    "max_duration",
    "endpoint_silence",
    "shutdown",
    "end_of_stream",
    # Accepted only for backwards-compatible database reads/tests.
    "maximum_utterance",
}


def _owned_float32(samples: Any) -> Any:
    import numpy as np

    audio = np.array(
        np.asarray(samples, dtype=np.float32).reshape(-1),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains non-finite samples")
    audio.setflags(write=False)
    return audio


@dataclass(frozen=True)
class CanarySegment:
    """One independently decoded in-memory segment of a larger utterance."""

    session_id: str
    utterance_id: str
    segment_id: str
    utterance_sequence_number: int
    segment_index: int
    samples: Any
    sample_rate: int
    segment_start_sample: int
    segment_end_sample: int
    decision_sample: int
    last_speech_end_sample: int
    forced_boundary_sample: int | None
    endpoint_reason: str
    is_final_segment_of_utterance: bool
    reliable: bool
    dropped_audio_chunks: int
    recording_closed_monotonic: float
    audio_started_monotonic: float
    audio_ended_monotonic: float
    audio_started_at_utc: str
    audio_ended_at_utc: str
    utterance_start_sample: int
    utterance_captured_end_sample: int
    # First sample this segment uniquely owns. For segment 0 it equals
    # segment_start_sample; after a forced split it is the forced boundary, so
    # segment_start_sample..segment_unique_start_sample is the retained
    # look-back that the previous segment already submitted.
    segment_unique_start_sample: int | None = None

    def __post_init__(self) -> None:
        if self.endpoint_reason not in _ENDPOINT_REASONS:
            raise ValueError(f"Unsupported endpoint reason: {self.endpoint_reason}")
        if self.sample_rate != 16_000:
            raise ValueError("Canary segments must be exactly 16 kHz")
        if self.segment_unique_start_sample is not None and not (
            self.segment_start_sample
            <= self.segment_unique_start_sample
            <= self.segment_end_sample
        ):
            raise ValueError("Unique start must lie inside the submitted interval")
        if self.utterance_sequence_number <= 0 or self.segment_index < 0:
            raise ValueError("Invalid utterance or segment sequence")
        if self.segment_start_sample < 0 or self.segment_end_sample <= self.segment_start_sample:
            raise ValueError("Invalid segment sample interval")
        audio = _owned_float32(self.samples)
        if len(audio) != self.segment_end_sample - self.segment_start_sample:
            raise ValueError("Segment interval and audio length disagree")
        object.__setattr__(self, "samples", audio)

    @property
    def submitted_audio_seconds(self) -> float:
        return len(self.samples) / float(self.sample_rate)


@dataclass(frozen=True)
class CanaryUtteranceEnd:
    """Finalization control when shutdown has no not-yet-submitted audio."""

    session_id: str
    utterance_id: str
    utterance_sequence_number: int
    decision_sample: int
    endpoint_reason: str
    reliable: bool
    dropped_audio_chunks: int
    recording_closed_monotonic: float
    audio_ended_monotonic: float
    audio_ended_at_utc: str
    utterance_start_sample: int

    def __post_init__(self) -> None:
        if self.endpoint_reason not in {"shutdown", "end_of_stream"}:
            raise ValueError("Utterance-end controls are shutdown-only")
        if self.utterance_sequence_number <= 0:
            raise ValueError("Invalid utterance sequence")
        if self.decision_sample < self.utterance_start_sample:
            raise ValueError("Invalid utterance end sample")


@dataclass(frozen=True)
class SegmentTranscript:
    """Canary output and model-only timing for one segment call.

    This value is short-lived: it still references the decoded ``CanarySegment``
    (and therefore its audio array) so callers can inspect the exact input that
    produced ``text``. The long-lived utterance accumulator must convert it to a
    :class:`SegmentSummary` immediately and drop this object, so that segment
    audio is never retained until the utterance ends.
    """

    segment: CanarySegment
    text: str
    processing_seconds: float
    queue_wait_seconds: float
    stt_started_at_utc: str
    stt_completed_at_utc: str

    def summarize(self) -> "SegmentSummary":
        """Project onto scalars only, releasing the segment audio reference."""

        return SegmentSummary.from_transcript(self)


@dataclass(frozen=True)
class SegmentSummary:
    """Audio-free record of one decoded segment.

    Every field is a scalar or a short string. Retaining a list of these for a
    logical utterance costs O(metadata + text), never O(total audio duration),
    which is what keeps continuous speech safe on a 4 GB Raspberry Pi.
    """

    session_id: str
    utterance_id: str
    segment_id: str
    utterance_sequence_number: int
    segment_index: int
    sample_rate: int
    segment_start_sample: int
    segment_end_sample: int
    decision_sample: int
    forced_boundary_sample: int | None
    endpoint_reason: str
    is_final_segment_of_utterance: bool
    reliable: bool
    dropped_audio_chunks: int
    recording_closed_monotonic: float
    audio_started_monotonic: float
    audio_ended_monotonic: float
    audio_started_at_utc: str
    audio_ended_at_utc: str
    utterance_start_sample: int
    utterance_captured_end_sample: int
    text: str
    processing_seconds: float
    queue_wait_seconds: float
    stt_started_at_utc: str
    stt_completed_at_utc: str

    @classmethod
    def from_transcript(cls, transcript: SegmentTranscript) -> "SegmentSummary":
        segment = transcript.segment
        return cls(
            session_id=segment.session_id,
            utterance_id=segment.utterance_id,
            segment_id=segment.segment_id,
            utterance_sequence_number=segment.utterance_sequence_number,
            segment_index=segment.segment_index,
            sample_rate=segment.sample_rate,
            segment_start_sample=segment.segment_start_sample,
            segment_end_sample=segment.segment_end_sample,
            decision_sample=segment.decision_sample,
            forced_boundary_sample=segment.forced_boundary_sample,
            endpoint_reason=segment.endpoint_reason,
            is_final_segment_of_utterance=segment.is_final_segment_of_utterance,
            reliable=segment.reliable,
            dropped_audio_chunks=segment.dropped_audio_chunks,
            recording_closed_monotonic=segment.recording_closed_monotonic,
            audio_started_monotonic=segment.audio_started_monotonic,
            audio_ended_monotonic=segment.audio_ended_monotonic,
            audio_started_at_utc=segment.audio_started_at_utc,
            audio_ended_at_utc=segment.audio_ended_at_utc,
            utterance_start_sample=segment.utterance_start_sample,
            utterance_captured_end_sample=segment.utterance_captured_end_sample,
            text=transcript.text,
            processing_seconds=transcript.processing_seconds,
            queue_wait_seconds=transcript.queue_wait_seconds,
            stt_started_at_utc=transcript.stt_started_at_utc,
            stt_completed_at_utc=transcript.stt_completed_at_utc,
        )

    @classmethod
    def from_segment(cls, segment: CanarySegment) -> "SegmentSummary":
        """Summarize a segment whose inference failed before producing text."""

        return cls(
            session_id=segment.session_id,
            utterance_id=segment.utterance_id,
            segment_id=segment.segment_id,
            utterance_sequence_number=segment.utterance_sequence_number,
            segment_index=segment.segment_index,
            sample_rate=segment.sample_rate,
            segment_start_sample=segment.segment_start_sample,
            segment_end_sample=segment.segment_end_sample,
            decision_sample=segment.decision_sample,
            forced_boundary_sample=segment.forced_boundary_sample,
            endpoint_reason=segment.endpoint_reason,
            is_final_segment_of_utterance=segment.is_final_segment_of_utterance,
            reliable=segment.reliable,
            dropped_audio_chunks=segment.dropped_audio_chunks,
            recording_closed_monotonic=segment.recording_closed_monotonic,
            audio_started_monotonic=segment.audio_started_monotonic,
            audio_ended_monotonic=segment.audio_ended_monotonic,
            audio_started_at_utc=segment.audio_started_at_utc,
            audio_ended_at_utc=segment.audio_ended_at_utc,
            utterance_start_sample=segment.utterance_start_sample,
            utterance_captured_end_sample=segment.utterance_captured_end_sample,
            text="",
            processing_seconds=0.0,
            queue_wait_seconds=0.0,
            stt_started_at_utc="",
            stt_completed_at_utc="",
        )

    @property
    def submitted_audio_seconds(self) -> float:
        """Exact submitted duration from the sample interval, not from audio."""

        return (
            self.segment_end_sample - self.segment_start_sample
        ) / float(self.sample_rate)


@dataclass(frozen=True)
class RecordedUtterance:
    """Complete utterance metadata used by the existing STT-failure storage path."""

    session_id: str
    batch_id: str
    sequence_number: int
    samples: Any
    sample_rate: int
    audio_started_monotonic: float
    audio_ended_monotonic: float
    audio_started_at_utc: str
    audio_ended_at_utc: str
    captured_at_utc: str
    recording_closed_monotonic: float
    endpoint_reason: str
    reliable: bool
    dropped_audio_chunks: int
    captured_audio_seconds: float
    overlap_with_previous: bool = False
    # The segment-based STT-failure path knows the exact submitted duration
    # from scalar sample intervals and carries no audio. Setting this lets it
    # pass an empty array instead of allocating a multi-megabyte zero buffer
    # purely to represent a duration.
    submitted_audio_seconds_override: float | None = None
    # Same split relationship metadata as STTResult, for the failure path.
    logical_utterance_id: str = ""
    continuation_part_index: int = 1

    def __post_init__(self) -> None:
        if self.endpoint_reason not in _ENDPOINT_REASONS:
            raise ValueError(f"Unsupported endpoint reason: {self.endpoint_reason}")
        if self.sequence_number <= 0 or self.sample_rate <= 0:
            raise ValueError("Invalid utterance identity or sample rate")
        if (
            self.submitted_audio_seconds_override is not None
            and float(self.submitted_audio_seconds_override) < 0.0
        ):
            raise ValueError("Submitted audio duration cannot be negative")
        object.__setattr__(self, "samples", _owned_float32(self.samples))

    @property
    def submitted_audio_seconds(self) -> float:
        if self.submitted_audio_seconds_override is not None:
            return float(self.submitted_audio_seconds_override)
        return len(self.samples) / float(self.sample_rate)


@dataclass(frozen=True)
class STTResult:
    """One final merged transcript for one complete spoken utterance."""

    final_text: str
    duration_seconds: float
    processing_seconds: float
    finalization_seconds: float
    rtf: float
    callback_count: int
    reliable: bool = True
    error: str | None = None
    session_id: str = ""
    batch_id: str = ""
    sequence_number: int = 0
    endpoint_reason: str = ""
    dropped_audio_chunks: int = 0
    audio_started_monotonic: float = 0.0
    audio_ended_monotonic: float = 0.0
    recording_closed_monotonic: float = 0.0
    audio_started_at_utc: str = ""
    audio_ended_at_utc: str = ""
    stt_started_at_utc: str = ""
    stt_completed_at_utc: str = ""
    captured_audio_seconds: float = 0.0
    submitted_audio_seconds: float = 0.0
    unique_submitted_audio_seconds: float = 0.0
    submitted_audio_rtf: float = float("nan")
    queue_wait_seconds: float = 0.0
    total_stt_seconds: float = 0.0
    deduplicated_word_count: int = 0
    segment_count: int = 0
    blank_segment_indices: tuple[int, ...] = ()
    # Split relationship metadata. ``sequence_number`` is always the next
    # chronological session number, so it can no longer encode the relationship
    # between the parts of a safety-split utterance; these two fields carry it
    # instead. For an ordinary utterance the logical id is the acoustic
    # utterance id and the part index is 1.
    logical_utterance_id: str = ""
    continuation_part_index: int = 1


@dataclass(frozen=True)
class TranscriptFragment:
    fragment_id: str
    text: str
    arrival_time_seconds: float
    source_audio_id: str
    stt_result: STTResult | None = None
    session_id: str = ""
    batch_id: str = ""
    sequence_number: int = 0


@dataclass
class BufferOutput:
    buffer_id: str
    text: str
    fragments: list[TranscriptFragment]
    started_at_seconds: float
    flushed_at_seconds: float
    flush_reason: str
    session_id: str = ""
    batch_id: str = ""
    sequence_number: int = 0

    @property
    def fragment_count(self) -> int:
        return len(self.fragments)

    @property
    def buffer_waiting_seconds(self) -> float:
        if not self.fragments:
            return 0.0
        return max(
            0.0,
            self.flushed_at_seconds
            - max(fragment.arrival_time_seconds for fragment in self.fragments),
        )


@dataclass(frozen=True)
class MaskTask:
    buffer: BufferOutput
    enqueued_monotonic: float


@dataclass
class PIIResult:
    original_text: str
    masked_text: str
    latency_seconds: float
    entities: list[dict[str, Any]]
    safe_to_release: bool
    privacy_status: str
    error: str | None = None
    raw_model_output: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveOutput:
    buffer: BufferOutput
    pii: PIIResult
    total_seconds: float
    queue_overflow_count: int = 0
    mask_started_at_utc: str = ""
    mask_completed_at_utc: str = ""
    mask_queue_wait_seconds: float = 0.0

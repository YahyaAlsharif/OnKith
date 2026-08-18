"""Typed records shared by capture, STT, masking, and persistence lanes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecordedUtterance:
    """One completed, bounded recording ready for the STT worker.

    The owned sample array is made contiguous and read-only at construction so
    later workers cannot accidentally mutate audio that has already been
    identified and queued.
    """

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

    def __post_init__(self) -> None:
        import numpy as np

        if self.endpoint_reason not in {
            "endpoint_silence",
            "maximum_utterance",
            "shutdown",
        }:
            raise ValueError(f"Unsupported endpoint reason: {self.endpoint_reason}")
        if self.sequence_number <= 0:
            raise ValueError("sequence_number must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        audio = np.array(
            np.asarray(self.samples, dtype=np.float32).reshape(-1),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        audio.setflags(write=False)
        object.__setattr__(self, "samples", audio)

    @property
    def submitted_audio_seconds(self) -> float:
        return len(self.samples) / float(self.sample_rate)


@dataclass(frozen=True)
class STTResult:
    """One finalized transcript with its recording identity and timings."""

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
    queue_wait_seconds: float = 0.0
    total_stt_seconds: float = 0.0
    deduplicated_word_count: int = 0


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

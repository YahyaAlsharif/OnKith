"""Small runtime records shared by live STT, buffering, and masking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class STTResult:
    final_text: str
    duration_seconds: float
    processing_seconds: float
    finalization_seconds: float
    rtf: float
    callback_count: int
    reliable: bool = True
    error: str | None = None


@dataclass(frozen=True)
class TranscriptFragment:
    fragment_id: str
    text: str
    arrival_time_seconds: float
    source_audio_id: str
    stt_result: STTResult | None = None


@dataclass
class BufferOutput:
    buffer_id: str
    text: str
    fragments: list[TranscriptFragment]
    started_at_seconds: float
    flushed_at_seconds: float
    flush_reason: str

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

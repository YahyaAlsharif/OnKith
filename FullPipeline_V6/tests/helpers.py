from __future__ import annotations

import time

import numpy as np

from app.live_audio import AudioChunk
from integrated_pipeline.schemas import (
    CanarySegment,
    RecordedUtterance,
    SegmentTranscript,
    STTResult,
)


def chunk(
    index: int,
    *,
    rms: float,
    value: float = 0.0,
    sample_rate: int = 100,
) -> AudioChunk:
    frames = sample_rate // 10
    return AudioChunk(
        samples=np.full(frames, value, dtype=np.float32),
        sample_rate=sample_rate,
        captured_at=index / 10.0,
        rms=rms,
        peak=abs(value),
    )


def recording(
    sequence: int,
    *,
    session_id: str = "session-test",
    overlap: bool = False,
    endpoint_reason: str = "endpoint_silence",
) -> RecordedUtterance:
    now = time.monotonic()
    return RecordedUtterance(
        session_id=session_id,
        batch_id=f"{session_id}-batch-{sequence:06d}",
        sequence_number=sequence,
        samples=np.ones(100, dtype=np.float32),
        sample_rate=100,
        audio_started_monotonic=now - 1.0,
        audio_ended_monotonic=now,
        audio_started_at_utc="2026-07-29T10:00:00Z",
        audio_ended_at_utc="2026-07-29T10:00:01Z",
        captured_at_utc="2026-07-29T10:00:01Z",
        recording_closed_monotonic=now,
        endpoint_reason=endpoint_reason,
        reliable=True,
        dropped_audio_chunks=0,
        captured_audio_seconds=1.0,
        overlap_with_previous=overlap,
    )


def stt_result(
    item: RecordedUtterance,
    text: str,
    *,
    processing_seconds: float = 0.01,
    finalization_seconds: float = 0.01,
) -> STTResult:
    total = processing_seconds + finalization_seconds
    return STTResult(
        final_text=text,
        duration_seconds=item.submitted_audio_seconds,
        processing_seconds=processing_seconds,
        finalization_seconds=finalization_seconds,
        rtf=total / item.submitted_audio_seconds,
        callback_count=1,
        reliable=item.reliable,
        session_id=item.session_id,
        batch_id=item.batch_id,
        sequence_number=item.sequence_number,
        endpoint_reason=item.endpoint_reason,
        dropped_audio_chunks=item.dropped_audio_chunks,
        audio_started_monotonic=item.audio_started_monotonic,
        audio_ended_monotonic=item.audio_ended_monotonic,
        recording_closed_monotonic=item.recording_closed_monotonic,
        audio_started_at_utc=item.audio_started_at_utc,
        audio_ended_at_utc=item.audio_ended_at_utc,
        stt_started_at_utc="2026-07-29T10:00:01Z",
        stt_completed_at_utc="2026-07-29T10:00:02Z",
        captured_audio_seconds=item.captured_audio_seconds,
        submitted_audio_seconds=item.submitted_audio_seconds,
        queue_wait_seconds=0.0,
        total_stt_seconds=total,
    )


def canary_segment(
    index: int,
    *,
    final: bool,
    utterance: int = 1,
    start: int | None = None,
    end: int | None = None,
    reliable: bool = True,
    dropped: int = 0,
) -> CanarySegment:
    start = index * 100_000 if start is None else start
    end = start + 16_000 if end is None else end
    now = time.monotonic()
    utterance_id = f"session-test-utterance-{utterance:06d}"
    return CanarySegment(
        session_id="session-test",
        utterance_id=utterance_id,
        segment_id=f"{utterance_id}-segment-{index:06d}",
        utterance_sequence_number=utterance,
        segment_index=index,
        samples=np.full(end - start, index + 1, dtype=np.float32),
        sample_rate=16_000,
        segment_start_sample=start,
        segment_end_sample=end,
        decision_sample=end + (39_936 if final else 0),
        last_speech_end_sample=end - 5_632,
        forced_boundary_sample=None if final else end - 5_632,
        endpoint_reason="endpoint_silence" if final else "max_duration",
        is_final_segment_of_utterance=final,
        reliable=reliable,
        dropped_audio_chunks=dropped,
        recording_closed_monotonic=now,
        audio_started_monotonic=now - 2.0,
        audio_ended_monotonic=now,
        audio_started_at_utc="2026-08-05T10:00:00Z",
        audio_ended_at_utc="2026-08-05T10:00:02Z",
        utterance_start_sample=0,
        utterance_captured_end_sample=end + (39_936 if final else 0),
    )


def segment_transcript(segment: CanarySegment, text: str) -> SegmentTranscript:
    return SegmentTranscript(
        segment=segment,
        text=text,
        processing_seconds=0.1,
        queue_wait_seconds=0.02,
        stt_started_at_utc="2026-08-05T10:00:02Z",
        stt_completed_at_utc="2026-08-05T10:00:03Z",
    )

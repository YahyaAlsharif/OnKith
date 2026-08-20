"""Whisper-to-DeBERTa live privacy pipeline (V4).

V3's architecture with faster-whisper ``base.en`` in place of Canary. The
archived Canary build is in ``FullPipeline_V3/``.
"""

from .config import (
    BUFFER_INACTIVITY_SECONDS,
    MAXIMUM_BUFFER_DURATION_SECONDS,
    MAXIMUM_BUFFER_TOKENS,
    MAXIMUM_LOGICAL_UTTERANCE_SECONDS,
    MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS,
    OPERATIONAL_AUDIO_START_SAMPLE,
    PipelineConfig,
)
from .inactivity_buffer import InactivityTranscriptBuffer
from .model_v2_masker import ModelV2OnnxMasker
from .pii_adapter import PIIMaskerAdapter
from .pipeline import FailClosedMasker
from .schemas import (
    BufferOutput,
    CanarySegment,
    CanaryUtteranceEnd,
    PIIResult,
    RecordedUtterance,
    SegmentSummary,
    SegmentTranscript,
    STTResult,
    TranscriptFragment,
)
from .silero_vad import SileroOnnxVAD
from .sqlite_storage import SQLiteTranscriptStore
from .transcript_merger import merge_segment_transcripts
from .whisper_adapter import WhisperAdapter, describe_whisper_backend, name_hint

__all__ = [
    "BUFFER_INACTIVITY_SECONDS",
    "MAXIMUM_BUFFER_DURATION_SECONDS",
    "MAXIMUM_BUFFER_TOKENS",
    "MAXIMUM_LOGICAL_UTTERANCE_SECONDS",
    "MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS",
    "OPERATIONAL_AUDIO_START_SAMPLE",
    "BufferOutput",
    "CanarySegment",
    "CanaryUtteranceEnd",
    "FailClosedMasker",
    "InactivityTranscriptBuffer",
    "ModelV2OnnxMasker",
    "PIIMaskerAdapter",
    "PIIResult",
    "PipelineConfig",
    "RecordedUtterance",
    "SQLiteTranscriptStore",
    "SegmentSummary",
    "SegmentTranscript",
    "SileroOnnxVAD",
    "STTResult",
    "TranscriptFragment",
    "WhisperAdapter",
    "describe_whisper_backend",
    "merge_segment_transcripts",
    "name_hint",
]

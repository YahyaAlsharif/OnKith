"""Validated configuration for the pinned Whisper/Silero live deployment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUFFER_INACTIVITY_SECONDS = 3.0
MAXIMUM_BUFFER_TOKENS = 256
MAXIMUM_BUFFER_DURATION_SECONDS = 15.0
PII_WORD_POLICY = "whole_word_union"
ENABLE_STRUCTURED_PII = True

#: Optional key under ``pii`` holding the per-label masking policy. Absent or
#: empty means every detected label is masked, which is the historical
#: behaviour and the safe default.
MASK_POLICY_KEY = "mask_policy"
LIVE_STAGE5_MODE = "immediate"
SESSION_DATABASE_FILENAME = "session_transcript.db"

# Privacy model V2: DeBERTa-v3-xsmall INT8, 31 entity types / 63 BIO labels.
# The packaged copy under models/deberta_pii/ must stay byte-identical to the
# canonical artifact in model/DeBERTa-v3-xsmall/artifacts/model_v2/, which is
# what the 40,909-row held-out comparison measured (typed F1 0.9588, leakage
# 0.0017, overmasking 0.0029). PIIMaskerAdapter.verify_artifact enforces it.
PII_MODEL_ID = "microsoft/deberta-v3-xsmall"
PII_ONTOLOGY_VERSION = "model_v2.ontology.1"
PII_ARTIFACT_SHA256 = "bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8"
PII_ARTIFACT_BYTES = 82_278_782
PII_LABEL_COUNT = 63

# Speech recognition, V4: faster-whisper `base.en` on CTranslate2, CPU, INT8.
# V3 pinned NVIDIA Canary 180M Flash to an exact .nemo SHA-256 because that
# artifact was fetched once and committed. faster-whisper resolves `base.en`
# through CTranslate2's own model cache instead, so what is pinned here is the
# package version, the model name and the decoding settings - the things that
# change what comes out. The archived Canary build is in
# archive/FullPipeline_V3/.
WHISPER_ENGINE = "faster-whisper"
WHISPER_RUNTIME = "ctranslate2"
WHISPER_MODEL = "base.en"
WHISPER_MODEL_ID = "Systran/faster-whisper-base.en"
WHISPER_PACKAGE_VERSION = "1.2.1"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_BEAM_SIZE = 5
WHISPER_LANGUAGE = "en"

SAMPLE_RATE = 16_000
VAD_FRAME_SAMPLES = 512
INPUT_CHUNK_SAMPLES = 1_600
CALIBRATION_SAMPLES = 12_000
PRE_ROLL_SAMPLES = 6_144
START_CONFIRM_SAMPLES = 3_072
MINIMUM_SPEECH_SAMPLES = 3_072
DROPOUT_BRIDGE_SAMPLES = 2_048
ENDPOINT_SILENCE_SAMPLES = 39_936
MAXIMUM_UNIQUE_SEGMENT_SAMPLES = 136_192
FORCED_OVERLAP_SAMPLES = 5_632
TRAILING_CONTEXT_SAMPLES = 5_632

# No operational (post-calibration) audio may start before this sample. The
# first 12,000 samples are startup calibration noise and must never reach
# the recognizer, so every pre-roll and retained-audio calculation clamps to it.
OPERATIONAL_AUDIO_START_SAMPLE = CALIBRATION_SAMPLES

# Safety boundary for pathological never-ending speech. Whichever limit is
# reached first closes the current *logical downstream* utterance at a segment
# boundary; the acoustic segmentation state machine is untouched.
MAXIMUM_LOGICAL_UTTERANCE_SECONDS = 120.0
MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS = 16


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class PipelineConfig:
    """Runtime settings resolved from the single deployment JSON file."""

    root: Path = field(default_factory=package_root)
    config_path: Path | None = None
    data: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.config_path = (
            self.root / "config" / "live_config.json"
            if self.config_path is None
            else Path(self.config_path).expanduser().resolve()
        )
        self.data = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.validate()

    @property
    def whisper(self) -> dict[str, Any]:
        return self.data["whisper"]

    @property
    def silero(self) -> dict[str, Any]:
        return self.data["silero"]

    @property
    def audio(self) -> dict[str, Any]:
        return self.data["audio"]

    @property
    def buffer(self) -> dict[str, Any]:
        return self.data["buffer"]

    @property
    def pii(self) -> dict[str, Any]:
        return self.data["pii"]

    @property
    def stage5(self) -> dict[str, Any]:
        return self.data["stage5"]

    @property
    def storage(self) -> dict[str, Any]:
        return self.data["storage"]

    @property
    def silero_model_path(self) -> Path:
        return (self.root / self.silero["artifact"]).resolve()

    @property
    def pii_model_path(self) -> Path:
        return (self.root / self.pii["artifact"]).resolve()

    @property
    def pii_tokenizer_path(self) -> Path:
        return (self.root / self.pii["tokenizer"]).resolve()

    @property
    def pii_label_map_path(self) -> Path:
        return (self.root / self.pii["label_map"]).resolve()

    @property
    def session_database_path(self) -> Path:
        return (self.root / self.storage["filename"]).resolve()

    @property
    def buffer_inactivity_seconds(self) -> float:
        return float(self.buffer["inactivity_seconds"])

    @property
    def maximum_buffer_tokens(self) -> int:
        return int(self.buffer["maximum_tokens"])

    @property
    def maximum_buffer_duration_seconds(self) -> float:
        return float(self.buffer["maximum_duration_seconds"])

    @property
    def pii_word_policy(self) -> str:
        return str(self.pii["word_policy"])

    @property
    def enable_structured_pii(self) -> bool:
        return bool(self.pii["structured_enabled"])

    @property
    def mask_policy(self) -> dict[str, Any]:
        """Per-label masking policy; an absent key means "mask everything".

        Deliberately not part of :meth:`validate`'s exact-match contract. The
        policy is operator-tunable configuration rather than a pinned deployment
        invariant, and ``MaskPolicy.from_config`` already fails closed on
        anything malformed.
        """

        raw = self.pii.get(MASK_POLICY_KEY)
        return raw if isinstance(raw, dict) else {}

    @property
    def unmasked_labels(self) -> tuple[str, ...]:
        """Entity types the operator has explicitly chosen not to mask."""

        return tuple(
            sorted(label for label, enabled in self.mask_policy.items() if enabled is False)
        )

    def validate(
        self,
        *,
        require_artifacts: bool = False,
        require_stt_artifacts: bool = False,
        require_privacy_artifacts: bool = False,
    ) -> None:
        """Reject settings that differ from the authoritative production contract."""

        required = {
            "whisper.engine": (self.whisper["engine"], WHISPER_ENGINE),
            "whisper.runtime": (self.whisper["runtime"], WHISPER_RUNTIME),
            "whisper.package_version": (
                self.whisper["package_version"],
                WHISPER_PACKAGE_VERSION,
            ),
            "whisper.model": (self.whisper["model"], WHISPER_MODEL),
            "whisper.model_id": (self.whisper["model_id"], WHISPER_MODEL_ID),
            "whisper.device": (self.whisper["device"], WHISPER_DEVICE),
            "whisper.compute_type": (self.whisper["compute_type"], WHISPER_COMPUTE_TYPE),
            "whisper.beam_size": (int(self.whisper["beam_size"]), WHISPER_BEAM_SIZE),
            "whisper.language": (self.whisper["language"], WHISPER_LANGUAGE),
            "whisper.sample_rate": (int(self.whisper["sample_rate"]), SAMPLE_RATE),
            "silero.package_version": (self.silero["package_version"], "6.0.0"),
            "silero.sha256": (
                self.silero["sha256"],
                "597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004",
            ),
            "silero.size_bytes": (int(self.silero["size_bytes"]), 2_327_524),
            "silero.opset": (int(self.silero["opset"]), 16),
            "silero.sample_rate": (int(self.silero["sample_rate"]), SAMPLE_RATE),
            "silero.frame_samples": (int(self.silero["frame_samples"]), VAD_FRAME_SAMPLES),
            "silero.threshold": (float(self.silero["threshold"]), 0.5),
            "audio.sample_rate": (int(self.audio["sample_rate"]), SAMPLE_RATE),
            "audio.input_chunk_seconds": (float(self.audio["input_chunk_seconds"]), 0.1),
            "audio.input_chunk_samples": (int(self.audio["input_chunk_samples"]), INPUT_CHUNK_SAMPLES),
            "audio.calibration_seconds": (float(self.audio["calibration_seconds"]), 0.75),
            "audio.calibration_samples": (int(self.audio["calibration_samples"]), CALIBRATION_SAMPLES),
            "audio.calibration_percentile": (float(self.audio["calibration_percentile"]), 95.0),
            "audio.energy_multiplier": (float(self.audio["energy_multiplier"]), 2.25),
            "audio.minimum_energy_gate": (float(self.audio["minimum_energy_gate"]), 1e-5),
            "audio.pre_roll_samples": (int(self.audio["pre_roll_samples"]), PRE_ROLL_SAMPLES),
            "audio.start_confirm_samples": (int(self.audio["start_confirm_samples"]), START_CONFIRM_SAMPLES),
            "audio.minimum_speech_samples": (int(self.audio["minimum_speech_samples"]), MINIMUM_SPEECH_SAMPLES),
            "audio.dropout_bridge_samples": (int(self.audio["dropout_bridge_samples"]), DROPOUT_BRIDGE_SAMPLES),
            "audio.endpoint_silence_samples": (int(self.audio["endpoint_silence_samples"]), ENDPOINT_SILENCE_SAMPLES),
            "audio.maximum_unique_segment_samples": (
                int(self.audio["maximum_unique_segment_samples"]),
                MAXIMUM_UNIQUE_SEGMENT_SAMPLES,
            ),
            "audio.forced_overlap_samples": (int(self.audio["forced_overlap_samples"]), FORCED_OVERLAP_SAMPLES),
            "audio.trailing_context_samples": (
                int(self.audio["trailing_context_samples"]),
                TRAILING_CONTEXT_SAMPLES,
            ),
            "audio.adjacent_submitted_overlap_samples": (
                int(self.audio["adjacent_submitted_overlap_samples"]),
                11_264,
            ),
            "audio.queue_max_chunks": (int(self.audio["queue_max_chunks"]), 100),
            "audio.completed_segment_queue_max": (int(self.audio["completed_segment_queue_max"]), 4),
            "audio.save_debug_audio": (bool(self.audio["save_debug_audio"]), False),
            "audio.require_direct_16000_hz": (bool(self.audio["require_direct_16000_hz"]), True),
            "audio.streaming_resampler_validated": (
                bool(self.audio["streaming_resampler_validated"]),
                False,
            ),
            "buffer.inactivity_seconds": (self.buffer_inactivity_seconds, BUFFER_INACTIVITY_SECONDS),
            "buffer.maximum_tokens": (self.maximum_buffer_tokens, MAXIMUM_BUFFER_TOKENS),
            "buffer.maximum_duration_seconds": (
                self.maximum_buffer_duration_seconds,
                MAXIMUM_BUFFER_DURATION_SECONDS,
            ),
            "stage5.mode": (self.stage5["mode"], LIVE_STAGE5_MODE),
            "storage.filename": (self.storage["filename"], SESSION_DATABASE_FILENAME),
            "storage.journal_mode": (self.storage["journal_mode"], "WAL"),
            "storage.synchronous": (self.storage["synchronous"], "NORMAL"),
            "storage.foreign_keys": (bool(self.storage["foreign_keys"]), True),
            "storage.busy_timeout_ms": (int(self.storage["busy_timeout_ms"]), 5000),
            "pii.provider": (self.pii["provider"], "CPUExecutionProvider"),
            "pii.model_id": (self.pii["model_id"], PII_MODEL_ID),
            "pii.ontology_version": (self.pii["ontology_version"], PII_ONTOLOGY_VERSION),
            "pii.sha256": (self.pii["sha256"], PII_ARTIFACT_SHA256),
            "pii.size_bytes": (int(self.pii["size_bytes"]), PII_ARTIFACT_BYTES),
            "pii.context_limit": (int(self.pii["context_limit"]), 512),
            "pii.overflow_stride": (int(self.pii["overflow_stride"]), 128),
            "pii.label_count": (int(self.pii["label_count"]), PII_LABEL_COUNT),
            "pii.word_policy": (self.pii_word_policy, PII_WORD_POLICY),
            "pii.structured_enabled": (self.enable_structured_pii, ENABLE_STRUCTURED_PII),
        }
        wrong = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in required.items()
            if actual != expected
        ]
        if wrong:
            raise ValueError("Deployment configuration mismatch:\n" + "\n".join(wrong))

        if require_artifacts or require_stt_artifacts:
            # Only Silero is a committed file. Whisper's weights live in the
            # CTranslate2 / Hugging Face cache, so "is the recognizer present"
            # is an import-and-load question rather than a path question, and
            # WhisperAdapter answers it with a message naming the fix.
            paths = (self.silero_model_path,)
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Required STT artifacts are missing:\n" + "\n".join(missing)
                )
        if require_artifacts or require_privacy_artifacts:
            paths = (
                self.pii_model_path,
                self.pii_tokenizer_path / "tokenizer.json",
                self.pii_tokenizer_path / "tokenizer_config.json",
                self.pii_label_map_path,
            )
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Required privacy artifacts are missing:\n" + "\n".join(missing)
                )

    def public_dict(self) -> dict[str, Any]:
        """Return fixed settings plus exact effective durations."""

        result = json.loads(json.dumps(self.data))
        result["audio"]["effective_seconds"] = {
            "input_chunk": INPUT_CHUNK_SAMPLES / SAMPLE_RATE,
            "pre_roll": PRE_ROLL_SAMPLES / SAMPLE_RATE,
            "start_confirm": START_CONFIRM_SAMPLES / SAMPLE_RATE,
            "minimum_speech": MINIMUM_SPEECH_SAMPLES / SAMPLE_RATE,
            "dropout_bridge": DROPOUT_BRIDGE_SAMPLES / SAMPLE_RATE,
            "endpoint_silence": ENDPOINT_SILENCE_SAMPLES / SAMPLE_RATE,
            "maximum_unique_segment": MAXIMUM_UNIQUE_SEGMENT_SAMPLES / SAMPLE_RATE,
            "forced_overlap": FORCED_OVERLAP_SAMPLES / SAMPLE_RATE,
            "trailing_context": TRAILING_CONTEXT_SAMPLES / SAMPLE_RATE,
            "adjacent_submitted_overlap": 11_264 / SAMPLE_RATE,
        }
        return result

"""Validated configuration for the pinned Parakeet-TDT/Silero deployment."""

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

# Speech recognition: the exact official hybrid Parakeet checkpoint used through
# its TDT/RNNT branch. The revision and .nemo hash are the artifact measured by
# the supplied experiment, so setup cannot silently cache a different model.
PARAKEET_ENGINE = "nvidia-parakeet"
PARAKEET_RUNTIME = "nemo"
PARAKEET_MODEL = "parakeet-tdt_ctc-110m"
PARAKEET_MODEL_ID = "nvidia/parakeet-tdt_ctc-110m"
PARAKEET_REVISION = "431a349f3051ab85c22b9b7a2741b5fe77065665"
PARAKEET_CHECKPOINT_FILENAME = "parakeet-tdt_ctc-110m.nemo"
PARAKEET_CHECKPOINT_SHA256 = (
    "1e6b5c3bcaa390b479bd8112512ede515d2ef5b61495568f2c266d6d5e680c3a"
)
PARAKEET_CHECKPOINT_BYTES = 459_243_520
PARAKEET_PACKAGE_VERSION = "2.7.3"
PARAKEET_TORCH_VERSION = "2.7.1"
PARAKEET_DEVICE = "cpu"
PARAKEET_PRECISION = "fp32"
PARAKEET_DECODER_FAMILY = "TDT"
PARAKEET_DECODER_TYPE = "rnnt"
PARAKEET_DECODER_STRATEGY = "greedy_batch"
PARAKEET_GREEDY_MAX_SYMBOLS = 10
PARAKEET_TEMPERATURE = 1.0

SAMPLE_RATE = 16_000
VAD_FRAME_SAMPLES = 512
VAD_SPEECH_THRESHOLD = 0.50
VAD_NEGATIVE_THRESHOLD = 0.35
INPUT_CHUNK_SAMPLES = 1_600
CALIBRATION_SAMPLES = 12_000
SPEECH_PAD_SAMPLES = 480
# Speech pad is applied symmetrically to the configured pre-roll and trailing
# context, matching the measured Parakeet segmentation configuration.
PRE_ROLL_SAMPLES = 6_400 + SPEECH_PAD_SAMPLES
START_CONFIRM_SAMPLES = 3_200
MINIMUM_SPEECH_SAMPLES = 4_000
MINIMUM_SILENCE_SAMPLES = 1_600
# Preserve the existing segmenter's name/architecture; its short dropout bridge
# is the configured minimum-silence interval.
DROPOUT_BRIDGE_SAMPLES = MINIMUM_SILENCE_SAMPLES
ENDPOINT_SILENCE_SAMPLES = 40_000
MAXIMUM_UNIQUE_SEGMENT_SAMPLES = 136_000
FORCED_OVERLAP_SAMPLES = 6_240
TRAILING_CONTEXT_SAMPLES = 5_600 + SPEECH_PAD_SAMPLES
ADJACENT_SUBMITTED_OVERLAP_SAMPLES = (
    FORCED_OVERLAP_SAMPLES + TRAILING_CONTEXT_SAMPLES
)

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
    def parakeet(self) -> dict[str, Any]:
        return self.data["parakeet"]

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
            "parakeet.engine": (self.parakeet["engine"], PARAKEET_ENGINE),
            "parakeet.runtime": (self.parakeet["runtime"], PARAKEET_RUNTIME),
            "parakeet.package_version": (
                self.parakeet["package_version"],
                PARAKEET_PACKAGE_VERSION,
            ),
            "parakeet.torch_version": (
                self.parakeet["torch_version"],
                PARAKEET_TORCH_VERSION,
            ),
            "parakeet.model": (self.parakeet["model"], PARAKEET_MODEL),
            "parakeet.model_id": (self.parakeet["model_id"], PARAKEET_MODEL_ID),
            "parakeet.revision": (self.parakeet["revision"], PARAKEET_REVISION),
            "parakeet.checkpoint_filename": (
                self.parakeet["checkpoint_filename"],
                PARAKEET_CHECKPOINT_FILENAME,
            ),
            "parakeet.sha256": (
                self.parakeet["sha256"],
                PARAKEET_CHECKPOINT_SHA256,
            ),
            "parakeet.size_bytes": (
                int(self.parakeet["size_bytes"]),
                PARAKEET_CHECKPOINT_BYTES,
            ),
            "parakeet.device": (self.parakeet["device"], PARAKEET_DEVICE),
            "parakeet.precision": (
                self.parakeet["precision"],
                PARAKEET_PRECISION,
            ),
            "parakeet.decoder_family": (
                self.parakeet["decoder_family"],
                PARAKEET_DECODER_FAMILY,
            ),
            "parakeet.decoder_type": (
                self.parakeet["decoder_type"],
                PARAKEET_DECODER_TYPE,
            ),
            "parakeet.decoder_strategy": (
                self.parakeet["decoder_strategy"],
                PARAKEET_DECODER_STRATEGY,
            ),
            "parakeet.greedy_max_symbols": (
                int(self.parakeet["greedy_max_symbols"]),
                PARAKEET_GREEDY_MAX_SYMBOLS,
            ),
            "parakeet.temperature": (
                float(self.parakeet["temperature"]),
                PARAKEET_TEMPERATURE,
            ),
            "parakeet.timestamps": (bool(self.parakeet["timestamps"]), False),
            "parakeet.preserve_alignments": (
                bool(self.parakeet["preserve_alignments"]),
                False,
            ),
            "parakeet.sample_rate": (int(self.parakeet["sample_rate"]), SAMPLE_RATE),
            "silero.package_version": (self.silero["package_version"], "6.0.0"),
            "silero.sha256": (
                self.silero["sha256"],
                "597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004",
            ),
            "silero.size_bytes": (int(self.silero["size_bytes"]), 2_327_524),
            "silero.opset": (int(self.silero["opset"]), 16),
            "silero.sample_rate": (int(self.silero["sample_rate"]), SAMPLE_RATE),
            "silero.frame_samples": (int(self.silero["frame_samples"]), VAD_FRAME_SAMPLES),
            "silero.threshold": (
                float(self.silero["threshold"]),
                VAD_SPEECH_THRESHOLD,
            ),
            "silero.negative_threshold": (
                float(self.silero["negative_threshold"]),
                VAD_NEGATIVE_THRESHOLD,
            ),
            "audio.sample_rate": (int(self.audio["sample_rate"]), SAMPLE_RATE),
            "audio.input_chunk_seconds": (float(self.audio["input_chunk_seconds"]), 0.1),
            "audio.input_chunk_samples": (int(self.audio["input_chunk_samples"]), INPUT_CHUNK_SAMPLES),
            "audio.calibration_seconds": (float(self.audio["calibration_seconds"]), 0.75),
            "audio.calibration_samples": (int(self.audio["calibration_samples"]), CALIBRATION_SAMPLES),
            "audio.calibration_percentile": (float(self.audio["calibration_percentile"]), 95.0),
            "audio.energy_multiplier": (float(self.audio["energy_multiplier"]), 2.25),
            "audio.minimum_energy_gate": (float(self.audio["minimum_energy_gate"]), 1e-5),
            "audio.pre_roll_requested_seconds": (
                float(self.audio["pre_roll_requested_seconds"]),
                0.40,
            ),
            "audio.speech_pad_requested_seconds": (
                float(self.audio["speech_pad_requested_seconds"]),
                0.03,
            ),
            "audio.pre_roll_samples": (int(self.audio["pre_roll_samples"]), PRE_ROLL_SAMPLES),
            "audio.speech_pad_samples": (
                int(self.audio["speech_pad_samples"]),
                SPEECH_PAD_SAMPLES,
            ),
            "audio.start_confirm_samples": (int(self.audio["start_confirm_samples"]), START_CONFIRM_SAMPLES),
            "audio.start_confirm_requested_seconds": (
                float(self.audio["start_confirm_requested_seconds"]),
                0.20,
            ),
            "audio.minimum_speech_samples": (int(self.audio["minimum_speech_samples"]), MINIMUM_SPEECH_SAMPLES),
            "audio.minimum_speech_requested_seconds": (
                float(self.audio["minimum_speech_requested_seconds"]),
                0.25,
            ),
            "audio.minimum_silence_samples": (
                int(self.audio["minimum_silence_samples"]),
                MINIMUM_SILENCE_SAMPLES,
            ),
            "audio.minimum_silence_requested_seconds": (
                float(self.audio["minimum_silence_requested_seconds"]),
                0.10,
            ),
            "audio.dropout_bridge_samples": (int(self.audio["dropout_bridge_samples"]), DROPOUT_BRIDGE_SAMPLES),
            "audio.endpoint_silence_requested_seconds": (
                float(self.audio["endpoint_silence_requested_seconds"]),
                2.50,
            ),
            "audio.endpoint_silence_samples": (int(self.audio["endpoint_silence_samples"]), ENDPOINT_SILENCE_SAMPLES),
            "audio.maximum_unique_segment_requested_seconds": (
                float(self.audio["maximum_unique_segment_requested_seconds"]),
                8.50,
            ),
            "audio.maximum_unique_segment_samples": (
                int(self.audio["maximum_unique_segment_samples"]),
                MAXIMUM_UNIQUE_SEGMENT_SAMPLES,
            ),
            "audio.forced_overlap_requested_seconds": (
                float(self.audio["forced_overlap_requested_seconds"]),
                0.39,
            ),
            "audio.forced_overlap_samples": (int(self.audio["forced_overlap_samples"]), FORCED_OVERLAP_SAMPLES),
            "audio.trailing_context_requested_seconds": (
                float(self.audio["trailing_context_requested_seconds"]),
                0.35,
            ),
            "audio.trailing_context_samples": (
                int(self.audio["trailing_context_samples"]),
                TRAILING_CONTEXT_SAMPLES,
            ),
            "audio.adjacent_submitted_overlap_samples": (
                int(self.audio["adjacent_submitted_overlap_samples"]),
                ADJACENT_SUBMITTED_OVERLAP_SAMPLES,
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
            # Only Silero is committed. The exact Parakeet .nemo checkpoint is
            # verified in the Hugging Face cache by ParakeetAdapter at load.
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
            "speech_pad": SPEECH_PAD_SAMPLES / SAMPLE_RATE,
            "start_confirm": START_CONFIRM_SAMPLES / SAMPLE_RATE,
            "minimum_speech": MINIMUM_SPEECH_SAMPLES / SAMPLE_RATE,
            "minimum_silence": MINIMUM_SILENCE_SAMPLES / SAMPLE_RATE,
            "dropout_bridge": DROPOUT_BRIDGE_SAMPLES / SAMPLE_RATE,
            "endpoint_silence": ENDPOINT_SILENCE_SAMPLES / SAMPLE_RATE,
            "maximum_unique_segment": MAXIMUM_UNIQUE_SEGMENT_SAMPLES / SAMPLE_RATE,
            "forced_overlap": FORCED_OVERLAP_SAMPLES / SAMPLE_RATE,
            "trailing_context": TRAILING_CONTEXT_SAMPLES / SAMPLE_RATE,
            "adjacent_submitted_overlap": (
                ADJACENT_SUBMITTED_OVERLAP_SAMPLES / SAMPLE_RATE
            ),
        }
        return result

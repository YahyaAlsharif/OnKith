"""Validated configuration for the pinned Canary/Silero live deployment."""

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

CANARY_MODEL_ID = "nvidia/canary-180m-flash"
CANARY_ARTIFACT_FILE = "canary-180m-flash.nemo"
CANARY_REVISION = "c770bfa0f70f66ad89dbf9827985f999675943a3"
CANARY_SHA256 = "7b97de18b718ef01bf3398715ce8d18486c4d83961e230230b26f5712e112675"
CANARY_BACKEND = "nemo_cpu_fp32"
CANARY_NEMO_VERSION = "2.5.3"
CANARY_TORCH_VERSION = "2.7.1"

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
# Canary, so every pre-roll and retained-audio calculation clamps to it.
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
    def canary(self) -> dict[str, Any]:
        return self.data["canary"]

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
    def canary_model_path(self) -> Path:
        return (self.root / self.canary["artifact"]).resolve()

    @property
    def canary_manifest_path(self) -> Path:
        return (self.root / self.canary["manifest"]).resolve()

    @property
    def silero_model_path(self) -> Path:
        return (self.root / self.silero["artifact"]).resolve()

    @property
    def tinybert_model_path(self) -> Path:
        return (self.root / self.pii["artifact"]).resolve()

    @property
    def tinybert_tokenizer_path(self) -> Path:
        return (self.root / self.pii["tokenizer"]).resolve()

    @property
    def tinybert_label_map_path(self) -> Path:
        return (self.root / self.pii["id2label"]).resolve()

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
            "canary.model_id": (self.canary["model_id"], CANARY_MODEL_ID),
            "canary.artifact_file": (self.canary["artifact_file"], CANARY_ARTIFACT_FILE),
            "canary.revision": (self.canary["revision"], CANARY_REVISION),
            "canary.sha256": (self.canary["sha256"], CANARY_SHA256),
            "canary.backend": (self.canary["backend"], CANARY_BACKEND),
            "canary.nemo_version": (self.canary["nemo_version"], CANARY_NEMO_VERSION),
            "canary.torch_version": (self.canary["torch_version"], CANARY_TORCH_VERSION),
            "canary.strategy": (self.canary["strategy"], "beam"),
            "canary.beam_size": (int(self.canary["beam_size"]), 2),
            "canary.return_best_hypothesis": (bool(self.canary["return_best_hypothesis"]), True),
            "canary.temperature": (float(self.canary["temperature"]), 1.0),
            "canary.max_generation_delta": (int(self.canary["max_generation_delta"]), 50),
            "canary.batch_size": (int(self.canary["batch_size"]), 1),
            "canary.source_lang": (self.canary["source_lang"], "en"),
            "canary.target_lang": (self.canary["target_lang"], "en"),
            "canary.task": (self.canary["task"], "asr"),
            "canary.pnc": (self.canary["pnc"], "no"),
            "canary.timestamps": (bool(self.canary["timestamps"]), False),
            "canary.timestamp_prompt": (self.canary["timestamp_prompt"], "no"),
            "canary.num_workers": (int(self.canary["num_workers"]), 0),
            "canary.return_hypotheses": (bool(self.canary["return_hypotheses"]), True),
            "canary.enable_chunking": (bool(self.canary["enable_chunking"]), False),
            "canary.precision": (self.canary["precision"], "fp32"),
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
            "pii.context_limit": (int(self.pii["context_limit"]), 512),
            "pii.overflow_stride": (int(self.pii["overflow_stride"]), 128),
            "pii.label_count": (int(self.pii["label_count"]), 67),
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
            paths = (
                self.canary_model_path,
                self.canary_manifest_path,
                self.silero_model_path,
            )
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Required STT artifacts are missing:\n" + "\n".join(missing)
                )
        if require_artifacts or require_privacy_artifacts:
            paths = (
                self.tinybert_model_path,
                self.tinybert_tokenizer_path / "tokenizer.json",
                self.tinybert_tokenizer_path / "tokenizer_config.json",
                self.tinybert_label_map_path,
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

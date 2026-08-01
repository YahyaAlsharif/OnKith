"""Validated deployment configuration with package-relative artifact paths."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUFFER_INACTIVITY_SECONDS = 3.0
MAXIMUM_BUFFER_TOKENS = 256
MAXIMUM_BUFFER_DURATION_SECONDS = 15.0
MOONSHINE_SESSION_POLICY = "isolated_cached_resources"
PII_WORD_POLICY = "whole_word_union"
ENABLE_STRUCTURED_PII = True
LIVE_STAGE5_MODE = "immediate"
SESSION_DATABASE_FILENAME = "session_transcript.db"


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
    def moonshine(self) -> dict[str, Any]:
        return self.data["moonshine"]

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
    def tinybert_model_path(self) -> Path:
        return (self.root / self.pii["artifact"]).resolve()

    @property
    def tinybert_tokenizer_path(self) -> Path:
        return (self.root / self.pii["tokenizer"]).resolve()

    @property
    def tinybert_label_map_path(self) -> Path:
        return (self.root / self.pii["id2label"]).resolve()

    @property
    def moonshine_config_path(self) -> Path:
        return (self.root / "moonshine_runtime" / "final_config.json").resolve()

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
    def moonshine_session_policy(self) -> str:
        return str(self.moonshine["session_policy"])

    @property
    def pii_word_policy(self) -> str:
        return str(self.pii["word_policy"])

    @property
    def enable_structured_pii(self) -> bool:
        return bool(self.pii["structured_enabled"])

    def validate(self, *, require_artifacts: bool = False) -> None:
        """Reject any configuration that changes the accepted deployment."""

        required = {
            "moonshine.language": (self.moonshine["language"], "en"),
            "moonshine.model_arch": (
                self.moonshine["model_arch"],
                "medium_streaming",
            ),
            "moonshine.package_version": (
                self.moonshine["package_version"],
                "0.0.69",
            ),
            "moonshine.session_policy": (
                self.moonshine_session_policy,
                MOONSHINE_SESSION_POLICY,
            ),
            "moonshine.rnnoise_enabled": (
                self.moonshine["rnnoise_enabled"],
                False,
            ),
            "moonshine.chunk_seconds": (
                float(self.moonshine["chunk_seconds"]),
                0.1,
            ),
            "moonshine.update_interval_seconds": (
                float(self.moonshine["update_interval_seconds"]),
                0.25,
            ),
            "moonshine.vad_threshold": (
                float(self.moonshine["vad_threshold"]),
                0.5,
            ),
            "moonshine.vad_window_duration_seconds": (
                float(self.moonshine["vad_window_duration_seconds"]),
                0.3,
            ),
            "moonshine.vad_max_segment_duration_seconds": (
                float(self.moonshine["vad_max_segment_duration_seconds"]),
                10.0,
            ),
            "moonshine.max_tokens_per_second": (
                float(self.moonshine["max_tokens_per_second"]),
                6.5,
            ),
            "audio.calibration_seconds": (
                float(self.audio["calibration_seconds"]),
                2.0,
            ),
            "audio.pre_roll_seconds": (
                float(self.audio["pre_roll_seconds"]),
                0.4,
            ),
            "audio.start_confirm_seconds": (
                float(self.audio["start_confirm_seconds"]),
                0.2,
            ),
            "audio.endpoint_silence_seconds": (
                float(self.audio["endpoint_silence_seconds"]),
                2.5,
            ),
            "audio.minimum_speech_seconds": (
                float(self.audio["minimum_speech_seconds"]),
                0.6,
            ),
            "audio.maximum_utterance_seconds": (
                float(self.audio["maximum_utterance_seconds"]),
                8.5,
            ),
            "audio.minimum_start_rms": (
                float(self.audio["minimum_start_rms"]),
                0.006,
            ),
            "audio.noise_multiplier": (
                float(self.audio["noise_multiplier"]),
                4.0,
            ),
            "audio.continue_ratio": (
                float(self.audio["continue_ratio"]),
                0.6,
            ),
            "audio.queue_max_chunks": (
                int(self.audio["queue_max_chunks"]),
                100,
            ),
            "audio.save_debug_audio": (
                bool(self.audio["save_debug_audio"]),
                False,
            ),
            "audio.forced_split_overlap_seconds": (
                float(self.audio["forced_split_overlap_seconds"]),
                0.35,
            ),
            "audio.trailing_silence_context_seconds": (
                float(self.audio["trailing_silence_context_seconds"]),
                0.35,
            ),
            "audio.completed_recording_queue_max": (
                int(self.audio["completed_recording_queue_max"]),
                4,
            ),
            "buffer.inactivity_seconds": (
                self.buffer_inactivity_seconds,
                BUFFER_INACTIVITY_SECONDS,
            ),
            "buffer.maximum_tokens": (
                self.maximum_buffer_tokens,
                MAXIMUM_BUFFER_TOKENS,
            ),
            "buffer.maximum_duration_seconds": (
                self.maximum_buffer_duration_seconds,
                MAXIMUM_BUFFER_DURATION_SECONDS,
            ),
            "stage5.mode": (self.stage5["mode"], LIVE_STAGE5_MODE),
            "storage.filename": (
                self.storage["filename"],
                SESSION_DATABASE_FILENAME,
            ),
            "storage.journal_mode": (self.storage["journal_mode"], "WAL"),
            "storage.synchronous": (
                self.storage["synchronous"],
                "NORMAL",
            ),
            "storage.foreign_keys": (
                bool(self.storage["foreign_keys"]),
                True,
            ),
            "storage.busy_timeout_ms": (
                int(self.storage["busy_timeout_ms"]),
                5000,
            ),
            "pii.provider": (self.pii["provider"], "CPUExecutionProvider"),
            "pii.context_limit": (int(self.pii["context_limit"]), 512),
            "pii.overflow_stride": (int(self.pii["overflow_stride"]), 128),
            "pii.label_count": (int(self.pii["label_count"]), 67),
            "pii.word_policy": (self.pii_word_policy, PII_WORD_POLICY),
            "pii.structured_enabled": (
                self.enable_structured_pii,
                ENABLE_STRUCTURED_PII,
            ),
        }
        wrong = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in required.items()
            if actual != expected
        ]
        if wrong:
            raise ValueError("Deployment configuration mismatch:\n" + "\n".join(wrong))

        if require_artifacts:
            paths = (
                self.tinybert_model_path,
                self.tinybert_tokenizer_path / "tokenizer.json",
                self.tinybert_tokenizer_path / "tokenizer_config.json",
                self.tinybert_label_map_path,
                self.moonshine_config_path,
            )
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Required deployment artifacts are missing:\n" + "\n".join(missing)
                )

    def public_dict(self) -> dict[str, Any]:
        """Return configuration values without machine-specific paths."""

        return json.loads(json.dumps(self.data))

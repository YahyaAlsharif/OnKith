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
    def light(self) -> dict[str, Any]:
        # Absent rather than required, because a deployment with no lamp
        # wired to it is a valid deployment and the defaults in
        # LightController cover it.
        return self.data.get("light", {})

    @property
    def stage5(self) -> dict[str, Any]:
        return self.data["stage5"]

    @property
    def wake(self) -> dict[str, Any]:
        """What the device answers to.

        Not checked against an accepted value the way everything else here
        is, because unlike everything else here it is meant to be changed. A
        device that cannot be renamed is one with its name compiled in.
        """

        return self.data.get("wake", {})

    @property
    def never_masked_entities(self) -> tuple[str, ...]:
        """Categories the classifier finds and the device releases anyway.

        Asked what time it was in Saudi Arabia, the model was handed "what
        time is it right now in [CITY_1] Arabia" and answered, correctly,
        that it did not know. Riyadh scores 0.996 and Jeddah scores 1.0, so
        no confidence threshold reaches this: the classifier is not
        mistaken, it is doing what it was trained to do.

        A city or a country on its own does not identify anybody, and
        redacting it costs every question about the time, the weather or
        the way somewhere. A street, a building number and a postcode still
        do identify somebody, so they are still masked. That line is the
        judgement here, and it lives in config so it can be moved.
        """

        configured = self.pii.get("never_mask", ["CITY", "COUNTRY", "TIMEZONE"])
        return tuple(
            str(name).strip().upper() for name in configured if str(name).strip()
        )

    @property
    def never_masked_words(self) -> tuple[str, ...]:
        """The device's own names, which are not somebody's personal data.

        TinyBERT reads Onkith as a person's given name, correctly as far
        as it knows, and redacts it. Every utterance to this device begins
        with its name, so every utterance arrived at the assistant as
        "[GIVENNAME_1], turn the light off". The reply is rehydrated on the
        way out, so the model answering "users like you, [GIVENNAME_1]"
        says the name out loud, which is the thing the wake word strip was
        written to stop.

        Seeded from the wake list rather than written out, so renaming the
        device keeps this true. The trade is real and worth stating: Sada
        and Echo are names people have, so a person called Echo is no
        longer masked by name in a transcript. Only an exact whole-span
        match is released, so "Echo Alshuaibi" still goes, and the device
        being unable to hear its own name on every single utterance is the
        larger harm.
        """

        wake = self.wake
        words = [*(wake.get("names") or []), *(wake.get("spellings") or [])]
        words.extend(self.pii.get("never_mask_words", []))
        return tuple(
            sorted({str(word).strip().casefold() for word in words if str(word).strip()})
        )

    @property
    def stt(self) -> dict[str, Any]:
        """Which recogniser listens, and how.

        Not checked against an accepted value, for the same reason the wake
        words are not: it is the one setting being changed on purpose while
        two recognisers are compared side by side. Absent, it means the
        Moonshine deployment everything else in this file describes.
        """

        return self.data.get("stt", {})

    @property
    def stt_engine(self) -> str:
        return str(self.stt.get("engine", "moonshine")).strip().casefold()

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
            "audio.speech_tail_seconds": (
                float(self.audio["speech_tail_seconds"]),
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

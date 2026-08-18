"""Small Moonshine SDK helpers used by the deployment adapter."""

from __future__ import annotations

import importlib.metadata
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class MoonshineRuntimeConfig:
    model_language: str
    model_arch: str
    tested_moonshine_voice_version: str
    rnnoise_enabled: bool
    update_interval_seconds: float
    vad_threshold: float
    vad_window_duration_seconds: float
    vad_max_segment_duration_seconds: float
    max_tokens_per_second: float
    block_seconds: float
    calibration_seconds: float
    pre_roll_seconds: float
    start_confirm_seconds: float
    end_silence_seconds: float
    min_speech_seconds: float
    max_utterance_seconds: float
    min_start_rms: float
    noise_multiplier: float
    continue_ratio: float
    queue_max_chunks: int
    save_debug_audio: bool

    @classmethod
    def from_json(cls, path: Path) -> "MoonshineRuntimeConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.model_language != "en" or self.model_arch != "medium_streaming":
            raise ValueError("English Moonshine Medium Streaming is required")
        if self.tested_moonshine_voice_version != "0.0.69":
            raise ValueError("moonshine-voice 0.0.69 is required")
        if self.rnnoise_enabled:
            raise ValueError("RNNoise must remain disabled")
        if self.save_debug_audio:
            raise ValueError("Debug audio saving must remain disabled")
        required = {
            "update_interval_seconds": (self.update_interval_seconds, 0.25),
            "vad_threshold": (self.vad_threshold, 0.5),
            "vad_window_duration_seconds": (
                self.vad_window_duration_seconds,
                0.3,
            ),
            "vad_max_segment_duration_seconds": (
                self.vad_max_segment_duration_seconds,
                10.0,
            ),
            "max_tokens_per_second": (self.max_tokens_per_second, 6.5),
            "block_seconds": (self.block_seconds, 0.1),
            "calibration_seconds": (self.calibration_seconds, 2.0),
            "pre_roll_seconds": (self.pre_roll_seconds, 0.4),
            "start_confirm_seconds": (self.start_confirm_seconds, 0.2),
            "end_silence_seconds": (self.end_silence_seconds, 0.85),
            "min_speech_seconds": (self.min_speech_seconds, 0.6),
            "max_utterance_seconds": (self.max_utterance_seconds, 20.0),
            "min_start_rms": (self.min_start_rms, 0.006),
            "noise_multiplier": (self.noise_multiplier, 4.0),
            "continue_ratio": (self.continue_ratio, 0.6),
            "queue_max_chunks": (self.queue_max_chunks, 100),
        }
        wrong = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in required.items()
            if actual != expected
        ]
        if wrong:
            raise ValueError(
                "Moonshine deployment configuration changed: " + ", ".join(wrong)
            )


def verify_package_version(expected: str = "0.0.69") -> str:
    installed = importlib.metadata.version("moonshine-voice")
    if installed != expected:
        raise RuntimeError(
            f"moonshine-voice {expected} is required; installed {installed}"
        )
    return installed


def line_id_from_event(event: Any) -> int:
    line = event.line
    for name in ("line_id", "lineId", "id"):
        value = getattr(line, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return id(line)


class TranscriptCollector:
    """Fresh, thread-safe text state for exactly one Moonshine utterance."""

    def __init__(self, partial_callback: Callable[[str], None] | None = None):
        self._lock = threading.Lock()
        self._lines: OrderedDict[int, str] = OrderedDict()
        self._last_visible = ""
        self._partial_callback = partial_callback
        self.callback_count = 0

    def update(self, event: Any) -> None:
        key = line_id_from_event(event)
        text = (getattr(event.line, "text", "") or "").strip()
        callback_text: str | None = None
        with self._lock:
            self.callback_count += 1
            self._lines[key] = text
            visible = " ".join(value for value in self._lines.values() if value).strip()
            if visible and visible != self._last_visible:
                self._last_visible = visible
                callback_text = visible
        if callback_text and self._partial_callback is not None:
            self._partial_callback(callback_text)

    def final_text(self) -> str:
        with self._lock:
            return " ".join(
                value for value in self._lines.values() if value
            ).strip()

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._last_visible = ""
            self._partial_callback = None

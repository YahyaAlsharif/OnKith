"""Deterministic transcript buffering for the fixed inactivity policy."""

from __future__ import annotations

import math
import re
import threading
from dataclasses import replace

from .config import (
    BUFFER_INACTIVITY_SECONDS,
    MAXIMUM_BUFFER_DURATION_SECONDS,
    MAXIMUM_BUFFER_TOKENS,
)
from .schemas import BufferOutput, TranscriptFragment

_VALID_REASONS = {
    "inactivity_timeout",
    "maximum_tokens",
    "maximum_duration",
    "shutdown",
    "manual",
}


class InactivityTranscriptBuffer:
    """Thread-safe logical buffer; a wall-clock worker calls ``flush_due``."""

    def __init__(
        self,
        inactivity_seconds: float = BUFFER_INACTIVITY_SECONDS,
        maximum_tokens: int = MAXIMUM_BUFFER_TOKENS,
        maximum_duration_seconds: float = MAXIMUM_BUFFER_DURATION_SECONDS,
    ):
        if float(inactivity_seconds) != BUFFER_INACTIVITY_SECONDS:
            raise ValueError("Live deployment inactivity must be exactly 3.0 seconds")
        if maximum_tokens <= 0 or maximum_duration_seconds <= 0:
            raise ValueError("Buffer safety caps must be positive")
        self.inactivity_seconds = float(inactivity_seconds)
        self.maximum_tokens = int(maximum_tokens)
        self.maximum_duration_seconds = float(maximum_duration_seconds)
        self._lock = threading.RLock()
        self._fragments: list[TranscriptFragment] = []
        self._started_at: float | None = None
        self._last_arrival: float | None = None
        self._sequence = 0

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @property
    def pending_fragment_count(self) -> int:
        with self._lock:
            return len(self._fragments)

    @property
    def next_deadline(self) -> float | None:
        with self._lock:
            if self._last_arrival is None:
                return None
            return self._last_arrival + self.inactivity_seconds

    def _token_count(self) -> int:
        return sum(len(fragment.text.split()) for fragment in self._fragments)

    def add_fragment(self, fragment: TranscriptFragment) -> list[BufferOutput]:
        """Merge before the deadline; flush first at or after the deadline."""

        timestamp = float(fragment.arrival_time_seconds)
        if not math.isfinite(timestamp):
            raise ValueError("Fragment arrival time must be finite")
        text = self._clean(fragment.text)
        if not text:
            return []
        with self._lock:
            if self._last_arrival is not None and timestamp < self._last_arrival:
                raise ValueError("Fragments must use nondecreasing monotonic time")

            outputs: list[BufferOutput] = []
            if self._last_arrival is not None:
                deadline = self._last_arrival + self.inactivity_seconds
                if timestamp >= deadline:
                    output = self._flush_locked(timestamp, "inactivity_timeout")
                    if output is not None:
                        outputs.append(output)

            incoming_tokens = len(text.split())
            if self._fragments and self._token_count() + incoming_tokens > self.maximum_tokens:
                output = self._flush_locked(timestamp, "maximum_tokens")
                if output is not None:
                    outputs.append(output)
            elif (
                self._fragments
                and self._started_at is not None
                and timestamp - self._started_at >= self.maximum_duration_seconds
            ):
                output = self._flush_locked(timestamp, "maximum_duration")
                if output is not None:
                    outputs.append(output)

            normalized = replace(fragment, text=text)
            if not self._fragments:
                self._started_at = timestamp
            self._fragments.append(normalized)
            self._last_arrival = timestamp

            if incoming_tokens >= self.maximum_tokens:
                output = self._flush_locked(timestamp, "maximum_tokens")
                if output is not None:
                    outputs.append(output)
            return outputs

    def flush_due(self, current_time_seconds: float) -> BufferOutput | None:
        with self._lock:
            if self._last_arrival is None:
                return None
            deadline = self._last_arrival + self.inactivity_seconds
            if float(current_time_seconds) < deadline:
                return None
            return self._flush_locked(
                float(current_time_seconds), "inactivity_timeout"
            )

    def flush(
        self, current_time_seconds: float, reason: str = "manual"
    ) -> BufferOutput | None:
        with self._lock:
            return self._flush_locked(float(current_time_seconds), reason)

    def _flush_locked(
        self, current_time_seconds: float, reason: str
    ) -> BufferOutput | None:
        if reason not in _VALID_REASONS:
            raise ValueError(f"Unsupported flush reason: {reason}")
        if not self._fragments:
            return None
        if not math.isfinite(current_time_seconds):
            raise ValueError("Flush time must be finite")
        assert self._started_at is not None
        if current_time_seconds < self._started_at:
            raise ValueError("A buffer cannot flush before it starts")
        self._sequence += 1
        fragments = list(self._fragments)
        output = BufferOutput(
            buffer_id=f"buffer-{self._sequence:06d}",
            text=" ".join(fragment.text for fragment in fragments),
            fragments=fragments,
            started_at_seconds=self._started_at,
            flushed_at_seconds=current_time_seconds,
            flush_reason=reason,
        )
        self._fragments.clear()
        self._started_at = None
        self._last_arrival = None
        return output

    def reset(self) -> None:
        with self._lock:
            self._fragments.clear()
            self._started_at = None
            self._last_arrival = None

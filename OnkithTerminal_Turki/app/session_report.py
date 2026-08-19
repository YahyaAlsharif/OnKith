"""One artifact describing a run, instead of six lines nobody reads.

Everything here was already being measured. It was printed as separate
tagged lines during shutdown, in raw form, and under the live screen
those lines go through the status log and scroll away, so in practice
the numbers existed and nobody had them. This collects them, keeps
distributions rather than only totals, and prints one report.

What it deliberately does not contain is accuracy. Word error rate,
wake recall and intent accuracy all need ground truth, and a live
session has none: nobody typed in what they actually said. Those come
from `eval/run_eval.py` against the corpus, and the report says so
rather than offering a number that looks like accuracy and is not.

Averages are avoided where a percentile will do. A mean latency hides
the one utterance in twenty that took four seconds, and that utterance
is the one somebody standing in front of the device remembers.
"""

from __future__ import annotations

import json
import platform
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Below this many samples a p95 is the maximum wearing a hat, so the
# report prints the raw values instead of implying a distribution.
MEANINGFUL_SAMPLE_COUNT = 5


class Series:
    """A distribution kept without keeping every reading forever."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: float | None) -> None:
        if value is None:
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        # NaN sorts unpredictably and would corrupt every percentile
        # below it. RTF arrives as nan whenever the audio length was
        # zero, which happens on an empty final.
        if number != number:
            return
        self.values.append(number)

    def __len__(self) -> int:
        return len(self.values)

    def percentile(self, fraction: float) -> float | None:
        if not self.values:
            return None
        ordered = sorted(self.values)
        index = min(
            len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1)))
        )
        return ordered[index]

    @property
    def mean(self) -> float | None:
        return sum(self.values) / len(self.values) if self.values else None

    @property
    def peak(self) -> float | None:
        return max(self.values) if self.values else None

    def summary(self) -> dict[str, float | int | None]:
        return {
            "count": len(self.values),
            "p50": self.percentile(0.50),
            "p95": self.percentile(0.95),
            "max": self.peak,
            "mean": self.mean,
        }


@dataclass
class SessionReport:
    """Collects a run and renders it. Every observer is failure-safe.

    These are called from the capture callback, the STT thread and the
    assistant thread. None of them may raise into a caller whose real
    job is to keep audio moving, so nothing here validates and nothing
    here throws.
    """

    session_id: str = ""
    started_at: float = field(default_factory=time.time)
    started_at_utc: str = ""
    stt_engine: str = ""
    voice: str = ""
    assistant_model: str = ""
    microphone: str = ""

    def __post_init__(self) -> None:
        self.started_at_utc = self.started_at_utc or datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")
        self.audio_seconds = Series()
        self.stt_seconds = Series()
        self.stt_queue_wait = Series()
        self.rtf = Series()
        self.mask_seconds = Series()
        self.mask_queue_wait = Series()
        self.classifier_seconds = Series()
        self.end_to_end = Series()
        self.reply_seconds = Series()
        self.input_level = Series()

        self.utterances = 0
        self.unreliable_utterances = 0
        self.blocked_utterances = 0
        self.empty_finals = 0
        self.commands_recognised = 0
        self.commands_by_intent: Counter[str] = Counter()
        self.fell_through = 0
        self.near_misses: Counter[str] = Counter()
        self.entities: Counter[str] = Counter()
        self.replies = 0
        self.reply_failures = 0
        self.dropped_chunks = 0
        self.threshold = 0.0
        self.threshold_capped = False
        self.audit: dict[str, Any] = {}

    # -- observers ---------------------------------------------------

    def observe_level(self, rms: Any) -> None:
        """Called from the capture callback, so it stays cheap."""
        self.input_level.add(rms)

    def observe_event(self, name: str, payload: dict[str, Any]) -> None:
        try:
            self._observe_event(name, payload)
        except Exception:
            pass

    def _observe_event(self, name: str, payload: dict[str, Any]) -> None:
        if name == "calibration_complete":
            self.threshold = float(payload.get("start_threshold") or 0.0)
            self.threshold_capped = bool(payload.get("capped"))
        elif name == "stt_finalized":
            self.rtf.add(payload.get("rtf"))
            self.stt_queue_wait.add(payload.get("queue_wait_seconds"))
            if not payload.get("reliable", True):
                self.unreliable_utterances += 1
        elif name == "queue_overflow":
            self.dropped_chunks += int(payload.get("new_dropped_chunks") or 0)
        elif name == "empty_final_ignored":
            self.empty_finals += 1
        elif name == "onkith_command_detected":
            self.commands_recognised += 1
            self.commands_by_intent[str(payload.get("intent") or "?")] += 1
        elif name == "onkith_no_command":
            self.fell_through += 1
            # The recogniser's own spelling of something close to the
            # device's name. One line in config/live_config.json fixes
            # it, and nobody can write that line without seeing this.
            nearest = payload.get("nearest")
            if nearest:
                self.near_misses[str(nearest)] += 1

    def observe_output(self, result: Any) -> None:
        try:
            self._observe_output(result)
        except Exception:
            pass

    def _observe_output(self, result: Any) -> None:
        self.utterances += 1
        pii = result.pii
        if not pii.safe_to_release:
            self.blocked_utterances += 1
            return
        for entity in pii.entities:
            self.entities[str(entity.get("entity_type") or "?")] += 1
        self.mask_seconds.add(pii.latency_seconds)
        self.mask_queue_wait.add(result.mask_queue_wait_seconds)
        self.end_to_end.add(result.total_seconds)
        details = pii.raw_model_output or {}
        self.classifier_seconds.add(
            details.get("pii_model_latency_seconds")
        )
        for fragment in result.buffer.fragments:
            stt = fragment.stt_result
            if stt is None:
                continue
            self.stt_seconds.add(stt.total_stt_seconds)
            self.audio_seconds.add(stt.submitted_audio_seconds)

    def observe_reply(self, seconds: float, failed: bool = False) -> None:
        if failed:
            self.reply_failures += 1
        else:
            self.replies += 1
            self.reply_seconds.add(seconds)

    # -- rendering ---------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at_utc": self.started_at_utc,
            "duration_seconds": time.time() - self.started_at,
            "platform": f"{platform.system()} {platform.machine()}",
            "stt_engine": self.stt_engine,
            "voice": self.voice,
            "assistant_model": self.assistant_model,
            "microphone": self.microphone,
            "hearing": {
                "utterances": self.utterances,
                "commands_recognised": self.commands_recognised,
                "commands_by_intent": dict(self.commands_by_intent),
                "fell_through_to_assistant": self.fell_through,
                "near_misses": dict(self.near_misses),
                "empty_finals": self.empty_finals,
                "threshold": self.threshold,
                "threshold_capped": self.threshold_capped,
                "input_level": self.input_level.summary(),
            },
            "latency_seconds": {
                "submitted_audio": self.audio_seconds.summary(),
                "stt_total": self.stt_seconds.summary(),
                "stt_queue_wait": self.stt_queue_wait.summary(),
                "stt_rtf": self.rtf.summary(),
                "mask_queue_wait": self.mask_queue_wait.summary(),
                "pii_classifier": self.classifier_seconds.summary(),
                "mask_total": self.mask_seconds.summary(),
                "assistant_reply": self.reply_seconds.summary(),
                "endpoint_to_screen": self.end_to_end.summary(),
            },
            "throughput": dict(self.audit),
            "privacy": {
                "entities": dict(self.entities),
                "blocked_utterances": self.blocked_utterances,
                "unreliable_utterances": self.unreliable_utterances,
                "dropped_chunks": self.dropped_chunks,
            },
            "assistant": {
                "replies": self.replies,
                "failures": self.reply_failures,
            },
        }

    def text(self) -> str:
        lines: list[str] = ["Onkith session report", ""]
        duration = time.time() - self.started_at
        for label, value in (
            ("session", self.session_id),
            ("started", self.started_at_utc),
            ("ran for", _duration(duration)),
            ("platform", f"{platform.system()} {platform.machine()}"),
            ("recogniser", self.stt_engine),
            ("voice", self.voice),
            ("assistant", self.assistant_model),
            ("microphone", self.microphone),
        ):
            if value:
                lines.append(f"  {label:<20} {value}")

        lines.extend(["", "Hearing"])
        lines.append(f"  {'utterances released':<20} {self.utterances}")
        lines.append(
            f"  {'commands recognised':<20} {self.commands_recognised}"
            + (
                "   " + ", ".join(
                    f"{name} {count}"
                    for name, count in self.commands_by_intent.most_common()
                )
                if self.commands_by_intent
                else ""
            )
        )
        lines.append(f"  {'sent to the model':<20} {self.fell_through}")
        if self.empty_finals:
            lines.append(f"  {'empty transcripts':<20} {self.empty_finals}")
        if self.near_misses:
            lines.append(
                f"  {'near misses':<20} "
                + ", ".join(
                    f'"{text}" x{count}'
                    for text, count in self.near_misses.most_common(5)
                )
            )
            lines.append(
                f"  {'':<20} add these to wake.spellings in "
                "config/live_config.json"
            )
        lines.append(
            f"  {'speech threshold':<20} {self.threshold:.5f}"
            + ("  (capped: the room was noisy)" if self.threshold_capped else "")
        )
        # Never measured before, and the suspicion that the microphone is
        # too quiet has been open since the first Pi run. A peak well
        # under the threshold is a device that cannot hear anybody.
        if len(self.input_level):
            lines.append(
                f"  {'input level':<20} "
                f"mean {self.input_level.mean:.4f}  "
                f"p95 {self.input_level.percentile(0.95):.4f}  "
                f"peak {self.input_level.peak:.4f}"
            )
            if self.input_level.peak is not None and self.threshold > 0:
                headroom = self.input_level.peak / self.threshold
                # A low ratio in a session where somebody did speak is a
                # microphone that cannot clear its own gate. In a session
                # where nothing was captured it means only that the room
                # was quiet, and calling that a fault would send somebody
                # to replace a working microphone.
                if headroom >= 2.0:
                    verdict = ""
                elif self.utterances:
                    verdict = "   barely clears its own threshold"
                else:
                    verdict = "   nothing crossed it: a quiet room, or a quiet mic"
                lines.append(
                    f"  {'loudest / threshold':<20} {headroom:.1f}x{verdict}"
                )

        lines.extend(["", "Latency, seconds"])
        lines.append(
            f"  {'':<22} {'p50':>8} {'p95':>8} {'max':>8} {'n':>4}"
        )
        for label, series in (
            ("audio submitted", self.audio_seconds),
            ("STT total", self.stt_seconds),
            ("STT queue wait", self.stt_queue_wait),
            ("STT real-time factor", self.rtf),
            ("mask queue wait", self.mask_queue_wait),
            ("PII classifier", self.classifier_seconds),
            ("masking total", self.mask_seconds),
            ("assistant reply", self.reply_seconds),
            ("speech to screen", self.end_to_end),
        ):
            lines.append(_latency_row(label, series))
        if 0 < len(self.end_to_end) < MEANINGFUL_SAMPLE_COUNT:
            lines.append(
                f"  under {MEANINGFUL_SAMPLE_COUNT} utterances, so p95 is "
                "the maximum under another name"
            )

        lines.extend(["", "Keeping up"])
        audit = self.audit
        for label, key, ceiling in (
            ("audio queue high", "audio_queue_high_water_mark", None),
            ("recording backlog", "completed_queue_high_water_mark", None),
            ("backlog overloads", "completed_queue_overload_count", 0),
            ("dropped chunks", "queue_overflow_count", 0),
            ("capture callback avg", "callback_average_seconds", None),
            ("capture callback max", "callback_max_seconds", None),
            ("mask queue high", "mask_queue_high_water_mark", None),
            ("DB queue high", "db_queue_high_water_mark", None),
            ("DB wait max", "db_maximum_queue_wait_seconds", None),
            ("DB write max", "db_maximum_write_seconds", None),
            ("DB operations", "db_operation_count", None),
            ("model reloads", "model_resource_loads", None),
            ("utterance sessions", "utterance_sessions", None),
        ):
            if key not in audit:
                continue
            value = audit[key]
            text = f"{value:.6f}s" if isinstance(value, float) else str(value)
            flag = "   <- not zero" if ceiling == 0 and value else ""
            lines.append(f"  {label:<20} {text}{flag}")
        if self.unreliable_utterances:
            lines.append(
                f"  {'unreliable output':<20} {self.unreliable_utterances}"
                "   <- affected by capture overflow"
            )
        if "audit_error" in audit:
            lines.append(f"  {'incomplete':<20} {audit['audit_error']}")

        lines.extend(["", "Privacy"])
        lines.append(
            f"  {'entities masked':<20} "
            + (
                ", ".join(
                    f"{name} {count}"
                    for name, count in self.entities.most_common()
                )
                or "none"
            )
        )
        lines.append(f"  {'blocked fail-closed':<20} {self.blocked_utterances}")

        lines.extend(["", "Assistant"])
        lines.append(f"  {'replies':<20} {self.replies}")
        lines.append(
            f"  {'failures':<20} {self.reply_failures}"
            + ("   <- see the warnings above" if self.reply_failures else "")
        )

        lines.extend(
            [
                "",
                "Accuracy is not in this report. Word error rate, wake recall",
                "and intent accuracy need ground truth, and nobody typed in",
                "what was actually said. Measure them on the corpus:",
                "",
                "  .venv/bin/python eval/run_eval.py --engine whisper --no-record",
            ]
        )
        return "\n".join(lines)

    def write_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_dict(), handle, indent=2, sort_keys=True)


def _duration(seconds: float) -> str:
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {remainder:02d}s"
    if minutes:
        return f"{minutes}m {remainder:02d}s"
    return f"{remainder}s"


def _latency_row(label: str, series: Series) -> str:
    if not len(series):
        return f"  {label:<22}        -        -        -    0"
    return (
        f"  {label:<22} "
        f"{series.percentile(0.50):8.3f} "
        f"{series.percentile(0.95):8.3f} "
        f"{series.peak:8.3f} "
        f"{len(series):4d}"
    )

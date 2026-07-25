"""Privacy-aware terminal rendering."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from typing import Any

from integrated_pipeline.schemas import LiveOutput


class TerminalUI:
    def __init__(
        self,
        *,
        show_raw: bool = False,
        show_partials: bool = False,
        json_output: bool = False,
        quiet: bool = False,
        colour: bool = True,
    ):
        self.show_raw = show_raw
        self.show_partials = show_partials
        self.json_output = json_output
        self.quiet = quiet
        self.colour = colour and sys.stdout.isatty()

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.colour else text

    def line(self, tag: str, message: str, *, force: bool = False) -> None:
        if self.quiet and not force:
            return
        if self.json_output:
            print(json.dumps({"event": tag.lower(), "message": message}))
            return
        print(f"{self._paint(f'[{tag}]', '1;36')} {message}", flush=True)

    def event(self, name: str, data: dict[str, Any]) -> None:
        if name == "microphone_open":
            self.line(
                "MIC",
                f"{data['name']}, device {data['device_id']}, "
                f"{data['sample_rate']} Hz",
            )
        elif name == "calibration_started":
            self.line(
                "CALIBRATION",
                f"Stay silent for {data['seconds']:.1f} seconds",
            )
        elif name == "calibration_complete":
            self.line(
                "READY",
                "Listening continuously "
                f"(threshold={data['start_threshold']:.5f})",
            )
        elif name == "speech_started":
            self.line("SPEECH", "Voice detected")
        elif name == "stt_finalized":
            reliability = "reliable" if data["reliable"] else "UNRELIABLE"
            rtf = data["rtf"]
            rtf_text = f"{rtf:.3f}" if math.isfinite(rtf) else "n/a"
            self.line(
                "STT",
                f"Utterance finalized ({reliability}, "
                f"callbacks={data['callback_count']}, RTF={rtf_text})",
            )
        elif name == "buffer_waiting":
            self.line(
                "BUFFER",
                f"Waiting up to {data['seconds']:.1f} seconds for more context",
            )
        elif name == "queue_overflow":
            self.line(
                "WARNING",
                "Audio queue overflow: "
                f"{data['new_dropped_chunks']} new chunk(s) dropped; "
                "the affected utterance is unreliable",
                force=True,
            )
        elif name in {"utterance_error", "microphone_status"}:
            self.line("WARNING", str(data), force=True)
        elif name == "empty_final_ignored":
            self.line("STT", "Empty finalized fragment ignored")

    def partial(self, text: str) -> None:
        if not self.show_partials:
            return
        if self.json_output:
            print(json.dumps({"event": "partial", "text": text}))
        else:
            print(f"[PARTIAL] {text}", flush=True)

    def output(self, result: LiveOutput) -> None:
        pii = result.pii
        if not pii.safe_to_release:
            self.line(
                "BLOCKED",
                "PII masking failed; transcript was not released",
                force=True,
            )
            if pii.error and not self.json_output:
                print(f"[ERROR] {pii.error}", file=sys.stderr)
            return

        counts = Counter(entity["entity_type"] for entity in pii.entities)
        stt_processing = sum(
            fragment.stt_result.processing_seconds
            for fragment in result.buffer.fragments
            if fragment.stt_result is not None
        )
        stt_finalization = sum(
            fragment.stt_result.finalization_seconds
            for fragment in result.buffer.fragments
            if fragment.stt_result is not None
        )
        audio_duration = sum(
            fragment.stt_result.duration_seconds
            for fragment in result.buffer.fragments
            if fragment.stt_result is not None
        )
        aggregate_rtf = (
            (stt_processing + stt_finalization) / audio_duration
            if audio_duration
            else float("nan")
        )
        details = pii.raw_model_output
        tinybert_seconds = float(details.get("tinybert_latency_seconds", 0.0))
        structured_seconds = float(details.get("structured_latency_seconds", 0.0))
        if self.json_output:
            payload: dict[str, Any] = {
                "event": "masked",
                "masked_text": pii.masked_text,
                "safe_to_release": True,
                "privacy_status": pii.privacy_status,
                "entities": dict(counts),
                "fragment_count": result.buffer.fragment_count,
                "flush_reason": result.buffer.flush_reason,
                "latency": {
                    "stt_processing_seconds": stt_processing,
                    "stt_finalization_seconds": stt_finalization,
                    "audio_duration_seconds": audio_duration,
                    "stt_rtf": aggregate_rtf,
                    "buffer_wait_seconds": result.buffer.buffer_waiting_seconds,
                    "tinybert_seconds": tinybert_seconds,
                    "structured_seconds": structured_seconds,
                    "pii_seconds": pii.latency_seconds,
                    "total_seconds": result.total_seconds,
                },
                "queue_overflow_count": result.queue_overflow_count,
            }
            if self.show_raw:
                payload["raw_text"] = result.buffer.text
            print(json.dumps(payload, ensure_ascii=False))
            return

        if self.show_raw:
            print(f"[RAW] {result.buffer.text}")
        print(f"{self._paint('[MASKED]', '1;32')} {pii.masked_text}", flush=True)
        entity_text = (
            ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
            if counts
            else "none"
        )
        if not self.quiet:
            print(f"[ENTITIES] {entity_text}")
            print(
                "[LATENCY] "
                f"audio={audio_duration:.3f}s, "
                f"STT={stt_processing:.3f}s, "
                f"finalize={stt_finalization:.3f}s, "
                f"RTF={aggregate_rtf:.3f}, "
                f"buffer_wait={result.buffer.buffer_waiting_seconds:.3f}s, "
                f"TinyBERT={tinybert_seconds:.3f}s, "
                f"structured={structured_seconds:.3f}s, "
                f"PII_total={pii.latency_seconds:.3f}s, "
                f"end_to_end={result.total_seconds:.3f}s, "
                f"queue_overflows={result.queue_overflow_count}"
            )
            unreliable = any(
                fragment.stt_result is not None
                and not fragment.stt_result.reliable
                for fragment in result.buffer.fragments
            )
            if unreliable:
                print("[WARNING] This output includes audio affected by queue overflow.")
            print("[READY] Listening continuously", flush=True)

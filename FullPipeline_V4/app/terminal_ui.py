"""Privacy-aware terminal rendering."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from typing import Any, Sequence

from app.digit_normalizer import normalize_spoken_digits
from integrated_pipeline.config import SAMPLE_RATE
from integrated_pipeline.pipeline import BLOCKED_MARKER
from integrated_pipeline.schemas import LiveOutput
from integrated_pipeline.sqlite_storage import StoredTranscriptBatch


def _calibration_threshold(data: dict[str, Any]) -> float:
    """Read the frozen speech gate out of a calibration_complete payload.

    The segmenter emits the gate as ``energy_gate``. ``start_threshold`` is
    accepted as a legacy alias. Never index these directly: a missing key used
    to raise KeyError inside the live UI and abort the whole session right
    after calibration.
    """

    for key in ("energy_gate", "start_threshold"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


class TerminalUI:
    def __init__(
        self,
        *,
        show_raw: bool = False,
        show_partials: bool = False,
        json_output: bool = False,
        quiet: bool = False,
        colour: bool = True,
        live_ui: Any | None = None,
    ):
        self.show_raw = show_raw
        self.show_partials = show_partials
        self.json_output = json_output
        self.quiet = quiet
        self.colour = colour and sys.stdout.isatty()
        self.live_ui = live_ui
        self._conversation_started = False

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.colour else text

    def line(self, tag: str, message: str, *, force: bool = False) -> None:
        if self.quiet and not force:
            return
        if self.json_output:
            print(json.dumps({"event": tag.lower(), "message": message}))
            return
        if self.live_ui is not None:
            if self._conversation_started and not force:
                return
            if tag in {"ERROR", "BLOCKED"}:
                self.live_ui.error(message)
            elif tag == "WARNING":
                self.live_ui.warn(message)
            elif tag in {"MODEL", "READY"}:
                self.live_ui.ok(message)
            else:
                self.live_ui.info(f"{tag.lower()}: {message}")
            return
        print(f"{self._paint(f'[{tag}]', '1;36')} {message}", flush=True)

    def event(self, name: str, data: dict[str, Any]) -> None:
        if self.live_ui is not None and not self.json_output:
            if name == "calibration_started":
                self.live_ui.set_state("boot", detail="microphone calibration")
            elif name == "calibration_complete":
                self._conversation_started = True
                self.live_ui.ready(threshold=_calibration_threshold(data))
                return
            elif name == "recording_started":
                self.live_ui.set_state("speech")
            elif name == "segment_closed":
                self.live_ui.set_state("thinking", detail="transcribing")
            elif name == "stage5_submitted":
                self.live_ui.set_state(
                    "buffer", detail="immediate Stage 5"
                )
                return
            elif name == "masking_started":
                self.live_ui.set_state(
                    "masking", detail="privacy classification"
                )
                return
            elif name == "onkith_command_only":
                self.live_ui.set_state("listening")
            elif name in {
                "segmentation_error",
                "stt_error",
                "mask_output_error",
                "database_error",
            }:
                self.live_ui.set_state("error", sticky=2.0)
            if name == "onkith_status":
                reply = self._onkith_reply(str(data["message"]))
                if reply is not None:
                    self.live_ui.onkith(reply)
                return

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
                f"(threshold={_calibration_threshold(data):.5f})",
            )
        elif name == "recording_started":
            self.line(
                "RECORDING",
                "Voice detected; recording utterance "
                f"{int(data.get('utterance_sequence_number', 0)):06d}",
            )
        elif name == "segment_closed":
            reliability = "reliable" if data.get("reliable", True) else "UNRELIABLE"
            captured = (
                int(data.get("decision_sample", 0))
                - int(data.get("segment_start_sample", 0))
            ) / float(SAMPLE_RATE)
            self.line(
                "SEGMENT",
                f"{data.get('segment_id', data.get('utterance_id', ''))} closed "
                f"after {data.get('reason', 'unknown')} "
                f"(captured={max(0.0, captured):.3f}s, "
                f"submitted={float(data.get('submitted_audio_seconds', 0.0)):.3f}s, "
                f"{reliability})",
            )
        elif name == "forced_split_continuation":
            overlap = (
                float(data.get("adjacent_submitted_overlap_samples", 0)) / SAMPLE_RATE
            )
            self.line(
                "RECORDING",
                "Continuous speech carried into the next segment "
                f"with {overlap:.3f}s overlap",
            )
        elif name == "stt_finalized":
            reliability = "reliable" if data["reliable"] else "UNRELIABLE"
            rtf = data["rtf"]
            rtf_text = f"{rtf:.3f}" if math.isfinite(rtf) else "n/a"
            self.line(
                "STT",
                f"{data['batch_id']} finalized ({reliability}, "
                f"queue_wait={data['queue_wait_seconds']:.3f}s, "
                f"RTF={rtf_text}, "
                f"dedup_words={data['deduplicated_word_count']})",
            )
        elif name == "queue_overflow":
            self.line(
                "WARNING",
                "Audio queue overflow: "
                f"{data['new_dropped_chunks']} new chunk(s) dropped; "
                "the affected recording is unreliable",
                force=True,
            )
        elif name == "completed_queue_overload":
            self.line(
                "WARNING",
                "Completed-recording STT backlog reached its configured "
                f"bound; {int(data.get('pending_recordings', 0))} recording(s) "
                "retained in lossless spill memory",
                force=True,
            )
        elif name in {
            "segmentation_error",
            "stt_error",
            "mask_output_error",
            "microphone_status",
        }:
            self.line("WARNING", str(data), force=True)
        elif name == "database_error":
            self.line("DB", str(data), force=True)
        elif name == "empty_final_ignored":
            batch = data.get("batch_id", "unknown")
            self.line("STT", f"{batch} produced no finalized text")
        elif name == "onkith_command_detected":
            self.line(
                "ONKITH",
                f'Wake word detected as: "{data["wake_word"]}"; '
                f'command="{data["command_text"]}"; '
                f'intent={data["intent"]}',
            )
        elif name == "onkith_command_queued":
            self.line(
                "ONKITH",
                f'{data["intent"]} command queued',
            )
        elif name == "onkith_remaining_forwarded":
            message = "Command span removed; remaining speech sent to Stage 5"
            if self.show_raw:
                message += f': "{data["remaining_text"]}"'
            self.line("ONKITH", message)
        elif name == "onkith_command_only":
            self.line(
                "ONKITH",
                "Command-only transcript consumed; nothing sent to Stage 5",
            )
        elif name == "onkith_no_command":
            self.line(
                "ONKITH",
                "No command detected; original transcript sent to Stage 5",
            )
        elif name == "onkith_status":
            self.line("ONKITH", str(data["message"]))

    @staticmethod
    def _onkith_reply(message: str) -> str | None:
        if message.startswith("Opening: "):
            return f"Opening {message.removeprefix('Opening: ').strip()}"
        if message == "Player closed":
            return "Music stopped"
        if message.startswith("Music action failed: "):
            return message
        return None

    def partial(self, text: str) -> None:
        if not self.show_partials:
            return
        if self.json_output:
            print(json.dumps({"event": "post_recording_partial", "text": text}))
        else:
            print(f"[STT DEBUG PARTIAL] {text}", flush=True)

    def preview(self, render: Any) -> None:
        """Render one live masked preview line.

        Display only. The line is a repaint of recent speech, so each one
        supersedes the last; the authoritative transcript still arrives
        separately when the utterance ends. Never raises.
        """

        try:
            line = getattr(render, "line", "")
            if self.quiet or not line:
                return
            if self.json_output:
                print(
                    json.dumps(
                        {
                            "event": "preview",
                            "masked_text": line,
                            "blurred_words": getattr(render, "blurred_words", 0),
                            "redacted": bool(getattr(render, "redacted", False)),
                            "degraded": bool(getattr(render, "degraded", False)),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return
            if self.live_ui is not None:
                painter = getattr(self.live_ui, "preview", None) or getattr(
                    self.live_ui, "info", None
                )
                if painter is not None:
                    painter(line)
                    return
            print(f"[PREVIEW] {line}", flush=True)
        except Exception:
            # A display failure must never disturb the pipeline.
            pass

    def output(self, result: LiveOutput) -> None:
        pii = result.pii
        batch_id = result.buffer.batch_id
        if not pii.safe_to_release:
            if self.json_output:
                print(
                    json.dumps(
                        {
                            "event": "blocked",
                            "batch_id": batch_id,
                            "masked_text": BLOCKED_MARKER,
                            "safe_to_release": False,
                        }
                    )
                )
            elif self.live_ui is not None:
                self.live_ui.error(
                    "PII masking failed; transcript was not released",
                    detail=pii.error,
                )
                self.live_ui.set_state("listening")
            else:
                self.line(
                    "BLOCKED",
                    f"{batch_id}: PII masking failed; transcript was not released",
                    force=True,
                )
                if pii.error:
                    print(f"[ERROR] {pii.error}", file=sys.stderr)
            return

        # Two distinct strings from here down, and they must not be confused:
        #
        #   canonical_text - exactly what the masker produced. This is what
        #                    SQLite stores and Onkith consumes, so it is the
        #                    value machine readers must keep seeing.
        #   display_text   - the same text with spoken numbers rendered as
        #                    numerals, for humans only.
        #
        # pii.masked_text is never overwritten; the normalisation lives in a
        # local copy so nothing downstream inherits it by accident.
        canonical_text = pii.masked_text
        display_text = normalize_spoken_digits(canonical_text)
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
        stt_total = sum(
            fragment.stt_result.total_stt_seconds
            for fragment in result.buffer.fragments
            if fragment.stt_result is not None
        )
        audio_duration = sum(
            fragment.stt_result.submitted_audio_seconds
            for fragment in result.buffer.fragments
            if fragment.stt_result is not None
        )
        aggregate_rtf = (
            stt_total / audio_duration if audio_duration else float("nan")
        )
        details = pii.raw_model_output
        pii_model_seconds = float(details.get("pii_model_latency_seconds", 0.0))
        structured_seconds = float(details.get("structured_latency_seconds", 0.0))
        if self.json_output:
            payload: dict[str, Any] = {
                "event": "masked",
                "session_id": result.buffer.session_id,
                "batch_id": batch_id,
                "sequence_number": result.buffer.sequence_number,
                # Canonical first, unchanged, for existing machine consumers.
                # display_text is additive so older readers keep working.
                "masked_text": canonical_text,
                "display_text": display_text,
                "safe_to_release": True,
                "privacy_status": pii.privacy_status,
                "entities": dict(counts),
                "fragment_count": result.buffer.fragment_count,
                "flush_reason": result.buffer.flush_reason,
                "latency": {
                    "stt_processing_seconds": stt_processing,
                    "stt_finalization_seconds": stt_finalization,
                    "stt_total_seconds": stt_total,
                    "submitted_audio_seconds": audio_duration,
                    "stt_rtf": aggregate_rtf,
                    "buffer_wait_seconds": 0.0,
                    "mask_queue_wait_seconds": (
                        result.mask_queue_wait_seconds
                    ),
                    "pii_model_seconds": pii_model_seconds,
                    "structured_seconds": structured_seconds,
                    "pii_seconds": pii.latency_seconds,
                    "endpoint_to_display_seconds": result.total_seconds,
                },
                "queue_overflow_count": result.queue_overflow_count,
            }
            if self.show_raw:
                payload["raw_text"] = result.buffer.text
            print(json.dumps(payload, ensure_ascii=False))
            return

        if self.live_ui is not None:
            self.live_ui.utterance(
                display_text,
                entities=dict(counts),
                latency={
                    "end_to_end": result.total_seconds,
                    "rtf": aggregate_rtf,
                    "audio": audio_duration,
                    "stt": stt_processing,
                    "finalize": stt_finalization,
                    "buffer_wait": 0.0,
                    "pii_total": pii.latency_seconds,
                },
            )
            self.live_ui.set_state("listening")
            return

        if self.show_raw:
            print(f"[RAW] {result.buffer.text}")
        print(
            f"{self._paint('[MASKED]', '1;32')} "
            f"{batch_id}: {display_text}",
            flush=True,
        )
        entity_text = (
            ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
            if counts
            else "none"
        )
        if not self.quiet:
            print(f"[ENTITIES] {entity_text}")
            print(
                "[LATENCY] "
                f"submitted_audio={audio_duration:.3f}s, "
                f"STT_add={stt_processing:.3f}s, "
                f"STT_finalize={stt_finalization:.3f}s, "
                f"STT_total={stt_total:.3f}s, "
                f"RTF={aggregate_rtf:.3f}, "
                f"Stage5_wait=0.000s, "
                f"mask_queue_wait={result.mask_queue_wait_seconds:.3f}s, "
                f"DeBERTa={pii_model_seconds:.3f}s, "
                f"structured={structured_seconds:.3f}s, "
                f"PII_total={pii.latency_seconds:.3f}s, "
                f"endpoint_to_display={result.total_seconds:.3f}s, "
                f"queue_overflows={result.queue_overflow_count}"
            )
            unreliable = any(
                fragment.stt_result is not None
                and not fragment.stt_result.reliable
                for fragment in result.buffer.fragments
            )
            if unreliable:
                print(
                    "[WARNING] This output includes audio affected by "
                    "capture overflow."
                )
            print("[READY] Listening continuously", flush=True)

    def session_transcripts(
        self, rows: Sequence[StoredTranscriptBatch]
    ) -> None:
        raw_text = " ".join(
            row.raw_transcript for row in rows if row.raw_transcript
        ).strip()
        masked_parts = [
            (
                row.masked_transcript
                if row.status == "complete" and row.masked_transcript is not None
                else BLOCKED_MARKER
            )
            for row in rows
        ]
        # Same canonical/display split as output() above. The stored rows are
        # untouched; only this joined copy for printing gets normalised.
        canonical_text = " ".join(masked_parts).strip()
        display_text = normalize_spoken_digits(canonical_text)
        if self.json_output:
            payload: dict[str, Any] = {
                "event": "session_transcripts",
                "batch_count": len(rows),
                "masked_text": canonical_text,
                "display_text": display_text,
            }
            if self.show_raw:
                payload["raw_text"] = raw_text
            print(json.dumps(payload, ensure_ascii=False))
            return
        print("[SESSION RAW]")
        print(raw_text if self.show_raw else "<hidden; use --show-raw>")
        print()
        print("[SESSION MASKED]")
        print(display_text)

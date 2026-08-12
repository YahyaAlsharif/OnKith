"""Privacy-aware terminal rendering."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from typing import Any, Sequence

from integrated_pipeline.pipeline import BLOCKED_MARKER
from integrated_pipeline.schemas import LiveOutput
from integrated_pipeline.sqlite_storage import StoredTranscriptBatch


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
                if data.get("capped"):
                    # The one case where the threshold on the ready line is
                    # not the whole story, and the whole story is why
                    # nobody is being heard.
                    self.live_ui.warn(
                        "noisy calibration: threshold capped at "
                        f"{data['start_threshold']:.3f} from "
                        f"{data['measured_threshold']:.3f}. Restart "
                        "somewhere quieter."
                    )
                self.live_ui.ready(threshold=data["start_threshold"])
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
            if data.get("capped"):
                self.line(
                    "CALIBRATION",
                    "The room was noisy during calibration "
                    f"(would have been {data['measured_threshold']:.5f}, "
                    "which is above anything a voice reaches). Capped at "
                    f"{data['start_threshold']:.5f}. Restart somewhere "
                    "quieter for a cleaner one.",
                )
            self.line(
                "READY",
                "Listening continuously "
                f"(threshold={data['start_threshold']:.5f})",
            )
        elif name == "recording_started":
            self.line(
                "RECORDING",
                "Voice detected; recording next batch "
                f"{data['next_sequence_number']:06d}",
            )
        elif name == "segment_closed":
            reliability = (
                "reliable" if data["reliable"] else "UNRELIABLE"
            )
            self.line(
                "SEGMENT",
                f"{data['batch_id']} closed after {data['reason']} "
                f"(captured={data['captured_audio_seconds']:.3f}s, "
                f"submitted={data['submitted_audio_seconds']:.3f}s, "
                f"{reliability})",
            )
        elif name == "forced_split_continuation":
            self.line(
                "RECORDING",
                "Continuous speech carried into the next recording "
                f"with {data['overlap_seconds']:.3f}s overlap",
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
                f"bound; {data['pending_recordings']} recording(s) retained "
                "in lossless spill memory",
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
            message = "No command detected; original transcript sent to Stage 5"
            nearest = data.get("nearest")
            if nearest:
                # Naming it is the point. The fix is one line of config, and
                # nobody can write that line without seeing what the
                # recogniser actually put down.
                message += (
                    f'; closest was "{nearest}". If that was the name, add it '
                    "to wake.spellings in config/live_config.json."
                )
            self.line("ONKITH", message)
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
        tinybert_seconds = float(details.get("tinybert_latency_seconds", 0.0))
        structured_seconds = float(details.get("structured_latency_seconds", 0.0))
        if self.json_output:
            payload: dict[str, Any] = {
                "event": "masked",
                "session_id": result.buffer.session_id,
                "batch_id": batch_id,
                "sequence_number": result.buffer.sequence_number,
                "masked_text": pii.masked_text,
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
                    "tinybert_seconds": tinybert_seconds,
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
                pii.masked_text,
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
            f"{batch_id}: {pii.masked_text}",
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
                f"TinyBERT={tinybert_seconds:.3f}s, "
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
        masked_text = " ".join(masked_parts).strip()
        if self.json_output:
            payload: dict[str, Any] = {
                "event": "session_transcripts",
                "batch_count": len(rows),
                "masked_text": masked_text,
            }
            if self.show_raw:
                payload["raw_text"] = raw_text
            print(json.dumps(payload, ensure_ascii=False))
            return
        print("[SESSION RAW]")
        print(raw_text if self.show_raw else "<hidden; use --show-raw>")
        print()
        print("[SESSION MASKED]")
        print(masked_text)

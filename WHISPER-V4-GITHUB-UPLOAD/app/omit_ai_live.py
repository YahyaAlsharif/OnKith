#!/usr/bin/env python3
"""Main executable for Silero-segmented utterance STT and private storage."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onkith_ui import LiveUI, muted_stderr, quiet_boot

quiet_boot()

from app.live_audio import LiveAudioRuntime
from app.live_preview import LivePreview
from app.live_runtime import (
    LivePrivacyRuntime,
    LiveSTTRuntime,
    deduplicate_boundary,
)
from app.onkith_music import (
    OnkithMusicController,
    OnkithMusicWorker,
    intercept_onkith_command,
)
from app.post_mask_assistant import PostMaskAssistant
from app.terminal_ui import TerminalUI
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.whisper_adapter import WhisperAdapter
from integrated_pipeline.pii_adapter import PIIMaskerAdapter
from integrated_pipeline.pipeline import FailClosedMasker
from integrated_pipeline.supplemental_pii import SupplementalPIIMasker
from integrated_pipeline.schemas import PIIResult
from integrated_pipeline.silero_vad import SileroOnnxVAD
from integrated_pipeline.sqlite_storage import (
    SQLiteTranscriptStore,
    StoredTranscriptBatch,
)
from integrated_pipeline.structured_pii import StructuredPIIRecognizer


_LOCAL_ENV_KEYS = {
    "YOUTUBE_API_KEY",
    "YOUTUBE_REGION",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_FALLBACK_MODEL",
    "ONKITH_TTS",
    "ONKITH_PIPER_MODEL",
    "ONKITH_KOKORO_MODEL",
    "ONKITH_KOKORO_VOICES",
    "ONKITH_KOKORO_VOICE",
    "ONKITH_UI_BROWSER",
}


def load_local_env(path: Path = ROOT / ".env") -> None:
    """Load supported local settings without overriding the process environment."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in _LOCAL_ENV_KEYS:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Silero-segmented utterances through faster-whisper base.en and "
            "DeBERTa private terminal text"
        )
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument(
        "--show-partials",
        action="store_true",
        help="show verbose non-transcript diagnostics",
    )
    parser.add_argument("--text-test")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-colour", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the assistant and local TTS without opening its browser UI",
    )
    parser.add_argument(
        "--assistant-port",
        type=int,
        default=8765,
        help="localhost port for the additive assistant UI (default: 8765)",
    )
    parser.add_argument(
        "--no-assistant",
        action="store_true",
        help="disable only the post-mask Groq/TTS/UI add-on",
    )
    return parser


def static_self_test(config: PipelineConfig, *, require_artifacts: bool = True) -> None:
    """Check the fixed contract and the pure-logic invariants.

    ``require_artifacts=False`` lets an acceptance run exercise the logic on a
    host where the large model files are not present, so their absence can be
    reported as NOT RUN instead of masquerading as a contract failure.
    """

    config.validate(require_artifacts=require_artifacts)
    if int(config.audio["endpoint_silence_samples"]) != 39_936:
        raise AssertionError("Endpoint silence must be exactly 39,936 samples")
    if int(config.audio["maximum_unique_segment_samples"]) != 136_192:
        raise AssertionError("Maximum unique segment must be 136,192 samples")
    if config.stage5["mode"] != "immediate":
        raise AssertionError("Live Stage 5 must be immediate")
    if config.whisper["model"] != "base.en":
        raise AssertionError("Whisper model identity changed")
    if config.storage["filename"] != "session_transcript.db":
        raise AssertionError("Unexpected session database filename")

    recognizer = StructuredPIIRecognizer()
    spans = recognizer.find_spans(
        "Email me at rayyan@example.com and my phone number is 555 123 4567."
    )
    types = {span.entity_type for span in spans}
    if not {"EMAIL", "TELEPHONENUM"}.issubset(types):
        raise AssertionError(f"Structured PII smoke failed: {types}")

    deduplicated, count = deduplicate_boundary(
        "we cross the important boundary words",
        "boundary words without repeating the remainder",
    )
    if count != 2 or deduplicated != "without repeating the remainder":
        raise AssertionError("Forced-boundary transcript de-duplication failed")

    onkith = intercept_onkith_command(
        "normal words Onkith play Imagine Dragons Bones"
    )
    if (
        onkith.intent != "PLAY"
        or onkith.command_text != "play Imagine Dragons Bones"
        or onkith.remaining_text != "normal words"
    ):
        raise AssertionError("Onkith interceptor smoke failed")
    if intercept_onkith_command("Someone named Ankit called").command_detected:
        raise AssertionError("Onkith interceptor matched the name Ankit")

    class Broken:
        def mask_text(self, text: str) -> PIIResult:
            raise RuntimeError("intentional self-test")

    blocked = FailClosedMasker(Broken()).mask_text("secret raw value")
    if blocked.safe_to_release or "secret raw value" in blocked.masked_text:
        raise AssertionError("Fail-closed privacy smoke failed")


def print_devices(json_output: bool) -> int:
    rows = LiveAudioRuntime.list_devices()
    if json_output:
        print(json.dumps(rows))
    else:
        print("Available input devices:")
        for row in rows:
            marker = " <DEFAULT>" if row["default"] else ""
            print(
                f"[{row['id']}] {row['name']} | channels={row['channels']} | "
                f"default_sample_rate={row['default_sample_rate']} | "
                f"accepts_16000={row['accepts_16000_hz_request']} | "
                f"capture_status={row['capture_16000_hz_status']}{marker}"
            )
    return 0 if rows else 2


def available_ram_mib() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return None


def _record_error(errors: list[str], stage: str, exc: BaseException) -> None:
    errors.append(f"{stage}: {type(exc).__name__}: {exc}")


def _shutdown_pipeline(
    *,
    session_id: str,
    ui: TerminalUI,
    audio: LiveAudioRuntime | None,
    stt: LiveSTTRuntime | None,
    music: OnkithMusicWorker | None,
    assistant: PostMaskAssistant | None = None,
    privacy: LivePrivacyRuntime | None,
    storage: SQLiteTranscriptStore | None,
    errors: list[str],
) -> list[StoredTranscriptBatch]:
    """Drain capture -> STT -> PII -> SQLite without skipping later stages."""

    rows: list[StoredTranscriptBatch] = []
    if audio is not None:
        audio.stop()
    if stt is not None:
        try:
            stt.shutdown()
        except BaseException as exc:
            _record_error(errors, "STT drain", exc)
    # Preserve the original shutdown sequence when the add-on is disabled.
    if assistant is None and music is not None:
        try:
            music.shutdown()
        except BaseException as exc:
            _record_error(errors, "music shutdown", exc)
    if privacy is not None:
        try:
            privacy.shutdown()
        except BaseException as exc:
            _record_error(errors, "PII drain", exc)
    # Privacy may queue one last masked question while draining. Let the
    # assistant and local TTS finish it before closing music commands.
    if assistant is not None:
        try:
            assistant.shutdown()
        except BaseException as exc:
            _record_error(errors, "assistant shutdown", exc)
    if assistant is not None and music is not None:
        try:
            music.shutdown()
        except BaseException as exc:
            _record_error(errors, "music shutdown", exc)
    if storage is not None:
        try:
            storage.drain()
            storage.finish_session(
                session_id,
                utc_now_iso(),
                "complete" if not errors else "failed",
            )
            storage.drain()
            rows = storage.query_session(session_id)
            ui.session_transcripts(rows)
        except BaseException as exc:
            _record_error(errors, "SQLite drain/query", exc)
        try:
            storage.shutdown(checkpoint=True)
        except BaseException as exc:
            _record_error(errors, "SQLite close", exc)
    return rows


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = build_parser().parse_args(argv)
    config = PipelineConfig()
    live_ui: LiveUI | None = None
    full_live_session = not any(
        (
            args.list_devices,
            args.print_config,
            args.self_test,
            args.text_test is not None,
            args.json_output,
            args.quiet,
        )
    )
    if full_live_session:
        if args.no_colour:
            os.environ.setdefault("NO_COLOR", "1")
        live_ui = LiveUI(
            theme="onkith",
            motion=True,
            pet=True,
            verbose=args.show_partials,
        )
        live_ui.start()
    ui = TerminalUI(
        show_raw=args.show_raw,
        show_partials=args.show_partials,
        json_output=args.json_output,
        quiet=args.quiet,
        colour=not args.no_colour,
        live_ui=live_ui,
    )

    if args.list_devices:
        return print_devices(args.json_output)
    if args.print_config:
        print(json.dumps(config.public_dict(), indent=2))
        return 0
    if args.self_test:
        static_self_test(config)
        stt_test = WhisperAdapter(config)
        silero_test = SileroOnnxVAD(config)
        try:
            stt_test.load()
            silero_test.load()
            if stt_test.counters.warmup_count != 1:
                raise AssertionError("Whisper warmup did not run exactly once")
            if silero_test.load_count != 1:
                raise AssertionError("Silero did not load exactly once")
        finally:
            stt_test.close()
            silero_test.close()
        print("Self-test passed.")
        return 0

    pii_adapter = PIIMaskerAdapter(config)
    # Supplemental structured recognisers sit *inside* the fail-closed wrapper,
    # so their corrections reach the canonical masked text (and therefore
    # SQLite, Onkith and every display path), while any failure of theirs still
    # becomes a blocked, non-releasable result rather than a leak.
    supplemented = SupplementalPIIMasker(pii_adapter)
    safe_masker = FailClosedMasker(supplemented)
    if args.text_test is not None:
        pii = safe_masker.mask_text(args.text_test)
        if not pii.safe_to_release:
            ui.line(
                "BLOCKED",
                "PII masking failed; transcript was not released",
                force=True,
            )
            return 1
        if args.json_output:
            payload = {
                "event": "text_test",
                "masked_text": pii.masked_text,
                "safe_to_release": True,
                "privacy_status": pii.privacy_status,
            }
            if args.show_raw:
                payload["raw_text"] = args.text_test
            print(json.dumps(payload, ensure_ascii=False))
        elif args.show_raw:
            print(f"[RAW] {args.text_test}")
            print(pii.masked_text)
        else:
            print(pii.masked_text)
        return 0

    session_id = str(uuid.uuid4())
    session_started_at = utc_now_iso()
    ram = available_ram_mib()
    if live_ui is not None:
        requested_device = (
            f"input device {args.device}"
            if args.device is not None
            else "default input device"
        )
        live_ui.banner(
            system={
                "device": requested_device,
                "platform": f"{platform.system()} {platform.machine()}",
                "python": platform.python_version(),
            },
            config={
                "stt": "faster-whisper base.en, CTranslate2 CPU INT8 beam=5",
                "privacy": "DeBERTa-v3-xsmall INT8 (Model V2) + structured PII",
                "sample rate": "16000 Hz stream required (native capture unverified)",
                "max unique segment": "8.512s",
                "silence": "2.496s",
                "database": config.session_database_path.name,
            },
        )
        live_ui.info(f"session: {session_id}")
    else:
        ui.line("STARTUP", "Raspberry Pi recorded-utterance privacy pipeline")
        ui.line("SESSION", session_id)
        ui.line(
            "SYSTEM",
            f"Python {platform.python_version()}, "
            f"{platform.system()} {platform.machine()}, "
            f"available_RAM={ram if ram is not None else 'unknown'} MiB",
        )
        ui.line(
            "CONFIG",
            "Whisper=base.en CTranslate2/CPU/INT8 beam=5, Silero=ONNX, "
            "capture=16000Hz-requested, chunk=0.100s, "
            "silence_endpoint=2.496s, unique_segment_max=8.512s, "
            "Stage5=immediate, SQLite=WAL/NORMAL, "
            "PII=DeBERTa-v3-xsmall INT8+structured whole_word_union",
        )

    recognizer = WhisperAdapter(config)
    silero = SileroOnnxVAD(config)
    audio: LiveAudioRuntime | None = None
    privacy: LivePrivacyRuntime | None = None
    stt: LiveSTTRuntime | None = None
    music: OnkithMusicWorker | None = None
    assistant: PostMaskAssistant | None = None
    storage: SQLiteTranscriptStore | None = None
    preview: LivePreview | None = None
    errors: list[str] = []
    interrupted = False

    def emit_event(name: str, data: dict[str, Any]) -> None:
        """Preserve the terminal event and mirror state to the local add-on."""

        ui.event(name, data)
        if assistant is not None:
            assistant.observe_event(name, data)

    try:
        with (
            live_ui.task(
                "whisper base.en and silero",
                done_label="whisper and silero ready",
            )
            if live_ui is not None
            else nullcontext()
        ):
            with muted_stderr():
                recognizer.load()
                silero.load()
        if live_ui is None:
            ui.line(
                "MODEL",
                "Whisper base.en CTranslate2 CPU/INT8 beam=5 and Silero ONNX loaded "
                f"(Whisper loads={recognizer.counters.model_load_count}, "
                f"warmups={recognizer.counters.warmup_count}, Silero loads={silero.load_count})",
            )
        with (
            live_ui.task(
                "privacy classifier",
                done_label="privacy classifier ready",
            )
            if live_ui is not None
            else nullcontext()
        ):
            with muted_stderr():
                pii_adapter.load()
        if live_ui is None:
            ui.line(
                "MODEL",
                "DeBERTa-v3-xsmall INT8 loaded with CPUExecutionProvider "
                f"(resource loads={pii_adapter.load_count}, "
                f"path={config.pii_model_path})",
            )

        audio = LiveAudioRuntime(
            config,
            session_id,
            device=args.device,
            once=args.once,
            on_event=emit_event,
            on_level=(
                live_ui.set_level
                if live_ui is not None
                else lambda rms: None
            ),
            vad=silero,
        )
        with muted_stderr():
            audio.prepare_device()
        storage = SQLiteTranscriptStore(
            config.session_database_path,
            busy_timeout_ms=int(config.storage["busy_timeout_ms"]),
        )
        storage.start()
        storage.start_session(
            session_id,
            session_started_at,
            model_identity=recognizer.resource_identity or "",
            device_information=(
                f"id={audio.device_id};name={audio.device_name};"
                f"sample_rate={audio.sample_rate}"
            ),
        )
        ui.line("DB", f"{config.session_database_path.name} opened in WAL mode")

        music = OnkithMusicWorker(
            OnkithMusicController.from_environment(
                log=lambda message: emit_event(
                    "onkith_status", {"message": message}
                )
            ),
            log=lambda message: emit_event(
                "onkith_status", {"message": message}
            ),
        )
        if not args.no_assistant:
            assistant = PostMaskAssistant(
                terminal_output=ui.output,
                music=music,
                port=args.assistant_port,
                open_window=not args.headless,
                log=lambda message: emit_event(
                    "onkith_status", {"message": message}
                ),
            )
        privacy = LivePrivacyRuntime(
            config,
            safe_masker,
            storage,
            on_event=emit_event,
            on_output=(
                assistant.handle_output if assistant is not None else ui.output
            ),
            queue_overflow_count=lambda: (
                audio.overflow_count if audio is not None else 0
            ),
        )
        # Display-only live preview. It owns a worker thread, so neither the
        # STT lane nor the privacy lane waits on it, and it never touches
        # the transcript, Onkith or the database.
        preview = LivePreview(safe_masker, ui.preview)
        preview.start()
        stt = LiveSTTRuntime(
            audio.completed_segment_queue,
            recognizer,
            privacy,
            on_event=emit_event,
            music_worker=music,
            utc_now=utc_now_iso,
            preview=preview,
        )
        # Dependency-safe worker order: DB, add-on, PII/music, STT, microphone.
        if assistant is not None:
            assistant.start()
        privacy.start()
        music.start()
        stt.start()
        signal.signal(signal.SIGTERM, lambda signum, frame: audio.stop())
        audio.run()
    except KeyboardInterrupt:
        interrupted = True
        ui.line(
            "SHUTDOWN",
            "Ctrl+C received; draining STT, music, masking, and database",
        )
    except BaseException as exc:
        _record_error(errors, "runtime", exc)
        ui.line("ERROR", f"{type(exc).__name__}: {exc}", force=True)
    finally:
        if preview is not None:
            # Wipe the rolling raw buffer and the label salt before anything
            # else drains. Nothing in it is ever persisted.
            preview.close()
        _shutdown_pipeline(
            session_id=session_id,
            ui=ui,
            audio=audio,
            stt=stt,
            music=music,
            assistant=assistant,
            privacy=privacy,
            storage=storage,
            errors=errors,
        )
        if audio is not None:
            audio_audit = audio.audit()
            ui.line(
                "CAPTURE METRICS",
                f"callback_avg={audio_audit['callback_average_seconds']:.6f}s, "
                f"callback_max={audio_audit['callback_max_seconds']:.6f}s, "
                f"audio_queue_high_water="
                f"{audio_audit['audio_queue_high_water_mark']}, "
                f"recording_queue_high_water="
                f"{audio_audit['completed_queue_high_water_mark']}, "
                f"dropped_chunks={audio_audit['queue_overflow_count']}",
            )
        if storage is not None:
            db_audit = storage.audit()
            ui.line(
                "DB",
                f"operations={db_audit['operation_count']}, "
                f"queue_high_water={db_audit['queue_high_water_mark']}, "
                f"queue_wait_max="
                f"{db_audit['maximum_queue_wait_seconds']:.6f}s, "
                f"write_total={db_audit['total_write_seconds']:.6f}s, "
                f"write_max={db_audit['maximum_write_seconds']:.6f}s",
            )
        if privacy is not None:
            ui.line(
                "PII METRICS",
                f"mask_queue_high_water={privacy.queue_high_water_mark}",
            )
        if recognizer.loaded:
            counters = recognizer.counters
            ui.line(
                "COUNTERS",
                "model_resource_loads="
                f"{counters.model_load_count}, warmups={counters.warmup_count}, "
                f"segment_inferences={counters.inference_count}, "
                f"callbacks={counters.callback_count}, "
                f"model_time={counters.model_processing_seconds:.3f}s, "
                f"queue_wait={counters.total_queue_wait_seconds:.3f}s",
            )
        recognizer.close()
        silero.close()

    if errors:
        for error in errors:
            ui.line("ERROR", error, force=True)
        ui.line(
            "SHUTDOWN",
            "Shutdown completed with errors; clean shutdown was not reported",
            force=True,
        )
        if live_ui is not None:
            live_ui.stop()
        return 1
    ui.line(
        "SHUTDOWN",
        "Stopped cleanly" + (" after Ctrl+C" if interrupted else ""),
    )
    if live_ui is not None:
        live_ui.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Main executable for recorded-utterance STT and private text storage."""

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
from app.terminal_ui import TerminalUI
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.moonshine_adapter import MoonshineAdapter
from integrated_pipeline.pii_adapter import PIIMaskerAdapter
from integrated_pipeline.pipeline import FailClosedMasker
from integrated_pipeline.schemas import PIIResult
from integrated_pipeline.sqlite_storage import (
    SQLiteTranscriptStore,
    StoredTranscriptBatch,
)
from integrated_pipeline.structured_pii import StructuredPIIRecognizer


_LOCAL_ENV_KEYS = {"YOUTUBE_API_KEY", "YOUTUBE_REGION"}


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
            "Recorded utterances through Moonshine Medium Streaming and "
            "TinyBERT private terminal text"
        )
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument(
        "--show-partials",
        action="store_true",
        help="debug post-recording Moonshine callbacks (not live partials)",
    )
    parser.add_argument("--text-test")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-colour", action="store_true")
    return parser


def static_self_test(config: PipelineConfig) -> None:
    config.validate(require_artifacts=True)
    if float(config.audio["endpoint_silence_seconds"]) != 2.5:
        raise AssertionError("Recording silence endpoint must be 2.5 seconds")
    if float(config.audio["maximum_utterance_seconds"]) != 8.5:
        raise AssertionError("Maximum recording must be 8.5 seconds")
    if config.stage5["mode"] != "immediate":
        raise AssertionError("Live Stage 5 must be immediate")
    if config.moonshine["model_arch"] != "medium_streaming":
        raise AssertionError("Moonshine Medium Streaming changed")
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
                f"sample_rate={row['sample_rate']}{marker}"
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
    if music is not None:
        try:
            music.shutdown()
        except BaseException as exc:
            _record_error(errors, "music shutdown", exc)
    if privacy is not None:
        try:
            privacy.shutdown()
        except BaseException as exc:
            _record_error(errors, "PII drain", exc)
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
        print("Self-test passed.")
        return 0

    pii_adapter = PIIMaskerAdapter(config)
    safe_masker = FailClosedMasker(pii_adapter)
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
                "stt": "Moonshine English Medium Streaming",
                "privacy": "TinyBERT-4 INT8 + structured PII",
                "sample rate": "selected device default",
                "max recording": (
                    f"{config.audio['maximum_utterance_seconds']:.1f}s"
                ),
                "silence": (
                    f"{config.audio['endpoint_silence_seconds']:.1f}s"
                ),
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
            "Moonshine=medium_streaming, RNNoise=off, "
            f"chunk={config.moonshine['chunk_seconds']:.1f}s, "
            f"silence_endpoint="
            f"{config.audio['endpoint_silence_seconds']:.1f}s, "
            f"recording_max="
            f"{config.audio['maximum_utterance_seconds']:.1f}s, "
            "Stage5=immediate, SQLite=WAL/NORMAL, "
            "PII=TinyBERT-4 INT8+structured whole_word_union",
        )

    moonshine = MoonshineAdapter(config)
    audio: LiveAudioRuntime | None = None
    privacy: LivePrivacyRuntime | None = None
    stt: LiveSTTRuntime | None = None
    music: OnkithMusicWorker | None = None
    storage: SQLiteTranscriptStore | None = None
    errors: list[str] = []
    interrupted = False
    try:
        with (
            live_ui.task(
                "moonshine medium",
                done_label="moonshine medium ready",
            )
            if live_ui is not None
            else nullcontext()
        ):
            with muted_stderr():
                moonshine.load()
        if live_ui is None:
            ui.line(
                "MODEL",
                "Moonshine English Medium Streaming loaded "
                f"(resource loads="
                f"{moonshine.counters.model_resource_load_count})",
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
                "TinyBERT INT8 loaded with CPUExecutionProvider "
                f"(resource loads={pii_adapter.load_count}, "
                f"path={config.tinybert_model_path})",
            )

        audio = LiveAudioRuntime(
            config,
            session_id,
            device=args.device,
            once=args.once,
            on_event=ui.event,
            on_level=(
                live_ui.set_level
                if live_ui is not None
                else lambda rms: None
            ),
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
            model_identity=moonshine.resource_identity or "",
            device_information=(
                f"id={audio.device_id};name={audio.device_name};"
                f"sample_rate={audio.sample_rate}"
            ),
        )
        ui.line("DB", f"{config.session_database_path.name} opened in WAL mode")

        privacy = LivePrivacyRuntime(
            config,
            safe_masker,
            storage,
            on_event=ui.event,
            on_output=ui.output,
            queue_overflow_count=lambda: (
                audio.overflow_count if audio is not None else 0
            ),
        )
        music = OnkithMusicWorker(
            OnkithMusicController.from_environment(
                log=lambda message: ui.event(
                    "onkith_status", {"message": message}
                )
            ),
            log=lambda message: ui.event(
                "onkith_status", {"message": message}
            ),
        )
        stt = LiveSTTRuntime(
            audio.completed_recording_queue,
            moonshine,
            privacy,
            on_event=ui.event,
            on_partial=ui.partial if args.show_partials else None,
            music_worker=music,
        )
        # Dependency-safe worker order: DB, PII/music, STT, microphone.
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
        _shutdown_pipeline(
            session_id=session_id,
            ui=ui,
            audio=audio,
            stt=stt,
            music=music,
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
        if moonshine.loaded:
            counters = moonshine.counters
            ui.line(
                "COUNTERS",
                "model_resource_loads="
                f"{counters.model_resource_load_count}, "
                f"utterance_sessions={counters.utterance_session_count}, "
                f"callbacks={counters.callback_count}, "
                f"STT_add={counters.stt_processing_seconds:.3f}s, "
                f"finalization={counters.finalization_seconds:.3f}s, "
                f"STT_total={counters.total_stt_seconds:.3f}s, "
                f"queue_wait={counters.total_queue_wait_seconds:.3f}s",
            )
        moonshine.close()

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

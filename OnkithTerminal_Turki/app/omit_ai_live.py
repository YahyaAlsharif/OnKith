#!/usr/bin/env python3
"""Main executable for recorded-utterance STT and private text storage."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.environment import load_environment

# Before anything asks for a key. The device is started from a desktop
# launcher and from a boot service as well as from a terminal, and only the
# terminal ever had these exported.
ENVIRONMENT_NAMES = load_environment(ROOT)

from onkith_ui import LiveUI, muted_stderr, quiet_boot

quiet_boot()

from app.assistant import AssistantWorker
from app.audio_output import describe_device, format_device_report
from app.core_runtime import CoreRuntime
from app.core_server import is_headless_environment
from app.llm import GroqClient
from app.live_audio import LiveAudioRuntime
from app.live_runtime import (
    LivePrivacyRuntime,
    LiveSTTRuntime,
    deduplicate_boundary,
)
from app.onkith_music import (
    VOLUME_STEP,
    OnkithMusicController,
    OnkithMusicWorker,
    WakeWords,
    configure_wake_words,
    intercept_onkith_command,
    play_query_from,
    requested_level,
    without_wake_word,
)
from app.resources import ResourceMonitor
from app.session_report import SessionReport
from app.terminal_ui import TerminalUI
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.moonshine_adapter import MoonshineAdapter
from integrated_pipeline.whisper_adapter import WhisperAdapter
from integrated_pipeline.pii_adapter import PIIMaskerAdapter
from integrated_pipeline.pipeline import FailClosedMasker
from integrated_pipeline.schemas import LiveOutput, PIIResult
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
    "ONKITH_TTS",
    "ONKITH_PIPER_MODEL",
    "ONKITH_KOKORO_MODEL",
    "ONKITH_KOKORO_VOICES",
    "ONKITH_KOKORO_VOICE",
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
            "Recorded utterances through Moonshine Medium Streaming and "
            "on-device private terminal text"
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
    parser.add_argument(
        "--stt",
        choices=("moonshine", "whisper"),
        help="which recogniser listens (default: the one in live_config.json)",
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
        help="run the whole stack with no display and no Chromium",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="fixed port for the local surface server (default: ephemeral)",
    )
    parser.add_argument(
        "--device-report",
        action="store_true",
        help="print this device's OS release, audio stack, and playback tools",
    )
    parser.add_argument(
        "--surface-only",
        action="store_true",
        help=(
            "start only the screen and playback, with no microphone or "
            "models, for reviewing the player"
        ),
    )
    parser.add_argument(
        "--play",
        help="with --surface-only, seed the player with one search",
    )
    parser.add_argument(
        "--no-assistant",
        action="store_true",
        help="transcribe and mask without sending anything to the LLM",
    )
    parser.add_argument(
        "--report-file",
        help="also write the end-of-session report to this path as JSON",
    )
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
    assistant: "AssistantWorker | None",
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
    if assistant is not None:
        # Drained after masking, because everything it replies to comes
        # from there.
        try:
            assistant.shutdown()
        except BaseException as exc:
            _record_error(errors, "assistant drain", exc)
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


# The capture lane reports its own lifecycle and knows nothing about
# surfaces, which is the right way round: a websocket must never be able to
# stall a microphone. These are the two moments that tell a screen whether
# the device is hearing someone right now.
HEARING_BY_EVENT = {
    "recording_started": True,
    "segment_closed": False,
    # An utterance longer than the maximum is cut into pieces, and each cut
    # closes a segment while the person is still mid sentence. Without this
    # the screen would go dark on anyone who talks for more than a few
    # seconds. The cut closes before it continues, so the state does go false
    # and true again, both inside the same call.
    "forced_split_continuation": True,
}


def audio_event_bridge(core: Any, on_event: Any) -> Any:
    """Relay the capture lifecycle to the screen as well as the terminal."""

    def relay(name: str, payload: dict[str, Any]) -> None:
        on_event(name, payload)
        hearing = HEARING_BY_EVENT.get(name)
        if hearing is None or core is None:
            return
        try:
            core.set_listening(hearing)
        except Exception:
            # A surface that cannot be told must never stop the microphone.
            pass

    return relay


def _report_gate(core: Any, state: str) -> None:
    if core is None:
        return
    try:
        core.set_gate(state)
    except Exception:
        # A surface that cannot be told must never stop the pipeline.
        pass


def assistant_command(intent: str, raw_transcript: str, core: Any, music: Any) -> None:
    """Carry out an instruction the device did not recognise itself.

    The model sends a name and nothing else. Every argument is re-read
    here, from the raw transcript, because the copy the model saw had
    them taken out: "set the light to 10 percent" reaches it as
    "[AGE_1] percent" and "play Adele" as "play [GIVENNAME_1]". So the
    model is asked only what was meant, and the device supplies what was
    said.
    """

    if core is not None and intent in ("SHOW", "HIDE"):
        core.set_panel(intent == "SHOW")
        return
    if core is not None and intent in ("SHOW_DEVICES", "HIDE_DEVICES"):
        core.set_devices_panel(intent == "SHOW_DEVICES")
        return
    if core is not None and intent in ("LOUDER", "QUIETER"):
        core.step_volume(VOLUME_STEP if intent == "LOUDER" else -VOLUME_STEP)
        return
    if core is not None and intent in (
        "LIGHT_ON", "LIGHT_OFF", "BRIGHTER", "DIMMER", "LIGHT_LEVEL"
    ):
        core.handle_light(intent, requested_level(_words_of(raw_transcript)))
        return
    if music is None:
        return
    if intent == "CLOSE":
        music.submit_close()
    elif intent == "PLAY":
        # From the verb onward, not from the start of the sentence. The
        # recogniser may have mangled the title, and if it did there is
        # nothing here that can un-mangle it, but a bad search is a
        # better answer than a sentence explaining the wording.
        query = play_query_from(raw_transcript)
        if query:
            music.submit_play(query)


def _words_of(transcript: str) -> list[str]:
    from app.onkith_music import _WORD_RE, _normalise_word

    return [_normalise_word(m.group(0)) for m in _WORD_RE.finditer(transcript)]


def forward_masked_output(
    result: LiveOutput,
    assistant: "AssistantWorker | None",
    on_output: Any,
    core: Any = None,
    on_command: Any = None,
) -> None:
    """Display one finished utterance, then offer it to the assistant.

    This is the privacy boundary, and it now enforces two rules rather than
    one. Only text the masker cleared for release is ever handed on, so raw
    speech cannot reach a network model. And only speech that was addressed
    to the device is handed on at all: this thing sits in a room and hears
    everything said in it, so answering an overheard sentence is both the
    wrong behaviour and a conversation nobody agreed to send anywhere.

    Addressing is decided here, from the text, rather than carried down from
    the capture lane. The buffer can join several utterances into one release
    and the wake word survives that join, so reading the text is both simpler
    and more accurate than threading a flag through three schemas.
    """

    on_output(result)
    if not result.pii.safe_to_release:
        _report_gate(core, "blocked")
        return
    spoken_to = intercept_onkith_command(result.pii.original_text)
    addressed = spoken_to.command_detected and spoken_to.intent == "ASK"
    if assistant is None or not addressed:
        # Cleared, and it stayed here. Which is most of what this device
        # hears, and is meant to be.
        _report_gate(core, "held")
        # A command was still spoken to it, and the screen showed nothing
        # at all: the lamp changed and the conversation stayed blank,
        # which reads as not having been heard. Shown, not sent. Speech
        # that never addressed the device is still not shown, because it
        # was not talking to it.
        if assistant is not None and spoken_to.command_detected:
            assistant.note_command(result.pii.masked_text, result.pii.entities)
        return
    _report_gate(core, "sent")
    raw = result.pii.original_text
    # The name goes no further than here. It is how the device was
    # addressed, and a model reading "Onkith, what is your purpose"
    # takes it for who is speaking and answers "users like you, Onkith".
    # Stripped from the masked text, never the raw, so the boundary
    # above still holds.
    # The entity list travels with the masked text so the reply can be
    # rehydrated locally before it is spoken. It is never sent onward.
    assistant.submit(
        without_wake_word(result.pii.masked_text),
        result.pii.entities,
        # Closes over the raw transcript of this utterance so the model
        # can name an instruction without ever seeing its arguments.
        on_command=(
            (lambda intent: on_command(intent, raw))
            if on_command is not None
            else None
        ),
    )


def run_surface_only(args: argparse.Namespace) -> int:
    """Show the player without starting the microphone or the models.

    This exists so the screen can be reviewed on its own. It starts the
    same core the full session starts, minus capture, so what is on screen
    is the real surface driven by the real playback process.
    """

    core = CoreRuntime(
        headless=args.headless or is_headless_environment(),
        port=args.port,
        log=lambda message: print(f"[ONKITH] {message}"),
    )
    core.start()
    print(f"Player surface: {core.server.url}")
    if core.headless:
        print("Headless: open the URL above from a machine with a screen.")

    if args.play:
        controller = OnkithMusicController.from_environment(
            playback=core.playback,
            log=lambda message: print(f"[ONKITH] {message}"),
        )
        try:
            controller.play(args.play)
        except Exception as exc:
            print(f"[ONKITH] Could not start music: {type(exc).__name__}: {exc}")

    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()
    finally:
        core.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = build_parser().parse_args(argv)
    config = PipelineConfig()
    # Folded into the config rather than kept beside it, so --print-config
    # answers the question it is asked. It ran after that branch had
    # already printed, so checking which recogniser a run would use showed
    # the file and not the flag, and the flag is how the two get compared.
    if args.stt:
        config.data.setdefault("stt", {})["engine"] = args.stt
    # Before any transcript is looked at, so the device answers to whatever
    # this deployment decided to call it.
    configure_wake_words(WakeWords.from_config(config.wake))
    live_ui: LiveUI | None = None
    full_live_session = not any(
        (
            args.list_devices,
            args.print_config,
            args.self_test,
            args.device_report,
            args.surface_only,
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
    if args.device_report:
        report = describe_device()
        if args.json_output:
            print(json.dumps(report, indent=2))
        else:
            print(format_device_report(report))
        return 0
    if args.surface_only:
        return run_surface_only(args)

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
    # Read from the config, which the flag has already been folded into, so
    # the banner, --print-config and the adapter that actually loads cannot
    # name three different recognisers.
    engine = config.stt_engine
    stt_label = (
        f"Whisper {config.stt.get('whisper', {}).get('model', 'base.en')}"
        if engine == "whisper"
        else "Moonshine English Medium Streaming"
    )
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
                "stt": stt_label,
                "privacy": f"{config.pii_display_name} + structured PII",
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
            f"STT={stt_label}, RNNoise=off, "
            f"chunk={config.moonshine['chunk_seconds']:.1f}s, "
            f"silence_endpoint="
            f"{config.audio['endpoint_silence_seconds']:.1f}s, "
            f"recording_max="
            f"{config.audio['maximum_utterance_seconds']:.1f}s, "
            "Stage5=immediate, SQLite=WAL/NORMAL, "
            f"PII={config.pii_display_name}"
            "+structured whole_word_union",
        )

    moonshine = (
        WhisperAdapter(config) if engine == "whisper" else MoonshineAdapter(config)
    )
    audio: LiveAudioRuntime | None = None
    privacy: LivePrivacyRuntime | None = None
    stt: LiveSTTRuntime | None = None
    music: OnkithMusicWorker | None = None
    storage: SQLiteTranscriptStore | None = None
    core: CoreRuntime | None = None
    assistant: AssistantWorker | None = None
    errors: list[str] = []
    interrupted = False
    session_report = SessionReport(
        session_id=session_id,
        started_at_utc=session_started_at,
        stt_engine=stt_label,
    )
    # Every runtime already publishes what it did; until now it went to the
    # status log, where it scrolled away under the live screen. This taps
    # the same stream on the way past and keeps the distributions.
    def on_event(name: str, payload: dict[str, Any]) -> None:
        session_report.observe_event(name, payload)
        ui.event(name, payload)

    # Started before the models load, because loading them is the heaviest
    # minute of the run and a report that begins after it is a report that
    # misses the peak.
    monitor = ResourceMonitor(
        on_sample=(
            (lambda _sample: live_ui.set_usage(monitor.status_line()))
            if live_ui is not None
            else lambda _sample: None
        )
    )
    monitor.start()
    try:
        # The core owns audio, so it starts before the models load: the
        # screen is up while Moonshine and the masker are still loading.
        headless = args.headless or is_headless_environment()
        core = CoreRuntime(
            headless=headless,
            port=args.port,
            light_settings=config.light,
            log=lambda message: ui.event(
                "onkith_status", {"message": message}
            ),
        )
        core.start()
        ui.line("SURFACE", f"{core.server.url} ({core.server.connection_count} attached)")
        if headless:
            ui.line("SURFACE", "Headless: no display and no Chromium")

        # Named after whichever recogniser is actually loading. While two
        # are being compared, a log that always says moonshine is a log
        # that cannot be used to tell the runs apart.
        recogniser = (
            f"whisper {moonshine.model_name}"
            if isinstance(moonshine, WhisperAdapter)
            else "moonshine medium"
        )
        with (
            live_ui.task(recogniser, done_label=f"{recogniser} ready")
            if live_ui is not None
            else nullcontext()
        ):
            with muted_stderr():
                moonshine.load()
        if live_ui is None:
            ui.line(
                "MODEL",
                f"{recogniser} loaded "
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
                f"{config.pii_engine} INT8 loaded with CPUExecutionProvider "
                f"({config.pii_label_count} labels, "
                f"resource loads={pii_adapter.load_count}, "
                f"path={config.pii_model_path})",
            )

        audio = LiveAudioRuntime(
            config,
            session_id,
            device=args.device,
            once=args.once,
            on_event=audio_event_bridge(core, on_event),
            # Two readers of one number, and neither may be allowed to
            # stall the other: this runs on the sounddevice callback
            # thread. The core throttles its own copy; the terminal meter
            # wants every block it can get.
            # A third reader now, because whether the microphone is loud
            # enough to cross its own threshold has been guessed at since
            # the first Pi run and never once measured.
            on_level=(
                (
                    lambda rms: (
                        session_report.observe_level(rms),
                        live_ui.set_level(rms),
                        core.set_input_level(rms),
                    )
                )
                if live_ui is not None
                else (
                    lambda rms: (
                        session_report.observe_level(rms),
                        core.set_input_level(rms),
                    )
                )
            ),
            # So the device stops answering itself. The speaker and the
            # microphone are in one room, so without this every reply is
            # also an utterance to be transcribed and replied to.
            speaking=lambda: core.speech.speaking,
        )
        with muted_stderr():
            audio.prepare_device()
        # The device that was actually opened, which is not always the one
        # that was asked for. It goes to the screen as well as to the
        # session row, because a wrong microphone is invisible until
        # somebody notices the device is deaf.
        core.set_microphone(
            audio.device_name, audio.device_id, audio.sample_rate
        )
        session_report.microphone = f"{audio.device_name} @ {audio.sample_rate} Hz"
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
        if ENVIRONMENT_NAMES:
            # Names only. A key belongs in the file and nowhere else.
            ui.line("ENV", f".env supplied {', '.join(sorted(ENVIRONMENT_NAMES))}")

        llm = GroqClient(
            log=lambda message: ui.event(
                "onkith_status", {"message": message}
            )
        )
        if args.no_assistant:
            ui.line("ASSISTANT", "Disabled; nothing is sent to the LLM")
        elif not llm.configured:
            ui.line(
                "ASSISTANT",
                "GROQ_API_KEY is not set; transcripts stay on the device",
            )
        else:
            assistant = AssistantWorker(
                llm=llm,
                speech=core.speech,
                bus=core.bus,
                log=lambda message: ui.event(
                    "onkith_status", {"message": message}
                ),
                on_reply=session_report.observe_reply,
            )
            session_report.assistant_model = llm.settings.model
            ui.line(
                "ASSISTANT",
                f"Groq {llm.settings.model}, voice {core.speech.backend_name}",
            )
        session_report.voice = core.speech.backend_name

        def record_output(result: LiveOutput) -> None:
            session_report.observe_output(result)
            ui.output(result)

        privacy = LivePrivacyRuntime(
            config,
            safe_masker,
            storage,
            on_event=on_event,
            on_output=lambda result: forward_masked_output(
                result,
                assistant,
                record_output,
                core,
                # `music` is bound below and read when this runs, which
                # is after everything has started. The raw transcript
                # travels no further than this closure.
                on_command=lambda intent, raw: assistant_command(
                    intent, raw, core, music
                ),
            ),
            queue_overflow_count=lambda: (
                audio.overflow_count if audio is not None else 0
            ),
        )
        music = OnkithMusicWorker(
            OnkithMusicController.from_environment(
                playback=core.playback,
                log=lambda message: ui.event(
                    "onkith_status", {"message": message}
                ),
            ),
            log=lambda message: ui.event(
                "onkith_status", {"message": message}
            ),
        )
        stt = LiveSTTRuntime(
            audio.completed_recording_queue,
            moonshine,
            privacy,
            on_event=on_event,
            on_partial=ui.partial if args.show_partials else None,
            music_worker=music,
            on_panel=core.set_panel,
            on_devices_panel=core.set_devices_panel,
            on_volume=core.step_volume,
            on_light=core.handle_light,
        )
        # Dependency-safe worker order: DB, PII/music/assistant, STT,
        # microphone. The assistant consumes masked output, so it is ready
        # before anything can produce any.
        privacy.start()
        if assistant is not None:
            assistant.start()
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
        monitor.stop()
        _shutdown_pipeline(
            session_id=session_id,
            ui=ui,
            audio=audio,
            stt=stt,
            music=music,
            privacy=privacy,
            assistant=assistant,
            storage=storage,
            errors=errors,
        )
        if core is not None:
            try:
                core.shutdown()
            except BaseException as exc:
                _record_error(errors, "core shutdown", exc)
        # These used to be four tagged lines printed here. Under the live
        # screen they went to the status log and scrolled away, so the
        # numbers existed and nobody ever had them. They go into the
        # session report instead, which is printed after the screen has
        # released the terminal.
        _collect_audit(
            session_report,
            audio=audio,
            storage=storage,
            privacy=privacy,
            moonshine=moonshine,
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
        _print_usage_report(monitor, session_report, args)
        return 1
    ui.line(
        "SHUTDOWN",
        "Stopped cleanly" + (" after Ctrl+C" if interrupted else ""),
    )
    if live_ui is not None:
        live_ui.stop()
    _print_usage_report(monitor, session_report, args)
    return 0


def _collect_audit(
    report: SessionReport,
    *,
    audio: Any,
    storage: Any,
    privacy: Any,
    moonshine: Any,
) -> None:
    """Gather the end-of-run counters every worker keeps for itself.

    Each of these is a high-water mark rather than an average, because
    the question they answer is whether the board ever fell behind, and
    an average over a quiet hour hides the thirty seconds it did.
    """

    try:
        if audio is not None:
            report.audit.update(audio.audit())
        if storage is not None:
            # Named rather than copied wholesale: the store's audit also
            # carries its absolute path and its thread ids, and this
            # artifact is written to be attached to a report and sent to
            # somebody. Prefixed because the database and the capture
            # lane both call their backlog "queue_high_water_mark".
            db_audit = storage.audit()
            report.audit.update(
                {
                    f"db_{name}": db_audit[name]
                    for name in (
                        "operation_count",
                        "rejected_operation_count",
                        "queue_high_water_mark",
                        "maximum_queue_wait_seconds",
                        "total_write_seconds",
                        "maximum_write_seconds",
                    )
                    if name in db_audit
                }
            )
        if privacy is not None:
            report.audit["mask_queue_high_water_mark"] = (
                privacy.queue_high_water_mark
            )
        if moonshine.loaded:
            counters = moonshine.counters
            report.audit.update(
                {
                    "model_resource_loads": counters.model_resource_load_count,
                    "utterance_sessions": counters.utterance_session_count,
                    "stt_seconds_total": counters.total_stt_seconds,
                    "stt_queue_wait_total": counters.total_queue_wait_seconds,
                }
            )
    except Exception as error:
        report.audit["audit_error"] = f"{type(error).__name__}: {error}"


def _print_usage_report(
    monitor: ResourceMonitor,
    report: SessionReport,
    args: argparse.Namespace,
) -> None:
    """After the live screen has released the terminal, not before.

    The status bar owns the bottom three rows while it runs, so anything
    printed there during shutdown is overwritten by the next repaint.
    """
    if args.report_file:
        try:
            report.write_json(args.report_file)
        except OSError as error:
            print(f"Could not write {args.report_file}: {error}", file=sys.stderr)
    if args.json_output:
        print(json.dumps(report.as_dict(), ensure_ascii=False))
        return
    print()
    print(report.text())
    print()
    print(monitor.report())


if __name__ == "__main__":
    raise SystemExit(main())

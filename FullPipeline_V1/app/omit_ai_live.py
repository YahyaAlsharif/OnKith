#!/usr/bin/env python3
"""Main executable for continuous live STT-to-private-text processing."""

from __future__ import annotations

import argparse
import json
import platform
import signal
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.live_audio import LiveAudioRuntime
from app.live_runtime import LivePrivacyRuntime
from app.terminal_ui import TerminalUI
from integrated_pipeline.config import BUFFER_INACTIVITY_SECONDS, PipelineConfig
from integrated_pipeline.inactivity_buffer import InactivityTranscriptBuffer
from integrated_pipeline.moonshine_adapter import MoonshineAdapter
from integrated_pipeline.pii_adapter import PIIMaskerAdapter
from integrated_pipeline.pipeline import FailClosedMasker
from integrated_pipeline.schemas import PIIResult, TranscriptFragment
from integrated_pipeline.structured_pii import StructuredPIIRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Moonshine Medium Streaming to TinyBERT private terminal text"
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--show-partials", action="store_true")
    parser.add_argument("--text-test")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json-output", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-colour", action="store_true")
    return parser


def static_self_test(config: PipelineConfig) -> None:
    config.validate(require_artifacts=True)
    if BUFFER_INACTIVITY_SECONDS != 3.0:
        raise AssertionError("The inactivity policy changed")

    recognizer = StructuredPIIRecognizer()
    spans = recognizer.find_spans(
        "Email me at rayyan@example.com and my phone number is 555 123 4567."
    )
    types = {span.entity_type for span in spans}
    if not {"EMAIL", "TELEPHONENUM"}.issubset(types):
        raise AssertionError(f"Structured PII smoke failed: {types}")

    buffer = InactivityTranscriptBuffer()
    first = TranscriptFragment("f1", "hello", 10.0, "self-test")
    second = TranscriptFragment("f2", "world", 12.9, "self-test")
    assert buffer.add_fragment(first) == []
    assert buffer.add_fragment(second) == []
    output = buffer.flush_due(15.9)
    if output is None or output.text != "hello world":
        raise AssertionError("Fixed inactivity merge/flush failed")

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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig()
    ui = TerminalUI(
        show_raw=args.show_raw,
        show_partials=args.show_partials,
        json_output=args.json_output,
        quiet=args.quiet,
        colour=not args.no_colour,
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

    ui.line("STARTUP", "Raspberry Pi live privacy pipeline")
    ram = available_ram_mib()
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
        f"update={config.moonshine['update_interval_seconds']:.2f}s, "
        f"VAD={config.moonshine['vad_threshold']:.2f}, "
        f"buffer={config.buffer_inactivity_seconds:.1f}s, "
        "PII=TinyBERT-4 INT8+structured whole_word_union",
    )
    moonshine = MoonshineAdapter(config)
    privacy: LivePrivacyRuntime | None = None
    audio: LiveAudioRuntime | None = None
    try:
        moonshine.load()
        ui.line(
            "MODEL",
            "Moonshine Medium Streaming loaded "
            f"(resource loads={moonshine.counters.model_resource_load_count})",
        )
        pii_adapter.load()
        ui.line(
            "MODEL",
            "TinyBERT INT8 loaded with CPUExecutionProvider "
            f"(resource loads={pii_adapter.load_count}, "
            f"path={config.tinybert_model_path})",
        )
        privacy = LivePrivacyRuntime(
            config,
            safe_masker,
            on_event=ui.event,
            on_output=ui.output,
            queue_overflow_count=lambda: moonshine.counters.queue_overflow_count,
        )
        privacy.start()
        audio = LiveAudioRuntime(
            config,
            moonshine,
            device=args.device,
            once=args.once,
            on_event=ui.event,
            on_final=privacy.submit_stt,
            on_partial=ui.partial if args.show_partials else None,
        )
        signal.signal(signal.SIGTERM, lambda signum, frame: audio.stop())
        audio.run()
        if args.once:
            privacy.wait_for_output(
                timeout=config.buffer_inactivity_seconds + 2.0
            )
    except KeyboardInterrupt:
        ui.line("SHUTDOWN", "Ctrl+C received; flushing private-text buffer")
    except Exception as exc:
        ui.line("ERROR", f"{type(exc).__name__}: {exc}", force=True)
        return 1
    finally:
        if audio is not None:
            audio.stop()
        if privacy is not None:
            privacy.shutdown()
        if moonshine.loaded:
            counters = moonshine.counters
            ui.line(
                "COUNTERS",
                "model_resource_loads="
                f"{counters.model_resource_load_count}, "
                f"utterance_sessions={counters.utterance_session_count}, "
                f"callbacks={counters.callback_count}, "
                f"queue_overflows={counters.queue_overflow_count}, "
                f"stt_processing={counters.stt_processing_seconds:.3f}s, "
                f"finalization={counters.finalization_seconds:.3f}s",
            )
        moonshine.close()
    ui.line("SHUTDOWN", "Stopped cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fast Raspberry Pi checks with an optional live microphone level test."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import io
import json
import platform
import sys
import tempfile
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.live_audio import LiveAudioRuntime, microphone_rms_test
from app.omit_ai_live import static_self_test
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.moonshine_adapter import MoonshineAdapter
from integrated_pipeline.pii_adapter import PIIMaskerAdapter
from integrated_pipeline.pipeline import FailClosedMasker
from integrated_pipeline.schemas import STTResult
from integrated_pipeline.sqlite_storage import SQLiteTranscriptStore


def code_without_prose(path: Path) -> str:
    """The file with its comments and docstrings taken out.

    The boundary below is about what this package depends on, and a
    grep over the raw text cannot tell a dependency from a sentence
    about one. The Whisper adapter's docstring explains that it does
    not pull torch, and that sentence failed the check that the
    sentence exists to reassure people about. The previous answer was
    an exception for one filename, which lasted exactly until a second
    file had a reason to say the word.

    Comments and docstrings go. Ordinary string literals stay, so
    something reached for by name at runtime is still caught.

    Requirements files get the same treatment for the same reason. The
    line that pins kokoro-onnx has to say why plain kokoro is not
    allowed, and naming the thing it is not allowed to pull failed the
    check that the sentence exists to explain.
    """

    source = path.read_text(encoding="utf-8")
    if path.suffix != ".py":
        return "\n".join(
            line.split("#", 1)[0] for line in source.splitlines()
        ).lower()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Unparseable is not a reason to stop policing it.
        return source.lower()
    prose_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            prose_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    kept: list[str] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return source.lower()
    for token in tokens:
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and token.start[0] in prose_lines:
            continue
        kept.append(token.string)
    return " ".join(kept).lower()


def verify_deployment_boundary() -> None:
    """Verify this reduced package contains no research runtime or extra model."""

    blocked = [
        "data" + "sets",
        "load_" + "dataset",
        "libri" + "speech",
        "kag" + "gle",
        "pan" + "das",
        "mat" + "plotlib",
        "sk" + "learn",
        "seq" + "eval",
        "to" + "rch",
        "wer " + "benchmark",
    ]
    findings = []
    runtime_targets = [
        ROOT / "onkith_ui.py",
        ROOT / "app",
        ROOT / "integrated_pipeline",
        ROOT / "moonshine_runtime",
        ROOT / "scripts" / "check_pi.py",
        ROOT / "requirements-rpi.txt",
    ]
    for target in runtime_targets:
        files = [target] if target.is_file() else list(target.rglob("*.py"))
        for path in files:
            text = code_without_prose(path)
            for term in blocked:
                if term in text:
                    findings.append(f"{path.relative_to(ROOT)}: {term}")
    if findings:
        raise RuntimeError(
            "Deployment dependency-boundary failure: " + "; ".join(findings)
        )
    # Only the shipped runtime tree is policed. Third-party packages inside
    # the virtual environment carry their own sample models (onnxruntime
    # ships several) and are not part of what this project deploys.
    onnx_files = [
        path
        for path in ROOT.rglob("*.onnx")
        if ".venv" not in path.parts and "site-packages" not in path.parts
    ]
    # Both trained classifiers ship: Model V2 is what runs, Model V1 is the
    # way back if it is too heavy for the board. Anything else appearing
    # here is a model that arrived without being decided on.
    expected = {
        ROOT / "models" / "deberta_pii" / "model_int8.onnx",
        ROOT / "models" / "tinybert_pii" / "model_int8.onnx",
    }
    unexpected = sorted(str(path) for path in set(onnx_files) - expected)
    if unexpected:
        raise RuntimeError(f"Unexpected ONNX models in the tree: {unexpected}")
    missing = sorted(str(path) for path in expected if not path.is_file())
    if missing:
        raise RuntimeError(f"Required ONNX models are missing: {missing}")


def verify_lane_ownership() -> None:
    """Statically reject model/SQL work in capture and the old live timer."""

    live_audio = (ROOT / "app" / "live_audio.py").read_text(encoding="utf-8")
    forbidden_audio_calls = (
        "open_utterance(",
        ".add_audio(",
        "transcribe_recording(",
        "sqlite3",
        "SQLiteTranscriptStore",
    )
    found = [
        value for value in forbidden_audio_calls if value in live_audio
    ]
    if found:
        raise RuntimeError(
            f"Capture/segmentation lane owns forbidden work: {found}"
        )
    callback_source = live_audio.split("    def _audio_callback(", 1)[1].split(
        "    def _report_overflows(", 1
    )[0]
    if "self.on_level(rms)" not in callback_source:
        raise RuntimeError("Existing RMS is not connected to the live UI")
    if "print(" in callback_source or "self.on_event(" in callback_source:
        raise RuntimeError("Audio callback performs terminal/event output")

    live_runtime = (ROOT / "app" / "live_runtime.py").read_text(
        encoding="utf-8"
    )
    if "self.buffer_worker = WallClockBufferWorker(" in live_runtime:
        raise RuntimeError("Live runtime still instantiates Stage 5 timer")
    stt_source = live_runtime.split("class LiveSTTRuntime:", 1)[1].split(
        "class LivePrivacyRuntime:", 1
    )[0]
    if "self.storage" in stt_source or ".insert_raw(" in stt_source:
        raise RuntimeError("STT runtime still owns SQLite transcript storage")
    interception = stt_source.find(
        "interception = intercept_onkith_command(text)"
    )
    stage5_submit = stt_source.find(
        "self.privacy.submit_stt(result)", interception
    )
    if interception < 0 or stage5_submit < 0 or interception > stage5_submit:
        raise RuntimeError(
            "Onkith interception must occur after final STT and before Stage 5"
        )
    for forbidden in (
        "youtube_get(",
        "search_youtube(",
        "urllib.request",
        "subprocess.Popen",
    ):
        if forbidden in stt_source:
            raise RuntimeError(
                f"STT runtime owns forbidden music work: {forbidden}"
            )
    privacy_source = live_runtime.split("class LivePrivacyRuntime:", 1)[1]
    raw_enqueue = privacy_source.find("self.storage.insert_raw(result)")
    mask_enqueue = privacy_source.find("self.mask_queue.put(MaskTask(")
    if raw_enqueue < 0 or mask_enqueue < 0 or raw_enqueue > mask_enqueue:
        raise RuntimeError(
            "Stage 5 must enqueue raw storage before the immediate mask task"
        )
    if "daemon=True" in live_audio or "daemon=True" in live_runtime:
        raise RuntimeError("Live workers must not rely on daemon termination")

    music_source = (ROOT / "app" / "onkith_music.py").read_text(
        encoding="utf-8"
    )
    if 'os.environ["YOUTUBE_API_KEY"]' not in music_source:
        raise RuntimeError("YouTube API key must be read from the environment")
    worker_source = music_source.split("class OnkithMusicWorker:", 1)[1]
    if "daemon=False" not in worker_source:
        raise RuntimeError("Music worker must be joined during shutdown")

    entry_source = (ROOT / "app" / "omit_ai_live.py").read_text(
        encoding="utf-8"
    )
    quiet_boot = entry_source.find("quiet_boot()")
    first_project_import = entry_source.find("from app.live_audio import")
    if (
        quiet_boot < 0
        or first_project_import < 0
        or quiet_boot > first_project_import
    ):
        raise RuntimeError("quiet_boot must run before project/model imports")
    if entry_source.count("LiveUI(") != 1:
        raise RuntimeError("The application must own exactly one LiveUI")
    terminal_source = (ROOT / "app" / "terminal_ui.py").read_text(
        encoding="utf-8"
    )
    if "self.live_ui.utterance(" not in terminal_source:
        raise RuntimeError("Final masked output is not connected to LiveUI")

    sqlite_callers = []
    for directory in ("app", "integrated_pipeline", "moonshine_runtime"):
        for path in (ROOT / directory).rglob("*.py"):
            if path.name == "sqlite_storage.py":
                continue
            if "sqlite3" in path.read_text(encoding="utf-8"):
                sqlite_callers.append(str(path.relative_to(ROOT)))
    if sqlite_callers:
        raise RuntimeError(
            "SQLite calls exist outside the writer module: "
            + ", ".join(sqlite_callers)
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--with-mic", action="store_true")
    result.add_argument("--device", type=int)
    return result


def verify_sqlite_writer() -> None:
    """Exercise WAL, writer ownership, and ordered session reads."""

    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteTranscriptStore(
            Path(directory) / "session_transcript.db"
        )
        store.start()
        store.start_session("check-session", "2026-01-01T00:00:00Z")
        for sequence in (2, 1):
            store.insert_raw(
                STTResult(
                    final_text=f"batch {sequence}",
                    duration_seconds=1.0,
                    processing_seconds=0.1,
                    finalization_seconds=0.1,
                    rtf=0.2,
                    callback_count=1,
                    session_id="check-session",
                    batch_id=f"check-session-batch-{sequence:06d}",
                    sequence_number=sequence,
                    endpoint_reason="endpoint_silence",
                    audio_started_at_utc="2026-01-01T00:00:00Z",
                    audio_ended_at_utc="2026-01-01T00:00:01Z",
                    stt_started_at_utc="2026-01-01T00:00:01Z",
                    stt_completed_at_utc="2026-01-01T00:00:02Z",
                    captured_audio_seconds=1.0,
                    submitted_audio_seconds=1.0,
                    total_stt_seconds=0.2,
                )
            )
        store.drain()
        rows = store.query_session("check-session")
        if [row.sequence_number for row in rows] != [1, 2]:
            raise RuntimeError("SQLite session order check failed")
        if store.journal_mode != "wal" or store.synchronous_level != 1:
            raise RuntimeError("SQLite WAL/NORMAL check failed")
        if store.writer_thread_id != store.connection_thread_id:
            raise RuntimeError("SQLite connection owner check failed")
        store.shutdown()


def main() -> int:
    args = parser().parse_args()
    print(f"[CHECK] Python {platform.python_version()}")
    print(f"[CHECK] Architecture {platform.machine()}")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Target runtime requires Python 3.11")

    config = PipelineConfig()
    static_self_test(config)
    print("[OK] Configuration, structured recognizers, and fail-closed boundary")
    verify_deployment_boundary()
    print("[OK] Deployment has no research-runtime references or duplicate ONNX files")
    verify_lane_ownership()
    print(
        "[OK] STT submits to immediate Stage 5; Stage 5 owns raw/mask fan-out"
    )
    verify_sqlite_writer()
    print("[OK] SQLite WAL/NORMAL writer ownership and session ordering")

    pii = PIIMaskerAdapter(config)
    pii.load()
    probe = FailClosedMasker(pii).mask_text(
        "My email is rayyan@example.com and account number is 12345678."
    )
    if not probe.safe_to_release:
        raise RuntimeError("PII smoke was blocked")
    if "rayyan@example.com" in probe.masked_text or "12345678" in probe.masked_text:
        raise RuntimeError("Structured PII smoke leaked a value")
    print("[OK] Local tokenizer, PII INT8 classifier, and CPUExecutionProvider")

    package_version = importlib.metadata.version("moonshine-voice")
    if package_version != "0.0.69":
        raise RuntimeError(f"Unexpected moonshine-voice {package_version}")
    moonshine = MoonshineAdapter(config)
    moonshine.load()
    if moonshine.counters.model_resource_load_count != 1:
        raise RuntimeError("Moonshine resource did not load exactly once")
    print("[OK] Moonshine English Medium Streaming model is available")
    print(json.dumps(moonshine.audit(), indent=2))
    moonshine.close()

    devices = LiveAudioRuntime.list_devices()
    print("[CHECK] Input devices:")
    for row in devices:
        print(
            f"  [{row['id']}] {row['name']} | "
            f"{row['sample_rate']} Hz | channels={row['channels']}"
        )
    if not devices:
        print("[WARNING] No input device is currently visible.")
    if args.with_mic:
        microphone_rms_test(args.device, seconds=3.0)
    print("[OK] Raspberry Pi checks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

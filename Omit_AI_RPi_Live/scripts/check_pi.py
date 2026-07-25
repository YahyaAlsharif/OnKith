#!/usr/bin/env python3
"""Fast Raspberry Pi checks with an optional live microphone level test."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
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
        ROOT / "app",
        ROOT / "integrated_pipeline",
        ROOT / "moonshine_runtime",
        ROOT / "scripts" / "check_pi.py",
        ROOT / "requirements-rpi.txt",
    ]
    for target in runtime_targets:
        files = [target] if target.is_file() else list(target.rglob("*.py"))
        for path in files:
            text = path.read_text(encoding="utf-8").lower()
            for term in blocked:
                if term in text:
                    findings.append(f"{path.relative_to(ROOT)}: {term}")
    if findings:
        raise RuntimeError(
            "Deployment dependency-boundary failure: " + "; ".join(findings)
        )
    onnx_files = list(ROOT.rglob("*.onnx"))
    expected = ROOT / "models" / "tinybert_pii" / "model_int8.onnx"
    if onnx_files != [expected]:
        raise RuntimeError(f"Expected only {expected}; found {onnx_files}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--with-mic", action="store_true")
    result.add_argument("--device", type=int)
    return result


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

    pii = PIIMaskerAdapter(config)
    pii.load()
    probe = FailClosedMasker(pii).mask_text(
        "My email is rayyan@example.com and account number is 12345678."
    )
    if not probe.safe_to_release:
        raise RuntimeError("PII smoke was blocked")
    if "rayyan@example.com" in probe.masked_text or "12345678" in probe.masked_text:
        raise RuntimeError("Structured PII smoke leaked a value")
    print("[OK] Local tokenizer, TinyBERT INT8, and CPUExecutionProvider")

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

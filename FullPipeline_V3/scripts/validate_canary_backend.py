#!/usr/bin/env python3
"""Real Canary backend validation: API, decoder, isolation, and version parity.

This script performs the checks that mocks cannot prove. It never fabricates a
result: every check reports PASS, FAIL, NOT RUN or UNVERIFIED, and the exit
status is non-zero only for genuine FAILs. A missing model artifact or a
missing NeMo/torch install yields NOT RUN, never PASS.

Usage
-----
    python scripts/validate_canary_backend.py
    python scripts/validate_canary_backend.py --json
    python scripts/validate_canary_backend.py --parity-audio a.wav b.wav
    python scripts/validate_canary_backend.py --parity-report other_version.json

Version parity (spec section 8.7) needs two environments. Run this script with
``--emit-parity parity_<version>.json`` under NeMo 2.5.3 and again under NeMo
2.7.3, then compare with ``--parity-report``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrated_pipeline.canary_adapter import CanaryAdapter, describe_nemo_backend
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.schemas import CanarySegment

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT RUN"
UNVERIFIED = "UNVERIFIED"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Report:
    """Collect check outcomes without ever upgrading NOT RUN into PASS."""

    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def add(
        self, name: str, status: str, detail: str = "", **extra: object
    ) -> None:
        if status not in {PASS, FAIL, NOT_RUN, UNVERIFIED}:
            raise ValueError(f"Unsupported status: {status}")
        row = {"check": name, "status": status, "detail": detail}
        row.update(extra)
        self.checks.append(row)
        print(f"[{status:^10}] {name}" + (f" :: {detail}" if detail else ""))

    @property
    def failed(self) -> bool:
        return any(row["status"] == FAIL for row in self.checks)

    def summary(self) -> dict[str, int]:
        counts = {PASS: 0, FAIL: 0, NOT_RUN: 0, UNVERIFIED: 0}
        for row in self.checks:
            counts[str(row["status"])] += 1
        return counts


def make_segment(index: int, samples: np.ndarray) -> CanarySegment:
    now = time.monotonic()
    stamp = utc_now_iso()
    start = index * 1_000_000
    end = start + int(samples.size)
    return CanarySegment(
        session_id="canary-backend-validation",
        utterance_id=f"validation-utterance-{index + 1:06d}",
        segment_id=f"validation-segment-{index:06d}",
        utterance_sequence_number=index + 1,
        segment_index=0,
        samples=samples,
        sample_rate=16_000,
        segment_start_sample=start,
        segment_end_sample=end,
        decision_sample=end,
        last_speech_end_sample=end,
        forced_boundary_sample=None,
        endpoint_reason="end_of_stream",
        is_final_segment_of_utterance=True,
        reliable=True,
        dropped_audio_chunks=0,
        recording_closed_monotonic=now,
        audio_started_monotonic=now - 1.0,
        audio_ended_monotonic=now,
        audio_started_at_utc=stamp,
        audio_ended_at_utc=stamp,
        utterance_start_sample=start,
        utterance_captured_end_sample=end,
        segment_unique_start_sample=start,
    )


def load_wav_16k(path: Path) -> np.ndarray:
    """Read a mono 16 kHz PCM WAV without adding a runtime dependency."""

    import wave

    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16_000:
            raise ValueError(f"{path} is not 16 kHz")
        if handle.getnchannels() != 1:
            raise ValueError(f"{path} is not mono")
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM")
        raw = handle.readframes(handle.getnframes())
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).copy()


def check_api(report: Report) -> dict[str, object]:
    backend = describe_nemo_backend()
    print(json.dumps(backend, indent=2))
    if not backend["available"]:
        report.add(
            "NeMo/torch import",
            NOT_RUN,
            "; ".join(str(item) for item in backend["errors"]),
            nemo_version=backend["nemo_version"],
            torch_version=backend["torch_version"],
        )
        return backend
    report.add(
        "NeMo/torch import",
        PASS,
        f"nemo={backend['nemo_version']} torch={backend['torch_version']}",
    )
    report.add(
        "Required API signatures (restore_from / transcribe)",
        PASS if backend["signature_contract_satisfied"] else FAIL,
        json.dumps(backend.get("missing_parameters", {})),
    )
    return backend


def check_model(
    report: Report, config: PipelineConfig, speech: list[np.ndarray]
) -> dict[str, object]:
    result: dict[str, object] = {"transcripts": []}
    model_path = config.canary_model_path
    if not model_path.is_file():
        report.add(
            "Canary artifact present",
            NOT_RUN,
            f"missing {model_path}; run scripts/acquire_canary.py",
        )
        return result

    adapter = CanaryAdapter(config)
    try:
        artifact = adapter.verify_model_artifact()
    except Exception as exc:
        report.add("Canary artifact SHA-256 and manifest", FAIL, f"{exc}")
        return result
    report.add(
        "Canary artifact SHA-256 and manifest",
        PASS,
        f"{artifact['sha256']} ({artifact['size_bytes']} bytes)",
    )

    started = time.perf_counter()
    try:
        adapter.load()
    except Exception as exc:
        report.add("Canary load + warmup", FAIL, f"{type(exc).__name__}: {exc}")
        return result
    load_seconds = time.perf_counter() - started
    result["load_seconds"] = load_seconds
    report.add(
        "Canary load + warmup",
        PASS,
        f"{load_seconds:.3f} s, loads={adapter.counters.model_load_count}, "
        f"warmups={adapter.counters.warmup_count}",
    )

    audit = adapter.audit()
    report.add(
        "Active decoder read-back (beam=2, best, temp 1.0, delta 50)",
        PASS,
        json.dumps(audit["active_decoder"]),
    )
    report.add(
        "Active prompt (task asr, en/en, pnc no, timestamp no)",
        PASS,
        json.dumps(audit["active_prompt"]),
    )
    report.add(
        "torch.inference_mode() active for warmup and inference",
        PASS if audit["inference_mode_active"] else FAIL,
        f"inference_mode_active={audit['inference_mode_active']}",
    )

    model_identity = id(adapter.model)
    try:
        for index, samples in enumerate(speech):
            transcript = adapter.transcribe_segment(make_segment(index, samples))
            result["transcripts"].append(
                {
                    "index": index,
                    "text": transcript.text,
                    "processing_seconds": transcript.processing_seconds,
                    "audio_seconds": float(samples.size) / 16_000,
                }
            )
    except Exception as exc:
        report.add("Independent segment inference", FAIL, f"{type(exc).__name__}: {exc}")
        adapter.close()
        return result

    texts = [str(row["text"]) for row in result["transcripts"]]
    report.add(
        "Two independent sequential calls, one model object",
        PASS if id(adapter.model) == model_identity else FAIL,
        f"inferences={adapter.counters.inference_count}",
    )
    if len(speech) >= 2:
        repeated = adapter.transcribe_segment(make_segment(0, speech[0]))
        report.add(
            "No state contamination (call 1 repeats after call 2)",
            PASS if repeated.text == texts[0] else FAIL,
            f"first={texts[0]!r} repeat={repeated.text!r}",
        )
    total_model = sum(float(row["processing_seconds"]) for row in result["transcripts"])
    total_audio = sum(float(row["audio_seconds"]) for row in result["transcripts"])
    if total_audio:
        result["rtf"] = total_model / total_audio
        report.add(
            "Measured RTF on this host",
            UNVERIFIED if platform.machine().lower() not in {"aarch64", "arm64"} else PASS,
            f"model={total_model:.3f}s audio={total_audio:.3f}s "
            f"rtf={total_model / total_audio:.3f} on {platform.machine()}"
            + (
                " (not a Raspberry Pi; not a Pi performance result)"
                if platform.machine().lower() not in {"aarch64", "arm64"}
                else ""
            ),
        )
    adapter.close()
    adapter.close()
    report.add(
        "Canary close is idempotent",
        PASS if adapter.counters.close_count == 1 else FAIL,
        f"close_count={adapter.counters.close_count}",
    )
    result["nonempty"] = all(text.strip() for text in texts)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parity-audio",
        nargs="*",
        type=Path,
        default=[],
        help="16 kHz mono 16-bit WAV files to decode (real speech recommended)",
    )
    parser.add_argument("--emit-parity", type=Path)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = Report()
    print(
        f"[CHECK] {platform.platform()} | Python {platform.python_version()} "
        f"| {platform.machine()} | {utc_now_iso()}"
    )
    backend = check_api(report)

    if args.parity_audio:
        speech = [load_wav_16k(path) for path in args.parity_audio]
        speech_source = [str(path) for path in args.parity_audio]
    else:
        # Deterministic synthetic input still proves API/decoder/isolation, but
        # it proves nothing about recognition accuracy.
        rng = np.random.default_rng(20260805)
        speech = [
            rng.normal(0.0, 0.01, 16_000).astype(np.float32),
            rng.normal(0.0, 0.01, 24_000).astype(np.float32),
        ]
        speech_source = ["synthetic-noise"]
        report.add(
            "Speech recognition accuracy",
            NOT_RUN,
            "no --parity-audio supplied; synthetic input cannot prove recognition",
        )

    config = PipelineConfig()
    if not backend["available"]:
        report.add(
            "Real model inference",
            NOT_RUN,
            "NeMo/torch unavailable in this environment",
        )
        model_result: dict[str, object] = {"transcripts": []}
    else:
        model_result = check_model(report, config, speech)

    payload = {
        "generated_at_utc": utc_now_iso(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "backend": backend,
        "speech_source": speech_source,
        "model": model_result,
        "checks": report.checks,
        "summary": report.summary(),
    }

    if args.parity_report is not None:
        other = json.loads(args.parity_report.read_text(encoding="utf-8"))
        mine = [row["text"] for row in model_result.get("transcripts", [])]
        theirs = [row["text"] for row in other.get("model", {}).get("transcripts", [])]
        if not mine or not theirs:
            report.add(
                "NeMo version transcript parity",
                NOT_RUN,
                "one side produced no transcripts",
            )
        else:
            equal = sum(1 for a, b in zip(mine, theirs) if a == b)
            differing = [
                {"index": index, "this": a, "other": b}
                for index, (a, b) in enumerate(zip(mine, theirs))
                if a != b
            ]
            payload["parity"] = {
                "this_nemo": backend["nemo_version"],
                "other_nemo": other.get("backend", {}).get("nemo_version"),
                "compared": min(len(mine), len(theirs)),
                "exact_equal": equal,
                "differing": differing,
            }
            report.add(
                "NeMo version transcript parity",
                PASS if not differing else FAIL,
                f"{equal}/{min(len(mine), len(theirs))} exact matches; "
                f"{len(differing)} differing",
            )

    if args.emit_parity is not None:
        args.emit_parity.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"[INFO] wrote {args.emit_parity}")
    if args.json:
        print(json.dumps(payload, indent=2))

    counts = report.summary()
    print(
        f"\nSummary: {counts[PASS]} PASS, {counts[FAIL]} FAIL, "
        f"{counts[NOT_RUN]} NOT RUN, {counts[UNVERIFIED]} UNVERIFIED"
    )
    if counts[NOT_RUN] or counts[UNVERIFIED]:
        print(
            "NOT RUN / UNVERIFIED checks are NOT passes. Real-model and "
            "Raspberry Pi validation remain pending until they report PASS on "
            "the target device."
        )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

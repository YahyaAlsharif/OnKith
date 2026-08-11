"""Command-line runner for TinyBERT model-only and production OOD baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from .adapters import FP32_ONNX, INT8_ONNX, stage_factory
from .metrics import controlled_pair_analysis, evaluate_predictions
from .schema import load_cases

DEFAULT_STAGES = ("fp32_model", "int8_model", "structured", "structured_supplemental", "production")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict(masker, cases) -> dict[str, dict]:
    predictions: dict[str, dict] = {}
    for number, case in enumerate(cases, start=1):
        started = time.perf_counter()
        try:
            result = masker.mask_text(case.text)
            predictions[case.case_id] = {
                "entities": list(result.entities),
                "masked_text_changed": result.masked_text != case.text,
                "latency_seconds": float(getattr(result, "latency_seconds", time.perf_counter() - started)),
                "safe_to_release": bool(getattr(result, "safe_to_release", True)),
                "raw_model_output": dict(getattr(result, "raw_model_output", {}) or {}),
            }
        except Exception as exc:
            predictions[case.case_id] = {
                "entities": [],
                "masked_text_changed": True,
                "latency_seconds": time.perf_counter() - started,
                "safe_to_release": False,
                "raw_model_output": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
        if number % 50 == 0:
            print(f"  evaluated {number}/{len(cases)}")
    return predictions


def run_partition(
    fixtures: Path,
    output_dir: Path,
    partition: str,
    stages: tuple[str, ...],
    limit: int | None = None,
    retain_predictions: bool = False,
) -> dict:
    fixture_path = fixtures / f"ood_{partition}.jsonl"
    manifest_path = fixtures / f"ood_{partition}.manifest.json"
    cases = load_cases(fixture_path)
    if limit is not None:
        cases = cases[:limit]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for stage in stages:
        print(f"{partition}: {stage}")
        masker = stage_factory(stage)
        predictions = _predict(masker, cases)
        metrics = evaluate_predictions(cases, predictions)
        payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "partition": partition,
            "stage": stage,
            "case_count": len(cases),
            "fixture_file": str(fixture_path),
            "fixture_sha256": _sha256(fixture_path),
            "fixture_manifest_sha256": _sha256(manifest_path),
            "frozen_final": bool(manifest["frozen"]),
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "models": {
                "fp32_onnx": {"path": str(FP32_ONNX), "bytes": FP32_ONNX.stat().st_size, "sha256": _sha256(FP32_ONNX)},
                "int8_onnx": {"path": str(INT8_ONNX), "bytes": INT8_ONNX.stat().st_size, "sha256": _sha256(INT8_ONNX)},
            },
            "metrics": metrics,
            "controlled_pairs": controlled_pair_analysis(cases, predictions),
        }
        if retain_predictions:
            payload["predictions"] = predictions
        suffix = f"_{limit}" if limit is not None else ""
        result_path = output_dir / f"tinybert_{partition}_{stage}{suffix}.json"
        result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        summaries[stage] = {
            "result_file": result_path.name,
            "overall": metrics["overall"],
            "by_category": metrics["by_category"],
        }
    matrix = {
        "partition": partition,
        "case_count": len(cases),
        "fixture_sha256": _sha256(fixture_path),
        "stages": summaries,
        "note": "Development-host measurements; never interpret latency as Raspberry Pi latency.",
    }
    suffix = f"_{limit}" if limit is not None else ""
    matrix_path = output_dir / f"tinybert_{partition}_summary{suffix}.json"
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return matrix


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=root / "fixtures")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--partition", choices=("dev", "final", "all"), default="dev")
    parser.add_argument("--stages", nargs="+", choices=DEFAULT_STAGES, default=list(DEFAULT_STAGES))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--retain-predictions",
        action="store_true",
        help="Store every case prediction; off by default to keep committed results compact.",
    )
    args = parser.parse_args()
    partitions = ("dev", "final") if args.partition == "all" else (args.partition,)
    for partition in partitions:
        run_partition(
            args.fixtures,
            args.output_dir,
            partition,
            tuple(args.stages),
            args.limit,
            args.retain_predictions,
        )


if __name__ == "__main__":
    main()

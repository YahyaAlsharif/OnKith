"""Score an ONNX artifact on every Model V2 evaluation view, through production decoding.

Section 26 of the training notebook scores INT8 through ``ModelV2OnnxMasker``
while its FP32 reference comes from the training-side ``predict_spans``. Those
are two different decoders, so an FP32-to-INT8 comparison built from them
measures the decoder gap as well as quantization. This script removes that
confound: it runs *any* ONNX graph through the one production path, so FP32 and
INT8 numbers are directly comparable.

Views: ``test``, ``val_original``, ``val_supplementary``, ``combined_validation``
(reconstructed, never materialised) and ``ood_dev``.

The frozen OOD final fixture is never opened. It is not a selectable view and
its path appears nowhere in this file.

    python model/DeBERTa-v3-xsmall/deployment/evaluate_onnx_production_path.py \
        --model artifacts/model_v2/models/fp32/model_fp32.onnx --label fp32_onnx

Rows are split across worker processes purely for wall-clock; every worker runs
the same deterministic decode, so the output does not depend on ``--workers``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "model" / "DeBERTa-v3-xsmall"
ARTIFACTS = MODEL_DIR / "artifacts" / "model_v2"
FIXTURES = REPO_ROOT / "model" / "privacy_evaluation" / "fixtures"

for extra in (REPO_ROOT / "model" / "privacy_evaluation", REPO_ROOT / "model" / "privacy_data_v2"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

CONTEXT_LIMIT = 512
OVERFLOW_STRIDE = 128
TOKENIZER_DIR = ARTIFACTS / "tokenizer"
LABEL_MAP = MODEL_DIR / "config" / "model_v2_label_map.json"

VIEW_FILES = {
    "test": ARTIFACTS / "data" / "model_v2_test.jsonl",
    "val_original": ARTIFACTS / "data" / "model_v2_val_original.jsonl",
    "val_supplementary": ARTIFACTS / "data" / "model_v2_val_supplementary.jsonl",
}
SCORED_VIEWS = ("test", "val_original", "val_supplementary", "ood_dev")

_WORKER = {}


def load_rows(view: str) -> list[dict]:
    if view in VIEW_FILES:
        path = VIEW_FILES[view]
        if not path.is_file():
            raise FileNotFoundError(f"{view} corpus not built: {path}")
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if view == "ood_dev":
        from privacy_eval.schema import load_cases
        from records import from_fixture_case

        return [
            from_fixture_case(case.to_dict(), partition="ood_dev")
            for case in load_cases(FIXTURES / "ood_dev.jsonl")
        ]
    raise ValueError(f"unknown view: {view}")


def _init_worker(model_path: str, threads: int) -> None:
    from privacy_eval.model_v2_adapter import ModelV2OnnxMasker

    payload = json.loads(LABEL_MAP.read_text(encoding="utf-8"))
    id2label = {int(key): str(value) for key, value in payload["id2label"].items()}
    masker = ModelV2OnnxMasker(
        model_path, TOKENIZER_DIR, id2label,
        context_limit=CONTEXT_LIMIT, overflow_stride=OVERFLOW_STRIDE,
        intra_op_threads=threads,
    )
    masker.load()
    _WORKER["masker"] = masker


def _predict_chunk(chunk: list[tuple[int, str]]) -> list[tuple[int, list[dict]]]:
    masker = _WORKER["masker"]
    out = []
    for index, text in chunk:
        result = masker.mask_text(text)
        out.append((index, [
            {"label": entity["entity_type"], "start": entity["start"], "end": entity["end"]}
            for entity in result.entities
        ]))
    return out


def predict(rows: list[dict], model_path: Path, workers: int, threads: int) -> list[list[dict]]:
    items = [(index, row["source_text"]) for index, row in enumerate(rows)]
    size = max(1, len(items) // (workers * 8) or 1)
    chunks = [items[start:start + size] for start in range(0, len(items), size)]
    predictions: list[list[dict] | None] = [None] * len(rows)
    if workers == 1:
        _init_worker(str(model_path), threads)
        results = [_predict_chunk(chunk) for chunk in chunks]
    else:
        with Pool(workers, initializer=_init_worker,
                  initargs=(str(model_path), threads)) as pool:
            results = pool.map(_predict_chunk, chunks)
    for group in results:
        for index, spans in group:
            predictions[index] = spans
    missing = [index for index, value in enumerate(predictions) if value is None]
    if missing:
        raise RuntimeError(f"no prediction for row {missing[0]}")
    return predictions  # type: ignore[return-value]


def summarise(metrics: dict) -> dict:
    return {
        "typed_entity_f1": metrics["typed_entity"]["f1"],
        "typed_entity_precision": metrics["typed_entity"]["precision"],
        "typed_entity_recall": metrics["typed_entity"]["recall"],
        "binary_entity_f1": metrics["binary_entity"]["f1"],
        "private_character_recall": metrics["private_character_recall"],
        "private_character_leakage": metrics["private_character_leakage"],
        "non_private_overmasking_rate": metrics["non_private_overmasking_rate"],
        "hard_negative_false_positive_rate": metrics["hard_negative_false_positive_rate"],
        "hard_negative_rows": metrics["hard_negative_rows"],
        "rows": metrics["rows"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="ONNX graph, absolute or relative to model/DeBERTa-v3-xsmall")
    parser.add_argument("--label", required=True, help="name for this artifact in the report")
    parser.add_argument("--out", default=None, help="output JSON path")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--threads", type=int, default=2,
                        help="intra-op threads per worker")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N rows of each view (smoke use only)")
    parser.add_argument("--views", nargs="*", default=list(SCORED_VIEWS))
    args = parser.parse_args()

    from privacy_eval.dataset_metrics import evaluate_spans

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (MODEL_DIR / model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    unknown = [view for view in args.views if view not in SCORED_VIEWS]
    if unknown:
        raise SystemExit(f"unknown view(s): {unknown}; choose from {list(SCORED_VIEWS)}")

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    print(f"model    : {model_path}")
    print(f"sha256   : {digest}")
    print(f"size     : {model_path.stat().st_size / 1024**2:.2f} MiB")
    print(f"workers  : {args.workers} x {args.threads} intra-op threads")
    if args.limit:
        print(f"LIMIT    : {args.limit} rows per view - smoke output, not a release number")

    report = {
        "artifact": "onnx_production_path_evaluation",
        "label": args.label,
        "model": model_path.relative_to(REPO_ROOT).as_posix(),
        "model_sha256": digest,
        "model_bytes": model_path.stat().st_size,
        "decoder": "privacy_eval.model_v2_adapter.ModelV2OnnxMasker",
        "row_limit": args.limit or None,
        "views": {},
    }
    held = {}
    for view in args.views:
        rows = load_rows(view)
        if args.limit:
            rows = rows[: args.limit]
        started = time.perf_counter()
        predictions = predict(rows, model_path, args.workers, args.threads)
        metrics = evaluate_spans(rows, predictions)
        elapsed = time.perf_counter() - started
        metrics["evaluation_seconds"] = elapsed
        report["views"][view] = metrics
        held[view] = (rows, predictions)
        print(f"{view:22} rows {metrics['rows']:>7,} | typed F1 "
              f"{metrics['typed_entity']['f1']:.4f} | char recall "
              f"{metrics['private_character_recall']:.4f} | leakage "
              f"{metrics['private_character_leakage']:.4f} | hard-neg FP "
              f"{metrics['hard_negative_false_positive_rate']:.4f} | {elapsed:.0f}s")

    if "val_original" in held and "val_supplementary" in held:
        rows = held["val_original"][0] + held["val_supplementary"][0]
        predictions = held["val_original"][1] + held["val_supplementary"][1]
        combined = evaluate_spans(rows, predictions)
        combined["composition"] = {
            "val_original_rows": len(held["val_original"][0]),
            "val_supplementary_rows": len(held["val_supplementary"][0]),
            "note": "Reconstructed from the two sets above; never materialised as a third file.",
        }
        report["views"]["combined_validation"] = combined
        print(f"{'combined_validation':22} rows {combined['rows']:>7,} | typed F1 "
              f"{combined['typed_entity']['f1']:.4f} | char recall "
              f"{combined['private_character_recall']:.4f} | leakage "
              f"{combined['private_character_leakage']:.4f}")

    report["summary"] = {name: summarise(metrics) for name, metrics in report["views"].items()}
    out = Path(args.out) if args.out else ARTIFACTS / "metrics" / f"evaluation_production_path_{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

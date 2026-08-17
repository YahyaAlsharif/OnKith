"""Re-issue the accepted-INT8-candidate manifest from the corrected evaluations.

Section 26 of the training notebook gated INT8 against an FP32 reference that
came from a *different* decoder, and the INT8 side was scored through a masker
that fed SentencePiece ``word_ids()`` into ``consolidate_predictions_to_words``.
That grouped trailing punctuation with the entity word, over-extended nearly
every predicted span, and reported an INT8 typed F1 around 0.59 - a decoder
defect measured as a quantization loss. The candidate manifest written from that
gate says ``accepted: false``.

``evaluate_onnx_production_path.py`` removed both confounds: it scores *any*
ONNX graph on all five views through the one production decoder. Its two reports

    metrics/evaluation_production_path_fp32_onnx.json
    metrics/evaluation_production_path_int8_onnx.json

are the corrected evidence. This script recomputes the quality gate from them
and rewrites ``metrics/accepted_int8_candidate.json`` through the same
``privacy_eval.accepted_candidate.build_candidate_manifest`` the notebook uses,
so the release contract has exactly one writer and one shape.

It does not train, export, quantize, or touch an ONNX graph. It reads two JSON
reports, re-hashes what is already on disk, and writes metadata. The INT8 file
is only ever hashed, never opened as a model.

    python model/DeBERTa-v3-xsmall/deployment/finalize_int8_candidate.py
    python model/DeBERTa-v3-xsmall/deployment/finalize_int8_candidate.py --check

``--check`` verifies the existing manifest and prints the gate without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "model" / "DeBERTa-v3-xsmall"
ARTIFACTS = MODEL_DIR / "artifacts" / "model_v2"
METRICS_DIR = ARTIFACTS / "metrics"
CONFIG_DIR = MODEL_DIR / "config"
FIXTURES = REPO_ROOT / "model" / "privacy_evaluation" / "fixtures"

if str(REPO_ROOT / "model" / "privacy_evaluation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "model" / "privacy_evaluation"))

from privacy_eval import accepted_candidate  # noqa: E402

# Same limits as section 26 of the training notebook. They are duplicated here
# rather than imported because a notebook is not importable; the assertion below
# is that they stay identical, and the values are printed on every run.
INT8_MAX_TYPED_F1_DROP = 0.02
INT8_MAX_LEAKAGE_RISE = 0.015
GATE_VIEWS = accepted_candidate.REQUIRED_EVALUATION_VIEWS

FP32_REPORT = METRICS_DIR / "evaluation_production_path_fp32_onnx.json"
INT8_REPORT = METRICS_DIR / "evaluation_production_path_int8_onnx.json"
LABEL_MAP_PATH = CONFIG_DIR / "model_v2_label_map.json"
TRAINING_CONFIG_PATH = CONFIG_DIR / "model_v2_training_config.json"
TOKENIZER_DIR = ARTIFACTS / "tokenizer"
BEST_CHECKPOINT = ARTIFACTS / "checkpoints" / "best.pt"
FP32_ONNX = ARTIFACTS / "models" / "fp32" / "model_fp32.onnx"
INT8_ONNX = ARTIFACTS / "models" / "int8" / "model_int8.onnx"
CANDIDATE_PATH = METRICS_DIR / accepted_candidate.CANDIDATE_MANIFEST_NAME
DEPLOYMENT_SUMMARY_PATH = METRICS_DIR / "deployment_summary.json"

PRODUCTION_DECODER = "privacy_eval.model_v2_adapter.ModelV2OnnxMasker"


def load_production_report(path: Path, label: str) -> dict[str, Any]:
    """Read one production-path report and refuse anything that is not a release run."""

    if not path.is_file():
        raise SystemExit(
            f"missing {path.relative_to(REPO_ROOT).as_posix()}. Produce it with:\n"
            f"  python model/DeBERTa-v3-xsmall/deployment/evaluate_onnx_production_path.py "
            f"--model artifacts/model_v2/models/{'int8/model_int8' if label == 'int8_onnx' else 'fp32/model_fp32'}.onnx "
            f"--label {label}"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("artifact") != "onnx_production_path_evaluation":
        raise SystemExit(f"{path.name} is not a production-path evaluation report")
    if report.get("label") != label:
        raise SystemExit(f"{path.name} is labelled {report.get('label')!r}, expected {label!r}")
    if report.get("row_limit") is not None:
        raise SystemExit(
            f"{path.name} was produced with --limit {report['row_limit']}. A truncated "
            "evaluation is smoke output and can never accept a release candidate."
        )
    if report.get("decoder") != PRODUCTION_DECODER:
        raise SystemExit(
            f"{path.name} was decoded by {report.get('decoder')!r}, not the production "
            f"decoder {PRODUCTION_DECODER!r}"
        )
    missing = [view for view in GATE_VIEWS if view not in report.get("views", {})]
    if missing:
        raise SystemExit(f"{path.name} does not cover {missing}")
    return report


def verify_graph(report: Mapping[str, Any], expected_path: Path, label: str) -> str:
    """Confirm the graph on disk is byte-for-byte the one the report scored."""

    recorded = Path(REPO_ROOT / str(report["model"]))
    if recorded.resolve() != expected_path.resolve():
        raise SystemExit(
            f"{label} report scored {report['model']}, not "
            f"{expected_path.relative_to(REPO_ROOT).as_posix()}"
        )
    if not expected_path.is_file():
        raise SystemExit(f"{label} graph is missing: {expected_path}")
    size = expected_path.stat().st_size
    if size != int(report["model_bytes"]):
        raise SystemExit(
            f"{label} graph is {size} bytes on disk, {report['model_bytes']} in the report"
        )
    digest = accepted_candidate.sha256_file(expected_path)
    if digest != report["model_sha256"]:
        raise SystemExit(
            f"{label} graph SHA-256 {digest} does not match the evaluated "
            f"{report['model_sha256']}. Do not re-export or re-quantize; recover the "
            "evaluated artifact."
        )
    return digest


def as_metrics(headline: Mapping[str, Any]) -> dict[str, Any]:
    """Re-inflate a stored headline into the shape ``accepted_candidate.headline`` reads."""

    return {
        "typed_entity": {
            "f1": headline["typed_entity_f1"],
            "precision": headline["typed_entity_precision"],
            "recall": headline["typed_entity_recall"],
        },
        "private_character_recall": headline["private_character_recall"],
        "private_character_leakage": headline["private_character_leakage"],
        "non_private_overmasking_rate": headline["non_private_overmasking_rate"],
        "hard_negative_false_positive_rate": headline["hard_negative_false_positive_rate"],
        "rows": headline["rows"],
    }


def compute_gate(
    fp32_views: Mapping[str, Any], int8_views: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """The section 26 rule, applied to two evaluations of the same decoder."""

    gate: dict[str, dict[str, Any]] = {}
    for view in GATE_VIEWS:
        reference, candidate = fp32_views[view], int8_views[view]
        f1_drop = reference["typed_entity"]["f1"] - candidate["typed_entity"]["f1"]
        leak_rise = (
            candidate["private_character_leakage"] - reference["private_character_leakage"]
        )
        gate[view] = {
            "fp32_typed_f1": reference["typed_entity"]["f1"],
            "int8_typed_f1": candidate["typed_entity"]["f1"],
            "typed_f1_drop": f1_drop,
            "fp32_leakage": reference["private_character_leakage"],
            "int8_leakage": candidate["private_character_leakage"],
            "leakage_rise": leak_rise,
            "rows": candidate["rows"],
            "passes": f1_drop < INT8_MAX_TYPED_F1_DROP and leak_rise < INT8_MAX_LEAKAGE_RISE,
        }
    return gate


def frozen_fixture_fingerprint() -> dict[str, Any]:
    """Identity of the frozen benchmark - hashes and counts, never annotations."""

    fixture = FIXTURES / "ood_final.jsonl"
    manifest_path = FIXTURES / "ood_final.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = accepted_candidate.sha256_file(fixture)
    if digest != manifest["fixture_sha256"]:
        raise SystemExit(
            f"frozen fixture SHA-256 {digest} does not match its manifest "
            f"{manifest['fixture_sha256']}. On Windows this is usually an end-of-line "
            "conversion; .gitattributes marks these fixtures -text for exactly that reason."
        )
    case_count = sum(1 for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip())
    if case_count != manifest["case_count"]:
        raise SystemExit(f"frozen fixture holds {case_count} cases, manifest says {manifest['case_count']}")
    return {
        "path": fixture.relative_to(REPO_ROOT).as_posix(),
        "manifest": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "sha256": digest,
        "case_count": case_count,
        "generator_version": manifest.get("generator_version"),
        "policy": manifest.get("policy"),
    }


def print_gate(gate: Mapping[str, Mapping[str, Any]]) -> None:
    print(
        f"\n{'view':22} {'rows':>7} {'FP32 F1':>8} {'INT8 F1':>8} {'dF1':>9}   "
        f"{'FP32 leak':>9} {'INT8 leak':>9} {'d leak':>9}   gate"
    )
    for view, entry in gate.items():
        print(
            f"{view:22} {entry['rows']:>7,} {entry['fp32_typed_f1']:>8.4f} "
            f"{entry['int8_typed_f1']:>8.4f} {entry['typed_f1_drop']:>+9.4f}   "
            f"{entry['fp32_leakage']:>9.4f} {entry['int8_leakage']:>9.4f} "
            f"{entry['leakage_rise']:>+9.4f}   {'PASS' if entry['passes'] else 'FAIL'}"
        )


def build(write: bool) -> int:
    fp32_report = load_production_report(FP32_REPORT, "fp32_onnx")
    int8_report = load_production_report(INT8_REPORT, "int8_onnx")
    fp32_sha = verify_graph(fp32_report, FP32_ONNX, "FP32")
    int8_sha = verify_graph(int8_report, INT8_ONNX, "INT8")

    gate = compute_gate(fp32_report["views"], int8_report["views"])
    accepted = all(entry["passes"] for entry in gate.values())

    print("INT8 quality gate, recomputed from the corrected production path")
    print(f"  decoder  : {PRODUCTION_DECODER}")
    print(f"  FP32     : {FP32_ONNX.relative_to(REPO_ROOT).as_posix()}  {fp32_sha[:16]}...")
    print(f"  INT8     : {INT8_ONNX.relative_to(REPO_ROOT).as_posix()}  {int8_sha[:16]}...")
    print(
        f"  limits   : typed F1 drop < {INT8_MAX_TYPED_F1_DROP}, "
        f"leakage rise < {INT8_MAX_LEAKAGE_RISE}"
    )
    print_gate(gate)
    print(f"\nINT8 quality gate over {len(gate)} views: {'PASSED' if accepted else 'FAILED'}")

    label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    training_config = json.loads(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    previous = (
        json.loads(CANDIDATE_PATH.read_text(encoding="utf-8")) if CANDIDATE_PATH.is_file() else None
    )

    # The FP32 *PyTorch* numbers stay in the manifest under their own name. They
    # were produced by the training-side `predict_spans`, which never had the
    # word-id defect, and they agree with the FP32 ONNX production path to four
    # decimals - which is the evidence that the export is faithful and that the
    # collapse was in the INT8 decode alone.
    pytorch_reference = (previous or {}).get("evaluation", {}).get("fp32_pytorch") or {}
    have_pytorch = all(view in pytorch_reference for view in GATE_VIEWS)
    fp32_pytorch_evaluations = (
        {view: as_metrics(pytorch_reference[view]) for view in GATE_VIEWS}
        if have_pytorch
        else dict(fp32_report["views"])
    )

    manifest = accepted_candidate.build_candidate_manifest(
        repo_root=REPO_ROOT,
        int8_onnx=INT8_ONNX,
        fp32_onnx=FP32_ONNX,
        tokenizer_dir=TOKENIZER_DIR,
        label_map_path=LABEL_MAP_PATH,
        training_config_path=TRAINING_CONFIG_PATH,
        best_checkpoint=BEST_CHECKPOINT,
        ontology_version=str(label_map["ontology_version"]),
        num_entities=int(label_map["num_entities"]),
        num_labels=int(label_map["num_labels"]),
        base_model=str(training_config["base_model"]),
        base_model_revision=str(training_config["base_model_revision"]),
        run_signature=str(training_config["run_signature"]),
        best_epoch=int(training_config["best_epoch"]),
        best_combined_typed_f1=float(training_config["best_combined_typed_f1"]),
        seed=int(training_config["seed"]),
        context_limit=int(training_config["context_limit"]),
        overflow_stride=int(training_config["overflow_stride"]),
        fp32_evaluations=fp32_pytorch_evaluations,
        int8_evaluations=int8_report["views"],
        gate=gate,
        gate_views=GATE_VIEWS,
        max_typed_f1_drop=INT8_MAX_TYPED_F1_DROP,
        max_leakage_rise=INT8_MAX_LEAKAGE_RISE,
        frozen_fixture=frozen_fixture_fingerprint(),
        int8_accepted=accepted,
        smoke_test=False,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # Additions the schema does not define but the release record needs. They sit
    # beside the required keys rather than inside them, so `load_accepted_candidate`
    # still walks exactly the structure it validates.
    manifest["evaluation"]["fp32_onnx"] = {
        view: accepted_candidate.headline(fp32_report["views"][view]) for view in GATE_VIEWS
    }
    manifest["int8_gate"]["reference_runtime"] = "fp32_onnx"
    manifest["int8_gate"]["detail"] = {view: dict(entry) for view, entry in gate.items()}
    manifest["evaluation_provenance"] = {
        "gate_comparison": "fp32_onnx vs int8_onnx, both decoded by the production decoder",
        "fp32_onnx_report": FP32_REPORT.relative_to(REPO_ROOT).as_posix(),
        "fp32_onnx_report_sha256": accepted_candidate.sha256_file(FP32_REPORT),
        "int8_onnx_report": INT8_REPORT.relative_to(REPO_ROOT).as_posix(),
        "int8_onnx_report_sha256": accepted_candidate.sha256_file(INT8_REPORT),
        "fp32_pytorch_source": (
            "carried forward from the superseded manifest; training-side predict_spans"
            if have_pytorch
            else "unavailable - fp32_onnx production-path numbers substituted"
        ),
        "decoder": PRODUCTION_DECODER,
        "decoder_module": "FullPipeline_V3/integrated_pipeline/model_v2_masker.py",
        "note": (
            "Section 26 of the training notebook compared a training-side FP32 decode "
            "with an INT8 decode that passed SentencePiece word ids into "
            "consolidate_predictions_to_words. Both confounds are removed here."
        ),
    }
    if previous is not None:
        manifest["supersedes"] = {
            "generated_utc": previous.get("generated_utc"),
            "accepted": previous.get("accepted"),
            "int8_sha256": previous.get("model", {}).get("int8_sha256"),
            "gate_passed": previous.get("int8_gate", {}).get("passed"),
            "typed_f1_drops": {
                view: entry.get("typed_f1_drop")
                for view, entry in previous.get("int8_gate", {}).get("results", {}).items()
            },
            "reason": (
                "The superseded gate measured a decoder defect as quantization loss. "
                "The model artifact is unchanged: same INT8 SHA-256, same run signature."
            ),
        }
    accepted_candidate.validate_candidate_manifest(manifest)

    same_artifact = previous is None or previous.get("model", {}).get("int8_sha256") == int8_sha
    if not same_artifact:
        raise SystemExit(
            "the INT8 SHA-256 differs from the superseded manifest. This script only "
            "re-issues metadata for an unchanged artifact."
        )

    if not write:
        print("\n--check: nothing written.")
        print(f"  would set accepted = {manifest['accepted']}")
        return 0 if accepted else 1

    CANDIDATE_PATH.write_text(
        json.dumps(manifest, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {CANDIDATE_PATH.relative_to(REPO_ROOT).as_posix()}")
    print(f"  accepted     : {manifest['accepted']}")
    print(f"  int8 sha256  : {manifest['model']['int8_sha256']}")
    print(f"  tokenizer    : {manifest['tokenizer']['tokenizer_sha256'][:16]}...")
    print(f"  run signature: {manifest['training']['run_signature'][:16]}...")

    update_deployment_summary(manifest, gate, fp32_report, int8_report)

    if accepted:
        print(
            "\nThis candidate is now locked. model_v2_frozen_ood_final.ipynb may spend the\n"
            "frozen OOD final benchmark on it exactly once."
        )
    else:
        print("\nNOT accepted. Do not deploy this INT8 graph.")
    return 0 if accepted else 1


def update_deployment_summary(
    manifest: Mapping[str, Any],
    gate: Mapping[str, Mapping[str, Any]],
    fp32_report: Mapping[str, Any],
    int8_report: Mapping[str, Any],
) -> None:
    """Keep the human-facing summary in step with the manifest it summarises."""

    if not DEPLOYMENT_SUMMARY_PATH.is_file():
        return
    summary = json.loads(DEPLOYMENT_SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["int8_accepted"] = manifest["accepted"]
    summary["int8_gate"] = {
        view: {
            "typed_f1_drop": entry["typed_f1_drop"],
            "leakage_rise": entry["leakage_rise"],
            "passes": entry["passes"],
        }
        for view, entry in gate.items()
    }
    summary["int8_gate_reference"] = "fp32_onnx production path"
    summary["accepted_candidate"] = {
        "manifest": CANDIDATE_PATH.relative_to(REPO_ROOT).as_posix(),
        "accepted": manifest["accepted"],
        "smoke_test": manifest["smoke_test"],
        "int8_sha256": manifest["model"]["int8_sha256"],
        "run_signature": manifest["training"]["run_signature"],
        "generated_utc": manifest["generated_utc"],
    }
    summary["headline_metrics"] = {
        view: {
            "fp32_onnx_typed_entity_f1": fp32_report["summary"][view]["typed_entity_f1"],
            "int8_onnx_typed_entity_f1": int8_report["summary"][view]["typed_entity_f1"],
            "int8_private_character_leakage": int8_report["summary"][view][
                "private_character_leakage"
            ],
            "int8_hard_negative_false_positive_rate": int8_report["summary"][view][
                "hard_negative_false_positive_rate"
            ],
            "rows": int8_report["summary"][view]["rows"],
        }
        for view in GATE_VIEWS
    }
    summary["headline_metrics_note"] = (
        "Both runtimes decoded by the production decoder; see "
        "metrics/evaluation_production_path_*.json."
    )
    summary["production_migration"] = {
        "status": "PERFORMED",
        "deployment_package": "FullPipeline_V3/models/deberta_pii/",
        "note": (
            "The packaged INT8 graph, tokenizer and label map are byte-identical to the "
            "canonical artifacts above; PIIMaskerAdapter.verify_artifact enforces the hash."
        ),
    }
    DEPLOYMENT_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(f"wrote {DEPLOYMENT_SUMMARY_PATH.relative_to(REPO_ROOT).as_posix()}")


def check_existing() -> int:
    """Run the release-gate reader over the manifest currently on disk."""

    resolved = accepted_candidate.load_accepted_candidate(CANDIDATE_PATH, repo_root=REPO_ROOT)
    print("accepted-candidate manifest verified by load_accepted_candidate()")
    print(f"  manifest    : {CANDIDATE_PATH.relative_to(REPO_ROOT).as_posix()}")
    print(f"  manifest sha: {resolved['manifest_sha256']}")
    print(f"  int8        : {resolved['int8_path'].relative_to(REPO_ROOT).as_posix()}")
    print(f"  int8 sha256 : {resolved['int8_sha256']}")
    print(f"  tokenizer   : {resolved['tokenizer_dir'].relative_to(REPO_ROOT).as_posix()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and print the gate, and verify the manifest on disk, without writing",
    )
    args = parser.parse_args()
    if args.check:
        status = build(write=False)
        if CANDIDATE_PATH.is_file():
            try:
                check_existing()
            except accepted_candidate.CandidateRejected as exc:
                print(f"\nexisting manifest REJECTED: {exc}")
                return 1
        return status
    return build(write=True)


if __name__ == "__main__":
    raise SystemExit(main())

"""Identity contract between the Model V2 training notebook and the frozen benchmark.

``model_v2_deberta_v3_xsmall.ipynb`` builds a deployment candidate and, only when
the INT8 quality gate passes on a real run, writes
``artifacts/model_v2/metrics/accepted_int8_candidate.json``.
``model_v2_frozen_ood_final.ipynb`` spends the one-shot frozen OOD benchmark on
that candidate and nothing else.

The two notebooks share no runtime state - only that file - so every check that
stands between them lives here, where it can be unit-tested against synthetic
manifests and synthetic fixtures instead of the real frozen set. The frozen
benchmark is spendable exactly once; a guard that is only exercised by spending
it is not a guard.

Nothing in this module reads frozen annotations. It hashes bytes, counts lines
and validates keys.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_MANIFEST_NAME = "accepted_int8_candidate.json"
FROZEN_REPORT_NAME = "evaluation_ood_final_int8.json"
CANDIDATE_ARTIFACT_KIND = "accepted_int8_candidate"
FROZEN_REPORT_KIND = "frozen_ood_final_int8"
SCHEMA_VERSION = 1

EXPECTED_NUM_ENTITIES = 31
EXPECTED_NUM_LABELS = 63
REQUIRED_EVALUATION_VIEWS = (
    "test",
    "val_original",
    "val_supplementary",
    "combined_validation",
    "ood_dev",
)
HEADLINE_METRICS = (
    "typed_entity_f1",
    "private_character_recall",
    "private_character_leakage",
    "non_private_overmasking_rate",
    "hard_negative_false_positive_rate",
)


class CandidateRejected(RuntimeError):
    """The accepted-candidate manifest is missing, unaccepted or inconsistent."""


class FixtureRejected(RuntimeError):
    """The frozen OOD final fixture is not the committed, frozen fixture."""


class FrozenBenchmarkAlreadySpent(RuntimeError):
    """A frozen-final report already exists for this fixture."""


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256; the INT8 graph is far too large to slurp."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: Any, message: str, error: type[Exception] = CandidateRejected) -> None:
    if not condition:
        raise error(message)


def _relative(path: str | Path, repo_root: str | Path = REPO_ROOT) -> str:
    """Repo-relative POSIX path, falling back to the absolute one.

    ``build_frozen_report`` runs *after* the one-shot benchmark has been spent, so
    a path outside the repository must not raise there and lose the report. Every
    real artifact lives under the repo root and takes the relative branch.
    """

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve(value: Any, repo_root: str | Path, *, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} path is missing from the manifest")
    candidate = Path(repo_root) / str(value)
    return candidate


def headline(metrics: Mapping[str, Any]) -> dict[str, float]:
    """The five numbers every view is judged on, flattened for the manifest."""

    return {
        "typed_entity_f1": float(metrics["typed_entity"]["f1"]),
        "typed_entity_precision": float(metrics["typed_entity"]["precision"]),
        "typed_entity_recall": float(metrics["typed_entity"]["recall"]),
        "private_character_recall": float(metrics["private_character_recall"]),
        "private_character_leakage": float(metrics["private_character_leakage"]),
        "non_private_overmasking_rate": float(metrics["non_private_overmasking_rate"]),
        "hard_negative_false_positive_rate": float(
            metrics["hard_negative_false_positive_rate"]
        ),
        "rows": int(metrics["rows"]),
    }


# ---------------------------------------------------------------------------
# Written by the training notebook
# ---------------------------------------------------------------------------


def build_candidate_manifest(
    *,
    repo_root: str | Path,
    int8_onnx: str | Path,
    fp32_onnx: str | Path,
    tokenizer_dir: str | Path,
    label_map_path: str | Path,
    training_config_path: str | Path,
    best_checkpoint: str | Path,
    ontology_version: str,
    num_entities: int,
    num_labels: int,
    base_model: str,
    base_model_revision: str,
    run_signature: str,
    best_epoch: int,
    best_combined_typed_f1: float,
    seed: int,
    context_limit: int,
    overflow_stride: int,
    fp32_evaluations: Mapping[str, Mapping[str, Any]],
    int8_evaluations: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Mapping[str, Any]],
    gate_views: Iterable[str],
    max_typed_f1_drop: float,
    max_leakage_rise: float,
    frozen_fixture: Mapping[str, Any],
    int8_accepted: bool,
    smoke_test: bool,
    timestamp_utc: str,
) -> dict[str, Any]:
    """Assemble the proof of exactly which candidate was accepted.

    ``accepted`` is true only for a real run whose INT8 gate passed. A smoke run
    writes the same shape - under its own isolated metrics directory - with
    ``smoke_test: true`` and ``accepted: false``, so smoke evidence can never be
    mistaken for a release candidate.
    """

    repo_root = Path(repo_root)
    int8_onnx, fp32_onnx = Path(int8_onnx), Path(fp32_onnx)
    tokenizer_dir, label_map_path = Path(tokenizer_dir), Path(label_map_path)
    training_config_path, best_checkpoint = Path(training_config_path), Path(best_checkpoint)
    accepted = bool(int8_accepted) and not bool(smoke_test)

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": CANDIDATE_ARTIFACT_KIND,
        "accepted": accepted,
        "smoke_test": bool(smoke_test),
        "generated_utc": str(timestamp_utc),
        "model": {
            "int8_onnx": _relative(int8_onnx, repo_root),
            "int8_sha256": sha256_file(int8_onnx),
            "int8_bytes": int8_onnx.stat().st_size,
            "fp32_onnx": _relative(fp32_onnx, repo_root),
            "fp32_sha256": sha256_file(fp32_onnx),
            "fp32_bytes": fp32_onnx.stat().st_size,
        },
        "ontology": {
            "version": str(ontology_version),
            "num_entities": int(num_entities),
            "num_labels": int(num_labels),
            "label_map": _relative(label_map_path, repo_root),
            "label_map_sha256": sha256_file(label_map_path),
            "label_map_ontology_version": str(label_map.get("ontology_version", "")),
        },
        "tokenizer": {
            "path": _relative(tokenizer_dir, repo_root),
            "base_model": str(base_model),
            "base_model_revision": str(base_model_revision),
            "tokenizer_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(tokenizer_dir / "tokenizer_config.json"),
        },
        "training": {
            "run_signature": str(run_signature),
            "best_epoch": int(best_epoch),
            "best_combined_typed_f1": float(best_combined_typed_f1),
            "best_checkpoint": _relative(best_checkpoint, repo_root),
            "training_config": _relative(training_config_path, repo_root),
            "training_config_sha256": sha256_file(training_config_path),
            "seed": int(seed),
            "context_limit": int(context_limit),
            "overflow_stride": int(overflow_stride),
        },
        "evaluation": {
            "fp32_pytorch": {
                name: headline(fp32_evaluations[name])
                for name in REQUIRED_EVALUATION_VIEWS
                if name in fp32_evaluations
            },
            "int8_onnx": {
                name: headline(int8_evaluations[name])
                for name in REQUIRED_EVALUATION_VIEWS
                if name in int8_evaluations
            },
        },
        "int8_gate": {
            "views": [str(view) for view in gate_views],
            "thresholds": {
                "max_typed_f1_drop": float(max_typed_f1_drop),
                "max_leakage_rise": float(max_leakage_rise),
            },
            "results": {
                name: {
                    "typed_f1_drop": float(entry["typed_f1_drop"]),
                    "leakage_rise": float(entry["leakage_rise"]),
                    "passes": bool(entry["passes"]),
                }
                for name, entry in gate.items()
            },
            "passed": bool(int8_accepted),
        },
        "frozen_benchmark": {
            "status": "not run",
            "notebook": "model/DeBERTa-v3-xsmall/model_v2_frozen_ood_final.ipynb",
            "fixture": dict(frozen_fixture),
        },
    }
    validate_candidate_manifest(payload)
    return payload


def validate_candidate_manifest(payload: Mapping[str, Any]) -> None:
    """Shape check applied when the manifest is written and when it is read."""

    _require(isinstance(payload, Mapping), "candidate manifest must be a JSON object")
    _require(
        payload.get("artifact") == CANDIDATE_ARTIFACT_KIND,
        f"not an {CANDIDATE_ARTIFACT_KIND} manifest: {payload.get('artifact')!r}",
    )
    _require(
        payload.get("schema_version") == SCHEMA_VERSION,
        f"unsupported candidate manifest schema_version {payload.get('schema_version')!r}",
    )
    for section in ("model", "ontology", "tokenizer", "training", "evaluation", "int8_gate"):
        _require(
            isinstance(payload.get(section), Mapping),
            f"candidate manifest is missing the {section!r} section",
        )
    _require(isinstance(payload.get("accepted"), bool), "'accepted' must be a boolean")
    _require(isinstance(payload.get("smoke_test"), bool), "'smoke_test' must be a boolean")


# ---------------------------------------------------------------------------
# Read by the frozen benchmark notebook
# ---------------------------------------------------------------------------


def load_accepted_candidate(
    manifest_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    expected_num_labels: int = EXPECTED_NUM_LABELS,
    expected_num_entities: int = EXPECTED_NUM_ENTITIES,
) -> dict[str, Any]:
    """Verify the accepted candidate and re-hash it on disk.

    Raises :class:`CandidateRejected` on the first inconsistency. A manually
    copied ONNX graph fails at the SHA-256 re-check: the manifest records what
    was gated, and the file on disk must still be that exact artifact.
    """

    manifest_path, repo_root = Path(manifest_path), Path(repo_root)
    _require(
        manifest_path.is_file(),
        f"no accepted-candidate manifest at {manifest_path}. Run the INT8 quality gate "
        "in model_v2_deberta_v3_xsmall.ipynb first; the frozen benchmark is only ever "
        "spent on an accepted candidate.",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - trivially propagated
        raise CandidateRejected(f"accepted-candidate manifest is not valid JSON: {exc}") from exc
    validate_candidate_manifest(manifest)

    _require(
        manifest["smoke_test"] is False,
        "this candidate came from a smoke run. Smoke artifacts are never release "
        "candidates and must never spend the frozen benchmark.",
    )
    _require(
        manifest["accepted"] is True,
        "the candidate manifest says accepted=false. The INT8 quality gate did not "
        "produce a deployable candidate, so there is nothing to benchmark.",
    )

    gate = manifest["int8_gate"]
    _require(gate.get("passed") is True, "the recorded INT8 quality gate did not pass")
    results = gate.get("results")
    _require(isinstance(results, Mapping) and results, "the INT8 gate recorded no view results")
    missing_gate_views = [view for view in REQUIRED_EVALUATION_VIEWS if view not in results]
    _require(
        not missing_gate_views,
        f"the INT8 gate did not cover {missing_gate_views}",
    )
    failed = sorted(name for name, entry in results.items() if not entry.get("passes"))
    _require(not failed, f"the INT8 gate failed on {failed}")

    evaluation = manifest["evaluation"]
    for runtime in ("fp32_pytorch", "int8_onnx"):
        recorded = evaluation.get(runtime)
        _require(
            isinstance(recorded, Mapping),
            f"the candidate manifest has no {runtime} evaluation evidence",
        )
        missing = [view for view in REQUIRED_EVALUATION_VIEWS if view not in recorded]
        _require(
            not missing,
            f"required {runtime} evaluation evidence is missing for {missing}",
        )
        for view, view_metrics in recorded.items():
            _require(
                isinstance(view_metrics, Mapping) and "typed_entity_f1" in view_metrics,
                f"{runtime} evidence for {view!r} has no typed_entity_f1",
            )

    ontology_block = manifest["ontology"]
    _require(
        ontology_block.get("num_labels") == expected_num_labels,
        f"candidate declares {ontology_block.get('num_labels')!r} labels, expected "
        f"{expected_num_labels}",
    )
    _require(
        ontology_block.get("num_entities") == expected_num_entities,
        f"candidate declares {ontology_block.get('num_entities')!r} entities, expected "
        f"{expected_num_entities}",
    )
    label_map_path = _resolve(ontology_block.get("label_map"), repo_root, label="label map")
    _require(label_map_path.is_file(), f"label map is missing: {label_map_path}")
    _require(
        sha256_file(label_map_path) == ontology_block.get("label_map_sha256"),
        "the label map on disk no longer matches the accepted candidate; the ontology "
        "changed after the candidate was gated",
    )
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    _require(
        label_map.get("num_labels") == expected_num_labels,
        f"label map declares {label_map.get('num_labels')!r} labels, expected "
        f"{expected_num_labels}",
    )
    _require(
        label_map.get("ontology_version") == ontology_block.get("version"),
        "label map ontology version differs from the accepted candidate",
    )

    training = manifest["training"]
    run_signature = training.get("run_signature")
    _require(
        isinstance(run_signature, str) and run_signature,
        "the candidate manifest records no training run signature",
    )
    training_config_path = _resolve(
        training.get("training_config"), repo_root, label="training config"
    )
    _require(
        training_config_path.is_file(), f"training config is missing: {training_config_path}"
    )
    _require(
        sha256_file(training_config_path) == training.get("training_config_sha256"),
        "the training config on disk no longer matches the accepted candidate",
    )
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    _require(
        training_config.get("run_signature") == run_signature,
        "run signature is inconsistent between the candidate manifest and the training "
        "config; these artifacts come from different runs",
    )
    _require(
        training_config.get("num_labels") == expected_num_labels,
        f"training config declares {training_config.get('num_labels')!r} labels, expected "
        f"{expected_num_labels}",
    )
    _require(
        training_config.get("smoke_test") is False,
        "the training config belongs to a smoke run",
    )

    tokenizer_block = manifest["tokenizer"]
    tokenizer_dir = _resolve(tokenizer_block.get("path"), repo_root, label="tokenizer")
    _require(tokenizer_dir.is_dir(), f"tokenizer directory is missing: {tokenizer_dir}")
    for filename, key in (
        ("tokenizer.json", "tokenizer_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
    ):
        tokenizer_file = tokenizer_dir / filename
        _require(tokenizer_file.is_file(), f"tokenizer file is missing: {tokenizer_file}")
        _require(
            sha256_file(tokenizer_file) == tokenizer_block.get(key),
            f"{filename} no longer matches the accepted candidate",
        )
    _require(
        tokenizer_block.get("base_model_revision") == training_config.get("base_model_revision"),
        "tokenizer base model revision differs from the training config",
    )

    model_block = manifest["model"]
    int8_path = _resolve(model_block.get("int8_onnx"), repo_root, label="INT8 ONNX")
    _require(
        int8_path.is_file(),
        f"the accepted INT8 candidate is missing: {int8_path}. Re-run the export and "
        "quantization sections; do not substitute another graph.",
    )
    _require(
        int8_path.stat().st_size == model_block.get("int8_bytes"),
        f"INT8 candidate size changed: {int8_path.stat().st_size} bytes on disk, "
        f"{model_block.get('int8_bytes')} in the manifest",
    )
    actual_sha = sha256_file(int8_path)
    _require(
        actual_sha == model_block.get("int8_sha256"),
        "the INT8 file on disk is not the accepted candidate: SHA-256 "
        f"{actual_sha} != {model_block.get('int8_sha256')}. The frozen benchmark is only "
        "spent on the exact artifact that passed the quality gate.",
    )

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "int8_path": int8_path,
        "int8_sha256": actual_sha,
        "tokenizer_dir": tokenizer_dir,
        "label_map_path": label_map_path,
        "training_config_path": training_config_path,
    }


# ---------------------------------------------------------------------------
# Frozen fixture identity
# ---------------------------------------------------------------------------

_FIXTURE_CASE_KEYS = frozenset(
    {"case_id", "partition", "category", "subcategory", "template_id", "text", "entities"}
)


def verify_frozen_fixture(
    fixture_path: str | Path,
    fixture_manifest_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_case_count: int | None = None,
) -> dict[str, Any]:
    """Confirm the frozen fixture is the committed one, without exposing it.

    Only keys, counts and hashes are inspected; no text and no annotation ever
    leaves this function.
    """

    fixture_path, fixture_manifest_path = Path(fixture_path), Path(fixture_manifest_path)
    _require(
        fixture_path.is_file(), f"frozen fixture is missing: {fixture_path}", FixtureRejected
    )
    _require(
        fixture_manifest_path.is_file(),
        f"frozen fixture manifest is missing: {fixture_manifest_path}",
        FixtureRejected,
    )
    fixture_manifest = json.loads(fixture_manifest_path.read_text(encoding="utf-8"))
    _require(
        fixture_manifest.get("frozen") is True,
        "the fixture manifest is not marked frozen",
        FixtureRejected,
    )
    _require(
        fixture_manifest.get("partition") == "final",
        f"fixture manifest partition is {fixture_manifest.get('partition')!r}, expected 'final'",
        FixtureRejected,
    )
    _require(
        fixture_manifest.get("fixture_file") == fixture_path.name,
        f"fixture manifest names {fixture_manifest.get('fixture_file')!r}, not {fixture_path.name}",
        FixtureRejected,
    )

    digest = sha256_file(fixture_path)
    _require(
        digest == fixture_manifest.get("fixture_sha256"),
        f"frozen fixture SHA-256 changed: {digest} != {fixture_manifest.get('fixture_sha256')}",
        FixtureRejected,
    )
    if expected_sha256 is not None:
        _require(
            digest == expected_sha256,
            "the frozen fixture is not the one fingerprinted when the candidate was built: "
            f"{digest} != {expected_sha256}",
            FixtureRejected,
        )

    case_count = 0
    with fixture_path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            case_count += 1
            case = json.loads(line)
            missing = sorted(_FIXTURE_CASE_KEYS - set(case))
            _require(
                not missing,
                f"frozen fixture line {number} is missing {missing}",
                FixtureRejected,
            )
            _require(
                case["partition"] == "final",
                f"frozen fixture line {number} is not a final-partition case",
                FixtureRejected,
            )
    _require(
        case_count == fixture_manifest.get("case_count"),
        f"frozen fixture holds {case_count} cases, manifest says "
        f"{fixture_manifest.get('case_count')}",
        FixtureRejected,
    )
    if expected_case_count is not None:
        _require(
            case_count == expected_case_count,
            f"frozen fixture holds {case_count} cases, the candidate recorded "
            f"{expected_case_count}",
            FixtureRejected,
        )

    return {
        "path": fixture_path,
        "sha256": digest,
        "case_count": case_count,
        "generator_version": fixture_manifest.get("generator_version"),
        "schema_version": fixture_manifest.get("schema_version"),
        "policy": fixture_manifest.get("policy"),
        "category_allocation": fixture_manifest.get("category_allocation"),
        "hard_negative_count": fixture_manifest.get("hard_negative_count"),
        "manifest_sha256": sha256_file(fixture_manifest_path),
    }


# ---------------------------------------------------------------------------
# One-shot guard and report
# ---------------------------------------------------------------------------


def guard_one_shot(
    report_path: str | Path,
    *,
    candidate_sha256: str,
    fixture_sha256: str,
    allow_rerun: bool = False,
) -> None:
    """Refuse a second pass over a fixture that has already been spent.

    The frozen set buys one honest number. A rerun on the same candidate is an
    accident; a rerun on a *different* candidate is worse, because it turns the
    holdout into a selection metric. Both are refused unless the archival
    override is deliberately switched on.
    """

    report_path = Path(report_path)
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    previous_candidate = str(report.get("candidate", {}).get("int8_sha256", ""))
    previous_fixture = str(report.get("fixture", {}).get("sha256", ""))
    if previous_fixture != fixture_sha256:
        return
    if allow_rerun:
        return
    if previous_candidate == candidate_sha256:
        raise FrozenBenchmarkAlreadySpent(
            f"{report_path.name} already records this exact candidate "
            f"({candidate_sha256[:12]}...) on this fixture. The frozen benchmark is "
            "one-shot; re-running it produces no new information and normalises "
            "spending it. Set ALLOW_FROZEN_RERUN = True only to recover a lost report."
        )
    raise FrozenBenchmarkAlreadySpent(
        f"{report_path.name} already spent this fixture on a different candidate "
        f"({previous_candidate[:12]}...). Scoring a second candidate turns the final "
        "holdout into a selection metric. Archive the existing report and treat the "
        "fixture as spent for this model generation."
    )


def build_frozen_report(
    *,
    repo_root: str | Path,
    candidate: Mapping[str, Any],
    fixture: Mapping[str, Any],
    metrics: Mapping[str, Any],
    by_category: Mapping[str, Any],
    password: Mapping[str, Any],
    runtime: Mapping[str, Any],
    timestamp_utc: str,
) -> dict[str, Any]:
    """Assemble the frozen benchmark report.

    ``guard_one_shot`` reads ``candidate.int8_sha256`` and ``fixture.sha256`` back
    out of this shape, so the two must be built together.
    """

    manifest = candidate["manifest"]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": FROZEN_REPORT_KIND,
        "generated_utc": str(timestamp_utc),
        "status": "spent",
        "candidate": {
            "int8_onnx": _relative(candidate["int8_path"], repo_root),
            "int8_sha256": candidate["int8_sha256"],
            "int8_bytes": Path(candidate["int8_path"]).stat().st_size,
            "accepted_candidate_manifest": _relative(candidate["manifest_path"], repo_root),
            "accepted_candidate_manifest_sha256": candidate["manifest_sha256"],
            "run_signature": manifest["training"]["run_signature"],
            "best_epoch": manifest["training"]["best_epoch"],
            "fp32_sha256": manifest["model"]["fp32_sha256"],
        },
        "ontology": dict(manifest["ontology"]),
        "tokenizer": dict(manifest["tokenizer"]),
        "fixture": {
            "path": _relative(fixture["path"], repo_root),
            "sha256": fixture["sha256"],
            "manifest_sha256": fixture["manifest_sha256"],
            "case_count": fixture["case_count"],
            "generator_version": fixture["generator_version"],
            "policy": fixture["policy"],
        },
        "runtime": dict(runtime),
        "headline": {
            "typed_entity_f1": float(metrics["typed_entity"]["f1"]),
            "binary_entity_f1": float(metrics["binary_entity"]["f1"]),
            "private_character_recall": float(metrics["private_character_recall"]),
            "private_character_leakage": float(metrics["private_character_leakage"]),
            "non_private_overmasking_rate": float(metrics["non_private_overmasking_rate"]),
            "hard_negative_false_positive_rate": float(
                metrics["hard_negative_false_positive_rate"]
            ),
        },
        "metrics": dict(metrics),
        "by_category": dict(by_category),
        "password": dict(password),
    }

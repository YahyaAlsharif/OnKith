"""The committed reproducibility manifest.

Generated corpora stay local. This manifest is what makes that safe: a teammate
clones the repository, runs the dataset section of the Model V2 notebook, and
regenerates a byte-identical supplementary holdout. Anything needed for that -
pinned revisions, mappings, filters, seeds, split configuration, expected counts
and SHA-256 fingerprints - is recorded here.

``verify_manifest`` re-checks a regenerated build against a committed manifest
and reports precisely what drifted, instead of failing with a bare mismatch.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from .ontology import MODEL_V2_VERSION, label_map_payload
except ImportError:  # Direct notebook/script execution from this directory.
    from ontology import MODEL_V2_VERSION, label_map_payload

MANIFEST_SCHEMA_VERSION = 2

#: Fields whose drift means the regenerated holdout is not the committed one.
CRITICAL_FIELDS = (
    "ontology.num_labels",
    "ontology.entities",
    "sources.base.revision",
    "split.supplementary.seed",
    "split.supplementary.holdout_fraction",
    "split.supplementary.factors",
    "artifacts.val_supplementary.sha256",
    "artifacts.val_supplementary.rows",
)


def _environment() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module in ("datasets", "transformers", "tokenizers", "torch", "onnxruntime", "numpy"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = None
    return versions


def _dig(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def build_manifest(
    *,
    sources: Mapping[str, Any],
    mapping_summary: Mapping[str, Any],
    selection_audit: Mapping[str, Any],
    password_audit: Mapping[str, Any],
    dedup_audit: Mapping[str, Any],
    firewall_audit: Mapping[str, Any],
    supplementary_split_audit: Mapping[str, Any],
    training_split_audit: Mapping[str, Any],
    augmentation_audit: Mapping[str, Any],
    partition_statistics: Mapping[str, Any],
    disjointness: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the manifest from the audits each stage already produced."""

    ontology = label_map_payload()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ontology_version": MODEL_V2_VERSION,
        "ontology": {
            "num_entities": ontology["num_entities"],
            "num_labels": ontology["num_labels"],
            "entities": ontology["entities"],
            "derivation": ontology["derivation"],
        },
        "sources": sources,
        "label_mapping": mapping_summary,
        "selection": selection_audit,
        "password": password_audit,
        "deduplication": dedup_audit,
        "contamination_firewall": firewall_audit,
        "split": {
            "supplementary": supplementary_split_audit,
            "training_side": training_split_audit,
        },
        "augmentation": augmentation_audit,
        "statistics": partition_statistics,
        "disjointness": disjointness,
        "artifacts": artifacts,
        "config": config,
        "environment": _environment(),
        "regeneration": {
            "entrypoint": "model/DeBERTa-v3-xsmall/model_v2_deberta_v3_xsmall.ipynb",
            "sections": "Run sections 1-8 (Setup through Augmentation) to rebuild every corpus.",
            "note": (
                "Corpora are gitignored by design. Re-running the dataset sections "
                "with this manifest's seeds and pinned revisions must reproduce "
                "artifacts.val_supplementary.sha256 exactly."
            ),
            "verify": "privacy_data_v2.manifest.verify_manifest(committed, regenerated)",
        },
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8")
    return {"path": str(path), "bytes": len(rendered.encode("utf-8"))}


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(
    committed: Mapping[str, Any],
    regenerated: Mapping[str, Any],
    *,
    fields: tuple[str, ...] = CRITICAL_FIELDS,
) -> dict[str, Any]:
    """Compare a fresh build against the committed manifest, field by field."""

    differences = []
    for field in fields:
        expected, actual = _dig(committed, field), _dig(regenerated, field)
        if expected != actual:
            differences.append(
                {
                    "field": field,
                    "committed": expected,
                    "regenerated": actual,
                }
            )
    return {
        "checked_fields": list(fields),
        "reproducible": not differences,
        "differences": differences,
        "interpretation": (
            "A difference in a source revision or a seed explains itself. A "
            "difference only in val_supplementary.sha256 means the selection or "
            "split logic changed and the committed holdout is no longer valid."
        ),
    }


def main() -> None:
    """``python manifest.py committed.json regenerated.json`` prints a verdict."""

    if len(sys.argv) != 3:
        raise SystemExit("usage: manifest.py <committed manifest> <regenerated manifest>")
    report = verify_manifest(read_manifest(sys.argv[1]), read_manifest(sys.argv[2]))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["reproducible"] else 1)


if __name__ == "__main__":
    main()

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

MANIFEST_SCHEMA_VERSION = 3

#: The version of the data-building contract itself, independent of the manifest
#: schema. Bump it whenever a change alters which rows end up in the final
#: artifacts, so a cache built by an older pipeline cannot be reused in silence.
#:
#: 1. original Model V2 build
#: 2. per-label masking policy and the 63-label ontology
#: 3. cross-partition contamination filter before the final train/test split,
#:    plus the narrowed ``email_domain`` and ``open_entity_value`` split keys
DATA_PIPELINE_VERSION = 3

#: Fields whose drift means the regenerated holdout is not the committed one.
CRITICAL_FIELDS = (
    "data_pipeline_version",
    "ontology.num_labels",
    "ontology.entities",
    "sources.base.revision",
    "split.supplementary.seed",
    "split.supplementary.holdout_fraction",
    "split.supplementary.factors",
    "contamination_filter.version",
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
    contamination_filter: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest from the audits each stage already produced."""

    ontology = label_map_payload()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "data_pipeline_version": DATA_PIPELINE_VERSION,
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
        "contamination_filter": dict(contamination_filter or {}),
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
            "sections": "Run sections 1-11 (Environment through manifest) to rebuild every corpus.",
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


class StaleCacheError(RuntimeError):
    """A cached corpus was produced by an older data pipeline."""


def cache_pipeline_version(manifest: Mapping[str, Any] | None) -> int:
    """The pipeline version a cached manifest was built by.

    Manifests written before the field existed report 0, which is exactly the
    behaviour we want: unversioned means stale.
    """

    if not manifest:
        return 0
    try:
        return int(manifest.get("data_pipeline_version", 0))
    except (TypeError, ValueError):
        return 0


def cache_is_current(manifest: Mapping[str, Any] | None) -> bool:
    return cache_pipeline_version(manifest) >= DATA_PIPELINE_VERSION


def describe_stale_cache(manifest: Mapping[str, Any] | None, *, path: str | Path = "") -> str:
    """Explain, in the terms of this fix, why a cache must be rebuilt."""

    found = cache_pipeline_version(manifest)
    location = f" ({path})" if path else ""
    return (
        f"Cached Model V2 data{location} was built by data pipeline version "
        f"{found or 'unversioned (pre-3)'}, but this build is version "
        f"{DATA_PIPELINE_VERSION}. Version 3 added the cross-partition "
        "contamination filter and narrowed the email_domain and "
        "open_entity_value split keys, so the cached train/test artifacts and "
        "the supplementary holdout no longer match the code that would read "
        "them. They are being rebuilt; nothing needs to be deleted by hand."
    )


def assert_cache_current(manifest: Mapping[str, Any] | None, *, path: str | Path = "") -> None:
    """Refuse to reuse a corpus built before the current pipeline version."""

    if not cache_is_current(manifest):
        raise StaleCacheError(describe_stale_cache(manifest, path=path))


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

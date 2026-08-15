"""Assemble the Model V2 partitions and prove they do not overlap.

The architecture, in one picture::

    AI4Privacy official English train (163,740)
        |-- pinned historical vectors --> 147,366 train / 16,374 test
        +
    supplementary pool --grouped 80/20--> 80% training-side / 20% val_supplementary
        |                                        |
        v                                        v
    training-side pool                    val_supplementary
        |-- grouped 90/10 on the supplementary part
        v
    train  /  test
        ^
        |  every training-side pool passes through ``filter_contamination``
        |  first, so an upstream duplicate cannot ride the pinned AI4Privacy
        |  split into training material

    val_original (40,909, untouched)  +  val_supplementary
        = Combined Model V2 validation, reconstructed at evaluation time

The combined validation set is deliberately *not* materialised. ``val_original``
is already deterministically addressable from the pinned upstream revision, so
copying 40,909 rows into a third file would only add a drift risk.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

try:
    from .dedup import ContaminationFirewall
    from .grouped_splits import MODEL_V2_SPLIT_FACTORS, split_by_factors
    from .records import text_fingerprint
except ImportError:  # Direct notebook/script execution from this directory.
    from dedup import ContaminationFirewall
    from grouped_splits import MODEL_V2_SPLIT_FACTORS, split_by_factors
    from records import text_fingerprint

#: Share of the supplementary pool held out for validation.
SUPPLEMENTARY_HOLDOUT_FRACTION = 0.20

#: Share of the supplementary training side used as training-side test. The
#: AI4Privacy side uses the pinned historical vectors instead, so the overall
#: test split stays comparable with TinyBERT.
SUPPLEMENTARY_TEST_FRACTION = 0.10

SEED_SUPPLEMENTARY_HOLDOUT = 20260814
SEED_SUPPLEMENTARY_TEST = 20260815

#: Version of the contamination-filter contract. Bumped whenever the filter
#: changes what it removes, so a manifest written by an older build cannot be
#: mistaken for a current one. See ``manifest.DATA_PIPELINE_VERSION``.
CONTAMINATION_FILTER_VERSION = 1

#: The candidate pools swept before the final train/test split, in the order
#: the audit reports them.
CONTAMINATION_CANDIDATE_ORDER = (
    "ai4privacy_train",
    "ai4privacy_test",
    "supplementary_training_side",
)

#: The protected partitions reported in the contamination audit, in the order
#: the audit reports them. A pool the firewall does not hold simply reports 0.
CONTAMINATION_PROTECTED_ORDER = (
    "val_original",
    "val_supplementary",
    "ood_dev",
    "ood_final",
)


def _repartition(records: Iterable[Mapping[str, Any]], partition: str) -> list[dict[str, Any]]:
    output = []
    for record in records:
        updated = dict(record)
        updated["partition"] = partition
        output.append(updated)
    return output


def split_supplementary(
    candidates: Sequence[Mapping[str, Any]],
    *,
    holdout_fraction: float = SUPPLEMENTARY_HOLDOUT_FRACTION,
    seed: int = SEED_SUPPLEMENTARY_HOLDOUT,
    factors: Sequence[str] = MODEL_V2_SPLIT_FACTORS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Grouped, label-aware 80/20 split of the deduplicated supplementary pool."""

    train_indices, holdout_indices, audit = split_by_factors(
        candidates,
        factors=factors,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    training_side = _repartition((candidates[index] for index in train_indices), "candidate")
    validation = _repartition(
        (candidates[index] for index in holdout_indices), "val_supplementary"
    )
    return training_side, validation, audit


def split_training_side(
    ai4privacy_train: Sequence[Mapping[str, Any]],
    ai4privacy_test: Sequence[Mapping[str, Any]],
    supplementary_training_side: Sequence[Mapping[str, Any]],
    *,
    test_fraction: float = SUPPLEMENTARY_TEST_FRACTION,
    seed: int = SEED_SUPPLEMENTARY_TEST,
    factors: Sequence[str] = MODEL_V2_SPLIT_FACTORS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build the final ``train`` and ``test`` splits.

    The AI4Privacy component keeps its pinned historical partition so the
    training-side test remains the exact 16,374 rows TinyBERT was measured on.
    The supplementary component is split by the same grouped mechanism.
    """

    if supplementary_training_side:
        supplementary_train_indices, supplementary_test_indices, audit = split_by_factors(
            supplementary_training_side,
            factors=factors,
            holdout_fraction=test_fraction,
            seed=seed,
        )
    else:
        supplementary_train_indices, supplementary_test_indices = [], []
        audit = {"factors": list(factors), "train_rows": 0, "holdout_rows": 0, "is_disjoint": True}

    train = _repartition(ai4privacy_train, "train") + _repartition(
        (supplementary_training_side[index] for index in supplementary_train_indices), "train"
    )
    test = _repartition(ai4privacy_test, "test") + _repartition(
        (supplementary_training_side[index] for index in supplementary_test_indices), "test"
    )
    audit = dict(audit)
    audit.update(
        ai4privacy_train_rows=len(ai4privacy_train),
        ai4privacy_test_rows=len(ai4privacy_test),
        supplementary_train_rows=len(supplementary_train_indices),
        supplementary_test_rows=len(supplementary_test_indices),
        total_train_rows=len(train),
        total_test_rows=len(test),
        ai4privacy_test_note=(
            "The AI4Privacy test rows are the pinned historical project-validation "
            "vectors, so TinyBERT and DeBERTa are measured on identical rows."
        ),
    )
    return train, test, audit


def build_firewall(
    *,
    val_original: Iterable[Mapping[str, Any]],
    val_supplementary: Iterable[Mapping[str, Any]] = (),
    ood_dev: Iterable[Mapping[str, Any]] = (),
    ood_final: Iterable[Mapping[str, Any]] = (),
) -> ContaminationFirewall:
    """Build the firewall from every evaluation set, before any training write."""

    firewall = ContaminationFirewall()
    firewall.add_protected_records("val_original", val_original)
    firewall.add_protected_records("val_supplementary", val_supplementary)
    firewall.add_protected_records("ood_dev", ood_dev)
    firewall.add_protected_records("ood_final", ood_final)
    return firewall


def filter_contamination(
    firewall: ContaminationFirewall,
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    stage: str,
    max_recorded_removals: int = 200,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Remove every training-side row that a protected set already holds.

    This exists because the firewall's selection-time check cannot see the whole
    problem. Supplementary rows are screened as they are chosen, but the
    AI4Privacy component is a *pinned* split: whatever upstream put in the
    official train split arrives unscreened, duplicates included. The sweep
    closes that gap before the final split is materialised.

    The protected sets always win. ``val_original`` in particular must stay at
    its full 40,909 rows, so a collision is resolved by removing the
    training-side copy and never by trimming the validation set.

    Returns ``(cleaned pools, audit)``. Deterministic: pools are swept in
    ``CONTAMINATION_CANDIDATE_ORDER`` first, then in the caller's order, and
    survivors keep their original order.
    """

    ordered = [name for name in CONTAMINATION_CANDIDATE_ORDER if name in candidates]
    ordered += [name for name in candidates if name not in ordered]

    cleaned: dict[str, list[dict[str, Any]]] = {}
    removals: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, int]] = {}
    by_source: Counter[tuple[str, str, str]] = Counter()

    for name in ordered:
        clean, removed = firewall.filter_records(candidates[name], label=name)
        cleaned[name] = clean
        removals.extend(removed)
        counts: Counter[str] = Counter()
        for entry in removed:
            counts[str(entry["protected_set"])] += 1
            by_source[(name, str(entry["protected_set"]), str(entry["source_dataset"]))] += 1
        matrix[name] = {
            protected: counts.get(protected, 0)
            for protected in CONTAMINATION_PROTECTED_ORDER
        }

    removed_by_pool = {name: len(candidates[name]) - len(cleaned[name]) for name in ordered}
    audit = {
        "version": CONTAMINATION_FILTER_VERSION,
        "stage": stage,
        "collision_type": "exact_normalized_text",
        "normalisation": (
            "NFKC, whitespace collapsed, casefolded, then blake2b-128 - the same "
            "fingerprint the firewall and the deduplicator use."
        ),
        "policy": (
            "The protected validation/OOD row is kept and the colliding "
            "training-side copy is removed. Protected sets are never modified."
        ),
        "candidate_rows_in": {name: len(candidates[name]) for name in ordered},
        "candidate_rows_out": {name: len(cleaned[name]) for name in ordered},
        "removed_rows_by_pool": removed_by_pool,
        "removed_rows_total": sum(removed_by_pool.values()),
        "collisions_by_candidate_and_protected_set": matrix,
        "collisions_by_source_dataset": {
            f"{candidate} -> {protected} [{source}]": count
            for (candidate, protected, source), count in sorted(by_source.items())
        },
        "unique_protected_rows_hit": len(
            {(entry["protected_set"], entry["protected_uid"]) for entry in removals}
        ),
        "removals": removals[:max_recorded_removals],
        "removals_truncated": len(removals) > max_recorded_removals,
    }
    return cleaned, audit


def assert_no_protected_overlap(
    firewall: ContaminationFirewall,
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    """Re-check every pool and return the (necessarily zero) overlap counts."""

    remaining = {}
    for name, records in pools.items():
        firewall.assert_clean(records, label=name)
        remaining[name] = 0
    return remaining


def render_contamination_audit(*audits: Mapping[str, Any]) -> str:
    """The concise pre-training summary printed by the notebook."""

    lines = ["cross-partition contamination audit"]
    for audit in audits:
        lines.append(f"  stage: {audit['stage']} (filter v{audit['version']}, {audit['collision_type']})")
        for candidate, counts in audit["collisions_by_candidate_and_protected_set"].items():
            rendered = "  ".join(f"{name}: {counts[name]}" for name in counts)
            lines.append(f"    {candidate:28} -> {rendered}")
        for name, removed in audit["removed_rows_by_pool"].items():
            kept = audit["candidate_rows_out"][name]
            lines.append(f"    removed from {name:22} {removed:>6,}  (kept {kept:,})")
        lines.append(f"    removed at this stage        {audit['removed_rows_total']:>6,}")
    return "\n".join(lines)


def assert_partitions_disjoint(partitions: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Prove that no normalised text appears in two partitions.

    Raises ``AssertionError`` naming both partitions and the offending row.
    Returns a per-partition fingerprint summary for the manifest.
    """

    owner: dict[str, str] = {}
    summary: dict[str, dict[str, int]] = {}
    for name, records in partitions.items():
        unique: set[str] = set()
        for record in records:
            fingerprint = text_fingerprint(str(record["source_text"]))
            unique.add(fingerprint)
            previous = owner.get(fingerprint)
            if previous is not None and previous != name:
                raise AssertionError(
                    f"Partition overlap: row {record.get('uid')!r} appears in both "
                    f"{previous!r} and {name!r}."
                )
            owner.setdefault(fingerprint, name)
        summary[name] = {
            "rows": len(records),
            "unique_texts": len(unique),
            "internal_duplicate_rows": len(records) - len(unique),
        }
    return {
        "partitions": dict(sorted(summary.items())),
        "distinct_texts_overall": len(owner),
        "is_disjoint": True,
    }


def partition_statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-partition statistics used for the committed data report."""

    label_spans: Counter[str] = Counter()
    label_rows: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    generations: Counter[str] = Counter()
    families: Counter[str] = Counter()
    unique_values: dict[str, set[str]] = defaultdict(set)
    email_domains: Counter[str] = Counter()
    empty_rows = 0

    for record in records:
        spans = record["privacy_mask"]
        empty_rows += int(not spans)
        sources[str(record.get("source_dataset", "unknown"))] += 1
        generations[str(record.get("generation_type", "original"))] += 1
        family = record.get("augmentation_family")
        if family:
            families[str(family)] += 1
        for label in {str(span["label"]) for span in spans}:
            label_rows[label] += 1
        for span in spans:
            label = str(span["label"])
            value = str(span["value"])
            label_spans[label] += 1
            unique_values[label].add(value.casefold())
            if label == "EMAIL":
                _, separator, domain = value.rpartition("@")
                if separator and domain:
                    email_domains[domain.casefold().rstrip(".")] += 1

    return {
        "rows": len(records),
        "empty_mask_rows": empty_rows,
        "empty_mask_fraction": empty_rows / len(records) if records else 0.0,
        "total_spans": sum(label_spans.values()),
        "spans_by_label": dict(sorted(label_spans.items())),
        "rows_by_label": dict(sorted(label_rows.items())),
        "unique_values_by_label": {
            label: len(values) for label, values in sorted(unique_values.items())
        },
        "rows_by_source": dict(sorted(sources.items())),
        "rows_by_generation_type": dict(sorted(generations.items())),
        "rows_by_augmentation_family": dict(sorted(families.items())),
        "unique_email_domains": len(email_domains),
        "top_email_domains": dict(email_domains.most_common(15)),
    }

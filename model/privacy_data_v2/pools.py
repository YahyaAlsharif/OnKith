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

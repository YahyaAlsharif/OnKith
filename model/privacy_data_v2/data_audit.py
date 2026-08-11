"""Reproducible audits for the pinned OnKith PII dataset.

The command-line path is offline-first and reuses the Hugging Face cache.  It
does not train a model or change the historical TinyBERT split.  The historical
split is reproduced only to establish the exact baseline that Model V2 must
improve upon.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

DATASET_NAME = "ai4privacy/pii-masking-openpii-1.5m"
DATASET_REVISION = "a785eb528e28be2693c3718a27e066970de5dadb"
SEED = 42
VALIDATION_FRACTION = 0.10


def _name_case(value: str) -> str:
    letters = "".join(character for character in value if character.isalpha())
    if not letters:
        return "no_letters"
    if letters.islower():
        return "lower"
    if letters.isupper():
        return "upper"
    if value.istitle():
        return "title"
    return "mixed"


def audit_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Audit entity support and surface families in one streaming pass."""

    row_count = empty_rows = total_spans = 0
    span_counts: Counter[str] = Counter()
    row_support: Counter[str] = Counter()
    source_datasets: Counter[str] = Counter()
    regions: Counter[str] = Counter()
    email_domains: Counter[str] = Counter()
    name_case: dict[str, Counter[str]] = {
        "GIVENNAME": Counter(),
        "SURNAME": Counter(),
    }
    unique_values: dict[str, set[str]] = {}

    def iter_records():
        if hasattr(records, "iter") and hasattr(records, "column_names"):
            required = ["privacy_mask", "source_dataset", "region"]
            slim = records.select_columns(
                [column for column in required if column in records.column_names]
            )
            for batch in slim.iter(batch_size=8_192):
                batch_size = len(batch["privacy_mask"])
                for index in range(batch_size):
                    yield {key: values[index] for key, values in batch.items()}
        else:
            yield from records

    for record in iter_records():
        row_count += 1
        spans = record.get("privacy_mask")
        if not isinstance(spans, list):
            raise TypeError(f"Row {row_count - 1} privacy_mask is not a list.")
        empty_rows += int(not spans)
        row_labels: set[str] = set()
        source_datasets[str(record.get("source_dataset", ""))] += 1
        regions[str(record.get("region", ""))] += 1
        for span in spans:
            label = str(span["label"])
            value = str(span["value"])
            span_counts[label] += 1
            total_spans += 1
            row_labels.add(label)
            unique_values.setdefault(label, set()).add(value.casefold())
            if label == "EMAIL":
                _, separator, domain = value.rpartition("@")
                if separator and domain:
                    email_domains[domain.casefold().rstrip(".")] += 1
            if label in name_case:
                name_case[label][_name_case(value)] += 1
        row_support.update(row_labels)

    support = dict(sorted(span_counts.items()))
    rare = {
        "singleton": sorted(label for label, count in span_counts.items() if count == 1),
        "fewer_than_10": sorted(label for label, count in span_counts.items() if count < 10),
        "fewer_than_100": sorted(label for label, count in span_counts.items() if count < 100),
        "fewer_than_1000": sorted(label for label, count in span_counts.items() if count < 1000),
    }
    return {
        "rows": row_count,
        "empty_mask_rows": empty_rows,
        "empty_mask_fraction": empty_rows / row_count if row_count else 0.0,
        "total_spans": total_spans,
        "span_counts": support,
        "row_support": dict(sorted(row_support.items())),
        "unique_value_counts": {
            label: len(values) for label, values in sorted(unique_values.items())
        },
        "rare_labels": rare,
        "email_domains": dict(email_domains.most_common()),
        "name_case": {
            label: dict(counter) for label, counter in name_case.items()
        },
        "source_datasets": dict(source_datasets.most_common()),
        "regions": dict(regions.most_common()),
    }


def extract_row_entity_types(privacy_masks: Sequence[list[dict]]) -> tuple[list[tuple[int, ...]], list[str], np.ndarray]:
    """Build the same entity-presence matrix used by the TinyBERT notebook."""

    row_types = [frozenset(str(span["label"]) for span in mask) for mask in privacy_masks]
    labels = sorted({label for values in row_types for label in values})
    label_to_column = {label: index for index, label in enumerate(labels)}
    indicator = np.zeros((len(row_types), len(labels)), dtype=np.uint8)
    labels_by_row: list[tuple[int, ...]] = []
    for row_index, values in enumerate(row_types):
        columns = tuple(label_to_column[label] for label in sorted(values))
        labels_by_row.append(columns)
        if columns:
            indicator[row_index, list(columns)] = 1
    return labels_by_row, labels, indicator


def historical_multilabel_split(
    indicator: np.ndarray,
    labels_by_row: Sequence[tuple[int, ...]],
    *,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact deterministic TinyBERT split, retained for historical continuity."""

    row_count, label_count = indicator.shape
    validation_rows = int(np.ceil(row_count * validation_fraction))
    training_rows = row_count - validation_rows
    label_support = indicator.sum(axis=0).astype(int)
    if np.any(label_support == 0):
        raise ValueError("The multilabel indicator contains an empty label column.")

    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(row_count)
    tie_rank = np.empty(row_count, dtype=np.int64)
    tie_rank[tie_order] = np.arange(row_count)
    split_order = rng.permutation(2)
    split_priority = np.empty(2, dtype=np.int64)
    split_priority[split_order] = np.arange(2)

    assignments = np.full(row_count, -1, dtype=np.int8)
    remaining_rows = np.array([training_rows, validation_rows], dtype=float)
    remaining_targets = np.outer(
        remaining_rows / row_count,
        label_support.astype(float),
    )
    remaining_support = label_support.copy()
    rows_by_label = [np.flatnonzero(indicator[:, index]) for index in range(label_count)]

    def assign(row_index: int, split_index: int) -> None:
        assignments[row_index] = split_index
        remaining_rows[split_index] -= 1
        for label_index in labels_by_row[row_index]:
            remaining_targets[split_index, label_index] -= 1
            remaining_support[label_index] -= 1

    empty = np.array(
        sorted(
            (index for index, labels in enumerate(labels_by_row) if not labels),
            key=lambda index: tie_rank[index],
        ),
        dtype=np.int64,
    )
    validation_empty = set(empty[: int(np.ceil(len(empty) * validation_fraction))].tolist())
    for row_index in empty:
        assign(int(row_index), 1 if int(row_index) in validation_empty else 0)

    uncovered = set(range(label_count))
    while uncovered:
        rarest = min(uncovered, key=lambda index: (label_support[index], index))
        candidates = [
            int(row_index) for row_index in rows_by_label[rarest]
            if assignments[row_index] == -1
        ]
        anchor = max(
            candidates,
            key=lambda row_index: (
                sum(
                    1.0 / label_support[label_index]
                    for label_index in labels_by_row[row_index]
                    if label_index in uncovered
                ),
                -tie_rank[row_index],
            ),
        )
        assign(anchor, 0)
        uncovered.difference_update(labels_by_row[anchor])

    while np.any(remaining_support > 0):
        active = np.flatnonzero(remaining_support > 0)
        rarest = min(
            active,
            key=lambda index: (remaining_support[index], label_support[index], index),
        )
        candidates = sorted(
            (
                int(row_index) for row_index in rows_by_label[rarest]
                if assignments[row_index] == -1
            ),
            key=lambda index: tie_rank[index],
        )
        for row_index in candidates:
            eligible = np.flatnonzero(remaining_rows > 0)
            split_index = max(
                eligible,
                key=lambda candidate: (
                    remaining_targets[candidate, rarest],
                    remaining_rows[candidate],
                    -split_priority[candidate],
                ),
            )
            assign(row_index, int(split_index))

    for row_index in sorted(np.flatnonzero(assignments == -1), key=lambda index: tie_rank[index]):
        eligible = np.flatnonzero(remaining_rows > 0)
        split_index = max(
            eligible,
            key=lambda candidate: (remaining_rows[candidate], -split_priority[candidate]),
        )
        assign(int(row_index), int(split_index))

    if np.any(assignments == -1) or np.any(remaining_rows != 0):
        raise RuntimeError("The historical split did not assign every row exactly once.")
    return np.flatnonzero(assignments == 0), np.flatnonzero(assignments == 1)


def _subset_records(dataset, indices: Sequence[int]):
    for index in indices:
        yield dataset[int(index)]


def _cached_english_split(dataset_split, expected_rows: int):
    """Reuse the notebook's materialized English-filter index without filtering.

    Hugging Face stores a filtered Dataset as an Arrow table of source-row
    indices. Reusing it avoids rescanning the full 1.6M multilingual rows on
    every audit. Every candidate is verified against the pinned source split;
    a missing or stale cache fails clearly instead of downloading or guessing.
    """

    import pyarrow.ipc as ipc

    cache_files = dataset_split.cache_files
    if not cache_files:
        raise FileNotFoundError("The pinned source split has no local cache file.")
    cache_root = Path(cache_files[0]["filename"]).resolve().parent
    candidates: list[list[int]] = []
    for path in sorted(cache_root.glob("cache-*.arrow")):
        try:
            with path.open("rb") as stream:
                table = ipc.open_stream(stream).read_all()
        except (OSError, ValueError):
            continue
        if table.schema.names != ["indices"] or table.num_rows != expected_rows:
            continue
        indices = [int(value) for value in table.column("indices").to_pylist()]
        if indices and min(indices) >= 0 and max(indices) < len(dataset_split):
            candidates.append(indices)

    for indices in candidates:
        selected = dataset_split.select(indices)
        if all(language == "en" for language in selected["language"]):
            return selected
    raise FileNotFoundError(
        f"No verified {expected_rows:,}-row English-filter index cache was found "
        f"under {cache_root}. Run the pinned TinyBERT data filter once first."
    )


def _cached_index_vector(
    cache_root: Path, *, expected_rows: int, upper_bound: int
) -> list[int]:
    """Read one verified cached Dataset.select index vector."""

    import pyarrow.ipc as ipc

    matches: list[list[int]] = []
    for path in sorted(cache_root.glob("cache-*.arrow")):
        try:
            with path.open("rb") as stream:
                table = ipc.open_stream(stream).read_all()
        except (OSError, ValueError):
            continue
        if table.schema.names != ["indices"] or table.num_rows != expected_rows:
            continue
        indices = [int(value) for value in table.column("indices").to_pylist()]
        if (
            len(set(indices)) == expected_rows
            and indices
            and min(indices) >= 0
            and max(indices) < upper_bound
        ):
            matches.append(indices)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one verified {expected_rows:,}-row cached index vector "
            f"under {cache_root}, found {len(matches)}."
        )
    return matches[0]


def build_current_report() -> dict[str, object]:
    """Load only the pinned cached dataset and reproduce the current baseline."""

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION)
    official_train = _cached_english_split(dataset["train"], expected_rows=163_740)
    official_validation = _cached_english_split(
        dataset["validation"], expected_rows=40_909
    )

    labels = sorted(
        {
            str(span["label"])
            for mask in official_train["privacy_mask"]
            for span in mask
        }
    )
    cache_root = Path(dataset["train"].cache_files[0]["filename"]).resolve().parent
    train_indices = _cached_index_vector(
        cache_root,
        expected_rows=147_366,
        upper_bound=len(official_train),
    )
    validation_indices = _cached_index_vector(
        cache_root,
        expected_rows=16_374,
        upper_bound=len(official_train),
    )
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("Cached historical train and validation indices overlap.")
    if set(train_indices) | set(validation_indices) != set(range(len(official_train))):
        raise RuntimeError("Cached historical indices do not cover English training.")
    project_train = official_train.select(train_indices)
    project_validation = official_train.select(validation_indices)
    train_audit = audit_records(project_train)
    validation_audit = audit_records(project_validation)

    supported = set(labels)
    unsupported_final_types = sorted(
        {
            str(span["label"])
            for mask in official_validation["privacy_mask"]
            for span in mask
        }
        - supported
    )
    scored_final_indices = [
        index
        for index, mask in enumerate(official_validation["privacy_mask"])
        if not any(str(span["label"]) in unsupported_final_types for span in mask)
    ]
    scored_final = official_validation.select(scored_final_indices)
    final_audit = audit_records(scored_final)
    return {
        "dataset": {
            "name": DATASET_NAME,
            "revision": DATASET_REVISION,
            "language": "en",
        },
        "historical_split": {
            "seed": SEED,
            "validation_fraction": VALIDATION_FRACTION,
            "indices_source": "verified cached TinyBERT Dataset.select vectors",
            "official_english_train_rows": len(official_train),
            "official_english_validation_rows": len(official_validation),
            "project_train_rows": len(train_indices),
            "project_validation_rows": len(validation_indices),
            "scored_final_rows": len(scored_final_indices),
            "unsupported_final_types": unsupported_final_types,
            "excluded_final_rows": len(official_validation) - len(scored_final_indices),
        },
        "project_train": train_audit,
        "project_validation": validation_audit,
        "scored_final": final_audit,
        "interpretation": {
            "rare_support_warning": (
                "Counts below 100 spans describe inadequate coverage, not a class "
                "that can be repaired by duplicating the same examples."
            ),
            "historical_split_limitation": (
                "The historical split balances row-level entity presence only; it "
                "does not hold out values, domains, templates, formats, regions, "
                "or source generators."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    report = build_current_report()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output.resolve())
    else:
        print(rendered)


if __name__ == "__main__":
    main()

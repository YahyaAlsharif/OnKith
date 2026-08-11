"""Leakage-aware, one-factor-at-a-time dataset splitting and auditing."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

try:
    from .augmentation import FROZEN_FINAL_PARTITION, validate_record
except ImportError:  # Direct notebook/script execution from this directory.
    from augmentation import FROZEN_FINAL_PARTITION, validate_record

FactorExtractor = Callable[[Mapping[str, object]], set[str]]


def canonical_template(record: Mapping[str, object]) -> str:
    """Replace entity values with labels to expose generator/template reuse."""

    validate_record(record)
    text = str(record["source_text"])
    parts: list[str] = []
    cursor = 0
    for span in record["privacy_mask"]:
        parts.append(text[cursor : int(span["start"])])
        parts.append(f"{{{span['label']}}}")
        cursor = int(span["end"])
    parts.append(text[cursor:])
    return re.sub(r"\s+", " ", "".join(parts)).strip().lower()


def _normalise_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def entity_value_factors(record: Mapping[str, object]) -> set[str]:
    validate_record(record)
    return {
        f"{span['label']}:{_normalise_value(str(span['value']))}"
        for span in record["privacy_mask"]
    }


def email_domain_factors(record: Mapping[str, object]) -> set[str]:
    validate_record(record)
    factors = set()
    for span in record["privacy_mask"]:
        if span["label"] != "EMAIL":
            continue
        _, separator, domain = str(span["value"]).rpartition("@")
        if separator and domain:
            factors.add(domain.casefold().rstrip("."))
    return factors


def formatting_family_factors(record: Mapping[str, object]) -> set[str]:
    validate_record(record)
    factors = set()
    for span in record["privacy_mask"]:
        value = str(span["value"])
        shape = re.sub(r"[A-Za-z]", "A", value)
        shape = re.sub(r"\d", "9", shape)
        shape = re.sub(r"A+", "A", shape)
        shape = re.sub(r"9+", "9", shape)
        factors.add(f"{span['label']}:{shape}")
    return factors


def single_field_factor(field: str) -> FactorExtractor:
    def extract(record: Mapping[str, object]) -> set[str]:
        value = record.get(field)
        return {str(value).strip().casefold()} if value not in {None, ""} else set()
    return extract


FACTOR_EXTRACTORS: dict[str, FactorExtractor] = {
    "entity_value": entity_value_factors,
    "email_domain": email_domain_factors,
    "formatting_family": formatting_family_factors,
    "canonical_template": lambda record: {canonical_template(record)},
    "region": single_field_factor("region"),
    "source_dataset": single_field_factor("source_dataset"),
}


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _components(
    records: Sequence[Mapping[str, object]], extractor: FactorExtractor
) -> list[list[int]]:
    union_find = _UnionFind(len(records))
    first_row_by_factor: dict[str, int] = {}
    for row_index, record in enumerate(records):
        factors = extractor(record) or {f"__row_without_factor__:{row_index}"}
        for factor in factors:
            previous = first_row_by_factor.setdefault(factor, row_index)
            union_find.union(row_index, previous)
    grouped: dict[int, list[int]] = defaultdict(list)
    for row_index in range(len(records)):
        grouped[union_find.find(row_index)].append(row_index)
    return list(grouped.values())


def _component_rank(component: Sequence[int], seed: int) -> str:
    return hashlib.sha256(f"{seed}|{tuple(component)}".encode("utf-8")).hexdigest()


def split_by_factor(
    records: Sequence[Mapping[str, object]],
    *,
    factor: str,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], dict[str, object]]:
    """Split connected factor groups without letting a factor cross splits."""

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be strictly between zero and one.")
    if factor not in FACTOR_EXTRACTORS:
        raise ValueError(f"Unknown grouping factor: {factor}")
    if any(
        str(record.get("partition", "")).lower() == FROZEN_FINAL_PARTITION
        or bool(record.get("frozen", False))
        for record in records
    ):
        raise ValueError("Frozen OOD-final records cannot enter split preparation.")
    for record in records:
        validate_record(record)

    target = round(len(records) * holdout_fraction)
    components = sorted(
        _components(records, FACTOR_EXTRACTORS[factor]),
        key=lambda component: (-len(component), _component_rank(component, seed)),
    )
    train: list[int] = []
    holdout: list[int] = []
    for component in components:
        current_distance = abs(target - len(holdout))
        candidate_distance = abs(target - (len(holdout) + len(component)))
        (holdout if candidate_distance < current_distance else train).extend(component)
    train.sort()
    holdout.sort()
    audit = audit_factor_overlap(records, train, holdout, factor=factor)
    audit.update(
        seed=int(seed),
        target_holdout_rows=int(target),
        component_count=len(components),
        largest_component_rows=max(map(len, components), default=0),
    )
    return train, holdout, audit


def audit_factor_overlap(
    records: Sequence[Mapping[str, object]],
    train_indices: Iterable[int],
    holdout_indices: Iterable[int],
    *,
    factor: str,
) -> dict[str, object]:
    if factor not in FACTOR_EXTRACTORS:
        raise ValueError(f"Unknown grouping factor: {factor}")
    extractor = FACTOR_EXTRACTORS[factor]

    def collect(indices: Iterable[int]) -> set[str]:
        result: set[str] = set()
        for index in indices:
            result.update(extractor(records[int(index)]))
        return result

    train_indices = list(map(int, train_indices))
    holdout_indices = list(map(int, holdout_indices))
    train_factors, holdout_factors = collect(train_indices), collect(holdout_indices)
    overlap = train_factors & holdout_factors
    return {
        "factor": factor,
        "train_rows": len(train_indices),
        "holdout_rows": len(holdout_indices),
        "train_factor_count": len(train_factors),
        "holdout_factor_count": len(holdout_factors),
        "overlap_count": len(overlap),
        "overlap_examples": sorted(overlap)[:20],
        "is_disjoint": not overlap,
    }

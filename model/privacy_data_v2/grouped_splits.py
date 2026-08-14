"""Leakage-aware, one-factor-at-a-time dataset splitting and auditing."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Sequence

try:
    from .augmentation import FROZEN_FINAL_PARTITION, PROTECTED_PARTITIONS, validate_record
except ImportError:  # Direct notebook/script execution from this directory.
    from augmentation import FROZEN_FINAL_PARTITION, PROTECTED_PARTITIONS, validate_record

FactorExtractor = Callable[[Mapping[str, object]], set[str]]


def _reject_protected(records: Sequence[Mapping[str, object]]) -> None:
    """Evaluation data must never be re-split into training material."""

    for record in records:
        partition = str(record.get("partition", "")).lower()
        if partition == FROZEN_FINAL_PARTITION or bool(record.get("frozen", False)):
            raise ValueError("Frozen OOD-final records cannot enter split preparation.")
        if partition in PROTECTED_PARTITIONS:
            raise ValueError(
                f"Records in the protected partition {partition!r} cannot enter "
                "split preparation; they are already held out."
            )


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


#: Labels drawn from small closed vocabularies. Their values are public facts,
#: not private identities: knowing that "Saudi riyal" is a currency is not
#: memorising anyone's data. Grouping on them would also be self-defeating -
#: there are only a dozen currencies and four Saudi timezone spellings, so
#: holding a value out removes that label from validation entirely, and because
#: the same handful of values recurs across the corpus it welds the whole pool
#: into one component.
#:
#: Both ``model/data`` builders reached the same conclusion independently and
#: recorded it as an audited split exception.
CLOSED_VOCABULARY_LABELS: frozenset[str] = frozenset(
    {"BANKNAME", "COUNTRY", "CURRENCY", "TIMEZONE", "GENDER", "SEX", "TITLE"}
)


def open_entity_value_factors(record: Mapping[str, object]) -> set[str]:
    """Entity values excluding closed vocabularies. See the note above."""

    validate_record(record)
    return {
        f"{span['label']}:{_normalise_value(str(span['value']))}"
        for span in record["privacy_mask"]
        if str(span["label"]) not in CLOSED_VOCABULARY_LABELS
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


#: Entity types whose surface form is a structured identifier. Only these are
#: grouped by numeric shape; a name that happens to contain a digit is not a
#: formatting family.
STRUCTURED_FORMAT_LABELS: frozenset[str] = frozenset(
    {
        "ACCOUNTNUM", "CREDITCARDNUMBER", "DRIVERLICENSENUM", "IDCARDNUM",
        "PASSPORTNUM", "SOCIALNUM", "TAXNUM", "TELEPHONENUM", "ZIPCODE",
        "BUILDINGNUM", "IPV4",
    }
)

_SHAPE_RUN = re.compile(r"[A-Za-z]+|\d+")


def _numeric_shape(value: str) -> str:
    """Collapse letter and digit runs in a single pass.

    One ``re.sub`` over an alternation, because substituting letters first would
    inject digits that a second digit pass would then rewrite - turning
    ``Person0`` into ``a9d3`` and silently merging unrelated values.
    """

    return _SHAPE_RUN.sub(
        lambda match: (
            f"L{len(match.group())}" if match.group()[0].isalpha() else f"D{len(match.group())}"
        ),
        value,
    )


def numeric_pattern_family_factors(record: Mapping[str, object]) -> set[str]:
    """Group structured identifiers by digit-run shape and separator layout.

    ``+966 55 123 4567`` and ``+966 50 987 6543`` share the family
    ``TELEPHONENUM:+D3 D2 D3 D4``, so one generator's numeric style cannot appear
    on both sides of a split.
    """

    validate_record(record)
    factors = set()
    for span in record["privacy_mask"]:
        label = str(span["label"])
        if label not in STRUCTURED_FORMAT_LABELS:
            continue
        value = str(span["value"])
        if not any(character.isdigit() for character in value):
            continue
        factors.add(f"{label}:{_numeric_shape(value)}")
    return factors


def organisation_family_factors(record: Mapping[str, object]) -> set[str]:
    """Group organisations by their normalised head word.

    ``Horizon Logistics Cooperative`` and ``Horizon Analytics Cooperative`` come
    from one compositional generator and must not straddle a split.

    ``BANKNAME`` is deliberately excluded even though it looks similar: it is a
    closed vocabulary (twelve licensed Saudi banks), so its head words recur
    throughout the corpus and grouping on them would connect the entire pool.
    See :data:`CLOSED_VOCABULARY_LABELS`.
    """

    validate_record(record)
    factors = set()
    for span in record["privacy_mask"]:
        if span["label"] != "ORGANISATION":
            continue
        words = re.findall(r"[A-Za-z]+", str(span["value"]).casefold())
        words = [word for word in words if word not in {"al", "the", "of", "and", "for"}]
        if words:
            factors.add(f"{span['label']}:{words[0]}")
    return factors


def single_field_factor(field: str) -> FactorExtractor:
    def extract(record: Mapping[str, object]) -> set[str]:
        value = record.get(field)
        return {str(value).strip().casefold()} if value not in {None, ""} else set()
    return extract


FACTOR_EXTRACTORS: dict[str, FactorExtractor] = {
    "entity_value": entity_value_factors,
    "open_entity_value": open_entity_value_factors,
    "email_domain": email_domain_factors,
    "formatting_family": formatting_family_factors,
    "canonical_template": lambda record: {canonical_template(record)},
    "numeric_pattern_family": numeric_pattern_family_factors,
    "organisation_family": organisation_family_factors,
    "region": single_field_factor("region"),
    "source_dataset": single_field_factor("source_dataset"),
}

#: The factor set used for the authoritative supplementary 80/20 split.
#:
#: Three deliberate exclusions:
#:
#: ``region`` and ``source_dataset``
#:     Holding either out would force an entire source or region onto one side.
#:     That is coverage loss, not leakage control.
#:
#: ``formatting_family`` and ``numeric_pattern_family``
#:     Both are too coarse to be *hard* constraints. ``formatting_family``
#:     collapses any run of letters to a single ``A``, so every alphabetic name
#:     shares ``GIVENNAME:A``. ``numeric_pattern_family`` is narrower but still
#:     groups every Saudi IBAN under one shape and every five-digit postcode
#:     under another - holding those out would leave ``ACCOUNTNUM`` and
#:     ``ZIPCODE`` with no validation at all, which is worse than the leakage it
#:     prevents. Both are reported as diagnostics instead, so format leakage
#:     stays visible without collapsing the pool into one component.
MODEL_V2_SPLIT_FACTORS: tuple[str, ...] = (
    "open_entity_value",
    "email_domain",
    "canonical_template",
    "organisation_family",
)

#: Measured and reported, never enforced. See the note above.
MODEL_V2_DIAGNOSTIC_FACTORS: tuple[str, ...] = (
    "formatting_family",
    "numeric_pattern_family",
    "entity_value",
)


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


def _multi_components(
    records: Sequence[Mapping[str, object]], factors: Sequence[str]
) -> list[list[int]]:
    """Connect rows that share ANY declared factor, namespaced per factor."""

    union_find = _UnionFind(len(records))
    first_row_by_factor: dict[str, int] = {}
    for row_index, record in enumerate(records):
        keys: set[str] = set()
        for factor in factors:
            for value in FACTOR_EXTRACTORS[factor](record):
                keys.add(f"{factor}::{value}")
        if not keys:
            keys = {f"__row_without_factor__::{row_index}"}
        for key in keys:
            previous = first_row_by_factor.setdefault(key, row_index)
            union_find.union(row_index, previous)
    grouped: dict[int, list[int]] = defaultdict(list)
    for row_index in range(len(records)):
        grouped[union_find.find(row_index)].append(row_index)
    return [sorted(component) for component in grouped.values()]


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
    _reject_protected(records)
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


def split_by_factors(
    records: Sequence[Mapping[str, object]],
    *,
    factors: Sequence[str] = MODEL_V2_SPLIT_FACTORS,
    holdout_fraction: float,
    seed: int,
    min_independent_groups: int = 10,
) -> tuple[list[int], list[int], dict[str, object]]:
    """Split so that no declared factor crosses the boundary, label-aware.

    Rows are connected when they share *any* declared factor, so a near-identical
    entity value, template, formatting family or numeric style cannot appear on
    both sides. Components are then assigned in two deterministic passes:

    1. label reservation - every label with at least ``min_independent_groups``
       independent components contributes its share of components to the
       holdout, so weak labels are actually validatable;
    2. size balancing - remaining components are assigned greedily to approach
       ``holdout_fraction``.

    Labels with too few independent groups are reported in the audit rather than
    forced, because splitting two near-identical rows is not real validation.
    """

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be strictly between zero and one.")
    unknown = [factor for factor in factors if factor not in FACTOR_EXTRACTORS]
    if unknown:
        raise ValueError(f"Unknown grouping factors: {unknown}")
    if not factors:
        raise ValueError("At least one grouping factor is required.")
    _reject_protected(records)
    for record in records:
        validate_record(record)

    components = sorted(
        _multi_components(records, factors),
        key=lambda component: (-len(component), _component_rank(component, seed)),
    )
    target = round(len(records) * holdout_fraction)

    labels_by_component: list[set[str]] = [
        {
            str(span["label"])
            for row_index in component
            for span in records[row_index]["privacy_mask"]
        }
        for component in components
    ]
    components_by_label: dict[str, list[int]] = defaultdict(list)
    for component_index, labels in enumerate(labels_by_component):
        for label in labels:
            components_by_label[label].append(component_index)

    assigned: dict[int, str] = {}
    holdout_rows = 0
    reservation: dict[str, dict[str, int]] = {}

    # Pass 1: reserve holdout components for labels that have enough groups.
    for label in sorted(components_by_label, key=lambda name: (len(components_by_label[name]), name)):
        available = components_by_label[label]
        required = max(1, round(len(available) * holdout_fraction))
        if len(available) < min_independent_groups:
            reservation[label] = {
                "independent_components": len(available),
                "required_holdout_components": 0,
                "reserved_holdout_components": 0,
                "sufficient": False,
            }
            continue
        already = sum(1 for index in available if assigned.get(index) == "holdout")
        candidates = sorted(
            (index for index in available if index not in assigned),
            key=lambda index: (len(components[index]), _component_rank(components[index], seed)),
        )
        reserved = 0
        for component_index in candidates:
            if already + reserved >= required:
                break
            assigned[component_index] = "holdout"
            holdout_rows += len(components[component_index])
            reserved += 1
        reservation[label] = {
            "independent_components": len(available),
            "required_holdout_components": int(required),
            "reserved_holdout_components": already + reserved,
            "sufficient": (already + reserved) >= required,
        }

    # Pass 2: balance the remaining components toward the target size.
    for component_index, component in enumerate(components):
        if component_index in assigned:
            continue
        current_distance = abs(target - holdout_rows)
        candidate_distance = abs(target - (holdout_rows + len(component)))
        if candidate_distance < current_distance:
            assigned[component_index] = "holdout"
            holdout_rows += len(component)
        else:
            assigned[component_index] = "train"

    # Pass 3: the greedy above selects nothing when every component is more than
    # twice the target - each one looks worse than staying at zero. An oversized
    # holdout is far better than no holdout, so take the smallest component.
    forced_component = None
    if holdout_rows == 0 and target > 0 and len(components) > 1:
        forced_component = min(
            range(len(components)),
            key=lambda index: (len(components[index]), _component_rank(components[index], seed)),
        )
        assigned[forced_component] = "holdout"
        holdout_rows += len(components[forced_component])

    train: list[int] = []
    holdout: list[int] = []
    for component_index, component in enumerate(components):
        (holdout if assigned[component_index] == "holdout" else train).extend(component)
    train.sort()
    holdout.sort()

    largest = max(map(len, components), default=0)
    audit = audit_factors_overlap(records, train, holdout, factors=factors)
    audit.update(
        seed=int(seed),
        holdout_fraction=float(holdout_fraction),
        target_holdout_rows=int(target),
        component_count=len(components),
        largest_component_rows=largest,
        largest_component_fraction=(largest / len(records) if records else 0.0),
        min_independent_groups=int(min_independent_groups),
        label_reservation=dict(sorted(reservation.items())),
        labels_without_independent_groups=sorted(
            label for label, info in reservation.items() if not info["sufficient"]
        ),
        holdout_shortfall_ratio=(
            (target - len(holdout)) / target if target else 0.0
        ),
        forced_smallest_component=forced_component is not None,
    )

    # A grouped split of highly connected data can collapse: if every row shares
    # a template or an entity value, no subset can be held out without leaking,
    # and the greedy pass assigns everything to one side. That still reports
    # is_disjoint=True, which would be a dangerously quiet failure - an empty or
    # tiny holdout is not a validation set. Fail loudly instead.
    if target > 0 and not holdout:
        raise ValueError(
            f"Grouped split produced an empty holdout against a target of {target}. "
            f"The pool is too connected to split on {list(factors)}: it forms "
            f"{len(components)} component(s) and the largest holds {largest} of "
            f"{len(records)} rows ({largest / len(records):.1%}). Diversify the data or "
            "narrow the factor list; an empty holdout is not a validation set."
        )
    return train, holdout, audit


def audit_factors_overlap(
    records: Sequence[Mapping[str, object]],
    train_indices: Iterable[int],
    holdout_indices: Iterable[int],
    *,
    factors: Sequence[str],
    diagnostic_factors: Sequence[str] = MODEL_V2_DIAGNOSTIC_FACTORS,
) -> dict[str, object]:
    """Audit every enforced factor, plus diagnostics and per-label support.

    ``diagnostic_factors`` are measured but not enforced, so surface-format
    leakage stays visible in the report even though it does not constrain the
    split. ``is_disjoint`` reflects the enforced factors only.
    """

    train_indices = list(map(int, train_indices))
    holdout_indices = list(map(int, holdout_indices))
    per_factor = {
        factor: audit_factor_overlap(records, train_indices, holdout_indices, factor=factor)
        for factor in factors
    }
    diagnostics = {
        factor: audit_factor_overlap(records, train_indices, holdout_indices, factor=factor)
        for factor in diagnostic_factors
        if factor not in factors
    }

    def label_spans(indices: Sequence[int]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for index in indices:
            for span in records[index]["privacy_mask"]:
                counts[str(span["label"])] += 1
        return dict(sorted(counts.items()))

    return {
        "factors": list(factors),
        "diagnostic_factors": list(diagnostics),
        "train_rows": len(train_indices),
        "holdout_rows": len(holdout_indices),
        "holdout_row_fraction": (
            len(holdout_indices) / (len(train_indices) + len(holdout_indices))
            if records
            else 0.0
        ),
        "per_factor": per_factor,
        "per_diagnostic_factor": diagnostics,
        "is_disjoint": all(audit["is_disjoint"] for audit in per_factor.values()),
        "diagnostic_note": (
            "Diagnostic factors are measured, not enforced. Overlap here means "
            "train and holdout share a surface format, which is expected when a "
            "label has only one realistic format."
        ),
        "train_label_spans": label_spans(train_indices),
        "holdout_label_spans": label_spans(holdout_indices),
    }


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

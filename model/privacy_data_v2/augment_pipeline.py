"""Training-only augmentation, with caps that keep diversity honest.

Only conservative, span-safe, field-proven transforms are used. Every one of
them rebuilds the text and recalculates offsets through ``augmentation``, and
every produced row is re-validated. Deliberately excluded: paraphrasing, LLM
rewriting, synonym substitution and backtranslation, all of which can silently
move or destroy an entity boundary.

Two guardrails make "diversity" mean something:

``per-family caps``
    No single transform may dominate the training split.

``parent diversity floor``
    A family must draw from at least ``MIN_PARENTS_PER_FAMILY`` distinct source
    rows. Producing a thousand variants of five rows is oversampling, not
    coverage, and the audit reports it as a failure instead of a success.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    from .augmentation import (
        add_conversational_frame,
        apply_case_variant,
        build_email_values,
        diversify_entity_values,
        generate_hard_negatives,
        normalized_whitespace,
        punctuation_free,
        speak_structured_entities,
        validate_record,
    )
    from .ontology import ENTITIES, STRUCTURED_ENTITIES
    from .records import PROTECTED_PARTITIONS, make_record, validate_model_v2_record
except ImportError:  # Direct notebook/script execution from this directory.
    from augmentation import (
        add_conversational_frame,
        apply_case_variant,
        build_email_values,
        diversify_entity_values,
        generate_hard_negatives,
        normalized_whitespace,
        punctuation_free,
        speak_structured_entities,
        validate_record,
    )
    from ontology import ENTITIES, STRUCTURED_ENTITIES
    from records import PROTECTED_PARTITIONS, make_record, validate_model_v2_record

RESOURCES = Path(__file__).resolve().parent / "resources"

#: Maximum share of the final training split that may be augmented.
MAX_AUGMENTED_SHARE = 0.25

#: Maximum share of the final training split that any single family may hold.
MAX_FAMILY_SHARE = 0.08

#: A family drawing on fewer distinct parent rows than this is oversampling.
MIN_PARENTS_PER_FAMILY = 50

#: Target share of the training split made of no-PII hard negatives. AI4Privacy
#: English supplies only 44 empty rows in 163,740, and the frozen OOD baseline
#: shows a 0.6528 hard-negative false-positive rate, so this is a priority.
HARD_NEGATIVE_SHARE = 0.05


def load_resources() -> dict[str, Any]:
    """Load the curated inventories used by the training-side transforms."""

    entity_values = json.loads((RESOURCES / "entity_values.json").read_text(encoding="utf-8"))
    email_parts = json.loads((RESOURCES / "email_domains.json").read_text(encoding="utf-8"))
    entity_values = dict(entity_values)
    entity_values["EMAIL"] = build_email_values(
        email_parts["local_parts"], email_parts["domains"]
    )
    return {
        "entity_values": entity_values,
        "email_parts": email_parts,
        "frames": json.loads((RESOURCES / "conversational_frames.json").read_text(encoding="utf-8")),
        "hard_negatives": json.loads(
            (RESOURCES / "hard_negative_templates.json").read_text(encoding="utf-8")
        ),
        "passwords": json.loads(
            (RESOURCES / "password_disclosures.json").read_text(encoding="utf-8")
        ),
    }


def restrict_inventories(
    entity_values: Mapping[str, Sequence[str]], forbidden: Mapping[str, Iterable[str]]
) -> dict[str, list[str]]:
    """Remove any value that also appears in a protected evaluation set.

    Augmenting training rows with an entity value that the supplementary
    holdout also uses would quietly break the grouped split, so the inventories
    are filtered before a single row is generated.
    """

    blocked = {
        label: {str(value).strip().casefold() for value in values}
        for label, values in forbidden.items()
    }
    return {
        label: [
            value
            for value in values
            if str(value).strip().casefold() not in blocked.get(label, set())
        ]
        for label, values in entity_values.items()
    }


# --------------------------------------------------------------------------
# PASSWORD disclosure rendering
# --------------------------------------------------------------------------


def render_template(
    template: str,
    inventories: Mapping[str, Sequence[str]],
    rng: random.Random,
) -> tuple[str, list[dict[str, Any]]] | None:
    """Render a ``{LABEL}`` template into text plus exact spans.

    Offsets are computed while the string is assembled, so alignment is exact by
    construction rather than by a later search.
    """

    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    index = 0
    while index < len(template):
        start_brace = template.find("{", index)
        if start_brace == -1:
            literal = template[index:]
            parts.append(literal)
            cursor += len(literal)
            break
        end_brace = template.find("}", start_brace)
        if end_brace == -1:
            return None
        literal = template[index:start_brace]
        parts.append(literal)
        cursor += len(literal)

        label = template[start_brace + 1 : end_brace]
        if label not in ENTITIES:
            return None
        candidates = list(inventories.get(label, ()))
        if not candidates:
            return None
        value = str(rng.choice(candidates))
        spans.append({"label": label, "start": cursor, "end": cursor + len(value), "value": value})
        parts.append(value)
        cursor += len(value)
        index = end_brace + 1
    return "".join(parts), spans


def generate_password_disclosures(
    inventories: Mapping[str, Sequence[str]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build conversational credential-disclosure rows with exact spans.

    Generated rows are checked against ``password_rules`` so the training data
    obeys exactly the same admission standard applied to external corpora.
    """

    try:
        from .password_rules import classify_password_span
    except ImportError:  # Direct notebook/script execution.
        from password_rules import classify_password_span

    templates = json.loads(
        (RESOURCES / "password_disclosures.json").read_text(encoding="utf-8")
    )["templates"]
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < count and attempts < count * 40:
        attempts += 1
        template = rng.choice(templates)
        rendered = render_template(template["template"], inventories, rng)
        if rendered is None:
            continue
        text, spans = rendered
        key = text.strip().casefold()
        if key in seen:
            continue
        password_span = next((span for span in spans if span["label"] == "PASSWORD"), None)
        if password_span is None:
            continue
        decision = classify_password_span(text, password_span["start"], password_span["end"])
        if not decision.accept:
            continue
        seen.add(key)
        digest = hashlib.sha256(f"password|{seed}|{text}".encode("utf-8")).hexdigest()[:16]
        rows.append(
            make_record(
                uid=f"password-disclosure-{digest}",
                source_text=text,
                privacy_mask=sorted(spans, key=lambda span: (span["start"], span["end"])),
                source_dataset="onkith_password_disclosure_v1",
                partition="train",
                generation_type="synthetic",
                augmentation_family=f"password_{template['family']}",
                generator_seed=int(seed),
            )
        )
    return rows


def generate_model_v2_hard_negatives(*, seed: int) -> list[dict[str, Any]]:
    """Curated no-PII rows, converted into canonical Model V2 records."""

    templates = json.loads(
        (RESOURCES / "hard_negative_templates.json").read_text(encoding="utf-8")
    )
    rows = []
    for row in generate_hard_negatives(templates, seed=seed):
        rows.append(
            make_record(
                uid=str(row["uid"]),
                source_text=str(row["source_text"]),
                privacy_mask=[],
                source_dataset="onkith_hard_negative_v1",
                partition="train",
                generation_type="synthetic",
                augmentation_family="hard_negative",
                hard_negative_category=str(row["hard_negative_category"]),
                generator_seed=int(seed),
            )
        )
    return rows


# --------------------------------------------------------------------------
# Transform registry
# --------------------------------------------------------------------------

Transform = Callable[[Mapping[str, Any], random.Random], dict[str, Any] | None]


def _entity_value_transform(inventories: Mapping[str, Sequence[str]]) -> Transform:
    def transform(record: Mapping[str, Any], rng: random.Random) -> dict[str, Any] | None:
        result = diversify_entity_values(record, inventories, seed=rng.randrange(2**31))
        return result if result.get("augmentation_family") != "entity_value_noop" else None

    return transform


def _case_transform(variant: str) -> Transform:
    def transform(record: Mapping[str, Any], rng: random.Random) -> dict[str, Any] | None:
        return apply_case_variant(record, variant)

    return transform


def _frame_transform(frames: Sequence[Mapping[str, str]]) -> Transform:
    def transform(record: Mapping[str, Any], rng: random.Random) -> dict[str, Any] | None:
        frame = rng.choice(list(frames))
        return add_conversational_frame(record, frame["prefix"], frame.get("suffix", ""))

    return transform


def _spoken_transform(record: Mapping[str, Any], rng: random.Random) -> dict[str, Any] | None:
    labels = {str(span["label"]) for span in record["privacy_mask"]}
    if not (labels & (STRUCTURED_ENTITIES | {"EMAIL"})):
        return None
    result = speak_structured_entities(record)
    return result if result.get("augmentation_family") != "spoken_structured_noop" else None


def build_transforms(resources: Mapping[str, Any]) -> dict[str, Transform]:
    """The conservative, span-safe transform set used for Model V2 training."""

    inventories = resources["entity_values"]
    return {
        "entity_value": _entity_value_transform(inventories),
        "case_lower": _case_transform("lower"),
        "case_upper": _case_transform("upper"),
        "case_mixed": _case_transform("mixed"),
        "stt_punctuation_free": lambda record, rng: punctuation_free(record),
        "stt_whitespace": lambda record, rng: normalized_whitespace(record),
        "conversational_frame": _frame_transform(resources["frames"]),
        "spoken_structured": _spoken_transform,
    }


@dataclass
class AugmentationPlan:
    """Caps expressed against the size of the base training split."""

    base_rows: int
    max_augmented_share: float = MAX_AUGMENTED_SHARE
    max_family_share: float = MAX_FAMILY_SHARE
    min_parents_per_family: int = MIN_PARENTS_PER_FAMILY
    families: tuple[str, ...] = ()
    seed: int = 42

    @property
    def max_augmented_rows(self) -> int:
        return int(self.base_rows * self.max_augmented_share)

    @property
    def max_rows_per_family(self) -> int:
        return int(self.base_rows * self.max_family_share)

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_rows": self.base_rows,
            "max_augmented_share": self.max_augmented_share,
            "max_augmented_rows": self.max_augmented_rows,
            "max_family_share": self.max_family_share,
            "max_rows_per_family": self.max_rows_per_family,
            "min_parents_per_family": self.min_parents_per_family,
            "families": list(self.families),
            "seed": self.seed,
        }


@dataclass
class AugmentationResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    produced: Counter = field(default_factory=Counter)
    parents: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    failures: Counter = field(default_factory=Counter)

    def audit(self, plan: AugmentationPlan) -> dict[str, Any]:
        thin = sorted(
            family
            for family, count in self.produced.items()
            if count and len(self.parents[family]) < plan.min_parents_per_family
        )
        return {
            "plan": plan.as_dict(),
            "augmented_rows": len(self.rows),
            "rows_by_family": dict(sorted(self.produced.items())),
            "distinct_parents_by_family": {
                family: len(parents) for family, parents in sorted(self.parents.items())
            },
            "skipped_by_reason": dict(sorted(self.failures.items())),
            "families_below_parent_floor": thin,
            "parent_floor_satisfied": not thin,
        }


def augment_training_rows(
    training_rows: Sequence[Mapping[str, Any]],
    resources: Mapping[str, Any],
    plan: AugmentationPlan,
) -> AugmentationResult:
    """Augment the training split only, respecting every cap.

    Parents are visited in a seeded shuffle so families draw from across the
    corpus instead of from one contiguous block.
    """

    protected = [
        record for record in training_rows if str(record.get("partition")) in PROTECTED_PARTITIONS
    ]
    if protected:
        raise ValueError(
            f"{len(protected)} protected rows reached augmentation; "
            "only the training split may be augmented."
        )

    transforms = build_transforms(resources)
    families = plan.families or tuple(transforms)
    unknown = [family for family in families if family not in transforms]
    if unknown:
        raise ValueError(f"Unknown augmentation families: {unknown}")

    result = AugmentationResult()
    order = list(range(len(training_rows)))
    random.Random(plan.seed).shuffle(order)

    for family in families:
        transform = transforms[family]
        rng = random.Random(f"{plan.seed}|{family}".__hash__() & 0x7FFFFFFF)
        produced = 0
        for position in order:
            if produced >= plan.max_rows_per_family:
                break
            if len(result.rows) >= plan.max_augmented_rows:
                break
            record = training_rows[position]
            try:
                candidate = transform(record, rng)
            except ValueError:
                result.failures[f"{family}:transform_rejected"] += 1
                continue
            if candidate is None:
                result.failures[f"{family}:not_applicable"] += 1
                continue
            if str(candidate["source_text"]).strip() == str(record["source_text"]).strip():
                result.failures[f"{family}:no_change"] += 1
                continue
            candidate = dict(candidate)
            candidate["partition"] = "train"
            candidate["generation_type"] = "augmented"
            candidate["augmentation_family"] = family
            candidate.setdefault("parent_uid", record.get("uid"))
            candidate["uid"] = f"aug:{family}:{record.get('uid')}"
            candidate.setdefault("selection_reason", list(record.get("selection_reason", ())))
            try:
                validate_model_v2_record(candidate)
            except (ValueError, TypeError):
                result.failures[f"{family}:invalid_result"] += 1
                continue
            result.rows.append(candidate)
            result.produced[family] += 1
            result.parents[family].add(record.get("uid"))
            produced += 1
        if produced == 0:
            result.failures[f"{family}:produced_nothing"] += 1
    return result

"""Deterministic, span-aware data diversification for Model V2.

Every transformation rebuilds text from left to right and recalculates entity
offsets. Frozen OOD-final records are rejected at the mutation boundary so
evaluation data cannot accidentally enter training preparation.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

FROZEN_FINAL_PARTITION = "ood_final"

#: Partitions that must never be mutated. Augmentation may only ever touch the
#: training side; every evaluation set - the original AI4Privacy validation, the
#: supplementary holdout, the training-side test split and both OOD suites - is
#: rejected at the mutation boundary so a transform the model saw during
#: training can never leak into a reported number.
PROTECTED_PARTITIONS = frozenset(
    {
        "test",
        "val_original",
        "val_supplementary",
        "ood_dev",
        FROZEN_FINAL_PARTITION,
    }
)

_DIGIT_WORDS = (
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
)
_STRUCTURED_TYPES = {
    "ACCOUNTNUM", "CREDITCARDNUMBER", "DRIVERLICENSENUM", "IDCARDNUM",
    "PASSPORTNUM", "SOCIALNUM", "TAXNUM", "TELEPHONENUM",
}


def _stable_index(seed: int, identity: object, size: int) -> int:
    if size <= 0:
        raise ValueError("Cannot choose from an empty inventory.")
    digest = hashlib.sha256(f"{seed}|{identity!r}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def _copy_metadata(record: Mapping[str, object]) -> dict[str, object]:
    return {
        key: deepcopy(value)
        for key, value in record.items()
        if key not in {"source_text", "masked_text", "privacy_mask"}
    }


def _assert_mutable(record: Mapping[str, object]) -> None:
    partition = str(record.get("partition", "")).lower()
    if partition == FROZEN_FINAL_PARTITION or bool(record.get("frozen", False)):
        raise ValueError("Frozen OOD-final records may not be augmented.")
    if partition in PROTECTED_PARTITIONS:
        raise ValueError(
            f"Records in the protected partition {partition!r} may not be augmented. "
            "Only the training split is augmentable."
        )


def validate_record(record: Mapping[str, object]) -> None:
    """Validate exact spans without silently repairing or dropping evidence."""

    text = record.get("source_text")
    spans = record.get("privacy_mask")
    if not isinstance(text, str):
        raise TypeError("source_text must be a string.")
    if not isinstance(spans, list):
        raise TypeError("privacy_mask must be a list.")

    previous_end = 0
    for index, span in enumerate(spans):
        if not isinstance(span, Mapping):
            raise TypeError(f"privacy_mask[{index}] must be a mapping.")
        label, value = span.get("label"), span.get("value")
        start, end = span.get("start"), span.get("end")
        if not isinstance(label, str) or not label:
            raise ValueError(f"privacy_mask[{index}] has no valid label.")
        if not isinstance(value, str) or not value:
            raise ValueError(f"privacy_mask[{index}] has no non-empty value.")
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError(f"privacy_mask[{index}] bounds must be integers.")
        if not 0 <= start < end <= len(text):
            raise ValueError(f"privacy_mask[{index}] bounds are invalid.")
        if text[start:end] != value:
            raise ValueError(f"privacy_mask[{index}] does not match source_text.")
        if start < previous_end:
            raise ValueError(
                "Augmentation input spans must be non-overlapping and sorted. "
                "Run the existing strict annotation normalization first."
            )
        previous_end = end


def replace_spans(
    record: Mapping[str, object],
    replacements: Mapping[int, str],
    *,
    family: str,
) -> dict[str, object]:
    """Replace selected entity values and recalculate every downstream offset."""

    _assert_mutable(record)
    validate_record(record)
    text = str(record["source_text"])
    spans = list(record["privacy_mask"])
    invalid = set(replacements) - set(range(len(spans)))
    if invalid:
        raise IndexError(f"Replacement indices are outside privacy_mask: {invalid}")

    parts: list[str] = []
    updated_spans: list[dict[str, object]] = []
    cursor = 0
    output_length = 0
    for index, original in enumerate(spans):
        start, end = int(original["start"]), int(original["end"])
        gap = text[cursor:start]
        value = replacements.get(index, str(original["value"]))
        if not isinstance(value, str) or not value:
            raise ValueError(f"Replacement {index} must be non-empty text.")
        parts.extend((gap, value))
        output_length += len(gap)
        updated = dict(original)
        updated.update(
            start=output_length,
            end=output_length + len(value),
            value=value,
        )
        updated_spans.append(updated)
        output_length += len(value)
        cursor = end
    parts.append(text[cursor:])

    result = _copy_metadata(record)
    result["source_text"] = "".join(parts)
    result["privacy_mask"] = updated_spans
    result["augmentation_family"] = family
    result["parent_uid"] = record.get("parent_uid", record.get("uid"))
    validate_record(result)
    return result


def diversify_entity_values(
    record: Mapping[str, object],
    inventories: Mapping[str, Sequence[str]],
    *,
    seed: int,
) -> dict[str, object]:
    """Replace supported entities using a stable seed and row identity."""

    _assert_mutable(record)
    validate_record(record)
    replacements: dict[int, str] = {}
    identity = record.get("uid", record.get("source_text"))
    for index, span in enumerate(record["privacy_mask"]):
        label = str(span["label"])
        candidates = [
            value for value in inventories.get(label, ())
            if value and value != span["value"]
        ]
        if candidates:
            replacements[index] = candidates[
                _stable_index(seed, (identity, index, label), len(candidates))
            ]
    if not replacements:
        result = deepcopy(dict(record))
        result["augmentation_family"] = "entity_value_noop"
        result["generator_seed"] = int(seed)
        return result
    result = replace_spans(record, replacements, family="entity_value")
    result["generator_seed"] = int(seed)
    return result


def build_email_values(
    local_parts: Sequence[str], domains: Sequence[str]
) -> list[str]:
    """Build deterministic synthetic email values from explicit inventories."""

    values: list[str] = []
    for local in local_parts:
        local = str(local).strip()
        if not local or "@" in local or any(char.isspace() for char in local):
            raise ValueError(f"Invalid synthetic email local part: {local!r}")
        for domain in domains:
            domain = str(domain).strip().casefold().rstrip(".")
            if (
                not domain
                or "@" in domain
                or any(char.isspace() for char in domain)
                or "." not in domain
            ):
                raise ValueError(f"Invalid synthetic email domain: {domain!r}")
            values.append(f"{local}@{domain}")
    return values


def _mixed_case(value: str) -> str:
    letter_index = 0
    output: list[str] = []
    for character in value:
        if character.isascii() and character.isalpha():
            output.append(character.upper() if letter_index % 2 else character.lower())
            letter_index += 1
        else:
            output.append(character)
    return "".join(output)


def apply_case_variant(
    record: Mapping[str, object], variant: str
) -> dict[str, object]:
    """Change entity casing only; non-ASCII values are preserved safely."""

    transforms: dict[str, Callable[[str], str]] = {
        "lower": lambda value: "".join(
            char.lower() if char.isascii() else char for char in value
        ),
        "upper": lambda value: "".join(
            char.upper() if char.isascii() else char for char in value
        ),
        "mixed": _mixed_case,
    }
    if variant not in transforms:
        raise ValueError(f"Unknown case variant: {variant}")
    replacements = {
        index: transforms[variant](str(span["value"]))
        for index, span in enumerate(record["privacy_mask"])
    }
    return replace_spans(record, replacements, family=f"case_{variant}")


def apply_gap_transform(
    record: Mapping[str, object],
    transform: Callable[[str], str],
    *,
    family: str,
) -> dict[str, object]:
    """Transform only non-PII text while preserving entity surface forms."""

    _assert_mutable(record)
    validate_record(record)
    text = str(record["source_text"])
    parts: list[str] = []
    updated_spans: list[dict[str, object]] = []
    cursor = output_length = 0
    for original in record["privacy_mask"]:
        gap = transform(text[cursor : int(original["start"])])
        value = str(original["value"])
        parts.extend((gap, value))
        output_length += len(gap)
        updated = dict(original)
        updated.update(start=output_length, end=output_length + len(value))
        updated_spans.append(updated)
        output_length += len(value)
        cursor = int(original["end"])
    parts.append(transform(text[cursor:]))
    result = _copy_metadata(record)
    result["source_text"] = "".join(parts)
    result["privacy_mask"] = updated_spans
    result["augmentation_family"] = family
    result["parent_uid"] = record.get("parent_uid", record.get("uid"))
    validate_record(result)
    return result


def punctuation_free(record: Mapping[str, object]) -> dict[str, object]:
    """Remove punctuation outside entities, approximating Canary PNC-off text."""

    return apply_gap_transform(
        record,
        lambda gap: re.sub(r"[^\w\s]", "", gap),
        family="stt_punctuation_free",
    )


def normalized_whitespace(record: Mapping[str, object]) -> dict[str, object]:
    """Collapse non-entity whitespace without changing entity values."""

    return apply_gap_transform(
        record,
        lambda gap: re.sub(r"\s+", " ", gap),
        family="stt_whitespace",
    )


def add_conversational_frame(
    record: Mapping[str, object], prefix: str, suffix: str = ""
) -> dict[str, object]:
    """Add realistic context and shift all entity offsets exactly once."""

    _assert_mutable(record)
    validate_record(record)
    prefix, suffix = str(prefix), str(suffix)
    result = _copy_metadata(record)
    result["source_text"] = prefix + str(record["source_text"]) + suffix
    result["privacy_mask"] = [
        {
            **dict(span),
            "start": int(span["start"]) + len(prefix),
            "end": int(span["end"]) + len(prefix),
        }
        for span in record["privacy_mask"]
    ]
    result["augmentation_family"] = "conversational_frame"
    result["parent_uid"] = record.get("parent_uid", record.get("uid"))
    validate_record(result)
    return result


def _speak_digits(value: str) -> str:
    return " ".join(_DIGIT_WORDS[int(char)] for char in value if char.isdigit())


def _speak_email(value: str) -> str:
    local, separator, domain = value.partition("@")
    if not separator:
        return value
    local = local.replace(".", " dot ").replace("_", " underscore ")
    local = local.replace("+", " plus ").replace("-", " dash ")
    domain = domain.replace(".", " dot ").replace("-", " dash ")
    return re.sub(r"\s+", " ", f"{local} at {domain}").strip()


def speak_structured_entities(record: Mapping[str, object]) -> dict[str, object]:
    """Create semantic spoken forms while retaining the original entity type."""

    replacements: dict[int, str] = {}
    for index, span in enumerate(record["privacy_mask"]):
        label, value = str(span["label"]), str(span["value"])
        if label == "EMAIL":
            replacements[index] = _speak_email(value)
        elif label in _STRUCTURED_TYPES and any(char.isdigit() for char in value):
            replacements[index] = _speak_digits(value)
    if not replacements:
        result = deepcopy(dict(record))
        result["augmentation_family"] = "spoken_structured_noop"
        return result
    return replace_spans(record, replacements, family="spoken_structured")


def load_json_resource(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_hard_negatives(
    templates: Iterable[Mapping[str, object]], *, seed: int
) -> list[dict[str, object]]:
    """Materialize realistic empty-mask rows in stable seeded order."""

    rows: list[dict[str, object]] = []
    for index, template in enumerate(templates):
        text, category = template.get("text"), template.get("category")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Hard-negative template {index} has no text.")
        if not isinstance(category, str) or not category:
            raise ValueError(f"Hard-negative template {index} has no category.")
        digest = hashlib.sha256(
            f"hard-negative|{seed}|{index}|{text}".encode("utf-8")
        ).hexdigest()[:16]
        row = {
            "uid": f"hard-negative-{digest}",
            "source_text": text,
            "privacy_mask": [],
            "language": "en",
            "source_dataset": "onkith_hard_negative_v1",
            "partition": "train_candidate",
            "augmentation_family": "hard_negative",
            "hard_negative_category": category,
            "generator_seed": int(seed),
        }
        validate_record(row)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["uid"]))

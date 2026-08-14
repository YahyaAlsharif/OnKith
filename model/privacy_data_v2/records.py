"""The one canonical Model V2 record shape, plus strict IO.

Three record shapes already exist in this repository:

* ``privacy_data_v2`` uses ``source_text`` / ``privacy_mask`` / ``label``;
* the ``model/data`` builders use ``text`` / ``spans`` / ``label``;
* the OOD fixtures use ``text`` / ``entities`` / ``entity_type``.

Model V2 standardises on the ``privacy_data_v2`` shape because
``augmentation.validate_record`` and ``grouped_splits`` already enforce it.
Thin adapters convert the other two shapes; nothing else is introduced.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:
    from .augmentation import PROTECTED_PARTITIONS, validate_record
    from .ontology import ENTITIES
except ImportError:  # Direct notebook/script execution from this directory.
    from augmentation import PROTECTED_PARTITIONS, validate_record
    from ontology import ENTITIES

#: Partitions the model may learn from.
TRAINABLE_PARTITIONS = frozenset({"train"})

#: ``PROTECTED_PARTITIONS`` is owned by ``augmentation`` so the mutation guard
#: and the record contract can never disagree about what is held out.

PARTITIONS = TRAINABLE_PARTITIONS | PROTECTED_PARTITIONS | {"candidate"}

GENERATION_TYPES = frozenset({"original", "replacement", "synthetic", "augmented"})

REQUIRED_FIELDS = ("uid", "source_text", "privacy_mask", "source_dataset", "partition")


def make_record(
    *,
    uid: str,
    source_text: str,
    privacy_mask: list[dict[str, Any]],
    source_dataset: str,
    partition: str,
    source_split: str | None = None,
    region: str | None = None,
    selection_reason: Iterable[str] = (),
    generation_type: str = "original",
    entity_source: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build one validated canonical record."""

    record = {
        "uid": str(uid),
        "source_text": str(source_text),
        "privacy_mask": [dict(span) for span in privacy_mask],
        "language": "en",
        "source_dataset": str(source_dataset),
        "source_split": source_split,
        "region": region,
        "selection_reason": sorted(set(selection_reason)),
        "partition": str(partition),
        "generation_type": str(generation_type),
        "entity_source": entity_source,
        **extra,
    }
    validate_model_v2_record(record)
    return record


def validate_model_v2_record(record: Mapping[str, Any]) -> None:
    """Validate spans, ontology membership, partition and generation type.

    ``augmentation.validate_record`` already proves exact, sorted,
    non-overlapping offsets. This adds the Model V2 specific contract.
    """

    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Record is missing required fields: {missing}")
    validate_record(record)

    supported = set(ENTITIES)
    for index, span in enumerate(record["privacy_mask"]):
        label = str(span["label"])
        if label not in supported:
            raise ValueError(
                f"privacy_mask[{index}] label {label!r} is outside the Model V2 "
                "ontology. Map it to a supported label or drop it to O first."
            )
    partition = str(record["partition"])
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition {partition!r}; expected one of {sorted(PARTITIONS)}.")
    generation = str(record.get("generation_type", "original"))
    if generation not in GENERATION_TYPES:
        raise ValueError(f"Unknown generation_type {generation!r}.")
    if partition in PROTECTED_PARTITIONS and generation == "augmented":
        raise ValueError(
            f"Partition {partition!r} is protected and must never hold augmented rows."
        )


def is_protected(record: Mapping[str, Any]) -> bool:
    return str(record.get("partition", "")) in PROTECTED_PARTITIONS or bool(
        record.get("frozen", False)
    )


def normalized_text(text: str) -> str:
    """The shared normalisation used by every dedup and contamination check."""

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def text_fingerprint(text: str) -> str:
    return hashlib.blake2b(normalized_text(text).encode("utf-8"), digest_size=16).hexdigest()


def labels_of(record: Mapping[str, Any]) -> set[str]:
    return {str(span["label"]) for span in record["privacy_mask"]}


# --------------------------------------------------------------------------
# Adapters from the other two repository record shapes
# --------------------------------------------------------------------------


def from_builder_row(
    row: Mapping[str, Any], *, partition: str, source_dataset: str | None = None
) -> dict[str, Any]:
    """Convert one ``model/data`` builder CSV row into a canonical record.

    Builder rows serialise ``spans`` and ``labels_present`` as JSON strings and
    carry their own ``final_split``. The caller owns splitting, so
    ``final_split`` is deliberately ignored and recorded only as provenance.
    """

    spans = row["spans"]
    if isinstance(spans, str):
        spans = json.loads(spans)
    mask = [
        {
            "label": str(span["label"]),
            "start": int(span["start"]),
            "end": int(span["end"]),
            "value": str(span["value"]),
        }
        for span in spans
    ]
    mask.sort(key=lambda span: (span["start"], span["end"]))
    return make_record(
        uid=str(row["id"]),
        source_text=str(row["text"]),
        privacy_mask=mask,
        source_dataset=str(source_dataset or row.get("source_dataset") or "unknown"),
        partition=partition,
        source_split=str(row.get("source_split") or "") or None,
        generation_type=str(row.get("generation_type") or "original"),
        entity_source=str(row.get("entity_source") or "") or None,
        builder_final_split=str(row.get("final_split") or "") or None,
    )


def from_fixture_case(case: Mapping[str, Any], *, partition: str) -> dict[str, Any]:
    """Convert one OOD fixture case into a canonical record (read-only use).

    Fixture cases are evaluation-only. The resulting record is marked frozen and
    protected so it can never reach a mutation or split entry point.
    """

    mask = sorted(
        (
            {
                "label": str(entity["entity_type"]),
                "start": int(entity["start"]),
                "end": int(entity["end"]),
                "value": str(entity["text"]),
            }
            for entity in case.get("entities", ())
        ),
        key=lambda span: (span["start"], span["end"]),
    )
    record = {
        "uid": str(case["case_id"]),
        "source_text": str(case["text"]),
        "privacy_mask": mask,
        "language": "en",
        "source_dataset": "onkith_ood_fixture",
        "source_split": str(case.get("partition") or ""),
        "region": None,
        "selection_reason": [],
        "partition": partition,
        "generation_type": "original",
        "entity_source": None,
        "frozen": partition == "ood_final",
        "category": case.get("category"),
        "subcategory": case.get("subcategory"),
        "tags": list(case.get("tags", ())),
    }
    validate_model_v2_record(record)
    return record


# --------------------------------------------------------------------------
# JSONL IO
# --------------------------------------------------------------------------


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Write records atomically and return a compact integrity summary."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows = spans = 0
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            validate_model_v2_record(record)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
            rows += 1
            spans += len(record["privacy_mask"])
    temporary.replace(path)
    return {
        "path": str(path),
        "rows": rows,
        "spans": spans,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path, *, validate: bool = True) -> list[dict[str, Any]]:
    records = list(read_jsonl(path))
    if validate:
        for record in records:
            validate_model_v2_record(record)
    return records

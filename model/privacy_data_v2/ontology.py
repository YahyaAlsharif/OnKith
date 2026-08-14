"""The canonical OnKith Model V2 label ontology.

This module is the single source of truth. Every builder, split, augmentation,
manifest, notebook cell, evaluator and exported model config reads its label
space from here so the ontology cannot drift between components.

Model V2 changes the historical TinyBERT head deliberately:

* ``ALLERGIES`` and ``USERNAME`` are removed. Both were unsupported in the
  pinned training distribution (3 and 16 spans in 147,366 rows).
* ``WEIGHT`` is removed for the same practical reason as ``HEIGHT``: a single
  training span, no external source that maps to it, and too little product
  value to maintain for this release.
* ``HEIGHT`` was never present in the TinyBERT head and stays out.
* ``PASSWORD`` is added. It is the only new Model V2 entity, and it is admitted
  under the conversational credential-disclosure rules in ``password_rules``.

The BIO ordering matches the historical convention exactly: ``O`` first, then
alphabetical entities with ``B-`` immediately before ``I-``.
"""

from __future__ import annotations

MODEL_V2_VERSION = "model_v2.ontology.1"

#: The 33 entity types carried by the TinyBERT-4 / DeBERTa-preflight 67-label head.
TINYBERT_ENTITIES: tuple[str, ...] = (
    "ACCOUNTNUM", "AGE", "ALLERGIES", "AMOUNT", "BANKNAME", "BUILDINGNUM",
    "CITY", "COUNTRY", "CREDITCARDNUMBER", "CURRENCY", "DATE",
    "DRIVERLICENSENUM", "EMAIL", "GENDER", "GIVENNAME", "IDCARDNUM", "IPV4",
    "ORGANISATION", "PASSPORTNUM", "SALARY", "SEX", "SOCIALNUM", "STREET",
    "SURNAME", "TAXNUM", "TELEPHONENUM", "TIME", "TIMEZONE", "TITLE", "URL",
    "USERNAME", "WEIGHT", "ZIPCODE",
)

#: Entities deliberately dropped from Model V2, with the reason recorded.
REMOVED_ENTITIES: dict[str, str] = {
    "ALLERGIES": "3 spans in the pinned project-train split; no supported source.",
    "USERNAME": "16 spans in the pinned project-train split; not an OnKith product need.",
    "WEIGHT": "1 span in the pinned project-train split; no source maps to it.",
}

#: Concepts that remain absent but were never part of the 67-label TinyBERT contract.
ABSENT_ENTITIES: dict[str, str] = {
    "HEIGHT": "Never present in the TinyBERT head; 1 span exists only in official validation.",
}

#: Entities added by Model V2, with the reason recorded.
ADDED_ENTITIES: dict[str, str] = {
    "PASSWORD": (
        "Protects disclosed credentials in conversational text. Admitted only "
        "through the credential-disclosure rules in password_rules.py."
    ),
}

#: The canonical Model V2 entity set: 31 types, sorted.
ENTITIES: tuple[str, ...] = tuple(
    sorted((set(TINYBERT_ENTITIES) - set(REMOVED_ENTITIES)) | set(ADDED_ENTITIES))
)

#: The canonical BIO label list: 63 labels, index-stable.
BIO_LABELS: tuple[str, ...] = ("O",) + tuple(
    f"{prefix}-{entity}" for entity in ENTITIES for prefix in ("B", "I")
)

LABEL2ID: dict[str, int] = {label: index for index, label in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {index: label for label, index in LABEL2ID.items()}

NUM_ENTITIES = len(ENTITIES)
NUM_LABELS = len(BIO_LABELS)

#: Labels shared with the TinyBERT head. Only these support a like-for-like
#: TinyBERT/DeBERTa comparison; aggregate numbers over different label spaces
#: are misleading and must never be reported as a single figure.
SHARED_WITH_TINYBERT: tuple[str, ...] = tuple(
    sorted(set(ENTITIES) & set(TINYBERT_ENTITIES))
)

#: Span support measured on the pinned project-train split. Source:
#: reports/current_dataset_audit.json (ai4privacy/pii-masking-openpii-1.5m
#: @ a785eb528e28be2693c3718a27e066970de5dadb, 147,366 English rows).
#: ``PASSWORD`` is 0 because AI4Privacy English contains no PASSWORD span at
#: all; it exists only in the supplementary sources.
BASELINE_TRAIN_SPAN_COUNTS: dict[str, int] = {
    "ACCOUNTNUM": 1, "AGE": 54531, "AMOUNT": 14, "BANKNAME": 4,
    "BUILDINGNUM": 52686, "CITY": 77306, "COUNTRY": 9,
    "CREDITCARDNUMBER": 34092, "CURRENCY": 15, "DATE": 134599,
    "DRIVERLICENSENUM": 31200, "EMAIL": 82477, "GENDER": 30361,
    "GIVENNAME": 135626, "IDCARDNUM": 36657, "IPV4": 2, "ORGANISATION": 18,
    "PASSPORTNUM": 22723, "PASSWORD": 0, "SALARY": 2, "SEX": 21483,
    "SOCIALNUM": 24539, "STREET": 55219, "SURNAME": 115723, "TAXNUM": 27777,
    "TELEPHONENUM": 60254, "TIME": 1848, "TIMEZONE": 2, "TITLE": 76206,
    "URL": 4, "ZIPCODE": 47289,
}

#: A label is weak when the pinned training distribution cannot teach it.
WEAK_LABEL_THRESHOLD = 100

#: Labels that supplementary data must strengthen by volume.
WEAK_LABELS: tuple[str, ...] = tuple(
    entity
    for entity in ENTITIES
    if BASELINE_TRAIN_SPAN_COUNTS[entity] < WEAK_LABEL_THRESHOLD
)

#: Labels with ample training volume but poor frozen-OOD behaviour. Source:
#: privacy_evaluation/OOD_BASELINE_FINDINGS.md (combined production, final
#: partition). These need surface/context diversity, not more rows.
OOD_WEAK_LABELS: dict[str, float] = {
    "ORGANISATION": 0.000,
    "IDCARDNUM": 0.111,
    "ZIPCODE": 0.118,
    "TAXNUM": 0.143,
    "DRIVERLICENSENUM": 0.143,
    "DATE": 0.275,
}

#: Structured identifier types, used by span-safe spoken/format augmentation.
STRUCTURED_ENTITIES: frozenset[str] = frozenset(
    {
        "ACCOUNTNUM", "CREDITCARDNUMBER", "DRIVERLICENSENUM", "IDCARDNUM",
        "PASSPORTNUM", "SOCIALNUM", "TAXNUM", "TELEPHONENUM", "ZIPCODE",
    }
) & frozenset(ENTITIES)


def is_supported(label: str) -> bool:
    """Return whether ``label`` is a Model V2 entity type (not a BIO tag)."""

    return label in set(ENTITIES)


def bio_tags(entity: str) -> tuple[str, str]:
    if not is_supported(entity):
        raise KeyError(f"{entity!r} is not a Model V2 entity type.")
    return f"B-{entity}", f"I-{entity}"


def label_map_payload() -> dict[str, object]:
    """The committed, human-reviewable ontology document."""

    return {
        "schema_version": 1,
        "ontology_version": MODEL_V2_VERSION,
        "num_entities": NUM_ENTITIES,
        "num_labels": NUM_LABELS,
        "entities": list(ENTITIES),
        "bio_labels": list(BIO_LABELS),
        "label2id": dict(LABEL2ID),
        "id2label": {str(index): label for index, label in ID2LABEL.items()},
        "derivation": {
            "base": "TinyBERT-4 / DeBERTa-preflight 67-label head (33 entities)",
            "removed": dict(REMOVED_ENTITIES),
            "absent": dict(ABSENT_ENTITIES),
            "added": dict(ADDED_ENTITIES),
            "note": (
                "HEIGHT was never in the 67-label head, so removing "
                "ALLERGIES, USERNAME and WEIGHT nets 33 - 3 = 30 entities; "
                "adding PASSWORD gives 31 entities and 63 BIO labels."
            ),
        },
        "shared_with_tinybert": list(SHARED_WITH_TINYBERT),
        "weak_labels": list(WEAK_LABELS),
        "ood_weak_labels": dict(OOD_WEAK_LABELS),
        "baseline_train_span_counts": dict(BASELINE_TRAIN_SPAN_COUNTS),
    }


def _self_check() -> None:
    """Fail fast at import time; a drifted ontology must never load quietly."""

    expected = (set(TINYBERT_ENTITIES) - {"ALLERGIES", "USERNAME", "WEIGHT"}) | {"PASSWORD"}
    if set(ENTITIES) != expected:
        raise RuntimeError("Model V2 entity set does not match its stated derivation.")
    if NUM_ENTITIES != 31 or NUM_LABELS != 63:
        raise RuntimeError(f"Expected 31 entities and 63 labels, got {NUM_ENTITIES}/{NUM_LABELS}.")
    if BIO_LABELS[0] != "O" or len(set(BIO_LABELS)) != NUM_LABELS:
        raise RuntimeError("BIO label list is malformed.")
    if set(BASELINE_TRAIN_SPAN_COUNTS) != set(ENTITIES):
        raise RuntimeError("Baseline span counts do not cover exactly the Model V2 entities.")
    if set(OOD_WEAK_LABELS) - set(ENTITIES):
        raise RuntimeError("OOD weak-label table references an unsupported entity.")


_self_check()

"""Auditable data-preparation helpers for the future OnKith privacy model."""

from .augmentation import (
    FROZEN_FINAL_PARTITION,
    add_conversational_frame,
    apply_case_variant,
    apply_gap_transform,
    build_email_values,
    diversify_entity_values,
    generate_hard_negatives,
    replace_spans,
    speak_structured_entities,
    validate_record,
)
from .data_audit import audit_records
from .grouped_splits import audit_factor_overlap, split_by_factor

__all__ = [
    "FROZEN_FINAL_PARTITION",
    "add_conversational_frame",
    "apply_case_variant",
    "apply_gap_transform",
    "audit_factor_overlap",
    "audit_records",
    "build_email_values",
    "diversify_entity_values",
    "generate_hard_negatives",
    "replace_spans",
    "speak_structured_entities",
    "split_by_factor",
    "validate_record",
]

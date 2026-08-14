"""Span metrics for dataset-shaped evaluation sets.

``metrics.evaluate_predictions`` scores the OOD fixtures, which carry category
and tag structure. The Model V2 training-side test and the two validation sets
are plain annotated records, so they need the same *definitions* with a simpler
input shape. Keeping the definitions identical is what makes the six evaluation
views comparable:

* **typed entity F1** - exact boundaries and the correct entity type;
* **binary entity F1** - exact boundaries, type ignored;
* **private-character recall** - the share of gold PII characters covered;
* **leakage** - ``1 - private-character recall``;
* **non-private overmasking** - predicted characters outside any gold span;
* **hard-negative false-positive rate** - the share of no-PII rows that received
  any prediction at all.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

Span = tuple[int, int, str]


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _typed(spans: Iterable[Mapping[str, Any]]) -> set[Span]:
    return {(int(s["start"]), int(s["end"]), str(s["label"])) for s in spans}


def _untyped(spans: Iterable[Mapping[str, Any]]) -> set[tuple[int, int]]:
    return {(int(s["start"]), int(s["end"])) for s in spans}


def _characters(spans: Iterable[Mapping[str, Any]]) -> set[int]:
    covered: set[int] = set()
    for span in spans:
        covered.update(range(int(span["start"]), int(span["end"])))
    return covered


def _prf(true_positives: int, predicted: int, gold: int) -> dict[str, float]:
    precision = _safe_div(true_positives, predicted)
    recall = _safe_div(true_positives, gold)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "support": int(gold),
        "predicted": int(predicted),
        "true_positives": int(true_positives),
    }


def evaluate_spans(
    gold_records: Sequence[Mapping[str, Any]],
    predicted_masks: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Score aligned gold records against predicted spans.

    ``predicted_masks[i]`` holds the spans predicted for ``gold_records[i]``,
    each a mapping with ``start``, ``end`` and ``label``.
    """

    if len(gold_records) != len(predicted_masks):
        raise ValueError("gold_records and predicted_masks must be the same length.")

    typed_tp = typed_pred = typed_gold = 0
    binary_tp = binary_pred = binary_gold = 0
    private_total = private_covered = 0
    predicted_characters = overmasked_characters = 0
    hard_negative_rows = hard_negative_hits = 0
    rows_with_any_gold = 0

    per_label_tp: defaultdict[str, int] = defaultdict(int)
    per_label_pred: defaultdict[str, int] = defaultdict(int)
    per_label_gold: defaultdict[str, int] = defaultdict(int)
    confusions: defaultdict[tuple[str, str], int] = defaultdict(int)

    for record, predicted in zip(gold_records, predicted_masks):
        gold_spans = list(record["privacy_mask"])
        gold_typed, predicted_typed = _typed(gold_spans), _typed(predicted)
        gold_binary, predicted_binary = _untyped(gold_spans), _untyped(predicted)

        typed_tp += len(gold_typed & predicted_typed)
        typed_pred += len(predicted_typed)
        typed_gold += len(gold_typed)
        binary_tp += len(gold_binary & predicted_binary)
        binary_pred += len(predicted_binary)
        binary_gold += len(gold_binary)

        for start, end, label in gold_typed:
            per_label_gold[label] += 1
            if (start, end, label) in predicted_typed:
                per_label_tp[label] += 1
        for start, end, label in predicted_typed:
            per_label_pred[label] += 1

        # Boundary-correct but type-wrong predictions, for the confusion table.
        gold_by_bounds = {(s, e): label for s, e, label in gold_typed}
        for start, end, label in predicted_typed:
            expected = gold_by_bounds.get((start, end))
            if expected is not None and expected != label:
                confusions[(expected, label)] += 1

        gold_characters = _characters(gold_spans)
        predicted_char_set = _characters(predicted)
        private_total += len(gold_characters)
        private_covered += len(gold_characters & predicted_char_set)
        predicted_characters += len(predicted_char_set)
        overmasked_characters += len(predicted_char_set - gold_characters)

        if gold_spans:
            rows_with_any_gold += 1
        else:
            hard_negative_rows += 1
            hard_negative_hits += int(bool(predicted))

    character_recall = _safe_div(private_covered, private_total)
    return {
        "rows": len(gold_records),
        "rows_with_pii": rows_with_any_gold,
        "typed_entity": _prf(typed_tp, typed_pred, typed_gold),
        "binary_entity": _prf(binary_tp, binary_pred, binary_gold),
        "private_character_recall": character_recall,
        "private_character_leakage": 1.0 - character_recall,
        "private_characters": private_total,
        "non_private_overmasking_rate": _safe_div(
            overmasked_characters, predicted_characters
        ),
        "overmasked_characters": overmasked_characters,
        "hard_negative_rows": hard_negative_rows,
        "hard_negative_false_positive_rate": _safe_div(
            hard_negative_hits, hard_negative_rows
        ),
        "per_label": {
            label: _prf(per_label_tp[label], per_label_pred[label], per_label_gold[label])
            for label in sorted(set(per_label_gold) | set(per_label_pred))
        },
        "boundary_correct_type_confusions": {
            f"{expected}->{predicted}": count
            for (expected, predicted), count in sorted(
                confusions.items(), key=lambda item: -item[1]
            )
        },
    }


def restrict_to_labels(
    records: Sequence[Mapping[str, Any]], labels: Iterable[str]
) -> list[dict[str, Any]]:
    """Drop spans outside ``labels`` so two label spaces can be compared fairly.

    Model V2 removes ``WEIGHT`` and adds ``PASSWORD``, so an aggregate TinyBERT
    comparison over the union would be meaningless. Restricting both sides to
    the shared ontology first is the only honest way to report one number.
    """

    allowed = set(labels)
    output = []
    for record in records:
        clipped = dict(record)
        clipped["privacy_mask"] = [
            span for span in record["privacy_mask"] if str(span["label"]) in allowed
        ]
        output.append(clipped)
    return output


def subset_by(
    records: Sequence[Mapping[str, Any]],
    predicted_masks: Sequence[Sequence[Mapping[str, Any]]],
    key: str,
) -> dict[str, dict[str, Any]]:
    """Evaluate separately per value of a record field, e.g. ``source_dataset``."""

    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[str(record.get(key, "unknown"))].append(index)
    return {
        name: evaluate_spans(
            [records[index] for index in indices],
            [predicted_masks[index] for index in indices],
        )
        for name, indices in sorted(grouped.items())
    }


def password_analysis(
    records: Sequence[Mapping[str, Any]],
    predicted_masks: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Separate credential disclosures from password-related hard negatives.

    ``PASSWORD`` is the one new Model V2 label, and its whole purpose is to fire
    on a disclosed secret while staying silent on advice about passwords. A
    single F1 hides that distinction, so both halves are reported.
    """

    disclosure_rows = [
        index
        for index, record in enumerate(records)
        if any(str(span["label"]) == "PASSWORD" for span in record["privacy_mask"])
    ]
    negative_rows = [
        index
        for index, record in enumerate(records)
        if not record["privacy_mask"]
        and "password" in str(record["source_text"]).casefold()
    ]

    disclosure = (
        evaluate_spans(
            [records[index] for index in disclosure_rows],
            [predicted_masks[index] for index in disclosure_rows],
        )
        if disclosure_rows
        else None
    )
    false_positives = sum(
        1
        for index in negative_rows
        if any(str(span["label"]) == "PASSWORD" for span in predicted_masks[index])
    )
    return {
        "disclosure_rows": len(disclosure_rows),
        "disclosure_metrics": disclosure,
        "password_hard_negative_rows": len(negative_rows),
        "password_hard_negative_false_positives": false_positives,
        "password_hard_negative_false_positive_rate": _safe_div(
            false_positives, len(negative_rows)
        ),
        "interpretation": (
            "disclosure_metrics measures whether real credentials are caught; the "
            "hard-negative rate measures whether talking about passwords wrongly "
            "triggers the label. Both must be good for PASSWORD to be useful."
        ),
    }

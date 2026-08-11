"""Span- and character-based privacy metrics for IID and OOD comparisons."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from .schema import EntitySpan, EvalCase


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _predicted_entities(record: dict, text: str) -> list[EntitySpan]:
    entities: list[EntitySpan] = []
    for item in record.get("entities", []):
        start = int(item.get("start", item.get("start_char", -1)))
        end = int(item.get("end", item.get("end_char", -1)))
        if start < 0 or end <= start or end > len(text):
            continue
        entities.append(
            EntitySpan(
                entity_type=str(item["entity_type"]),
                start=start,
                end=end,
                text=text[start:end],
            )
        )
    return entities


def _accumulate(cases: Iterable[EvalCase], predictions: dict[str, dict]) -> dict:
    typed_tp = typed_fp = typed_fn = 0
    binary_tp = binary_fp = binary_fn = 0
    private_chars = masked_private = non_private_chars = masked_non_private = 0
    positive_cases = predicted_positive_cases = detected_positive_cases = 0
    hard_negative_cases = hard_negative_fp_cases = hard_negative_fp_spans = 0
    exact_cases = 0
    total_cases = 0
    per_type: dict[str, Counter] = defaultdict(Counter)
    errors: list[dict] = []
    bio_repairs = 0
    blocked_cases = 0
    latencies: list[float] = []

    for case in cases:
        total_cases += 1
        record = predictions.get(case.case_id, {"entities": [], "missing_prediction": True})
        predicted = _predicted_entities(record, case.text)
        safe_to_release = bool(record.get("safe_to_release", True))
        if not safe_to_release:
            blocked_cases += 1
        latency = record.get("latency_seconds")
        if latency is not None:
            latencies.append(float(latency))
        bio_repairs += int((record.get("raw_model_output") or {}).get("bio_repairs", 0))

        gold_typed = {(item.start, item.end, item.entity_type) for item in case.entities}
        pred_typed = {(item.start, item.end, item.entity_type) for item in predicted}
        gold_binary = {(item.start, item.end) for item in case.entities}
        pred_binary = {(item.start, item.end) for item in predicted}
        typed_tp += len(gold_typed & pred_typed)
        typed_fp += len(pred_typed - gold_typed)
        typed_fn += len(gold_typed - pred_typed)
        binary_tp += len(gold_binary & pred_binary)
        binary_fp += len(pred_binary - gold_binary)
        binary_fn += len(gold_binary - pred_binary)
        exact_cases += int(gold_typed == pred_typed)

        all_types = {item[2] for item in gold_typed | pred_typed}
        for entity_type in all_types:
            gold_for_type = {item for item in gold_typed if item[2] == entity_type}
            pred_for_type = {item for item in pred_typed if item[2] == entity_type}
            per_type[entity_type]["tp"] += len(gold_for_type & pred_for_type)
            per_type[entity_type]["fp"] += len(pred_for_type - gold_for_type)
            per_type[entity_type]["fn"] += len(gold_for_type - pred_for_type)
            per_type[entity_type]["support"] += len(gold_for_type)

        gold_mask = bytearray(len(case.text))
        pred_mask = bytearray(len(case.text))
        for entity in case.entities:
            gold_mask[entity.start : entity.end] = b"\x01" * (entity.end - entity.start)
        for entity in predicted:
            pred_mask[entity.start : entity.end] = b"\x01" * (entity.end - entity.start)
        if not safe_to_release:
            # Fail-closed output releases no source characters. Treat the whole
            # text as protected for character privacy while keeping typed-span
            # metrics honest: a block is not an entity prediction.
            pred_mask[:] = b"\x01" * len(case.text)
        for gold, pred in zip(gold_mask, pred_mask):
            if gold:
                private_chars += 1
                masked_private += int(bool(pred))
            else:
                non_private_chars += 1
                masked_non_private += int(bool(pred))

        if case.entities:
            positive_cases += 1
            predicted_positive_cases += int(bool(predicted))
            detected_positive_cases += int(any(gold_mask[i] and pred_mask[i] for i in range(len(case.text))))
        else:
            hard_negative_cases += 1
            if predicted or not safe_to_release:
                hard_negative_fp_cases += 1
                hard_negative_fp_spans += len(predicted)

        if gold_typed != pred_typed:
            errors.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "subcategory": case.subcategory,
                    "tags": list(case.tags),
                    "false_negatives": [
                        {"start": start, "end": end, "entity_type": entity_type, "text": case.text[start:end]}
                        for start, end, entity_type in sorted(gold_typed - pred_typed)
                    ],
                    "false_positives": [
                        {"start": start, "end": end, "entity_type": entity_type, "text": case.text[start:end]}
                        for start, end, entity_type in sorted(pred_typed - gold_typed)
                    ],
                }
            )

    typed_precision = _safe_div(typed_tp, typed_tp + typed_fp)
    typed_recall = _safe_div(typed_tp, typed_tp + typed_fn)
    binary_precision = _safe_div(binary_tp, binary_tp + binary_fp)
    binary_recall = _safe_div(binary_tp, binary_tp + binary_fn)
    per_entity_type = {}
    for entity_type, counts in sorted(per_type.items()):
        precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
        per_entity_type[entity_type] = {
            **dict(counts),
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    latencies.sort()
    return {
        "case_count": total_cases,
        "typed_entity": {
            "tp": typed_tp, "fp": typed_fp, "fn": typed_fn,
            "precision": typed_precision,
            "recall": typed_recall,
            "f1": _f1(typed_precision, typed_recall),
        },
        "binary_entity_exact_boundary": {
            "tp": binary_tp, "fp": binary_fp, "fn": binary_fn,
            "precision": binary_precision,
            "recall": binary_recall,
            "f1": _f1(binary_precision, binary_recall),
        },
        "privacy_character": {
            "private_characters": private_chars,
            "masked_private_characters": masked_private,
            "private_character_recall": _safe_div(masked_private, private_chars),
            "private_character_leakage_rate": 1.0 - _safe_div(masked_private, private_chars) if private_chars else 0.0,
            "non_private_characters": non_private_chars,
            "masked_non_private_characters": masked_non_private,
            "non_private_character_overmasking_rate": _safe_div(masked_non_private, non_private_chars),
        },
        "case_level": {
            "exact_typed_case_rate": _safe_div(exact_cases, total_cases),
            "positive_cases": positive_cases,
            "predicted_positive_cases": predicted_positive_cases,
            "positive_case_detection_recall": _safe_div(detected_positive_cases, positive_cases),
            "hard_negative_cases": hard_negative_cases,
            "hard_negative_false_positive_cases": hard_negative_fp_cases,
            "hard_negative_false_positive_case_rate": _safe_div(hard_negative_fp_cases, hard_negative_cases),
            "hard_negative_false_positive_spans": hard_negative_fp_spans,
            "blocked_cases": blocked_cases,
        },
        "per_entity_type": per_entity_type,
        "bio_repairs": bio_repairs,
        "latency_seconds": {
            "count": len(latencies),
            "median": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
        },
        "error_count": len(errors),
        "errors": errors,
    }


def evaluate_predictions(cases: list[EvalCase], predictions: dict[str, dict]) -> dict:
    """Return overall, category, subcategory, and tag slices with one metric contract."""

    missing = sorted({case.case_id for case in cases} - set(predictions))
    extra = sorted(set(predictions) - {case.case_id for case in cases})
    if missing or extra:
        raise ValueError(f"Prediction keys differ from fixtures: missing={missing[:3]}, extra={extra[:3]}")
    overall = _accumulate(cases, predictions)
    errors = overall.pop("errors")
    by_category = {
        value: _without_errors(_accumulate([case for case in cases if case.category == value], predictions))
        for value in sorted({case.category for case in cases})
    }
    subcategory_keys = sorted({(case.category, case.subcategory) for case in cases})
    by_subcategory = {
        f"{category}/{subcategory}": _without_errors(
            _accumulate(
                [
                    case for case in cases
                    if case.category == category and case.subcategory == subcategory
                ],
                predictions,
            )
        )
        for category, subcategory in subcategory_keys
    }
    by_tag = {
        value: _without_errors(_accumulate([case for case in cases if value in case.tags], predictions))
        for value in sorted({tag for case in cases for tag in case.tags})
    }
    return {
        "overall": overall,
        "by_category": by_category,
        "by_subcategory": by_subcategory,
        "by_tag": by_tag,
        "errors": errors,
    }


def _without_errors(metrics: dict) -> dict:
    metrics.pop("errors", None)
    return metrics


def controlled_pair_analysis(cases: list[EvalCase], predictions: dict[str, dict]) -> dict:
    """Expose detection changes within deliberately related variant groups."""

    groups: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        if case.pair_id:
            groups[case.pair_id].append(case)
    rows = []
    for pair_id, members in sorted(groups.items()):
        outcomes = {}
        for case in sorted(members, key=lambda item: item.subcategory):
            predicted = _predicted_entities(predictions[case.case_id], case.text)
            gold_mask = {(entity.start, entity.end) for entity in case.entities}
            pred_mask = {(entity.start, entity.end) for entity in predicted}
            outcomes[case.subcategory] = {
                "typed_exact": {
                    (entity.start, entity.end, entity.entity_type) for entity in case.entities
                } == {
                    (entity.start, entity.end, entity.entity_type) for entity in predicted
                },
                "any_gold_boundary_detected": bool(gold_mask & pred_mask),
            }
        rows.append({"pair_id": pair_id, "outcomes": outcomes})
    return {"pair_count": len(rows), "pairs": rows}

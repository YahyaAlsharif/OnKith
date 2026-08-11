from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from augmentation import (  # noqa: E402
    add_conversational_frame,
    apply_case_variant,
    build_email_values,
    diversify_entity_values,
    generate_hard_negatives,
    normalized_whitespace,
    punctuation_free,
    replace_spans,
    speak_structured_entities,
    validate_record,
)


def example_record() -> dict:
    text = "Email Ana at ana.silva@example.com, account 123-45."
    return {
        "uid": 42,
        "source_text": text,
        "privacy_mask": [
            {"label": "GIVENNAME", "start": 6, "end": 9, "value": "Ana"},
            {
                "label": "EMAIL",
                "start": 13,
                "end": 34,
                "value": "ana.silva@example.com",
            },
            {"label": "ACCOUNTNUM", "start": 44, "end": 50, "value": "123-45"},
        ],
        "partition": "train_candidate",
    }


class SpanAwareAugmentationTests(unittest.TestCase):
    def test_replacement_recalculates_all_downstream_offsets(self) -> None:
        result = replace_spans(
            example_record(),
            {0: "Ana-María", 1: "ana+lab@subunit.example.cloud"},
            family="test",
        )
        validate_record(result)
        self.assertIn("Ana-María", result["source_text"])
        self.assertEqual(
            result["source_text"][result["privacy_mask"][2]["start"] : result["privacy_mask"][2]["end"]],
            "123-45",
        )

    def test_diversification_is_deterministic(self) -> None:
        inventory = {"GIVENNAME": ["Aiko", "Noura", "Yuna"]}
        first = diversify_entity_values(example_record(), inventory, seed=42)
        second = diversify_entity_values(example_record(), inventory, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["generator_seed"], 42)

    def test_email_inventory_crosses_local_parts_and_unseen_domains(self) -> None:
        values = build_email_values(
            ["first+receipts", "x9.y_test"],
            ["example.travel", "subunit.example.cloud"],
        )
        self.assertEqual(len(values), 4)
        self.assertIn("first+receipts@example.travel", values)
        self.assertIn("x9.y_test@subunit.example.cloud", values)
        with self.assertRaisesRegex(ValueError, "local part"):
            build_email_values(["bad local"], ["example.org"])

    def test_case_variants_preserve_valid_offsets(self) -> None:
        for variant in ("lower", "upper", "mixed"):
            with self.subTest(variant=variant):
                validate_record(apply_case_variant(example_record(), variant))

    def test_gap_transforms_do_not_damage_email_punctuation(self) -> None:
        noisy = example_record()
        noisy["source_text"] = noisy["source_text"].replace("Email ", "Email...  ")
        shift = len("Email...  ") - len("Email ")
        for span in noisy["privacy_mask"]:
            span["start"] += shift
            span["end"] += shift
        punct = punctuation_free(noisy)
        spaces = normalized_whitespace(noisy)
        self.assertIn("ana.silva@example.com", punct["source_text"])
        validate_record(punct)
        validate_record(spaces)

    def test_spoken_forms_keep_semantic_labels(self) -> None:
        result = speak_structured_entities(example_record())
        self.assertIn("ana dot silva at example dot com", result["source_text"])
        self.assertIn("one two three four five", result["source_text"])
        self.assertEqual(
            [span["label"] for span in result["privacy_mask"]],
            ["GIVENNAME", "EMAIL", "ACCOUNTNUM"],
        )
        validate_record(result)

    def test_conversational_frame_shifts_offsets(self) -> None:
        result = add_conversational_frame(example_record(), "um ", " thanks")
        self.assertEqual(result["privacy_mask"][0]["start"], 9)
        validate_record(result)

    def test_frozen_final_isolation(self) -> None:
        record = example_record()
        record["partition"] = "ood_final"
        with self.assertRaisesRegex(ValueError, "Frozen OOD-final"):
            apply_case_variant(record, "lower")

    def test_hard_negatives_are_realistic_empty_rows_and_deterministic(self) -> None:
        templates = json.loads(
            (ROOT / "resources" / "hard_negative_templates.json").read_text(encoding="utf-8")
        )
        first = generate_hard_negatives(templates, seed=42)
        second = generate_hard_negatives(templates, seed=42)
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 40)
        self.assertTrue(all(row["privacy_mask"] == [] for row in first))
        self.assertGreaterEqual(len({row["hard_negative_category"] for row in first}), 15)


if __name__ == "__main__":
    unittest.main()

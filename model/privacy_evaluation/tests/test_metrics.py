from __future__ import annotations

import sys
import unittest
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from privacy_eval.metrics import evaluate_predictions
from privacy_eval.schema import EntitySpan, EvalCase


def case(case_id, text, entities, category="positive"):
    return EvalCase(case_id, "dev", category, category, f"dev-{case_id}", text, tuple(entities), ())


class MetricsTests(unittest.TestCase):
    def test_exact_typed_match(self):
        item = case("one", "call Amal", [EntitySpan("GIVENNAME", 5, 9, "Amal")])
        result = evaluate_predictions([item], {"one": {"entities": [{"start": 5, "end": 9, "entity_type": "GIVENNAME"}]}})
        self.assertEqual(result["overall"]["typed_entity"]["f1"], 1.0)
        self.assertEqual(result["overall"]["privacy_character"]["private_character_recall"], 1.0)

    def test_wrong_type_preserves_binary_privacy_but_fails_typed_metric(self):
        item = case("one", "call Amal", [EntitySpan("GIVENNAME", 5, 9, "Amal")])
        result = evaluate_predictions([item], {"one": {"entities": [{"start": 5, "end": 9, "entity_type": "CITY"}]}})
        self.assertEqual(result["overall"]["typed_entity"]["f1"], 0.0)
        self.assertEqual(result["overall"]["binary_entity_exact_boundary"]["f1"], 1.0)
        self.assertEqual(result["overall"]["privacy_character"]["private_character_leakage_rate"], 0.0)

    def test_partial_span_counts_leakage_and_overmasking(self):
        item = case("one", "xx secret yy", [EntitySpan("IDCARDNUM", 3, 9, "secret")])
        result = evaluate_predictions([item], {"one": {"entities": [{"start": 2, "end": 7, "entity_type": "IDCARDNUM"}]}})
        privacy = result["overall"]["privacy_character"]
        self.assertAlmostEqual(privacy["private_character_recall"], 4 / 6)
        self.assertGreater(privacy["non_private_character_overmasking_rate"], 0.0)

    def test_hard_negative_false_positive_rate(self):
        items = [case("a", "clean one", [], "hard_negatives"), case("b", "clean two", [], "hard_negatives")]
        predictions = {
            "a": {"entities": [{"start": 0, "end": 5, "entity_type": "GIVENNAME"}]},
            "b": {"entities": []},
        }
        result = evaluate_predictions(items, predictions)
        self.assertEqual(result["overall"]["case_level"]["hard_negative_false_positive_case_rate"], 0.5)

    def test_prediction_key_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([case("a", "clean", [])], {})

    def test_subcategory_slices_are_namespaced_by_primary_category(self):
        first = case("a", "clean", [], "hard_negatives")
        second = EvalCase("b", "dev", "email", "hard_negatives", "dev-b", "clean", (), ())
        result = evaluate_predictions([first, second], {"a": {"entities": []}, "b": {"entities": []}})
        self.assertIn("hard_negatives/hard_negatives", result["by_subcategory"])
        self.assertIn("email/hard_negatives", result["by_subcategory"])

    def test_fail_closed_block_has_full_privacy_recall_without_fake_entities(self):
        item = case("one", "call Amal", [EntitySpan("GIVENNAME", 5, 9, "Amal")])
        result = evaluate_predictions([item], {"one": {"entities": [], "safe_to_release": False}})
        self.assertEqual(result["overall"]["typed_entity"]["recall"], 0.0)
        self.assertEqual(result["overall"]["privacy_character"]["private_character_recall"], 1.0)
        self.assertEqual(result["overall"]["case_level"]["blocked_cases"], 1)


if __name__ == "__main__":
    unittest.main()

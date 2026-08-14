"""Dataset-shaped metrics, the model-agnostic adapter, and the Model V2 notebook."""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "model" / "privacy_data_v2"))

from privacy_eval.dataset_metrics import (  # noqa: E402
    evaluate_spans,
    password_analysis,
    restrict_to_labels,
    subset_by,
)
from privacy_eval.model_v2_adapter import ModelV2OnnxMasker, repair_bio  # noqa: E402

NOTEBOOK = REPO_ROOT / "model" / "DeBERTa-v3-xsmall" / "model_v2_deberta_v3_xsmall.ipynb"


def record(text, spans, **extra):
    return {
        "source_text": text,
        "privacy_mask": [
            {"label": label, "start": text.index(value),
             "end": text.index(value) + len(value), "value": value}
            for label, value in spans
        ],
        **extra,
    }


class SpanMetricTests(unittest.TestCase):
    def test_perfect_prediction_scores_one(self):
        gold = [record("Call Ana today", [("GIVENNAME", "Ana")])]
        predicted = [[{"label": "GIVENNAME", "start": 5, "end": 8}]]
        metrics = evaluate_spans(gold, predicted)
        self.assertEqual(metrics["typed_entity"]["f1"], 1.0)
        self.assertEqual(metrics["private_character_recall"], 1.0)
        self.assertEqual(metrics["private_character_leakage"], 0.0)
        self.assertEqual(metrics["non_private_overmasking_rate"], 0.0)

    def test_missing_prediction_is_leakage(self):
        gold = [record("Call Ana today", [("GIVENNAME", "Ana")])]
        metrics = evaluate_spans(gold, [[]])
        self.assertEqual(metrics["typed_entity"]["recall"], 0.0)
        self.assertEqual(metrics["private_character_leakage"], 1.0)

    def test_right_boundary_wrong_type_fails_typed_but_passes_binary(self):
        gold = [record("Call Ana today", [("GIVENNAME", "Ana")])]
        predicted = [[{"label": "SURNAME", "start": 5, "end": 8}]]
        metrics = evaluate_spans(gold, predicted)
        self.assertEqual(metrics["typed_entity"]["f1"], 0.0)
        self.assertEqual(metrics["binary_entity"]["f1"], 1.0)
        # Characters are still protected even though the type is wrong.
        self.assertEqual(metrics["private_character_recall"], 1.0)
        self.assertIn("GIVENNAME->SURNAME", metrics["boundary_correct_type_confusions"])

    def test_hard_negative_false_positive_rate(self):
        gold = [record("i forgot my password", []), record("turn the lights off", [])]
        predicted = [[{"label": "PASSWORD", "start": 12, "end": 20}], []]
        metrics = evaluate_spans(gold, predicted)
        self.assertEqual(metrics["hard_negative_rows"], 2)
        self.assertEqual(metrics["hard_negative_false_positive_rate"], 0.5)

    def test_overmasking_counts_characters_outside_gold(self):
        gold = [record("Call Ana today", [("GIVENNAME", "Ana")])]
        predicted = [[{"label": "GIVENNAME", "start": 5, "end": 14}]]
        metrics = evaluate_spans(gold, predicted)
        self.assertGreater(metrics["non_private_overmasking_rate"], 0.0)
        self.assertEqual(metrics["overmasked_characters"], 6)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_spans([record("Call Ana", [("GIVENNAME", "Ana")])], [])

    def test_per_label_support_is_reported(self):
        gold = [
            record("Call Ana at ana@example.com",
                   [("GIVENNAME", "Ana"), ("EMAIL", "ana@example.com")]),
            record("Call Omar", [("GIVENNAME", "Omar")]),
        ]
        predicted = [
            [{"label": "GIVENNAME", "start": 5, "end": 8},
             {"label": "EMAIL", "start": 12, "end": 27}],
            [],
        ]
        metrics = evaluate_spans(gold, predicted)
        self.assertEqual(metrics["per_label"]["GIVENNAME"]["support"], 2)
        self.assertEqual(metrics["per_label"]["GIVENNAME"]["recall"], 0.5)
        self.assertEqual(metrics["per_label"]["EMAIL"]["f1"], 1.0)


class SharedOntologyTests(unittest.TestCase):
    def test_restricting_labels_enables_a_fair_comparison(self):
        import ontology

        gold = [record("my password is BlueHorse!92 and Ana called",
                       [("PASSWORD", "BlueHorse!92"), ("GIVENNAME", "Ana")])]
        shared = restrict_to_labels(gold, ontology.SHARED_WITH_TINYBERT)
        labels = {span["label"] for span in shared[0]["privacy_mask"]}
        self.assertEqual(labels, {"GIVENNAME"})
        self.assertNotIn("PASSWORD", labels)

    def test_the_original_records_are_not_mutated(self):
        gold = [record("Call Ana", [("GIVENNAME", "Ana")])]
        restrict_to_labels(gold, ["EMAIL"])
        self.assertEqual(len(gold[0]["privacy_mask"]), 1)


class PasswordAnalysisTests(unittest.TestCase):
    def test_disclosures_and_hard_negatives_are_separated(self):
        gold = [
            record("my password is BlueHorse!92", [("PASSWORD", "BlueHorse!92")]),
            record("i forgot my password", []),
            record("turn the lights off", []),
        ]
        predicted = [
            [{"label": "PASSWORD", "start": 15, "end": 27}],
            [{"label": "PASSWORD", "start": 12, "end": 20}],
            [],
        ]
        analysis = password_analysis(gold, predicted)
        self.assertEqual(analysis["disclosure_rows"], 1)
        self.assertEqual(analysis["password_hard_negative_rows"], 1)
        self.assertEqual(analysis["password_hard_negative_false_positives"], 1)
        self.assertEqual(analysis["password_hard_negative_false_positive_rate"], 1.0)
        self.assertEqual(
            analysis["disclosure_metrics"]["per_label"]["PASSWORD"]["recall"], 1.0
        )

    def test_a_clean_model_scores_zero_false_positives(self):
        gold = [record("i forgot my password", []), record("never share your password", [])]
        analysis = password_analysis(gold, [[], []])
        self.assertEqual(analysis["password_hard_negative_false_positive_rate"], 0.0)


class SubsetTests(unittest.TestCase):
    def test_metrics_can_be_split_by_source(self):
        gold = [
            record("Call Ana", [("GIVENNAME", "Ana")], source_dataset="ai4privacy"),
            record("Call Omar", [("GIVENNAME", "Omar")], source_dataset="gretel"),
        ]
        predicted = [[{"label": "GIVENNAME", "start": 5, "end": 8}], []]
        by_source = subset_by(gold, predicted, "source_dataset")
        self.assertEqual(by_source["ai4privacy"]["typed_entity"]["f1"], 1.0)
        self.assertEqual(by_source["gretel"]["typed_entity"]["f1"], 0.0)


class AdapterTests(unittest.TestCase):
    def test_bio_repair_matches_the_production_rule(self):
        self.assertEqual(
            repair_bio(["O", "I-CITY", "I-CITY", "O", "I-EMAIL"]),
            (["O", "B-CITY", "I-CITY", "O", "B-EMAIL"], 2),
        )
        self.assertEqual(repair_bio(["B-CITY", "I-CITY"]), (["B-CITY", "I-CITY"], 0))

    def test_label_map_must_be_contiguous(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            ModelV2OnnxMasker("missing.onnx", "missing", {0: "O", 2: "B-CITY"})

    def test_label_map_can_be_loaded_from_the_committed_file(self):
        committed = REPO_ROOT / "model" / "DeBERTa-v3-xsmall" / "config" / "model_v2_label_map.json"
        if not committed.exists():
            self.skipTest("label map not generated yet")
        masker = ModelV2OnnxMasker("missing.onnx", "missing", committed)
        self.assertEqual(masker.num_labels, 63)
        self.assertEqual(masker.id2label[0], "O")

    def test_adapter_is_not_pinned_to_a_label_count(self):
        """No literal label count in the code; docstrings may mention 67."""
        source = (ROOT / "privacy_eval" / "model_v2_adapter.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        constants = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, int)
        ]
        self.assertNotIn(67, constants)
        self.assertNotIn(63, constants)
        assignments = [
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        self.assertNotIn("EXPECTED_LABEL_COUNT", assignments)

    def test_adapter_reuses_production_decoding(self):
        source = (ROOT / "privacy_eval" / "model_v2_adapter.py").read_text(encoding="utf-8")
        self.assertIn("consolidate_predictions_to_words", source)
        self.assertIn("merge_pii_spans", source)

    def test_production_adapter_is_untouched_by_model_v2(self):
        """Model V2 must not have relaxed the pinned production contract."""
        adapter = (REPO_ROOT / "FullPipeline_V3" / "integrated_pipeline" / "pii_adapter.py")
        source = adapter.read_text(encoding="utf-8")
        self.assertIn("EXPECTED_LABEL_COUNT = 67", source)
        config = json.loads(
            (REPO_ROOT / "FullPipeline_V3" / "config" / "live_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["pii"]["label_count"], 67)


class RunnerTests(unittest.TestCase):
    def test_result_filenames_are_model_scoped(self):
        source = (ROOT / "privacy_eval" / "run.py").read_text(encoding="utf-8")
        self.assertIn('f"{model_id}_{partition}_{stage}', source)
        self.assertNotIn('f"tinybert_{partition}_{stage}', source)

    def test_default_model_id_preserves_existing_filenames(self):
        import inspect

        from privacy_eval.run import run_partition

        signature = inspect.signature(run_partition)
        self.assertEqual(signature.parameters["model_id"].default, "tinybert")


class NotebookTests(unittest.TestCase):
    """Static validation; the notebook itself is expensive and is not executed."""

    @classmethod
    def setUpClass(cls):
        if not NOTEBOOK.exists():
            raise unittest.SkipTest("Model V2 notebook not generated")
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.all_code = "\n".join(cls.code_cells)

    def test_every_code_cell_parses(self):
        for index, source in enumerate(self.code_cells):
            with self.subTest(cell=index):
                ast.parse(source)

    def test_notebook_has_no_stored_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_frozen_final_is_gated_and_defaults_off(self):
        self.assertIn("RUN_FROZEN_FINAL = False", self.all_code)
        self.assertIn("if RUN_FROZEN_FINAL:", self.all_code)

    def test_ontology_is_asserted(self):
        self.assertIn("len(ENTITIES) == 31", self.all_code)
        self.assertIn("NUM_LABELS == 63", self.all_code)

    def test_leakage_assertions_are_present(self):
        self.assertIn("assert_partitions_disjoint", self.all_code)
        self.assertIn("firewall.assert_clean", self.all_code)
        self.assertIn("len(val_original) == 40_909", self.all_code)

    def test_validation_sets_are_never_augmented(self):
        self.assertIn("augment_training_rows", self.all_code)
        # Augmentation is applied to the training pool only.
        self.assertNotIn("augment_training_rows(val_", self.all_code)
        self.assertNotIn("augment_training_rows(test_rows", self.all_code)

    def test_int8_has_a_quality_gate(self):
        self.assertIn("INT8_MAX_TYPED_F1_DROP", self.all_code)
        self.assertIn("INT8_ACCEPTED", self.all_code)

    def test_production_migration_is_explicitly_not_performed(self):
        self.assertIn("NOT PERFORMED", self.all_code)

    def test_training_budget_matches_the_agreed_configuration(self):
        self.assertIn("MAX_EPOCHS = 10", self.all_code)
        self.assertIn("EARLY_STOPPING_PATIENCE = 2", self.all_code)

    def test_production_windowing_contract_is_preserved(self):
        self.assertIn("CONTEXT_LIMIT = 512", self.all_code)
        self.assertIn("OVERFLOW_STRIDE = 128", self.all_code)

    def test_pi_claims_are_not_made_from_host_measurements(self):
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        self.assertIn("not** Raspberry Pi measurements", markdown)


if __name__ == "__main__":
    unittest.main()

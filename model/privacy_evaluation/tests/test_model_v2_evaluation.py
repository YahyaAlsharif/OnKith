"""Dataset-shaped metrics, the model-agnostic adapter, and the Model V2 notebook."""

from __future__ import annotations

import ast
import inspect
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
from privacy_eval.model_v2_adapter import (  # noqa: E402
    MASKER_SOURCE,
    ModelV2OnnxMasker,
    repair_bio,
)

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
        source = MASKER_SOURCE.read_text(encoding="utf-8")
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
        source = MASKER_SOURCE.read_text(encoding="utf-8")
        self.assertIn("consolidate_predictions_to_words", source)
        self.assertIn("merge_pii_spans", source)

    def test_evaluation_and_production_share_one_masker(self):
        """The name this harness imports must resolve into a pipeline.

        Replaces the old "Model V2 must not touch production" guard. Production
        *is* Model V2 now, and the whole point of the integration is that these
        metrics were measured through the code the deployment runs. A second
        copy of the decoder appearing under privacy_eval/ would silently break
        that.

        Byte-identity rather than path-identity, because there are now two
        deployable pipelines and both ship this decoder. Which root the bare
        ``integrated_pipeline`` import resolves to depends on what else the
        process has imported - ``benchmark_common.ensure_import_paths`` exists
        to make that deterministic for the benchmarks - so pinning the path
        would make this test a report on import order. Pinning the bytes is the
        guarantee that was actually wanted: whichever copy was loaded, it is the
        same decoder, and a divergence between the two pipelines fails here.
        """

        loaded = Path(inspect.getsourcefile(ModelV2OnnxMasker)).resolve()
        self.assertEqual(
            loaded.read_bytes(),
            MASKER_SOURCE.resolve().read_bytes(),
            f"{loaded} and {MASKER_SOURCE} have diverged",
        )
        self.assertEqual(loaded.name, MASKER_SOURCE.name)
        self.assertEqual(loaded.parent.name, "integrated_pipeline")

    def test_production_runs_the_model_v2_ontology(self):
        """The deployed pipeline is pinned to the 63-label Model V2 contract."""

        # Whichever pipeline is currently deployed, read from one place so
        # this cannot go on asserting about a version that has been archived.
        from privacy_eval.benchmark_common import PRODUCTION_ROOT

        source = (
            PRODUCTION_ROOT / "integrated_pipeline" / "pii_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertIn("EXPECTED_LABEL_COUNT = 63", source)
        config = json.loads(
            (PRODUCTION_ROOT / "config" / "live_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["pii"]["label_count"], 63)
        self.assertEqual(config["pii"]["model_id"], "microsoft/deberta-v3-xsmall")


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


class NotebookResumeLogicTests(unittest.TestCase):
    """Execute the notebook's own resume block against synthetic checkpoints.

    The block is short, self-contained and the part most likely to silently
    restart a run that had already stopped, so it is worth exercising rather
    than only grepping for.
    """

    START = "training_complete = False"
    END = "for epoch in planned_epochs:"

    @classmethod
    def setUpClass(cls):
        if not NOTEBOOK.exists():
            raise unittest.SkipTest("Model V2 notebook not generated")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        sources = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and cls.START in "".join(cell["source"])
        ]
        if len(sources) != 1:
            raise AssertionError(f"expected one resume block, found {len(sources)}")
        source = sources[0]
        cls.block = source[source.index(cls.START):source.index(cls.END)]

    def run_resume(self, payload, *, run_epochs=10, signature="sig"):
        recorded = []

        class Stub:
            def load_state_dict(self, state):
                recorded.append(state)

        namespace = {
            "RESUME_TRAINING": True,
            "LATEST_CHECKPOINT": SimpleNamespace(exists=lambda: True),
            "torch": SimpleNamespace(load=lambda *args, **kwargs: payload),
            "device": "cpu",
            "run_signature": signature,
            "model": Stub(), "optimizer": Stub(), "scheduler": Stub(), "scaler": Stub(),
            "RUN_EPOCHS": run_epochs,
            "start_epoch": 1,
            "print": lambda *args, **kwargs: None,
        }
        exec(self.block, namespace)  # noqa: S102 - the notebook's own source
        return namespace

    @staticmethod
    def checkpoint(**overrides):
        payload = {
            "run_signature": "sig", "epoch": 3, "model_state": {},
            "optimizer_state": {}, "scheduler_state": {}, "scaler_state": {},
            "history": [], "best_metric": 0.8, "best_epoch": 2,
            "patience_left": 2, "optimizer_steps": 300,
        }
        payload.update(overrides)
        return payload

    def test_an_interrupted_run_resumes_at_the_next_epoch(self):
        namespace = self.run_resume(self.checkpoint(epoch=3, patience_left=2))
        self.assertFalse(namespace["training_complete"])
        self.assertEqual(namespace["planned_epochs"], list(range(4, 11)))

    def test_a_run_that_already_stopped_early_starts_no_further_epoch(self):
        namespace = self.run_resume(
            self.checkpoint(epoch=5, patience_left=0, early_stopped=True)
        )
        self.assertTrue(namespace["training_complete"])
        self.assertEqual(namespace["planned_epochs"], [])
        self.assertEqual(namespace["best_epoch"], 2)

    def test_exhausted_patience_stops_even_without_the_new_flag(self):
        """Checkpoints written before `early_stopped` existed must still stop."""
        namespace = self.run_resume(self.checkpoint(epoch=5, patience_left=0))
        self.assertTrue(namespace["training_complete"])
        self.assertEqual(namespace["planned_epochs"], [])

    def test_a_completed_budget_starts_no_further_epoch(self):
        namespace = self.run_resume(self.checkpoint(epoch=10, patience_left=1))
        self.assertTrue(namespace["training_complete"])
        self.assertEqual(namespace["planned_epochs"], [])

    def test_a_foreign_checkpoint_is_refused(self):
        with self.assertRaises(RuntimeError):
            self.run_resume(self.checkpoint(run_signature="other"))

    def test_optimizer_scheduler_and_scaler_state_are_restored(self):
        namespace = self.run_resume(self.checkpoint(optimizer_steps=1234))
        self.assertEqual(namespace["optimizer_steps"], 1234)
        self.assertEqual(namespace["best_metric"], 0.8)
        self.assertEqual(namespace["patience_left"], 2)


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
        cls.markdown = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        cls.candidate_cell = next(
            source
            for source in cls.code_cells
            if "accepted_candidate.build_candidate_manifest(" in source
        )

    def test_every_code_cell_parses(self):
        for index, source in enumerate(self.code_cells):
            with self.subTest(cell=index):
                ast.parse(source)

    def test_notebook_has_no_stored_outputs(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_the_frozen_final_lives_in_a_separate_notebook(self):
        """Run All must never be able to spend the final holdout."""
        for removed in ("RUN_FROZEN_FINAL", 'partition="ood_final"',
                        "evaluation_ood_final_int8"):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, self.all_code)
        self.assertIn("model_v2_frozen_ood_final.ipynb", self.all_code)

    def test_smoke_mode_is_isolated_and_bounded(self):
        self.assertIn('MODEL_V2_SMOKE_TEST', self.all_code)
        self.assertIn('FINAL_ARTIFACTS / "smoke"', self.all_code)
        self.assertIn("SMOKE_TRAIN_STEPS", self.all_code)
        self.assertIn("optimizer_steps > 0", self.all_code)

    def test_saudi_prerequisite_is_checked_early(self):
        self.assertIn("sources.audit_saudi_csv(SAUDI_CSV)", self.all_code)
        self.assertIn("Saudi CSV hash does not match its manifest", self.all_code)

    def test_organized_deployment_paths_are_used(self):
        self.assertIn('MODEL_DIR / "deployment" / "verify_production_runtime.py"', self.all_code)
        self.assertIn('FP32_DIR / "model_fp32.onnx"', self.all_code)
        self.assertIn('INT8_DIR / "model_int8.onnx"', self.all_code)

    def test_ontology_is_asserted(self):
        self.assertIn("len(ENTITIES) == 31", self.all_code)
        self.assertIn("NUM_LABELS == 63", self.all_code)

    def test_leakage_assertions_are_present(self):
        self.assertIn("assert_partitions_disjoint", self.all_code)
        self.assertIn("firewall.assert_clean", self.all_code)
        self.assertIn("len(val_original) == 40_909", self.all_code)

    def test_the_contamination_filter_runs_before_the_final_split_and_the_write(self):
        """Filtering must precede materialisation, and never replace the assertion."""
        filter_index = self.all_code.index('stage="pre_split"')
        split_index = self.all_code.index("pools.split_training_side(")
        write_index = self.all_code.index("write_jsonl(TRAIN_PATH")
        self.assertLess(filter_index, split_index)
        self.assertLess(self.all_code.index('stage="pre_write"'), write_index)
        self.assertIn("pools.filter_contamination(", self.all_code)
        self.assertIn("assert_no_protected_overlap", self.all_code)
        # The hard assertions survive the fix rather than becoming warnings.
        self.assertIn('firewall.assert_clean(train_rows, label="train")', self.all_code)
        self.assertIn('firewall.assert_clean(test_rows, label="test")', self.all_code)

    def test_the_holdout_joins_the_firewall_once_it_exists(self):
        """A firewall that never sees val_supplementary cannot protect it."""
        self.assertIn('add_protected_records("val_supplementary", val_supplementary)',
                      self.all_code)

    def test_stale_caches_cannot_be_reused_in_silence(self):
        self.assertIn("cache_is_current", self.all_code)
        self.assertIn("assert_cache_current", self.all_code)
        self.assertIn("REBUILD_SUPPLEMENTARY", self.all_code)

    def test_the_tinybert_comparison_states_when_the_row_set_changed(self):
        self.assertIn("AI4_TEST_HISTORICAL_ROWS", self.all_code)
        self.assertIn("AI4_TEST_ROWS_REMOVED", self.all_code)
        self.assertIn("NOT exactly like-for-like", self.all_code)

    def test_validation_sets_are_never_augmented(self):
        self.assertIn("augment_training_rows", self.all_code)
        # Augmentation is applied to the training pool only.
        self.assertNotIn("augment_training_rows(val_", self.all_code)
        self.assertNotIn("augment_training_rows(test_rows", self.all_code)

    def test_int8_has_a_quality_gate(self):
        self.assertIn("INT8_MAX_TYPED_F1_DROP", self.all_code)
        self.assertIn("INT8_ACCEPTED", self.all_code)

    def test_int8_gate_covers_combined_validation_and_ood_dev(self):
        """A gate that ignores OOD dev cannot see quantization collapse."""
        self.assertIn("INT8_GATE_VIEWS", self.all_code)
        for view in ("test", "val_original", "val_supplementary",
                     "combined_validation", "ood_dev"):
            with self.subTest(view=view):
                self.assertIn(f'"{view}"', self.all_code)
        # The combined view is scored explicitly on INT8, not inferred.
        self.assertIn("combined_int8_metrics = evaluate_spans(", self.all_code)
        self.assertIn('int8_evaluations["combined_validation"] = combined_int8_metrics',
                      self.all_code)
        self.assertIn("for name in INT8_GATE_VIEWS:", self.all_code)

    def test_the_accepted_candidate_is_recorded_after_the_int8_gate(self):
        """The manifest proves which artifact cleared the gate, in that order."""
        gate_index = self.all_code.index("INT8_ACCEPTED = bool(gate)")
        candidate_index = self.all_code.index("accepted_candidate.build_candidate_manifest(")
        export_index = self.all_code.index("onnx_program = torch.onnx.export")
        quantize_index = self.all_code.index("quantize_dynamic(")
        self.assertLess(export_index, gate_index)
        self.assertLess(quantize_index, gate_index)
        self.assertLess(gate_index, candidate_index)

    def test_the_candidate_manifest_carries_the_release_identity(self):
        for provenance in ("int8_onnx=INT8_ONNX", "tokenizer_dir=TOKENIZER_DIR",
                           "label_map_path=LABEL_MAP_PATH",
                           "training_config_path=TRAINING_CONFIG_PATH",
                           "ontology_version=ontology.MODEL_V2_VERSION",
                           "num_labels=NUM_LABELS", "run_signature=run_signature",
                           "best_epoch=best_epoch", "gate=gate",
                           "frozen_fixture=FROZEN_FIXTURE_FINGERPRINT",
                           "timestamp_utc="):
            with self.subTest(provenance=provenance):
                self.assertIn(provenance, self.candidate_cell)

    def test_a_smoke_candidate_can_never_be_accepted(self):
        self.assertIn("int8_accepted=INT8_ACCEPTED, smoke_test=SMOKE_TEST",
                      self.candidate_cell)
        self.assertIn('assert not (SMOKE_TEST and candidate_manifest["accepted"])',
                      self.candidate_cell)

    def test_the_frozen_fixture_is_fingerprinted_but_never_scored(self):
        """Section 6 protects the frozen texts; it must not build cases from them."""
        self.assertIn('firewall.add_protected("ood_final"', self.all_code)
        self.assertIn("FROZEN_FIXTURE_FINGERPRINT", self.all_code)
        self.assertNotIn("load_cases(OOD_FINAL_PATH)", self.all_code)

    def test_export_does_not_move_the_trained_model(self):
        """Exporting on CPU must not strand later CUDA evaluation cells."""
        self.assertNotIn('model.to("cpu").float().eval()', self.all_code)
        self.assertIn("export_model = AutoModelForTokenClassification.from_pretrained(",
                      self.all_code)
        self.assertIn("next(model.parameters()).device.type == device.type", self.all_code)

    def test_resume_does_not_restart_a_run_that_stopped_early(self):
        self.assertIn('"early_stopped": patience_left <= 0', self.all_code)
        self.assertIn('resumed.get("early_stopped"', self.all_code)
        self.assertIn("planned_epochs = [] if training_complete", self.all_code)
        self.assertIn("for epoch in planned_epochs:", self.all_code)
        for state in ('resumed["optimizer_state"]', 'resumed["scheduler_state"]',
                      'resumed["scaler_state"]', 'resumed["best_metric"]',
                      'resumed["patience_left"]', 'resumed.get("run_signature")'):
            with self.subTest(state=state):
                self.assertIn(state, self.all_code)

    def test_tokenization_reports_its_memory_footprint(self):
        self.assertIn("tracemalloc.start()", self.all_code)
        self.assertIn("TOKENIZATION_MEMORY_WARN_GIB", self.all_code)
        self.assertIn("release_offset_mappings(train_features)", self.all_code)

    def test_markdown_describes_the_implemented_bio_alignment(self):
        """Sub-word continuations are labelled; -100 is for special/pad positions."""
        self.assertNotIn("continuations receive `-100`", self.markdown)
        self.assertIn("every token that covers source", self.markdown)

    def test_markdown_uses_the_current_artifact_paths(self):
        self.assertNotIn("`huggingface_model/`", self.markdown)
        self.assertNotIn("`onnx_model/`", self.markdown)
        self.assertIn("artifacts/model_v2/models/int8/", self.markdown)
        self.assertIn("artifacts/model_v2/models/fp32/", self.markdown)
        self.assertIn("artifacts/model_v2/checkpoints/", self.markdown)

    def test_markdown_separates_the_tinybert_comparison_subset(self):
        """The full Model V2 test set is larger than the 16,374-row subset."""
        self.assertIn("16,374", self.markdown)
        self.assertIn("AI4Privacy component", self.markdown)
        self.assertNotIn("identical rows to the TinyBERT baseline", self.markdown)

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

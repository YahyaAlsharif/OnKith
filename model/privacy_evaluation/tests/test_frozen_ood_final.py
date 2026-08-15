"""The accepted-candidate contract and the separated frozen-benchmark notebook.

Every guard here is exercised against synthetic manifests and synthetic frozen
fixtures. The real frozen OOD final set is spendable exactly once, so no test in
this module may read it - see ``test_the_real_frozen_fixture_is_never_read``.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from privacy_eval import accepted_candidate  # noqa: E402
from privacy_eval.accepted_candidate import (  # noqa: E402
    CandidateRejected,
    FixtureRejected,
    FrozenBenchmarkAlreadySpent,
)

MODEL_V2_DIR = REPO_ROOT / "model" / "DeBERTa-v3-xsmall"
MAIN_NOTEBOOK = MODEL_V2_DIR / "model_v2_deberta_v3_xsmall.ipynb"
FROZEN_NOTEBOOK = MODEL_V2_DIR / "model_v2_frozen_ood_final.ipynb"

ONTOLOGY_VERSION = "model_v2.ontology.1"
REVISION = "4b419818330868dff6a60ad3e6b1c730f8b8c0c6"


def metrics(f1=0.80, leakage=0.10):
    return {
        "rows": 120,
        "typed_entity": {"precision": f1, "recall": f1, "f1": f1},
        "binary_entity": {"precision": f1, "recall": f1, "f1": f1},
        "private_character_recall": 0.9,
        "private_character_leakage": leakage,
        "non_private_overmasking_rate": 0.05,
        "hard_negative_false_positive_rate": 0.2,
    }


def evaluations(f1=0.80):
    return {view: metrics(f1) for view in accepted_candidate.REQUIRED_EVALUATION_VIEWS}


def gate(passes=True):
    return {
        view: {"typed_f1_drop": 0.004, "leakage_rise": 0.001, "passes": passes}
        for view in accepted_candidate.REQUIRED_EVALUATION_VIEWS
    }


class CandidateFactory:
    """A throwaway repository root holding a plausible accepted candidate."""

    def __init__(self, root: Path):
        self.root = root
        self.metrics_dir = root / "metrics"
        self.models_dir = root / "models"
        self.tokenizer_dir = root / "tokenizer"
        for directory in (self.metrics_dir, self.models_dir, self.tokenizer_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.int8 = self.models_dir / "model_int8.onnx"
        self.int8.write_bytes(b"int8-graph-bytes")
        self.fp32 = self.models_dir / "model_fp32.onnx"
        self.fp32.write_bytes(b"fp32-graph-bytes")
        (self.tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (self.tokenizer_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        self.label_map = root / "model_v2_label_map.json"
        self.write_label_map(num_labels=63, ontology_version=ONTOLOGY_VERSION)
        self.training_config = root / "model_v2_training_config.json"
        self.write_training_config()
        self.manifest_path = self.metrics_dir / accepted_candidate.CANDIDATE_MANIFEST_NAME
        self.report_path = self.metrics_dir / accepted_candidate.FROZEN_REPORT_NAME

    def write_label_map(self, *, num_labels, ontology_version):
        self.label_map.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "ontology_version": ontology_version,
                    "num_entities": 31,
                    "num_labels": num_labels,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def write_training_config(self, **overrides):
        payload = {
            "base_model": "microsoft/deberta-v3-xsmall",
            "base_model_revision": REVISION,
            "num_labels": 63,
            "smoke_test": False,
            "run_signature": "a" * 64,
            "best_epoch": 4,
        }
        payload.update(overrides)
        self.training_config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def build(self, **overrides):
        arguments = dict(
            repo_root=self.root,
            int8_onnx=self.int8,
            fp32_onnx=self.fp32,
            tokenizer_dir=self.tokenizer_dir,
            label_map_path=self.label_map,
            training_config_path=self.training_config,
            best_checkpoint=self.models_dir / "best.pt",
            ontology_version=ONTOLOGY_VERSION,
            num_entities=31,
            num_labels=63,
            base_model="microsoft/deberta-v3-xsmall",
            base_model_revision=REVISION,
            run_signature="a" * 64,
            best_epoch=4,
            best_combined_typed_f1=0.83,
            seed=42,
            context_limit=512,
            overflow_stride=128,
            fp32_evaluations=evaluations(0.82),
            int8_evaluations=evaluations(0.81),
            gate=gate(True),
            gate_views=accepted_candidate.REQUIRED_EVALUATION_VIEWS,
            max_typed_f1_drop=0.02,
            max_leakage_rise=0.015,
            frozen_fixture={"sha256": "f" * 64, "case_count": 360},
            int8_accepted=True,
            smoke_test=False,
            timestamp_utc="2026-08-15T00:00:00Z",
        )
        arguments.update(overrides)
        return accepted_candidate.build_candidate_manifest(**arguments)

    def write(self, **overrides):
        manifest = self.build(**overrides)
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    def load(self):
        return accepted_candidate.load_accepted_candidate(
            self.manifest_path, repo_root=self.root
        )


class CandidateFactoryCase(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.factory = CandidateFactory(Path(self._temporary.name))


class AcceptedCandidateTests(CandidateFactoryCase):
    def test_a_real_passing_run_is_accepted(self):
        manifest = self.factory.write()
        self.assertTrue(manifest["accepted"])
        self.assertFalse(manifest["smoke_test"])
        loaded = self.factory.load()
        self.assertEqual(loaded["int8_sha256"], manifest["model"]["int8_sha256"])
        self.assertEqual(loaded["int8_path"], self.factory.int8)

    def test_a_missing_manifest_fails_loudly(self):
        with self.assertRaisesRegex(CandidateRejected, "no accepted-candidate manifest"):
            self.factory.load()

    def test_a_failed_int8_gate_is_never_accepted(self):
        manifest = self.factory.write(int8_accepted=False, gate=gate(False))
        self.assertFalse(manifest["accepted"])
        with self.assertRaisesRegex(CandidateRejected, "accepted=false"):
            self.factory.load()

    def test_a_smoke_run_is_never_accepted(self):
        manifest = self.factory.write(smoke_test=True)
        self.assertFalse(manifest["accepted"])
        self.assertTrue(manifest["smoke_test"])

    def test_a_smoke_candidate_is_refused_even_if_accepted_is_forced(self):
        """A hand-edited smoke manifest must still be refused."""
        manifest = self.factory.build(smoke_test=True)
        manifest["accepted"] = True
        self.factory.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(CandidateRejected, "smoke run"):
            self.factory.load()

    def test_a_missing_int8_file_fails(self):
        self.factory.write()
        self.factory.int8.unlink()
        with self.assertRaisesRegex(CandidateRejected, "accepted INT8 candidate is missing"):
            self.factory.load()

    def test_a_substituted_int8_graph_fails_on_its_hash(self):
        """A hand-copied ONNX model must not silently take the candidate's place."""
        self.factory.write()
        self.factory.int8.write_bytes(b"some-other-graph")
        with self.assertRaisesRegex(CandidateRejected, "not the accepted candidate"):
            self.factory.load()

    def test_a_resized_int8_graph_fails(self):
        self.factory.write()
        self.factory.int8.write_bytes(b"int8-graph-bytes-plus")
        with self.assertRaisesRegex(CandidateRejected, "size changed"):
            self.factory.load()

    def test_a_label_count_that_is_not_63_fails(self):
        self.factory.write_label_map(num_labels=62, ontology_version=ONTOLOGY_VERSION)
        self.factory.write(num_labels=62)
        with self.assertRaisesRegex(CandidateRejected, "labels, expected 63"):
            self.factory.load()

    def test_a_changed_label_map_fails_on_its_hash(self):
        self.factory.write()
        self.factory.label_map.write_text(
            self.factory.label_map.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(CandidateRejected, "label map on disk"):
            self.factory.load()

    def test_a_different_ontology_version_fails(self):
        """Even with a re-hashed manifest, the ontology version must agree."""
        manifest = self.factory.build()
        self.factory.write_label_map(num_labels=63, ontology_version="model_v2.ontology.2")
        manifest["ontology"]["label_map_sha256"] = accepted_candidate.sha256_file(
            self.factory.label_map
        )
        self.factory.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(CandidateRejected, "ontology version differs"):
            self.factory.load()

    def test_a_changed_tokenizer_fails(self):
        self.factory.write()
        (self.factory.tokenizer_dir / "tokenizer.json").write_text(
            '{"changed": true}', encoding="utf-8"
        )
        with self.assertRaisesRegex(CandidateRejected, "tokenizer.json no longer matches"):
            self.factory.load()

    def test_a_missing_tokenizer_config_fails(self):
        self.factory.write()
        (self.factory.tokenizer_dir / "tokenizer_config.json").unlink()
        with self.assertRaisesRegex(CandidateRejected, "tokenizer file is missing"):
            self.factory.load()

    def test_an_inconsistent_run_signature_fails(self):
        self.factory.write()
        self.factory.write_training_config(run_signature="b" * 64)
        with self.assertRaisesRegex(CandidateRejected, "training config on disk"):
            self.factory.load()

    def test_a_training_config_from_another_run_fails_the_signature_check(self):
        manifest = self.factory.build()
        self.factory.write_training_config(run_signature="b" * 64)
        manifest["training"]["training_config_sha256"] = accepted_candidate.sha256_file(
            self.factory.training_config
        )
        self.factory.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(CandidateRejected, "run signature is inconsistent"):
            self.factory.load()

    def test_missing_evaluation_evidence_fails(self):
        manifest = self.factory.build()
        manifest["evaluation"]["int8_onnx"].pop("ood_dev")
        self.factory.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(CandidateRejected, "evaluation evidence is missing"):
            self.factory.load()

    def test_a_gate_that_skipped_ood_dev_fails(self):
        partial = {
            view: entry
            for view, entry in gate(True).items()
            if view != "ood_dev"
        }
        self.factory.write(gate=partial)
        with self.assertRaisesRegex(CandidateRejected, r"did not cover \['ood_dev'\]"):
            self.factory.load()

    def test_a_foreign_artifact_kind_is_refused(self):
        self.factory.manifest_path.write_text(
            json.dumps({"artifact": "something_else", "schema_version": 1}), encoding="utf-8"
        )
        with self.assertRaisesRegex(CandidateRejected, "not an accepted_int8_candidate"):
            self.factory.load()

    def test_the_manifest_records_the_frozen_fixture_identity(self):
        manifest = self.factory.write()
        recorded = manifest["frozen_benchmark"]["fixture"]
        self.assertEqual(recorded["sha256"], "f" * 64)
        self.assertEqual(recorded["case_count"], 360)
        self.assertEqual(manifest["frozen_benchmark"]["status"], "not run")


class SyntheticFixture:
    """A frozen-looking fixture that is emphatically not the real one."""

    def __init__(self, root: Path, *, cases=3, partition="final"):
        self.path = root / "synthetic_final.jsonl"
        self.manifest_path = root / "synthetic_final.manifest.json"
        lines = [
            json.dumps(
                {
                    "case_id": f"synthetic-{index}",
                    "partition": partition,
                    "category": "names",
                    "subcategory": "written",
                    "template_id": "t1",
                    "text": f"placeholder case {index}",
                    "entities": [],
                    "tags": [],
                }
            )
            for index in range(cases)
        ]
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.write_manifest(cases=cases)

    def write_manifest(self, *, cases, frozen=True, partition="final"):
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "frozen": frozen,
                    "partition": partition,
                    "fixture_file": self.path.name,
                    "fixture_sha256": accepted_candidate.sha256_file(self.path),
                    "case_count": cases,
                    "generator_version": "1.0.0",
                    "hard_negative_count": 0,
                    "category_allocation": {"names": cases},
                    "policy": "synthetic",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def verify(self, **overrides):
        return accepted_candidate.verify_frozen_fixture(
            self.path, self.manifest_path, **overrides
        )


class FrozenFixtureVerificationTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.fixture = SyntheticFixture(self.root)

    def test_an_unchanged_fixture_verifies(self):
        result = self.fixture.verify()
        self.assertEqual(result["case_count"], 3)
        self.assertEqual(result["generator_version"], "1.0.0")

    def test_a_missing_fixture_fails(self):
        self.fixture.path.unlink()
        with self.assertRaisesRegex(FixtureRejected, "frozen fixture is missing"):
            self.fixture.verify()

    def test_a_missing_fixture_manifest_fails(self):
        self.fixture.manifest_path.unlink()
        with self.assertRaisesRegex(FixtureRejected, "fixture manifest is missing"):
            self.fixture.verify()

    def test_a_changed_fixture_fails_its_committed_hash(self):
        self.fixture.path.write_text(
            self.fixture.path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(FixtureRejected, "SHA-256 changed"):
            self.fixture.verify()

    def test_a_fixture_the_candidate_did_not_see_fails(self):
        with self.assertRaisesRegex(FixtureRejected, "not the one fingerprinted"):
            self.fixture.verify(expected_sha256="0" * 64)

    def test_a_changed_case_count_fails(self):
        self.fixture.verify()
        with self.assertRaisesRegex(FixtureRejected, "the candidate recorded"):
            self.fixture.verify(expected_case_count=360)

    def test_a_manifest_that_is_not_frozen_fails(self):
        self.fixture.write_manifest(cases=3, frozen=False)
        with self.assertRaisesRegex(FixtureRejected, "not marked frozen"):
            self.fixture.verify()

    def test_a_non_final_partition_fails(self):
        self.fixture.write_manifest(cases=3, partition="dev")
        with self.assertRaisesRegex(FixtureRejected, "expected 'final'"):
            self.fixture.verify()

    def test_a_case_missing_schema_keys_fails(self):
        self.fixture.path.write_text(
            json.dumps({"case_id": "x", "partition": "final"}) + "\n", encoding="utf-8"
        )
        self.fixture.write_manifest(cases=1)
        with self.assertRaisesRegex(FixtureRejected, "is missing"):
            self.fixture.verify()

    def test_a_case_from_another_partition_fails(self):
        elsewhere = self.root / "dev"
        elsewhere.mkdir()
        dev = SyntheticFixture(elsewhere, partition="dev")
        dev.write_manifest(cases=3, partition="final")
        with self.assertRaisesRegex(FixtureRejected, "not a final-partition case"):
            dev.verify()


class OneShotGuardTests(CandidateFactoryCase):
    def setUp(self):
        super().setUp()
        self.fixture = SyntheticFixture(Path(self._temporary.name))

    def report(self, *, candidate_sha, fixture_sha):
        self.factory.report_path.write_text(
            json.dumps(
                {
                    "artifact": accepted_candidate.FROZEN_REPORT_KIND,
                    "candidate": {"int8_sha256": candidate_sha},
                    "fixture": {"sha256": fixture_sha},
                }
            ),
            encoding="utf-8",
        )

    def guard(self, *, allow_rerun=False):
        accepted_candidate.guard_one_shot(
            self.factory.report_path,
            candidate_sha256="c" * 64,
            fixture_sha256="f" * 64,
            allow_rerun=allow_rerun,
        )

    def test_a_first_run_is_allowed(self):
        self.guard()

    def test_an_existing_report_for_the_same_candidate_refuses(self):
        self.report(candidate_sha="c" * 64, fixture_sha="f" * 64)
        with self.assertRaisesRegex(FrozenBenchmarkAlreadySpent, "one-shot"):
            self.guard()

    def test_a_second_candidate_on_a_spent_fixture_refuses(self):
        self.report(candidate_sha="d" * 64, fixture_sha="f" * 64)
        with self.assertRaisesRegex(FrozenBenchmarkAlreadySpent, "different candidate"):
            self.guard()

    def test_a_report_for_another_fixture_does_not_block(self):
        self.report(candidate_sha="c" * 64, fixture_sha="0" * 64)
        self.guard()

    def test_the_archival_override_is_the_only_way_through(self):
        self.report(candidate_sha="c" * 64, fixture_sha="f" * 64)
        self.guard(allow_rerun=True)

    def test_the_report_shape_feeds_the_guard(self):
        """build_frozen_report must write exactly what guard_one_shot reads."""
        self.factory.write()
        candidate = self.factory.load()
        fixture = self.fixture.verify()
        report = accepted_candidate.build_frozen_report(
            repo_root=self.factory.root,
            candidate=candidate,
            fixture=fixture,
            metrics=metrics(),
            by_category={"names": metrics()},
            password={"disclosure_rows": 0},
            runtime={"runtime": "onnx_int8"},
            timestamp_utc="2026-08-15T00:00:00Z",
        )
        self.factory.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        with self.assertRaises(FrozenBenchmarkAlreadySpent):
            accepted_candidate.guard_one_shot(
                self.factory.report_path,
                candidate_sha256=candidate["int8_sha256"],
                fixture_sha256=fixture["sha256"],
            )
        self.assertEqual(report["artifact"], accepted_candidate.FROZEN_REPORT_KIND)
        self.assertEqual(report["fixture"]["sha256"], fixture["sha256"])
        for key in ("typed_entity_f1", "private_character_recall",
                    "private_character_leakage", "non_private_overmasking_rate",
                    "hard_negative_false_positive_rate"):
            self.assertIn(key, report["headline"])

    def test_the_report_survives_a_path_outside_the_repository(self):
        """The report is built after the benchmark is spent; it must not raise."""
        self.factory.write()
        candidate = self.factory.load()
        elsewhere = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(elsewhere, ignore_errors=True))
        fixture = SyntheticFixture(elsewhere).verify()
        report = accepted_candidate.build_frozen_report(
            repo_root=self.factory.root,
            candidate=candidate,
            fixture=fixture,
            metrics=metrics(),
            by_category={},
            password={},
            runtime={"runtime": "onnx_int8"},
            timestamp_utc="2026-08-15T00:00:00Z",
        )
        self.assertTrue(report["fixture"]["path"].endswith("synthetic_final.jsonl"))
        self.assertEqual(report["fixture"]["sha256"], fixture["sha256"])


class NotebookSeparationTests(unittest.TestCase):
    """Static validation; neither notebook is executed here."""

    @classmethod
    def setUpClass(cls):
        if not MAIN_NOTEBOOK.exists() or not FROZEN_NOTEBOOK.exists():
            raise unittest.SkipTest("Model V2 notebooks not generated")
        cls.main, cls.main_code, cls.main_markdown = cls.read(MAIN_NOTEBOOK)
        cls.frozen, cls.frozen_code, cls.frozen_markdown = cls.read(FROZEN_NOTEBOOK)

    @staticmethod
    def read(path):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        return notebook, code, markdown

    # -- the main notebook must be safe to Run All --------------------------

    def test_the_main_notebook_cannot_run_the_frozen_benchmark(self):
        for removed in ("RUN_FROZEN_FINAL", 'partition="ood_final"',
                        "evaluation_ood_final_int8", "frozen_final_summary"):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, self.main_code)

    def test_the_main_notebook_keeps_the_contamination_firewall(self):
        """Fingerprints only: the firewall still protects the frozen texts."""
        self.assertIn('firewall.add_protected("ood_final"', self.main_code)
        self.assertIn("FROZEN_FIXTURE_FINGERPRINT", self.main_code)
        self.assertNotIn("load_cases(OOD_FINAL_PATH)", self.main_code)

    def test_the_main_notebook_writes_an_accepted_candidate_manifest(self):
        self.assertIn("accepted_candidate.build_candidate_manifest(", self.main_code)
        self.assertIn("CANDIDATE_MANIFEST_PATH.write_text", self.main_code)
        gate_index = self.main_code.index("INT8_ACCEPTED = bool(gate)")
        manifest_index = self.main_code.index("accepted_candidate.build_candidate_manifest(")
        self.assertLess(gate_index, manifest_index)

    def test_the_main_notebook_cannot_accept_a_smoke_candidate(self):
        self.assertIn("int8_accepted=INT8_ACCEPTED, smoke_test=SMOKE_TEST", self.main_code)
        self.assertIn(
            'assert not (SMOKE_TEST and candidate_manifest["accepted"])', self.main_code
        )

    def test_the_main_notebook_ends_with_the_deployment_summary(self):
        manifest_index = self.main_code.index("accepted_candidate.build_candidate_manifest(")
        runtime_index = self.main_code.index("verify_production_runtime.py")
        summary_index = self.main_code.index('METRICS_DIR / "deployment_summary.json"')
        self.assertLess(manifest_index, runtime_index)
        self.assertLess(runtime_index, summary_index)

    def test_the_main_notebook_points_at_the_separate_benchmark(self):
        self.assertIn("model_v2_frozen_ood_final.ipynb", self.main_markdown)
        self.assertIn("model_v2_frozen_ood_final.ipynb", self.main_code)

    # -- the frozen notebook ------------------------------------------------

    def test_the_frozen_notebook_is_valid_and_unexecuted(self):
        self.assertEqual(self.frozen["nbformat"], 4)
        for index, cell in enumerate(self.frozen["cells"]):
            if cell["cell_type"] != "code":
                continue
            with self.subTest(cell=index):
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_the_frozen_notebook_opens_with_an_unmissable_warning(self):
        first = "".join(self.frozen["cells"][0]["source"])
        self.assertEqual(self.frozen["cells"][0]["cell_type"], "markdown")
        self.assertTrue(first.startswith("# FINAL ONE-SHOT OOD BENCHMARK"))
        for warning in ("during training", "during model selection", "tune thresholds",
                        "choose augmentation", "unfinished checkpoints",
                        "repeatedly while developing", "no longer pristine"):
            with self.subTest(warning=warning):
                self.assertIn(warning, first)

    def test_the_frozen_notebook_verifies_before_it_loads(self):
        candidate = self.frozen_code.index("accepted_candidate.load_accepted_candidate(")
        fixture = self.frozen_code.index("accepted_candidate.verify_frozen_fixture(")
        guard = self.frozen_code.index("accepted_candidate.guard_one_shot(")
        load = self.frozen_code.index("load_cases(OOD_FINAL_PATH)")
        self.assertLess(candidate, fixture)
        self.assertLess(fixture, guard)
        self.assertLess(guard, load)

    def test_the_frozen_notebook_defaults_the_rerun_override_off(self):
        self.assertIn("ALLOW_FROZEN_RERUN = False", self.frozen_code)
        self.assertNotIn("ALLOW_FROZEN_RERUN = True", self.frozen_code)
        self.assertIn("allow_rerun=ALLOW_FROZEN_RERUN", self.frozen_code)

    def test_the_frozen_notebook_refuses_smoke_mode(self):
        self.assertIn("MODEL_V2_SMOKE_TEST", self.frozen_code)
        smoke = self.frozen_code.index("MODEL_V2_SMOKE_TEST")
        load = self.frozen_code.index("load_cases(OOD_FINAL_PATH)")
        self.assertLess(smoke, load)

    def test_the_frozen_notebook_scores_int8_only(self):
        self.assertIn("ModelV2OnnxMasker(", self.frozen_code)
        self.assertIn('candidate["int8_path"]', self.frozen_code)
        for forbidden in ("model_fp32.onnx", "AutoModelForTokenClassification",
                          "predict_spans", "evaluate_partition", "TinyBERTMasker"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.frozen_code)

    def test_the_frozen_notebook_does_not_build_or_change_the_candidate(self):
        for forbidden in ("quantize_dynamic", "torch.onnx.export", "AdamW",
                          "augment_training_rows", "save_pretrained", "build_firewall",
                          "split_training_side", "save_checkpoint"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.frozen_code)

    def test_the_frozen_notebook_writes_the_expected_report(self):
        self.assertIn("accepted_candidate.FROZEN_REPORT_NAME", self.frozen_code)
        self.assertIn("REPORT_PATH.write_text", self.frozen_code)
        self.assertEqual(
            accepted_candidate.FROZEN_REPORT_NAME, "evaluation_ood_final_int8.json"
        )
        self.assertIn("subset_by(ood_final, predictions", self.frozen_code)
        self.assertIn("password_analysis(ood_final, predictions)", self.frozen_code)

    def test_the_frozen_notebook_stays_small(self):
        """Auditability is the point; growth here is a design smell."""
        self.assertLessEqual(len(self.frozen["cells"]), 20)
        self.assertLessEqual(len(self.frozen_code.splitlines()), 200)

    # -- CI must not spend the benchmark ------------------------------------

    def test_the_real_frozen_fixture_is_never_read(self):
        """Every fixture in this module is synthetic; none may name the real one.

        Checked over string literals rather than raw text, so this guard does not
        trip over its own description of what it forbids.
        """
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        guard = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == self._testMethodName
        )
        forbidden_names = ("ood_final.jsonl", "ood_final.manifest.json")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if guard.lineno <= node.lineno <= guard.end_lineno:
                continue
            with self.subTest(literal=node.value[:40]):
                self.assertNotEqual(node.value, "fixtures")
                self.assertFalse(
                    any(node.value.endswith(name) for name in forbidden_names),
                    f"line {node.lineno} names the real frozen fixture",
                )


if __name__ == "__main__":
    unittest.main()

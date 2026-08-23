"""The CPU benchmark runner's contracts, without loading a model.

``Full_Pipeline_CPU_Benchmark.ipynb`` was replaced by
``Full_Pipeline_CPU_Benchmark.py`` after a smoke run of both produced identical
selected rows, identical predicted spans, identical metrics and an identical
predictions digest. These tests pin the properties that made that comparison
meaningful, so the surviving implementation cannot quietly drift away from what
was validated: the mode/row resolution, the run identity, the artifact identity
gate, and the deterministic waveform Part A is timed on.

Loading the INT8 session costs about forty seconds, so nothing here does. The
end-to-end behaviour is exercised by running the script itself in smoke mode.
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from privacy_eval import benchmark_common as common
from privacy_eval import cpu_benchmark as cpu
from privacy_eval import deployment_benchmark as bench


def parse(argv: list[str]) -> argparse.Namespace:
    return cpu.build_parser().parse_args(argv)


class OptionResolutionTests(unittest.TestCase):
    """The mode decides the row count, the probe size and the pipeline loop."""

    def setUp(self) -> None:
        self._environ = {
            key: os.environ.pop(key)
            for key in ("ONKITH_BENCHMARK_SMOKE", "ONKITH_BENCHMARK_DEVICE")
            if key in os.environ
        }

    def tearDown(self) -> None:
        os.environ.update(self._environ)

    def test_the_default_is_the_full_two_thousand_row_benchmark(self) -> None:
        options = cpu.options_from_args(parse([]))
        self.assertEqual(options.mode, "full")
        self.assertEqual(options.target_rows, cpu.DEFAULT_BENCHMARK_ROWS)
        self.assertTrue(options.run_full_pipeline)
        self.assertTrue(options.run_ood_dev)

    def test_smoke_mode_shrinks_the_rows_the_probe_and_the_pipeline_loop(self) -> None:
        options = cpu.options_from_args(parse(["--smoke-test"]))
        self.assertEqual(options.mode, "smoke")
        self.assertEqual(options.target_rows, cpu.DEFAULT_SMOKE_ROWS)
        self.assertLessEqual(options.latency_probe_rows, 10)
        self.assertLessEqual(options.pipeline_latency_runs, 5)

    def test_an_explicit_row_count_overrides_the_mode_default(self) -> None:
        self.assertEqual(cpu.options_from_args(parse(["--rows", "100"])).target_rows, 100)
        smoke = cpu.options_from_args(parse(["--smoke-test", "--rows", "7"]))
        self.assertEqual(smoke.target_rows, 7)
        self.assertEqual(smoke.mode, "smoke")

    def test_zero_rows_is_rejected_rather_than_producing_an_empty_run(self) -> None:
        with self.assertRaises(ValueError):
            cpu.options_from_args(parse(["--rows", "0"]))

    def test_the_environment_overrides_the_notebook_honoured_still_work(self) -> None:
        os.environ["ONKITH_BENCHMARK_SMOKE"] = "1"
        os.environ["ONKITH_BENCHMARK_DEVICE"] = "pi5"
        options = cpu.options_from_args(parse([]))
        self.assertTrue(options.smoke_test)
        self.assertEqual(options.device_label, "pi5")

    def test_a_falsey_environment_value_does_not_enable_smoke_mode(self) -> None:
        for value in ("", "0", "false", "False"):
            os.environ["ONKITH_BENCHMARK_SMOKE"] = value
            self.assertFalse(cpu.options_from_args(parse([])).smoke_test, value)

    def test_an_explicit_flag_beats_a_falsey_environment_value(self) -> None:
        os.environ["ONKITH_BENCHMARK_SMOKE"] = "0"
        self.assertTrue(cpu.options_from_args(parse(["--smoke-test"])).smoke_test)

    def test_the_optional_sections_can_be_switched_off(self) -> None:
        options = cpu.options_from_args(parse(["--no-pipeline", "--no-ood-dev"]))
        self.assertFalse(options.run_full_pipeline)
        self.assertFalse(options.run_ood_dev)


class RunIdentityTests(unittest.TestCase):
    MOMENT = datetime(2026, 8, 18, 8, 15, 0, tzinfo=timezone.utc)

    def test_the_run_id_carries_the_timestamp_platform_and_mode(self) -> None:
        run_id = common.make_run_id(self.MOMENT, "full")
        self.assertTrue(run_id.startswith("20260818T081500Z_"))
        self.assertTrue(run_id.endswith("_full"))

    def test_a_device_label_is_appended_and_made_filesystem_safe(self) -> None:
        run_id = common.make_run_id(self.MOMENT, "smoke", "Pi 5 / kitchen")
        self.assertTrue(run_id.endswith("_pi-5---kitchen"))

    def test_a_text_run_id_is_never_mistaken_for_an_audio_run_id(self) -> None:
        text = common.make_run_id(self.MOMENT, "full")
        audio = common.make_run_id(self.MOMENT, "full", prefix="audio")
        self.assertNotIn("_audio_", text)
        self.assertIn("_audio_", audio)

    def test_slugs_never_leak_a_path_separator_into_a_directory_name(self) -> None:
        self.assertNotIn("/", common.slug("linux/aarch64"))
        self.assertNotIn("\\", common.slug("a\\b"))
        self.assertEqual(common.slug(""), "unknown")


class SpanRecordingTests(unittest.TestCase):
    ENTITIES = [
        {"start": 3, "end": 7, "entity_type": "GIVENNAME", "source": "model_v2"},
        {"start": 20, "end": 31, "entity_type": "EMAIL", "source": "structured_email"},
    ]

    def test_the_production_view_keeps_every_detected_entity(self) -> None:
        self.assertEqual(len(cpu.spans_of(self.ENTITIES)), 2)

    def test_the_model_attributed_view_drops_the_structured_layer(self) -> None:
        """The release measured the model alone, so the comparison needs this split."""

        spans = cpu.spans_of(self.ENTITIES, model_only=True)
        self.assertEqual([span["label"] for span in spans], ["GIVENNAME"])

    def test_a_digest_depends_on_the_boundaries_and_the_type(self) -> None:
        base = cpu.spans_of(self.ENTITIES)
        self.assertEqual(cpu.span_digest(base), cpu.span_digest(cpu.spans_of(self.ENTITIES)))
        moved = [dict(base[0], end=8), base[1]]
        retyped = [dict(base[0], label="SURNAME"), base[1]]
        self.assertNotEqual(cpu.span_digest(base), cpu.span_digest(moved))
        self.assertNotEqual(cpu.span_digest(base), cpu.span_digest(retyped))

    def test_no_predictions_still_produces_a_stable_digest(self) -> None:
        self.assertEqual(cpu.span_digest([]), cpu.span_digest([]))


class DeterministicWaveformTests(unittest.TestCase):
    """Part A compares devices, so the audio must be identical on all of them."""

    def test_the_same_seed_gives_the_same_bytes(self) -> None:
        first = cpu.deterministic_speech(seed=7)
        second = cpu.deterministic_speech(seed=7)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_a_different_seed_gives_different_audio(self) -> None:
        self.assertNotEqual(
            cpu.deterministic_speech(seed=7).tobytes(),
            cpu.deterministic_speech(seed=8).tobytes(),
        )

    def test_the_waveform_is_float32_in_range_and_the_right_length(self) -> None:
        audio = cpu.deterministic_speech(seconds=3.0)
        self.assertEqual(audio.dtype.name, "float32")
        self.assertEqual(len(audio), 3 * common.SAMPLE_RATE)
        self.assertLessEqual(float(abs(audio).max()), 1.0)


class ArtifactIdentityTests(unittest.TestCase):
    """The gate that decides whether privacy numbers may be produced at all."""

    @classmethod
    def setUpClass(cls) -> None:
        # Not a bare sys.path.insert: another test module may already have
        # imported FullPipeline_V4's integrated_pipeline, and an insert does
        # not dislodge sys.modules. ensure_import_paths does.
        common.ensure_import_paths()
        from integrated_pipeline.config import PipelineConfig

        cls.config = PipelineConfig(root=common.PRODUCTION_ROOT)
        cls.manifest = common.verify_privacy_artifacts(
            cls.config, "unit-test", common.WarningLog()
        )

    def test_the_committed_artifact_passes_every_identity_check(self) -> None:
        self.assertTrue(self.manifest["ok"], self.manifest["identity_checks"])
        self.assertTrue(all(self.manifest["identity_checks"].values()))

    def test_the_hash_matches_the_pinned_release_artifact(self) -> None:
        self.assertEqual(
            self.manifest["measured_sha256"], str(self.config.pii["sha256"])
        )
        self.assertEqual(
            self.manifest["files"]["model_int8.onnx"]["size_bytes"],
            int(self.config.pii["size_bytes"]),
        )

    def test_the_label_map_is_the_model_v2_ontology(self) -> None:
        label_map = self.manifest["label_map"]
        self.assertEqual(label_map["num_entities"], 31)
        self.assertEqual(label_map["num_labels"], 63)
        self.assertEqual(label_map["ontology_version"], "model_v2.ontology.1")
        self.assertEqual(set(label_map["id2label"]), set(range(63)))

    def test_a_wrong_pinned_hash_fails_the_gate_rather_than_being_ignored(self) -> None:
        broken = mock.MagicMock(wraps=self.config)
        broken.pii = dict(self.config.pii, sha256="0" * 64)
        broken.pii_model_path = self.config.pii_model_path
        broken.pii_tokenizer_path = self.config.pii_tokenizer_path
        broken.pii_label_map_path = self.config.pii_label_map_path
        manifest = common.verify_privacy_artifacts(broken, "unit-test", common.WarningLog())
        self.assertFalse(manifest["ok"])
        self.assertFalse(manifest["identity_checks"]["live_config.pii.sha256"])

    def test_the_written_manifest_holds_no_runtime_conveniences(self) -> None:
        on_disk = common.artifact_manifest_for_disk(self.manifest)
        self.assertNotIn("label_map", on_disk)
        self.assertNotIn("release_metrics", on_disk)
        self.assertIn("identity_checks", on_disk)
        self.assertIn("files", on_disk)

    def test_the_release_reference_for_the_ood_crosscheck_is_present(self) -> None:
        """Without it the ood_dev section has nothing to reproduce."""

        released = self.manifest["release_metrics"].get("ood_dev")
        self.assertIsNotNone(released)
        self.assertAlmostEqual(released["typed_entity_f1"], 0.6882, places=4)


class SubsetReuseTests(unittest.TestCase):
    """The runner must score the committed rows, not reselect its own."""

    def test_smoke_rows_are_a_prefix_of_the_committed_fixture(self) -> None:
        rows, _ = bench.load_subset()
        prefix = bench.take(rows, cpu.DEFAULT_SMOKE_ROWS)
        self.assertEqual([row["row_id"] for row in prefix],
                         [row["row_id"] for row in rows[:cpu.DEFAULT_SMOKE_ROWS]])

    def test_the_hard_negatives_the_runner_adds_back_exist(self) -> None:
        """A 50-row prefix can hold none, leaving the FP rate no denominator."""

        rows, manifest = bench.load_subset()
        zero_pii = [row for row in rows if not row["privacy_mask"]]
        self.assertEqual(len(zero_pii), manifest["coverage"]["rows_without_pii"])
        self.assertGreater(len(zero_pii), 0)


class EntryPointTests(unittest.TestCase):
    def test_the_root_script_is_a_thin_wrapper_over_the_runner(self) -> None:
        text = (common.REPO_ROOT / "Full_Pipeline_CPU_Benchmark.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from privacy_eval.cpu_benchmark import main", text)
        # Docstring plus a dozen lines of path setup. The bound is a guard
        # against measurement logic creeping back into the entry point, not a
        # style rule, so it has room for the explanation a reader needs.
        self.assertLess(len(text.splitlines()), 80)

    def test_the_obsolete_notebook_is_gone(self) -> None:
        """One canonical runner. Two implementations would drift apart."""

        self.assertFalse((common.REPO_ROOT / "Full_Pipeline_CPU_Benchmark.ipynb").exists())

    def test_the_figures_notebook_is_kept(self) -> None:
        self.assertTrue((common.REPO_ROOT / "Research_Paper_Figures.ipynb").is_file())

    def test_both_benchmarks_expose_a_help_only_command_line(self) -> None:
        from privacy_eval import audio_benchmark as audio
        from privacy_eval import end_to_end_benchmark as e2e

        for parser in (cpu.build_parser(), audio.build_parser(), e2e.build_parser()):
            with self.assertRaises(SystemExit) as raised:
                parser.parse_args(["--help"])
            self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

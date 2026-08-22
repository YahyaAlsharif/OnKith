from __future__ import annotations

import unittest
import json

from integrated_pipeline.config import PipelineConfig


class CanaryConfigurationTests(unittest.TestCase):
    def test_exact_model_decoder_and_transcription_contract(self) -> None:
        config = PipelineConfig()
        self.assertEqual(config.canary["model_id"], "nvidia/canary-180m-flash")
        self.assertEqual(
            config.canary["revision"],
            "c770bfa0f70f66ad89dbf9827985f999675943a3",
        )
        self.assertEqual(
            config.canary["sha256"],
            "7b97de18b718ef01bf3398715ce8d18486c4d83961e230230b26f5712e112675",
        )
        self.assertEqual(config.canary["strategy"], "beam")
        self.assertEqual(config.canary["beam_size"], 2)
        self.assertTrue(config.canary["return_best_hypothesis"])
        self.assertEqual(config.canary["temperature"], 1.0)
        self.assertEqual(config.canary["max_generation_delta"], 50)
        self.assertFalse(config.canary["enable_chunking"])
        self.assertFalse(config.canary["timestamps"])
        self.assertEqual(config.canary["pnc"], "no")

    def test_exact_sample_counts(self) -> None:
        audio = PipelineConfig().audio
        expected = {
            "input_chunk_samples": 1_600,
            "calibration_samples": 12_000,
            "pre_roll_samples": 6_144,
            "start_confirm_samples": 3_072,
            "minimum_speech_samples": 3_072,
            "dropout_bridge_samples": 2_048,
            "endpoint_silence_samples": 39_936,
            "maximum_unique_segment_samples": 136_192,
            "forced_overlap_samples": 5_632,
            "trailing_context_samples": 5_632,
            "adjacent_submitted_overlap_samples": 11_264,
        }
        for key, value in expected.items():
            self.assertEqual(audio[key], value)

    def test_preserved_stage5_privacy_and_storage(self) -> None:
        config = PipelineConfig()
        self.assertEqual(config.stage5["mode"], "immediate")
        self.assertEqual(config.storage["journal_mode"], "WAL")
        self.assertEqual(config.storage["synchronous"], "NORMAL")
        self.assertEqual(config.storage["filename"], "session_transcript.db")
        self.assertEqual(config.pii["provider"], "CPUExecutionProvider")
        self.assertEqual(config.pii["word_policy"], "whole_word_union")

    def test_manifest_matches_configuration(self) -> None:
        config = PipelineConfig()
        manifest = json.loads(
            config.canary_manifest_path.read_text(encoding="utf-8")
        )
        for key in (
            "model_id",
            "revision",
            "artifact_file",
            "sha256",
            "backend",
            "nemo_version",
            "torch_version",
            "precision",
        ):
            self.assertEqual(manifest[key], config.canary[key])


if __name__ == "__main__":
    unittest.main()

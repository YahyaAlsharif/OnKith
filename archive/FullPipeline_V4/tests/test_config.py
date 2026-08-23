from __future__ import annotations

import unittest
import json

from integrated_pipeline.config import PipelineConfig


class WhisperConfigurationTests(unittest.TestCase):
    def test_exact_model_and_decoder_contract(self) -> None:
        config = PipelineConfig()
        self.assertEqual(config.whisper["engine"], "faster-whisper")
        self.assertEqual(config.whisper["runtime"], "ctranslate2")
        self.assertEqual(config.whisper["package_version"], "1.2.1")
        self.assertEqual(config.whisper["model"], "base.en")
        self.assertEqual(config.whisper["model_id"], "Systran/faster-whisper-base.en")
        self.assertEqual(config.whisper["device"], "cpu")
        self.assertEqual(config.whisper["compute_type"], "int8")
        self.assertEqual(config.whisper["beam_size"], 5)
        self.assertEqual(config.whisper["language"], "en")
        self.assertEqual(config.whisper["sample_rate"], 16_000)

    def test_the_hint_is_built_from_the_configured_names(self) -> None:
        # Null rather than a written-out string, so renaming the device in
        # `names` re-biases the recogniser instead of leaving a stale word
        # compiled into the adapter.
        config = PipelineConfig()
        self.assertIsNone(config.whisper["initial_prompt"])
        self.assertEqual(config.whisper["names"], ["onkith"])

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

    def test_no_committed_stt_artifact_is_claimed(self) -> None:
        """V3 pinned a committed .nemo file by SHA-256; V4 has no such file.

        faster-whisper resolves `base.en` through the CTranslate2 / Hugging
        Face cache, so the config must not advertise an artifact path that
        nothing writes - a manifest nobody produces is worse than none.
        """

        config = PipelineConfig()
        self.assertNotIn("artifact", config.whisper)
        self.assertNotIn("manifest", config.whisper)
        self.assertFalse(hasattr(config, "canary_model_path"))
        # Silero is still a committed file and is still required.
        self.assertTrue(config.silero_model_path.is_file())


if __name__ == "__main__":
    unittest.main()

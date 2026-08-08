from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

import numpy as np

from integrated_pipeline.canary_adapter import CanaryAdapter, describe_nemo_backend
from integrated_pipeline.config import PipelineConfig
from tests.helpers import canary_segment


class TransformerAEDBeamInfer:
    def __init__(self) -> None:
        self.beam_size = 2
        self.return_best_hypothesis = True
        self.beam_search = types.SimpleNamespace(max_delta_len=50)


class FakeModel:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.cfg = types.SimpleNamespace(
            decoding={
                "strategy": "greedy",
                "temperature": 0.5,
                "beam": {
                    "beam_size": 1,
                    "return_best_hypothesis": False,
                    "max_generation_delta": 1,
                },
            }
        )
        self.outputs = list(outputs or ["warmup", "first", "second"])
        self.inputs: list[np.ndarray] = []
        self.override_configs: list[object] = []

    def change_decoding_strategy(self, decoding: object) -> None:
        self.cfg.decoding = decoding
        self.decoding = types.SimpleNamespace(
            cfg=decoding,
            decoding=TransformerAEDBeamInfer(),
        )

    def transcribe(self, *, audio: list[np.ndarray], override_config: object):
        self.inputs.append(audio[0].copy())
        self.override_configs.append(override_config)
        return [types.SimpleNamespace(text=self.outputs.pop(0))]


def transcription_config(**kwargs: object) -> object:
    return types.SimpleNamespace(**kwargs)


class CanaryAdapterTests(unittest.TestCase):
    def test_load_once_warm_once_and_independent_calls(self) -> None:
        model = FakeModel()
        adapter = CanaryAdapter(
            PipelineConfig(),
            model_loader=lambda path: model,
            transcription_config_factory=transcription_config,
            verify_artifacts=False,
            verify_versions=False,
        )
        adapter.load()
        adapter.load()
        first = adapter.transcribe_segment(canary_segment(0, final=True))
        second = adapter.transcribe_segment(
            canary_segment(0, final=True, utterance=2)
        )
        self.assertEqual((first.text, second.text), ("first", "second"))
        self.assertEqual(adapter.counters.model_load_count, 1)
        self.assertEqual(adapter.counters.warmup_count, 1)
        self.assertEqual(adapter.counters.inference_count, 2)
        self.assertEqual(len(model.inputs), 3)
        self.assertIsNot(model.inputs[1], model.inputs[2])
        active = adapter.audit()["active_decoder"]
        self.assertEqual(active["strategy"], "beam")
        self.assertEqual(active["beam_size"], 2)
        self.assertEqual(active["engine"], "TransformerAEDBeamInfer")
        config = model.override_configs[-1]
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.num_workers, 0)
        self.assertFalse(config.timestamps)
        self.assertFalse(config.enable_chunking)
        adapter.close()
        adapter.close()
        self.assertEqual(adapter.counters.close_count, 1)

    def test_nbest_shape_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "N-best"):
            CanaryAdapter._extract_one_text([[types.SimpleNamespace(text="a")]])

    def test_prompt_contract_is_verified(self) -> None:
        def wrong_prompt(**kwargs: object) -> object:
            kwargs["prompt"] = [
                {
                    "role": "user",
                    # Punctuation-and-capitalisation silently turned on.
                    "slots": {
                        "source_lang": "en",
                        "target_lang": "en",
                        "task": "asr",
                        "pnc": "yes",
                        "timestamp": "no",
                    },
                }
            ]
            return types.SimpleNamespace(**kwargs)

        adapter = CanaryAdapter(
            PipelineConfig(),
            model_loader=lambda path: FakeModel(),
            transcription_config_factory=wrong_prompt,
            verify_artifacts=False,
            verify_versions=False,
        )
        with self.assertRaisesRegex(RuntimeError, "pnc="):
            adapter.load()

    def test_active_prompt_is_recorded_for_audit(self) -> None:
        adapter = CanaryAdapter(
            PipelineConfig(),
            model_loader=lambda path: FakeModel(),
            transcription_config_factory=transcription_config,
            verify_artifacts=False,
            verify_versions=False,
        )
        adapter.load()
        self.assertEqual(
            adapter.audit()["active_prompt"],
            [
                {
                    "role": "user",
                    "slots": {
                        "source_lang": "en",
                        "target_lang": "en",
                        "task": "asr",
                        "pnc": "no",
                        "timestamp": "no",
                    },
                }
            ],
        )
        adapter.close()


class InferenceModeTests(unittest.TestCase):
    """Warmup and every inference must run under torch.inference_mode()."""

    def test_inference_mode_wraps_warmup_and_every_inference(self) -> None:
        events: list[str] = []

        class Recorder:
            def __enter__(self):
                events.append("enter")
                return self

            def __exit__(self, *args):
                events.append("exit")
                return False

        fake_torch = types.SimpleNamespace(inference_mode=lambda: Recorder())
        model = FakeModel()
        adapter = CanaryAdapter(
            PipelineConfig(),
            model_loader=lambda path: model,
            transcription_config_factory=transcription_config,
            verify_artifacts=False,
            verify_versions=False,
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            adapter.load()
            self.assertEqual(events, ["enter", "exit"])  # warmup
            adapter.transcribe_segment(canary_segment(0, final=True))
            adapter.transcribe_segment(canary_segment(0, final=True, utterance=2))
        # One enter/exit pair per model call: warmup plus two inferences.
        self.assertEqual(events, ["enter", "exit"] * 3)
        self.assertEqual(len(model.inputs), 3)
        self.assertTrue(adapter.audit()["inference_mode_active"])
        adapter.close()

    def test_real_backend_refuses_to_run_without_torch(self) -> None:
        adapter = CanaryAdapter(
            PipelineConfig(), verify_artifacts=False, verify_versions=False
        )
        # No model_loader means the real NeMo path, which must never silently
        # fall back to running outside inference_mode.
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(ImportError):
                adapter._inference_context()


class BackendIntrospectionTests(unittest.TestCase):
    def test_describe_backend_reports_absence_without_raising(self) -> None:
        report = describe_nemo_backend()
        self.assertIn("nemo_version", report)
        self.assertIn("torch_version", report)
        self.assertIsInstance(report["available"], bool)
        if not report["available"]:
            # Absence is reported, never silently treated as a satisfied API.
            self.assertTrue(report["errors"])
            self.assertNotIn("signature_contract_satisfied", report)


if __name__ == "__main__":
    unittest.main()

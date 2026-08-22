from __future__ import annotations

import types
import unittest

import numpy as np

from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.silero_vad import SileroOnnxVAD


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def get_inputs(self):
        return [types.SimpleNamespace(name=name) for name in ("input", "state", "sr")]

    def get_outputs(self):
        return [types.SimpleNamespace(name=name) for name in ("output", "stateN")]

    def run(self, outputs, inputs):
        self.calls.append({key: value.copy() for key, value in inputs.items()})
        next_state = inputs["state"] + 1
        return np.asarray([[0.75]], dtype=np.float32), next_state


class SileroVADTests(unittest.TestCase):
    def test_exact_artifact_metadata(self) -> None:
        metadata = SileroOnnxVAD(PipelineConfig()).verify_artifact()
        self.assertEqual(metadata["size_bytes"], 2_327_524)
        self.assertEqual(
            metadata["sha256"],
            "597d30b3ec076608d059477bb14cfeffdf951bf5cae370d38f65d33bbfe82004",
        )
        self.assertEqual(metadata["opsets"][""], 16)

    def test_state_and_context_continue_until_explicit_reset(self) -> None:
        session = FakeSession()
        vad = SileroOnnxVAD(
            PipelineConfig(), session_factory=lambda path: session
        )
        self.assertEqual(vad.probability(np.ones(512, dtype=np.float32)), 0.75)
        self.assertEqual(vad.probability(np.full(512, 2, dtype=np.float32)), 0.75)
        self.assertEqual(session.calls[0]["input"].shape, (1, 576))
        self.assertTrue(np.all(session.calls[1]["input"][:, :64] == 1))
        self.assertTrue(np.all(session.calls[1]["state"] == 1))
        vad.reset()
        vad.probability(np.full(512, 3, dtype=np.float32))
        self.assertTrue(np.all(session.calls[2]["input"][:, :64] == 0))
        self.assertTrue(np.all(session.calls[2]["state"] == 0))

    def test_requires_exact_512_and_finite_probability(self) -> None:
        vad = SileroOnnxVAD(
            PipelineConfig(), session_factory=lambda path: FakeSession()
        )
        with self.assertRaises(ValueError):
            vad.probability(np.zeros(511, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

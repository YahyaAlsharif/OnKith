from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


class IIDBaselineTests(unittest.TestCase):
    def test_frozen_sources_still_match(self):
        baseline = json.loads((ROOT / "baselines" / "tinybert_iid_baseline.json").read_text(encoding="utf-8"))
        for source in baseline["sources"]:
            path = (ROOT / "baselines" / source["path"]).resolve()
            self.assertTrue(path.is_file(), path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["sha256"])

    def test_core_continuity_metrics_are_frozen(self):
        baseline = json.loads((ROOT / "baselines" / "tinybert_iid_baseline.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(baseline["onnx_fp32"]["typed_entity_f1"], 0.9565065826189437)
        self.assertAlmostEqual(baseline["onnx_int8"]["typed_entity_f1"], 0.9496185057036707)
        self.assertEqual(baseline["dataset"]["scored_final_test_rows"], 40908)


if __name__ == "__main__":
    unittest.main()

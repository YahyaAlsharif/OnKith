"""Replacing the recognizer did not change the masking implementation.

V4 is V3 with faster-whisper in place of Canary. Everything after the recognizer
is meant to be the same code, and "meant to be" is not evidence. This runs the
same fixed text through V3's privacy path and through V4's, and requires the
masked output, the consolidated spans and the rendered final text to be
identical.

V4 has exactly one deliberate addition, and it is why the parity is asserted on
these texts specifically: ``spoken_form_normalizer`` rewrites a spoken address -
"billing at example.org" - into the written form before the detectors see it,
because Whisper produces that shape and neither the model nor the structured
EMAIL pattern was trained or written for it. None of ``PROBE_TEXTS`` contains a
spoken address, so on every one of them V3 and V4 must still agree exactly; the
rewrite's own behaviour is pinned in
``FullPipeline_V4/tests/test_spoken_form_normalizer.py``.

**No speech recognizer is loaded on either side.** The probe imports the masker,
the merger, the wake-word repair, the buffer and the digit renderer, and nothing
else; running it against ``FullPipeline_V3`` cannot start Canary. The
Canary measurements in ``research_results/`` are frozen and are never
regenerated.

Each root is probed in its own interpreter because both packages are called
``integrated_pipeline``: importing the second into a process that already holds
the first would silently compare a pipeline against itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE = REPO_ROOT / "model" / "privacy_evaluation" / "scripts" / "masking_parity_probe.py"
V4_ROOT = REPO_ROOT / "FullPipeline_V4"
V3_ROOT = REPO_ROOT / "FullPipeline_V3"


def run_probe(root: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(PROBE), str(root)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"probe failed for {root.name}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return json.loads(result.stdout)


class MaskingParityTests(unittest.TestCase):
    v3: dict
    v4: dict

    @classmethod
    def setUpClass(cls) -> None:
        for root in (V3_ROOT, V4_ROOT):
            if not (root / "integrated_pipeline" / "pii_adapter.py").is_file():
                raise unittest.SkipTest(f"{root} is not present")
        cls.v3 = run_probe(V3_ROOT)
        cls.v4 = run_probe(V4_ROOT)

    def test_both_roots_carry_the_same_privacy_artifact(self) -> None:
        # The comparison only means anything if it is one model on both sides.
        self.assertEqual(
            self.v3["privacy_artifact_sha256"], self.v4["privacy_artifact_sha256"]
        )
        self.assertEqual(self.v3["label_count"], self.v4["label_count"])
        self.assertEqual(self.v3["word_policy"], self.v4["word_policy"])
        self.assertEqual(self.v3["structured_enabled"], self.v4["structured_enabled"])

    def test_the_probe_covered_both_transcript_surfaces(self) -> None:
        # Canary produced lowercase, unpunctuated text; Whisper punctuates and
        # capitalises. Both shapes have to be in the probe or the parity claim
        # only covers the surface one of them ever produced.
        inputs = [row["input"] for row in self.v4["rows"]]
        self.assertTrue(any(text.endswith(".") for text in inputs))
        self.assertTrue(any(text.islower() and "." not in text for text in inputs))

    def test_identical_text_gives_identical_masked_output(self) -> None:
        self.assertEqual(len(self.v3["rows"]), len(self.v4["rows"]))
        for old, new in zip(self.v3["rows"], self.v4["rows"]):
            with self.subTest(text=new["input"]):
                self.assertEqual(old["input"], new["input"])
                self.assertEqual(old["privacy_input"], new["privacy_input"])
                self.assertEqual(old["masked_text"], new["masked_text"])

    def test_identical_text_gives_identical_spans(self) -> None:
        for old, new in zip(self.v3["rows"], self.v4["rows"]):
            with self.subTest(text=new["input"]):
                self.assertEqual(old["entities"], new["entities"])

    def test_identical_text_gives_identical_released_text(self) -> None:
        # The spoken-digit renderer runs after masking, so it is the last thing
        # that can put private information back on screen. It has to match too.
        for old, new in zip(self.v3["rows"], self.v4["rows"]):
            with self.subTest(text=new["input"]):
                self.assertEqual(old["final_output"], new["final_output"])

    def test_the_wake_word_repair_behaves_the_same(self) -> None:
        for old, new in zip(self.v3["rows"], self.v4["rows"]):
            with self.subTest(text=new["input"]):
                self.assertEqual(old["wake_variant"], new["wake_variant"])

    def test_the_probe_never_imports_a_recognizer(self) -> None:
        """Reading the archived Canary pipeline must not be able to run it."""

        source = PROBE.read_text(encoding="utf-8")
        for forbidden in ("canary_adapter", "whisper_adapter", "faster_whisper", "nemo"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(f"import {forbidden}", source)
                self.assertNotIn(f"from {forbidden}", source)
                self.assertNotIn(f"integrated_pipeline.{forbidden}", source)


if __name__ == "__main__":
    unittest.main()

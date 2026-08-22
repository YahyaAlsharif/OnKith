"""Recomputing a saved run's entity outcomes without a recognizer or a model.

The reconstruction has to be exact, not approximate, or every number it produces
is a guess. The guard is each run folder's own recorded ``mask_coverage``:
replaying it from the reconstructed spans has to reproduce it to the bit, on
every committed audio run, Canary and Whisper alike.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "model" / "privacy_evaluation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "model" / "privacy_evaluation"))

from privacy_eval import rescore  # noqa: E402
from privacy_eval.benchmark_common import RESULTS_ROOT  # noqa: E402


def audio_run_folders() -> list[Path]:
    if not RESULTS_ROOT.is_dir():
        return []
    folders = []
    for directory in sorted(RESULTS_ROOT.glob("*")):
        config = directory / "benchmark_config.json"
        if not config.is_file():
            continue
        payload = json.loads(config.read_text(encoding="utf-8"))
        # A committed failed attempt is still useful environment evidence, but
        # it has no STT predictions to reconstruct or rescore.
        if (payload.get("benchmark") == "audio_pipeline"
                and payload.get("stt_available")):
            folders.append(directory)
    return folders


class SpanRecoveryTests(unittest.TestCase):
    """Reconstructing what the masker replaced, from its input and its output."""

    def test_one_replaced_region_is_found(self) -> None:
        spans = rescore.recover_predicted_spans(
            "My name is Alex Carter.", "My name is [GIVENNAME_1] Carter."
        )
        self.assertEqual(spans, [(11, 15)])

    def test_two_separated_regions_are_found(self) -> None:
        spans = rescore.recover_predicted_spans(
            "My name is Alex Carter.",
            "My name is [GIVENNAME_1] [SURNAME_1].",
        )
        self.assertEqual(spans, [(11, 15), (16, 22)])

    def test_adjacent_placeholders_give_one_combined_region(self) -> None:
        """``[PASSPORTNUM_1][TELEPHONENUM_1]`` is a real merger output.

        Nothing anchors the join between them, so they come back as one range.
        Character coverage is a union, so this is the same answer.
        """

        spans = rescore.recover_predicted_spans(
            "Call me back on 555-0147.",
            "Call me back on [PASSPORTNUM_1][TELEPHONENUM_1].",
        )
        self.assertEqual(spans, [(16, 24)])

    def test_a_region_at_the_end_of_the_text_is_found(self) -> None:
        spans = rescore.recover_predicted_spans(
            "Call Alex Carter", "Call [GIVENNAME_1] [SURNAME_1]"
        )
        self.assertEqual(rescore.covered_characters(spans), set(range(5, 9)) | set(range(10, 16)))

    def test_unmasked_text_recovers_nothing(self) -> None:
        self.assertEqual(
            rescore.recover_predicted_spans("what time is it", "what time is it"), []
        )

    def test_text_that_does_not_align_is_rejected_rather_than_guessed(self) -> None:
        with self.assertRaises(rescore.RecoveryFailed):
            rescore.recover_predicted_spans(
                "My name is Alex Carter.", "A completely different [SURNAME_1] sentence."
            )


class StoredRunTests(unittest.TestCase):
    """The guard, run against every completed audio folder in the repository."""

    def setUp(self) -> None:
        self.folders = audio_run_folders()
        if not self.folders:
            self.skipTest("no audio benchmark runs committed")

    def test_every_stored_run_reconstructs_exactly(self) -> None:
        for directory in self.folders:
            with self.subTest(run=directory.name):
                report = rescore.verify_recovery(directory)
                self.assertTrue(report["exact"], report)
                self.assertGreater(report["entities_checked"], 0)

    def test_every_stored_run_rescores(self) -> None:
        for directory in self.folders:
            with self.subTest(run=directory.name):
                scored = rescore.rescore_run(directory)
                self.assertIsNotNone(scored)
                self.assertEqual(scored["expected_entities"],
                                 sum(scored["outcomes"].values()))
                self.assertLessEqual(scored["entities_fully_masked"],
                                     scored["entities_protected"])
                self.assertLessEqual(scored["entities_protected"],
                                     scored["expected_entities"])

    def test_hard_negatives_exclude_the_unannotated_subset(self) -> None:
        """LibriSpeech carries no gold spans; counting it would invent negatives."""

        for directory in self.folders:
            with self.subTest(run=directory.name):
                scored = rescore.rescore_run(directory)
                if scored["clips_scored"] < 25:
                    continue  # a smoke run scores a handful of clips
                self.assertEqual(scored["hard_negative_clips"], 7)

    def test_scoring_leaves_the_folders_untouched(self) -> None:
        before = {
            path: path.stat().st_mtime_ns
            for directory in self.folders for path in sorted(directory.iterdir())
        }
        for directory in self.folders:
            rescore.rescore_run(directory)
        after = {
            path: path.stat().st_mtime_ns
            for directory in self.folders for path in sorted(directory.iterdir())
        }
        self.assertEqual(before, after)

    def test_the_module_never_imports_a_recognizer(self) -> None:
        source = (Path(rescore.__file__)).read_text(encoding="utf-8")
        for forbidden in ("faster_whisper", "whisper_adapter", "canary_adapter",
                          "nemo", "onnxruntime"):
            self.assertNotIn(forbidden, source)


class ComparabilityTests(unittest.TestCase):
    """What the rescoring is for: two engines, one definition."""

    def setUp(self) -> None:
        self.scored = [
            item for item in (rescore.rescore_run(d) for d in audio_run_folders())
            if item and item["clips_scored"] >= 25
        ]
        if not self.scored:
            self.skipTest("no full audio benchmark runs committed")

    def test_every_full_run_scores_the_same_entities(self) -> None:
        """A denominator that moved between arms would void every comparison."""

        self.assertEqual(
            {item["expected_entities"] for item in self.scored}, {60}
        )

    def test_a_canary_run_and_a_whisper_run_are_both_scored(self) -> None:
        engines = {item["engine"] for item in self.scored}
        self.assertIn("canary", engines)
        self.assertIn("whisper", engines)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

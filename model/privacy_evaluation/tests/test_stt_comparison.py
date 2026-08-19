"""The engine-comparable scoring, and the promise that it never runs a model.

Canary wrote spoken numbers as words because it ran with ``pnc: no``; Whisper
writes numerals. The benchmark's standing word error rate charges eight
substitutions for "five five five zero one eight eight" against "55550188",
which is a correct transcription. These tests pin the normalisation that makes
the two engines comparable, and pin the failures it must still catch.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = REPO_ROOT / "model" / "privacy_evaluation"
for extra in (str(EVALUATION_ROOT), str(REPO_ROOT / "FullPipeline_V4")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from privacy_eval import stt_comparison as compare

RESULTS_ROOT = REPO_ROOT / "research_results" / "cpu_benchmarks"


class NumberNormalisationTests(unittest.TestCase):
    def test_a_spoken_digit_run_becomes_the_number_it_is(self) -> None:
        self.assertEqual(
            compare.normalise_numbers("my number is five five five zero one four seven"),
            "my number is 5550147",
        )

    def test_a_compound_is_not_a_digit_run(self) -> None:
        # "sixty three" is 63, not 603. Getting this wrong would silently
        # invent a leaked identifier.
        self.assertEqual(compare.normalise_numbers("she is sixty three"), "she is 63")

    def test_teens_and_bare_tens(self) -> None:
        self.assertEqual(compare.normalise_numbers("eleven"), "11")
        self.assertEqual(compare.normalise_numbers("set a timer for twenty minutes"),
                         "set a timer for 20 minutes")

    def test_an_ordinal_matches_the_numeral_a_recogniser_writes(self) -> None:
        self.assertEqual(compare.normalise_numbers("on the nineteenth of january"),
                         "on the 19 of january")
        self.assertEqual(compare.normalise_numbers("on the 19th of January"),
                         "on the 19 of january")

    def test_a_decade_matches(self) -> None:
        self.assertEqual(compare.normalise_numbers("THE TWENTIES"),
                         compare.normalise_numbers("The 20s"))

    def test_and_inside_a_number_is_dropped_but_not_elsewhere(self) -> None:
        self.assertEqual(compare.normalise_numbers("five five and four seven"), "5547")
        self.assertEqual(compare.normalise_numbers("bread and butter"), "bread and butter")

    def test_the_two_spellings_of_one_utterance_score_zero(self) -> None:
        pairs = [
            ("Set a timer for twenty minutes.", "Set a timer for 20 minutes."),
            ("She is sixty three.", "She is 63."),
            ("on the nineteenth of January at ten past eleven",
             "on the 19th of January at 10 past 11"),
            ("my id card number is one two three four five six seven eight nine zero",
             "My ID card number is 1234567890."),
        ]
        for reference, hypothesis in pairs:
            with self.subTest(reference=reference):
                self.assertEqual(
                    compare.number_normalised_wer(reference, hypothesis)["wer"], 0.0
                )

    def test_a_real_digit_error_is_still_an_error(self) -> None:
        """The normalisation must not launder a wrong number into a match.

        This is the case the whole benchmark exists for: an inserted digit
        inside a telephone number.
        """

        reference = "her contact number is five five five zero one eight eight"
        hypothesis = "her contact number is 55550188"
        self.assertGreater(
            compare.number_normalised_wer(reference, hypothesis)["wer"], 0.0
        )
        digits = compare.digit_preservation(reference, hypothesis)
        self.assertEqual(digits["expected"], ["5550188"])
        self.assertEqual(digits["produced"], ["55550188"])
        self.assertEqual(digits["accuracy"], 0.0)

    def test_a_clip_without_a_long_number_is_not_scored_for_digits(self) -> None:
        # None, not zero: a clip with nothing to preserve must not drag the rate
        # down or prop it up.
        self.assertIsNone(compare.digit_preservation("what time is it", "what time is it"))

    def test_an_exactly_preserved_identifier_scores_one(self) -> None:
        digits = compare.digit_preservation(
            "my account is one two three four five six seven eight",
            "My account is 12345678.",
        )
        self.assertEqual(digits["accuracy"], 1.0)


class ReleasedIdentifierTests(unittest.TestCase):
    """The privacy question that does not depend on aligning two spellings."""

    def test_a_rebuilt_number_in_the_released_text_is_a_leak(self) -> None:
        row = {
            "clip_id": "probe",
            "reference_transcript": "Call me back on five five five, zero one four seven.",
            "final_masked_output": "call me back on 5550147",
        }
        leak = compare.released_identifier_leak(row)
        self.assertFalse(leak["clean"])
        self.assertEqual(leak["leaked"], ["5550147"])

    def test_a_masked_number_is_clean(self) -> None:
        row = {
            "clip_id": "probe",
            "reference_transcript": "Call me back on five five five, zero one four seven.",
            "final_masked_output": "Call me back on [TELEPHONENUM_1].",
        }
        self.assertTrue(compare.released_identifier_leak(row)["clean"])

    def test_a_number_left_as_digit_words_is_the_same_leak(self) -> None:
        # A masker that misses the spoken form has leaked the number just as
        # surely as one that misses the numeral; the placeholder index digits
        # in "[GIVENNAME_1]" are far too short to be mistaken for one.
        row = {
            "clip_id": "probe",
            "reference_transcript": "my account is one two three four five six seven eight",
            "final_masked_output": "[GIVENNAME_1] account is one two three four five six seven eight",
        }
        leak = compare.released_identifier_leak(row)
        self.assertFalse(leak["clean"])
        self.assertEqual(leak["leaked"], ["12345678"])

    def test_a_clip_with_nothing_to_protect_is_not_scored(self) -> None:
        self.assertIsNone(compare.released_identifier_leak({
            "clip_id": "probe",
            "reference_transcript": "what time is it",
            "final_masked_output": "what time is it",
        }))


class EngineIdentificationTests(unittest.TestCase):
    def test_a_run_without_the_key_is_canary(self) -> None:
        """The key was added in the change that replaced Canary.

        Every run folder predating it is a Canary run, so its absence is an
        identification rather than a guess.
        """

        self.assertEqual(compare.engine_of({}), "canary")
        self.assertEqual(compare.pipeline_of({}), "V3")

    def test_a_v4_run_says_so(self) -> None:
        config = {"stt_engine": "whisper", "pipeline_version": "V4"}
        self.assertEqual(compare.engine_of(config), "whisper")
        self.assertEqual(compare.pipeline_of(config), "V4")


class SafetyTests(unittest.TestCase):
    def test_the_module_never_imports_a_recognizer(self) -> None:
        """Scoring the frozen Canary folders must not be able to run Canary."""

        source = (
            EVALUATION_ROOT / "privacy_eval" / "stt_comparison.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("nemo", "canary_adapter", "whisper_adapter", "faster_whisper"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(f"import {forbidden}", source)
                self.assertNotIn(f"from {forbidden}", source)


class StoredCanaryRunTests(unittest.TestCase):
    """The historical Canary artifacts still parse with the current code."""

    @classmethod
    def setUpClass(cls) -> None:
        if not RESULTS_ROOT.is_dir():
            raise unittest.SkipTest("no result folders committed")
        cls.canary = [
            directory for directory in sorted(RESULTS_ROOT.glob("*"))
            if (directory / "benchmark_config.json").is_file()
            and compare.engine_of(
                json.loads((directory / "benchmark_config.json").read_text(encoding="utf-8"))
            ) == "canary"
            and (directory / "audio_predictions.jsonl").is_file()
        ]
        if not cls.canary:
            raise unittest.SkipTest("no stored Canary audio runs")

    def test_every_stored_canary_audio_run_scores(self) -> None:
        for directory in self.canary:
            with self.subTest(run=directory.name):
                scored = compare.score_run(directory)
                self.assertIsNotNone(scored)
                self.assertEqual(scored["engine"], "canary")
                self.assertGreater(scored["clips"], 0)
                self.assertIsNotNone(
                    scored["overall"]["aggregate_wer_number_normalised"]
                )

    def test_scoring_leaves_the_historical_folders_untouched(self) -> None:
        before = {
            path: path.stat().st_mtime_ns
            for directory in self.canary
            for path in sorted(directory.iterdir())
            if path.is_file()
        }
        for directory in self.canary:
            compare.score_run(directory)
        after = {path: path.stat().st_mtime_ns for path in before}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

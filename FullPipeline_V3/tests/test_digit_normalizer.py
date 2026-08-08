"""Render-time spoken-digit-to-numeral conversion. Display only.

See app/digit_normalizer.py for the rules. Every worked example from the
task spec has a corresponding case below.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import digit_normalizer
from app.digit_normalizer import normalize_spoken_digits
from integrated_pipeline.sqlite_storage import (
    CompleteMaskedTranscript,
    SQLiteTranscriptStore,
)
from tests.helpers import recording, stt_result
from tests.test_live_preview import FakeMasker, build


class Rule1DigitWordsTests(unittest.TestCase):
    def test_bare_digit_words_map_zero_through_nine(self) -> None:
        # A run needs >= 3 bare digits to convert (rule 5); use ten digits so
        # every mapping in rule 1 is exercised in one pass.
        text = "zero one two three four five six seven eight nine"
        self.assertEqual(normalize_spoken_digits(text), "0123456789")


class Rule2TensPlusUnitTests(unittest.TestCase):
    def test_tens_plus_unit_converts_regardless_of_run_length(self) -> None:
        self.assertEqual(
            normalize_spoken_digits("call back in twenty five minutes"),
            "call back in 25 minutes",
        )
        self.assertEqual(
            normalize_spoken_digits("forty five dollars"),
            "45 dollars",
        )

    # NOTE: the three tests below previously asserted the opposite. Standalone
    # tens/teens were left as words and times were out of scope. Both rules
    # were reversed by the current specification ("twenty minutes -> 20
    # minutes", "three thirty pm -> 3:30 pm"), so the expectations are inverted
    # here rather than relaxed - each still pins an exact output.

    def test_standalone_tens_convert(self) -> None:
        self.assertEqual(normalize_spoken_digits("thirty seconds"), "30 seconds")
        self.assertEqual(normalize_spoken_digits("twenty minutes"), "20 minutes")
        self.assertEqual(
            normalize_spoken_digits("I am forty years old"), "I am 40 years old"
        )

    def test_teens_convert_standalone(self) -> None:
        for word, expected in (
            ("ten", "10"),
            ("eleven", "11"),
            ("fifteen", "15"),
            ("nineteen", "19"),
        ):
            with self.subTest(word=word):
                self.assertEqual(
                    normalize_spoken_digits(f"wait {word} seconds"),
                    f"wait {expected} seconds",
                )

    def test_times_render_with_a_colon(self) -> None:
        self.assertEqual(
            normalize_spoken_digits("the meeting is at three thirty"),
            "the meeting is at 3:30",
        )
        self.assertEqual(normalize_spoken_digits("three thirty pm"), "3:30 pm")
        self.assertEqual(
            normalize_spoken_digits("meeting at one thirty"), "meeting at 1:30"
        )

    def test_a_bare_tens_pair_without_time_context_keeps_its_boundary(self) -> None:
        # No "at", no meridiem: two values, not a clock reading, and never
        # collapsed into 3020.
        self.assertEqual(normalize_spoken_digits("thirty twenty"), "30 20")


class Rule3OhTests(unittest.TestCase):
    def test_oh_is_zero_only_next_to_a_digit(self) -> None:
        self.assertEqual(normalize_spoken_digits("five oh five"), "505")
        self.assertEqual(normalize_spoken_digits("oh five five one"), "0551")

    def test_oh_resolves_repeatedly_so_a_resolved_oh_licenses_its_neighbour(
        self,
    ) -> None:
        self.assertEqual(normalize_spoken_digits("oh oh seven"), "007")

    def test_oh_untouched_without_a_digit_neighbour(self) -> None:
        self.assertEqual(normalize_spoken_digits("oh i see"), "oh i see")
        self.assertEqual(normalize_spoken_digits("oh really"), "oh really")
        self.assertEqual(normalize_spoken_digits("oh"), "oh")


class Rule4DoubleTripleTests(unittest.TestCase):
    def test_double_and_triple_before_a_digit(self) -> None:
        self.assertEqual(normalize_spoken_digits("double four"), "44")
        self.assertEqual(normalize_spoken_digits("triple seven"), "777")

    def test_double_before_a_resolved_oh(self) -> None:
        self.assertEqual(normalize_spoken_digits("double oh seven"), "007")

    def test_double_triple_untouched_without_a_following_digit(self) -> None:
        self.assertEqual(normalize_spoken_digits("double click"), "double click")
        self.assertEqual(normalize_spoken_digits("double check"), "double check")
        self.assertEqual(normalize_spoken_digits("triple bypass"), "triple bypass")

    def test_double_never_applies_to_a_tens_token(self) -> None:
        # "double" stays a word because no single digit follows it; "twenty" is
        # a standalone tens and converts on its own.
        self.assertEqual(normalize_spoken_digits("double twenty"), "double 20")

    def test_resolved_double_joins_the_surrounding_run(self) -> None:
        self.assertEqual(normalize_spoken_digits("room double four seven"), "room 447")
        self.assertEqual(normalize_spoken_digits("seven double four two"), "7442")


class Rule5GroupingAndThresholdTests(unittest.TestCase):
    def test_four_or_more_bare_digits_convert(self) -> None:
        self.assertEqual(
            normalize_spoken_digits("the code is four seven two nine"),
            "the code is 4729",
        )

    def test_short_bare_runs_are_left_alone(self) -> None:
        text = "one of the doctors said two things"
        self.assertEqual(normalize_spoken_digits(text), text)

    def test_two_bare_digits_do_not_convert(self) -> None:
        text = "room seven four please"
        self.assertEqual(normalize_spoken_digits(text), text)

    def test_three_bare_digits_do_convert(self) -> None:
        self.assertEqual(normalize_spoken_digits("room seven four two"), "room 742")


class Rule6OpaqueSpanTests(unittest.TestCase):
    def test_entity_placeholder_untouched(self) -> None:
        text = "my phone number is [TELEPHONENUM_1]"
        self.assertEqual(normalize_spoken_digits(text), text)

    def test_blur_token_untouched(self) -> None:
        text = "call me on ●●●●●"
        self.assertEqual(normalize_spoken_digits(text), text)

    def test_redacted_marker_untouched(self) -> None:
        text = "the code is [[REDACTED]]"
        self.assertEqual(normalize_spoken_digits(text), text)

    def test_opaque_atom_still_breaks_adjacency_around_it(self) -> None:
        # Digits on either side of an opaque span must not be read as one
        # run through it.
        text = "call seven [TELEPHONENUM_1] eight now"
        self.assertEqual(normalize_spoken_digits(text), text)


class IdempotencyTests(unittest.TestCase):
    _CASES = (
        "the code is four seven two nine",
        "one of the doctors said two things",
        "call back in twenty five minutes",
        "room double four seven",
        "seven double four two",
        "the meeting is at three thirty",
        "double oh seven",
        "oh oh seven",
        "my phone number is [TELEPHONENUM_1]",
        "call me on ●●●●●",
        "the code is [[REDACTED]]",
        "",
        "   ",
    )

    def test_normalising_twice_matches_normalising_once(self) -> None:
        for text in self._CASES:
            with self.subTest(text=text):
                once = normalize_spoken_digits(text)
                twice = normalize_spoken_digits(once)
                self.assertEqual(once, twice)


class MalformedInputTests(unittest.TestCase):
    def setUp(self) -> None:
        digit_normalizer._fallback_logged = False

    def test_non_string_input_is_returned_unchanged(self) -> None:
        self.assertIsNone(normalize_spoken_digits(None))
        self.assertEqual(normalize_spoken_digits(42), 42)
        self.assertEqual(normalize_spoken_digits(["four", "five"]), ["four", "five"])

    def test_empty_string_is_returned_unchanged(self) -> None:
        self.assertEqual(normalize_spoken_digits(""), "")

    def test_internal_failure_falls_back_to_the_original_text(self) -> None:
        with patch.object(
            digit_normalizer, "_normalize_impl", side_effect=RuntimeError("boom")
        ):
            self.assertEqual(
                normalize_spoken_digits("four seven two nine"),
                "four seven two nine",
            )

    def test_fallback_is_logged_once_per_session_not_once_per_call(self) -> None:
        with patch.object(
            digit_normalizer, "_normalize_impl", side_effect=RuntimeError("boom")
        ):
            with self.assertLogs(digit_normalizer.logger, level="WARNING") as cm:
                normalize_spoken_digits("four seven two nine")
                normalize_spoken_digits("five six seven")
                normalize_spoken_digits("eight nine zero")
        self.assertEqual(len(cm.output), 1)


class StorageUnaffectedTests(unittest.TestCase):
    """Storage keeps raw text; normalisation must never reach the writer."""

    def test_stored_transcript_is_byte_identical_with_and_without_the_renderer(
        self,
    ) -> None:
        raw_text = "my code is four seven two nine and my extension is double oh five"
        # The renderer would visibly change this text, so a passing test here
        # is actually proving something.
        self.assertNotEqual(normalize_spoken_digits(raw_text), raw_text)

        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteTranscriptStore(Path(tmp) / "session_transcript.db")
            store.start()
            try:
                store.start_session(
                    "session-digit-normalizer-test",
                    "2026-07-29T10:00:00Z",
                    model_identity="test",
                )
                item = recording(1, session_id="session-digit-normalizer-test")
                result = stt_result(item, raw_text)
                store.insert_raw(result)
                store.complete_masked(
                    CompleteMaskedTranscript(
                        session_id=result.session_id,
                        batch_id=result.batch_id,
                        masked_transcript=raw_text,
                        entities=[],
                        mask_started_at="2026-07-29T10:00:02Z",
                        mask_completed_at="2026-07-29T10:00:03Z",
                        mask_latency_seconds=0.1,
                        total_latency_seconds=0.5,
                    )
                )
                store.drain()
                row = store.query_session("session-digit-normalizer-test")[0]
            finally:
                store.shutdown()

        self.assertEqual(row.raw_transcript, raw_text)
        self.assertEqual(row.masked_transcript, raw_text)


class PreviewIntegrationTests(unittest.TestCase):
    """The two integration checks called for in the task spec."""

    def test_preview_output_identical_with_and_without_normalisation_when_no_digits_spoken(
        self,
    ) -> None:
        text = "we talked about the weather and then walked to the shop together"

        with patch(
            "app.live_preview.normalize_spoken_digits", side_effect=lambda value: value
        ):
            baseline_preview, baseline_renders = build()
            baseline_preview.add_segment(text, unique_seconds=8.512)

        actual_preview, actual_renders = build()
        actual_preview.add_segment(text, unique_seconds=8.512)

        self.assertEqual(baseline_renders[-1].line, actual_renders[-1].line)

    def test_masker_sees_spelled_out_digits_not_numerals(self) -> None:
        # Detection (_DIGIT_WORDS / _SEQUENCE_PATTERN in structured_pii.py)
        # only matches spelled-out digit words. If normalisation ever ran
        # before masking, this text would arrive at the masker as numerals
        # and structured detection over the raw buffer would stop working.
        masker = FakeMasker()
        preview, _ = build(masker, blur_cap_words=20)
        text = "my account number is four seven two nine"
        preview.add_segment(text, unique_seconds=8.512)
        self.assertIn("four seven two nine", masker.calls[-1])
        self.assertNotIn("4729", masker.calls[-1])

    def test_blur_word_count_and_redaction_are_unaffected_by_normalisation(
        self,
    ) -> None:
        # blurred_words/redacted are decided by `_render_line` against `raw`,
        # strictly before normalisation runs on the resulting `line`. Toggling
        # normalisation on and off must not move that decision either way -
        # if it did, the ordering described in live_preview.py would have
        # regressed and a word that should stay hidden could print instead.
        text = (
            "my account number is zero one two three four five six seven "
            "eight nine"
        )

        with patch(
            "app.live_preview.normalize_spoken_digits", side_effect=lambda v: v
        ):
            baseline_preview, baseline_renders = build(
                FakeMasker(), blur_cap_words=5
            )
            baseline_preview.add_segment(text, unique_seconds=8.512)

        actual_preview, actual_renders = build(FakeMasker(), blur_cap_words=5)
        actual_preview.add_segment(text, unique_seconds=8.512)

        self.assertEqual(
            baseline_renders[-1].blurred_words, actual_renders[-1].blurred_words
        )
        self.assertEqual(
            baseline_renders[-1].redacted, actual_renders[-1].redacted
        )
        self.assertTrue(actual_renders[-1].redacted)


if __name__ == "__main__":
    unittest.main()

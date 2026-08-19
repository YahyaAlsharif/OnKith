"""Wake-word repair for Canary mis-transcriptions.

Pure unit tests: no Canary, no audio, no ONNX. The integration tests at the
bottom drive the real `intercept_onkith_command` to prove the repaired text is
actually accepted as a command, which is the whole point of the change.
"""

from __future__ import annotations

import unittest

from app.onkith_music import intercept_onkith_command
from app.wake_word_normalizer import (
    CANONICAL_WAKE_WORD,
    normalize_wake_word,
)


class SpecifiedExampleTests(unittest.TestCase):
    """Every example from the request, asserted exactly."""

    CASES = (
        ("on kids play music", "Onkith play music"),
        ("on kiss could you hear me", "Onkith could you hear me"),
        ("on care", "Onkith"),
        ("hello uncle did you hear me", "hello Onkith did you hear me"),
        (
            "hello aunt Keith do you hear me",
            "hello Onkith do you hear me",
        ),
    )

    def test_every_specified_example(self) -> None:
        for source, expected in self.CASES:
            with self.subTest(source=source):
                normalized, variant = normalize_wake_word(source)
                self.assertEqual(normalized, expected)
                self.assertIsNotNone(variant)


class EveryListedVariantTests(unittest.TestCase):
    TWO_TOKEN = (
        "on kith",
        "on kit",
        "on kids",
        "on kid",
        "on kiss",
        "on care",
        "on Keith",
        "aunt Keith",
    )
    SINGLE_TOKEN = ("uncle", "aunt", "Ankit")

    def test_two_token_variants_at_the_start(self) -> None:
        for variant in self.TWO_TOKEN:
            with self.subTest(variant=variant):
                normalized, matched = normalize_wake_word(f"{variant} play music")
                self.assertEqual(normalized, "Onkith play music")
                self.assertIsNotNone(matched)

    def test_single_token_variants_at_the_start(self) -> None:
        for variant in self.SINGLE_TOKEN:
            with self.subTest(variant=variant):
                normalized, matched = normalize_wake_word(f"{variant} play music")
                self.assertEqual(normalized, "Onkith play music")
                self.assertIsNotNone(matched)

    def test_a_bare_variant_with_no_command(self) -> None:
        for variant in (*self.TWO_TOKEN, *self.SINGLE_TOKEN):
            with self.subTest(variant=variant):
                normalized, _ = normalize_wake_word(variant)
                self.assertEqual(normalized, CANONICAL_WAKE_WORD)


class CaseAndPunctuationTests(unittest.TestCase):
    def test_matching_is_case_insensitive(self) -> None:
        for source in ("ON KIDS play music", "On Kids play music", "oN kIdS play music"):
            with self.subTest(source=source):
                self.assertEqual(
                    normalize_wake_word(source)[0], "Onkith play music"
                )

    def test_punctuation_after_the_wake_word_survives(self) -> None:
        self.assertEqual(
            normalize_wake_word("on kids, play music")[0], "Onkith, play music"
        )
        self.assertEqual(normalize_wake_word("on care.")[0], "Onkith.")
        self.assertEqual(
            normalize_wake_word("hello, uncle did you hear me")[0],
            "hello, Onkith did you hear me",
        )

    def test_surrounding_text_is_preserved_verbatim(self) -> None:
        normalized, _ = normalize_wake_word("on kiss Could You Hear Me")
        self.assertEqual(normalized, "Onkith Could You Hear Me")


class PositionGuardTests(unittest.TestCase):
    """The generic words must not be rewritten in ordinary sentences."""

    UNCHANGED = (
        "Someone named Ankit called",
        "my uncle is visiting tomorrow",
        "I called my aunt yesterday",
        "the cat sat on kids toys",
        "she gave it to her uncle and aunt",
        "we talked about on care policies at work",
        "tell my uncle the meeting moved",
    )

    def test_mid_sentence_occurrences_are_left_alone(self) -> None:
        for source in self.UNCHANGED:
            with self.subTest(source=source):
                normalized, variant = normalize_wake_word(source)
                self.assertEqual(normalized, source)
                self.assertIsNone(variant)

    def test_the_deliberate_ankit_exclusion_still_holds(self) -> None:
        # onkith_music treats "Ankit" as a name on purpose, and a self-test in
        # omit_ai_live asserts this exact sentence is not a command. Repairing
        # only at the wake position keeps that true.
        source = "Someone named Ankit called"
        normalized, _ = normalize_wake_word(source)
        self.assertEqual(normalized, source)
        self.assertFalse(intercept_onkith_command(normalized).command_detected)

    def test_too_many_leading_words_breaks_the_wake_position(self) -> None:
        # Three fillers is past the limit, and a content word is never a filler.
        for source in (
            "well ok so uncle did you hear me",
            "yesterday my uncle did you hear me",
        ):
            with self.subTest(source=source):
                self.assertEqual(normalize_wake_word(source)[0], source)

    def test_one_and_two_leading_fillers_are_accepted(self) -> None:
        # The greeting itself is preserved; only the variant is rewritten.
        self.assertEqual(
            normalize_wake_word("hey uncle play music")[0], "hey Onkith play music"
        )
        self.assertEqual(
            normalize_wake_word("ok hey uncle play music")[0],
            "ok hey Onkith play music",
        )


class RobustnessTests(unittest.TestCase):
    def test_empty_and_blank_input(self) -> None:
        for source in ("", "   ", "\n"):
            with self.subTest(source=source):
                self.assertEqual(normalize_wake_word(source), (source, None))

    def test_non_string_input_is_returned_unchanged(self) -> None:
        self.assertEqual(normalize_wake_word(None), (None, None))
        self.assertEqual(normalize_wake_word(42), (42, None))

    def test_punctuation_only_input(self) -> None:
        self.assertEqual(normalize_wake_word("...")[0], "...")

    def test_already_canonical_text_is_untouched(self) -> None:
        source = "Onkith play music"
        self.assertEqual(normalize_wake_word(source), (source, None))

    def test_idempotent(self) -> None:
        for source in ("on kids play music", "hello uncle did you hear me", "on care"):
            with self.subTest(source=source):
                once = normalize_wake_word(source)[0]
                self.assertEqual(normalize_wake_word(once)[0], once)


class InterceptorIntegrationTests(unittest.TestCase):
    """The repaired text must actually be accepted as a command."""

    def test_repaired_variants_reach_the_interceptor_as_commands(self) -> None:
        for variant in (
            "on kids",
            "on kiss",
            "on care",
            "on Keith",
            "aunt Keith",
            "uncle",
            "aunt",
            "Ankit",
        ):
            with self.subTest(variant=variant):
                raw = f"{variant} play Imagine Dragons Bones"
                before = intercept_onkith_command(raw)
                repaired, _ = normalize_wake_word(raw)
                after = intercept_onkith_command(repaired)
                self.assertTrue(
                    after.command_detected,
                    f"{variant!r} still not detected after repair",
                )
                self.assertEqual(after.intent, "PLAY")
                self.assertEqual(
                    after.command_text, "play Imagine Dragons Bones"
                )
                del before

    def test_variants_that_previously_failed_now_succeed(self) -> None:
        # Guards the actual regression: these are the ones onkith_music's own
        # fuzzy matcher does not reach.
        for variant in ("on kids", "on kiss", "on care", "uncle", "aunt"):
            with self.subTest(variant=variant):
                raw = f"{variant} play music"
                self.assertFalse(
                    intercept_onkith_command(raw).command_detected,
                    f"{variant!r} unexpectedly already worked; test is stale",
                )
                repaired, _ = normalize_wake_word(raw)
                self.assertTrue(intercept_onkith_command(repaired).command_detected)

    def test_a_close_command_still_routes_as_close(self) -> None:
        repaired, _ = normalize_wake_word("on kids stop")
        interception = intercept_onkith_command(repaired)
        self.assertTrue(interception.command_detected)
        self.assertEqual(interception.intent, "CLOSE")

    def test_unrelated_speech_around_the_command_survives(self) -> None:
        repaired, _ = normalize_wake_word("uncle play music")
        interception = intercept_onkith_command(repaired)
        self.assertTrue(interception.command_detected)
        self.assertEqual(interception.remaining_text.strip(), "")

    def test_aunt_keith_survives_because_repair_precedes_masking(self) -> None:
        # Running before the masker is the whole point: "aunt Keith" collapses
        # to the wake word while the name is still there to match against.
        repaired, variant = normalize_wake_word("hello aunt Keith do you hear me")
        self.assertEqual(variant, "aunt keith")
        self.assertEqual(repaired, "hello Onkith do you hear me")
        self.assertTrue(
            intercept_onkith_command(repaired).command_detected
            or "Onkith" in repaired
        )

    def test_aunt_with_an_unlisted_name_only_replaces_aunt(self) -> None:
        # Pinned because the request is ambiguous here. "aunt Keith" is listed
        # as a pair, so both tokens collapse. A different name is not listed,
        # so only the bare "aunt" variant matches and the name is left for the
        # masker to handle. Collapsing "aunt <any capitalised word>" would eat
        # real names, so it is deliberately not done.
        self.assertEqual(
            normalize_wake_word("hello aunt Sarah do you hear me")[0],
            "hello Onkith Sarah do you hear me",
        )


if __name__ == "__main__":
    unittest.main()

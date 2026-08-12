"""The device can be called anything, and its name is only ever data.

The matcher used to know the shape of one particular word: a candidate had
to start on, an or un and end in h, which is true of Onkith and of nothing
else. Renaming the device stopped it answering to its own name.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.onkith_music import (
    WakeWords,
    configure_wake_words,
    intercept_onkith_command,
    nearest_wake_miss,
    wake_words,
)
from integrated_pipeline.config import PipelineConfig


class WakeWordTest(unittest.TestCase):
    def setUp(self) -> None:
        previous = wake_words()
        self.addCleanup(configure_wake_words, previous)

    def heard(self, transcript: str) -> bool:
        return intercept_onkith_command(transcript).command_detected

    def test_a_different_name_works_with_no_code_change(self) -> None:
        configure_wake_words(WakeWords(names=("jarvis",), spellings=(), never=()))
        self.assertTrue(self.heard("Jarvis play bones"))
        self.assertTrue(self.heard("jarvis what is the time"))
        # Near enough, the way a recogniser mishears a long name.
        self.assertTrue(self.heard("jarvus play bones"))
        # And the old name is now a stranger.
        self.assertFalse(self.heard("Onkith play bones"))

    def test_a_name_split_in_two_is_joined_back_up(self) -> None:
        configure_wake_words(WakeWords(names=("jarvis",), spellings=(), never=()))
        self.assertTrue(self.heard("Jar vis play bones"))

    def test_a_joined_pair_has_to_be_exact(self) -> None:
        """Otherwise it starts hearing its name in ordinary speech.

        Joining a pair is already a reconstruction. Allowing an edit on top
        of it stacks two guesses: with the device called Jarvis, "the jar is
        empty" joins to "jaris", which is one letter away from its name.
        """

        configure_wake_words(WakeWords(names=("jarvis",), spellings=(), never=()))
        self.assertFalse(self.heard("the jar is empty"))
        self.assertFalse(self.heard("a jar is on the shelf"))

    def test_a_short_name_is_never_matched_loosely(self) -> None:
        """A four letter name cannot spare an edit.

        At one edit Sada answers to "sad" and to "soda", and a device that
        wakes up on "sad" is worse than one that misses a call.
        """

        configure_wake_words(WakeWords(names=("sada",), spellings=(), never=()))
        self.assertTrue(self.heard("Sada play bones"))
        self.assertFalse(self.heard("that is so sad"))
        self.assertFalse(self.heard("pass the soda please"))

    def test_a_near_miss_is_measured_against_a_name_never_a_spelling(
        self,
    ) -> None:
        """Otherwise the accepted forms drift outward a letter at a time.

        A spelling is already one step from the name, so allowing an edit on
        top of it accepts words two steps away without ever saying so.
        """

        configure_wake_words(
            WakeWords(names=("onkith",), spellings=("onkithe",), never=())
        )
        self.assertTrue(self.heard("onkithe play bones"))
        # One edit from the spelling, two from the name. Long enough that the
        # length floor is not what rejects it.
        self.assertFalse(self.heard("onkathe play bones"))

    def test_a_blocked_word_never_wakes_it(self) -> None:
        configure_wake_words(
            WakeWords(names=("onkith",), spellings=("ankit",), never=("ankit",))
        )
        self.assertFalse(self.heard("Ankit called me"))

    def test_the_shipped_config_keeps_every_name_it_had(self) -> None:
        configure_wake_words(WakeWords.from_config(PipelineConfig().wake))
        for transcript in (
            "Onkith play bones",
            "Onkuth stop",
            "صدى play believer",
            "ستر close it",
            "Sitter stop the music",
            "On ket play believer",
            "Onkithh what time is it",
            # Echo, which is what sada means, added on request.
            "Echo what time is it",
        ):
            with self.subTest(transcript=transcript):
                self.assertTrue(self.heard(transcript))
        self.assertFalse(self.heard("Ankit called me"))

    def test_config_falls_back_to_the_built_in_name(self) -> None:
        self.assertEqual(WakeWords.from_config(None), WakeWords())
        self.assertEqual(WakeWords.from_config({}), WakeWords())
        # An empty list is a mistake, not an instruction to answer to nothing.
        self.assertEqual(WakeWords.from_config({"names": []}).names, WakeWords().names)

    def test_config_is_case_and_form_insensitive(self) -> None:
        wake = WakeWords.from_config({"names": ["  JARVIS  "]})
        self.assertEqual(wake.names, ("jarvis",))

    def test_what_the_recogniser_actually_writes_wakes_it(self) -> None:
        """Moonshine hears Onkith and writes "on kiss".

        It is consistent about it, so the device simply never answered and
        there was nothing on screen to say why. These are the spellings it
        was observed producing, on a Mac, on 2026-08-06.
        """

        configure_wake_words(WakeWords.from_config(PipelineConfig().wake))
        for transcript in (
            "On kiss. What time is it in Saudi now?",
            "On kiss What time is it in Riyadh now?",
            "Own kiss What times it is in Riyadh?",
            "On keys what is the weather",
        ):
            with self.subTest(transcript=transcript):
                result = intercept_onkith_command(transcript)
                self.assertTrue(result.command_detected)
                self.assertEqual(result.intent, "ASK")


class NearMissTest(unittest.TestCase):
    """A recogniser that mishears the name should say so out loud.

    Otherwise the device is silent for a reason nobody can see, and the
    person just repeats themselves at it.
    """

    def setUp(self) -> None:
        previous = wake_words()
        self.addCleanup(configure_wake_words, previous)
        configure_wake_words(WakeWords.from_config(PipelineConfig().wake))

    def test_a_plausible_mishearing_is_named(self) -> None:
        self.assertEqual(nearest_wake_miss("on kif what time is it"), ("onkif", 2))
        # Joined across two words, which is how it actually goes wrong.
        self.assertEqual(nearest_wake_miss("on kuth play bones")[0], "onkuth")

    def test_ordinary_speech_says_nothing(self) -> None:
        for transcript in (
            "the jar is empty",
            "one two three",
            "I was telling Sarah about the meeting",
            "we should stop by the shop later",
            "it is on the table",
            "what time is it in Riyadh",
            "put the kettle on the stove",
        ):
            with self.subTest(transcript=transcript):
                self.assertIsNone(nearest_wake_miss(transcript))

    def test_a_short_name_is_never_used_for_the_comparison(self) -> None:
        """Sitr is four letters and sits two edits from half the language.

        Measured against it, "is it" and "stop" both come back as plausible
        names, which is worse than saying nothing at all.
        """

        configure_wake_words(WakeWords(names=("sitr", "sada"), spellings=(), never=()))
        self.assertIsNone(nearest_wake_miss("what time is it"))
        self.assertIsNone(nearest_wake_miss("we should stop"))

    def test_a_blocked_word_is_never_offered_as_the_name(self) -> None:
        self.assertIsNone(nearest_wake_miss("someone named Ankit called me"))

    def test_it_never_wakes_anything_by_itself(self) -> None:
        # It is a diagnostic. Being near enough to report is not being near
        # enough to answer.
        self.assertIsNotNone(nearest_wake_miss("on kif what time is it"))
        self.assertFalse(
            intercept_onkith_command("on kif what time is it").command_detected
        )


class MatcherShapeTest(unittest.TestCase):
    def test_the_matcher_holds_no_opinion_about_the_name(self) -> None:
        source = (ROOT / "app" / "onkith_music.py").read_text(encoding="utf-8")
        matcher = source.split("def _answers_to(", maxsplit=1)[1].split(
            "\ndef ", maxsplit=1
        )[0]
        # The old rule was startswith on/an/un and endswith h, which is the
        # shape of one word and of nothing else.
        for shape in ('"on"', '"an"', '"un"', 'endswith', '"kith"'):
            with self.subTest(shape=shape):
                self.assertNotIn(shape, matcher)


if __name__ == "__main__":
    unittest.main()

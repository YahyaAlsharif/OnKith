"""The two spoken-address rewrites, and the guards that keep them narrow.

See integrated_pipeline/spoken_form_normalizer.py. Every case here is either a
form the audio benchmark observed Whisper produce, or a form from the written
fixtures that must not be touched.
"""

from __future__ import annotations

import unittest

from integrated_pipeline.spoken_form_normalizer import (
    describe,
    normalise_spoken_forms,
    project_offsets,
)


class SpokenEmailTests(unittest.TestCase):
    """The failure this exists for: the domain masked, the local part released."""

    def test_a_spoken_address_becomes_a_written_one(self) -> None:
        view = normalise_spoken_forms("Please forward the invoice to billing at example.org.")
        self.assertEqual(
            view.text, "Please forward the invoice to billing@example.org."
        )

    def test_a_dotted_local_part_survives_the_rewrite(self) -> None:
        view = normalise_spoken_forms("My email is alex.carter at example.com.")
        self.assertEqual(view.text, "My email is alex.carter@example.com.")

    def test_a_sentence_full_stop_is_not_part_of_the_domain(self) -> None:
        view = normalise_spoken_forms("It went to contracts at example.net from work.")
        self.assertEqual(view.text, "It went to contracts@example.net from work.")

    def test_the_edit_is_reported(self) -> None:
        view = normalise_spoken_forms("Write to billing at example.org today.")
        self.assertEqual(describe(view), [{
            "rule": "spoken_email",
            "original": "billing at example.org",
            "normalised": "billing@example.org",
            "start": 9,
            "end": 31,
        }])


class SpokenUrlTests(unittest.TestCase):
    def test_a_spoken_path_separator_becomes_a_slash(self) -> None:
        view = normalise_spoken_forms("The booking page is example.com slash booking.")
        self.assertEqual(view.text, "The booking page is example.com/booking.")


class GuardTests(unittest.TestCase):
    """Ordinary language, and addresses that were already written correctly."""

    def test_an_address_already_written_is_left_alone(self) -> None:
        """Without the guard this bites the front off a correct address.

        ``Email me at first.last@alpine-solutions.travel`` would otherwise be
        rewritten to ``Email me@first.last@alpine-solutions.travel``, breaking an
        address the model recognises perfectly well as it stands.
        """

        for text in (
            "Email me at first.last@alpine-solutions.travel.",
            "Email me at team.bot@deepfield.engineering.",
            "Email me at hello.world@mail.saffron.agency.",
        ):
            with self.subTest(text=text):
                self.assertFalse(normalise_spoken_forms(text).changed)

    def test_prose_that_only_looks_like_a_domain_is_left_alone(self) -> None:
        """"H.No." is a house number in the deployment benchmark, not a domain."""

        for text in (
            "Eric N. Robinson sustained damage at H.No. 26 Bains Chowk, Panihati.",
            "The story unfolds at H.No. 833, Wali Ganj.",
        ):
            with self.subTest(text=text):
                self.assertFalse(normalise_spoken_forms(text).changed)

    def test_ordinary_uses_of_at_are_left_alone(self) -> None:
        for text in (
            "Meet me at the cafe at half past four.",
            "Turn the kitchen lights off, please.",
            "Set a timer for twenty minutes.",
            "The recipe needs three hundred grams of flour and two eggs.",
            "Call me back on 555-0147.",
        ):
            with self.subTest(text=text):
                self.assertFalse(normalise_spoken_forms(text).changed)

    def test_the_rewrite_is_idempotent(self) -> None:
        once = normalise_spoken_forms("Write to billing at example.org.")
        twice = normalise_spoken_forms(once.text)
        self.assertFalse(twice.changed)
        self.assertEqual(twice.text, once.text)


class OffsetProjectionTests(unittest.TestCase):
    """The released transcript stays the transcript; only detection moves."""

    TEXT = "Please forward the invoice to billing at example.org before Friday."

    def test_the_original_text_is_never_modified(self) -> None:
        view = normalise_spoken_forms(self.TEXT)
        self.assertEqual(view.original, self.TEXT)

    def test_a_span_on_the_rewritten_address_covers_the_spoken_one(self) -> None:
        view = normalise_spoken_forms(self.TEXT)
        edit = view.edits[0]
        start, end = project_offsets(view, edit.normalised_start, edit.normalised_end)
        self.assertEqual(self.TEXT[start:end], "billing at example.org")

    def test_a_span_after_the_edit_lands_where_it_belongs(self) -> None:
        view = normalise_spoken_forms(self.TEXT)
        position = view.text.index("Friday")
        start, end = project_offsets(view, position, position + len("Friday"))
        self.assertEqual(self.TEXT[start:end], "Friday")

    def test_a_span_before_the_edit_is_unmoved(self) -> None:
        view = normalise_spoken_forms(self.TEXT)
        position = view.text.index("invoice")
        start, end = project_offsets(view, position, position + len("invoice"))
        self.assertEqual(self.TEXT[start:end], "invoice")

    def test_a_partial_span_inside_the_edit_widens_to_the_whole_form(self) -> None:
        """Masking half of a rewritten address would release the other half."""

        view = normalise_spoken_forms(self.TEXT)
        position = view.text.index("example.org")
        start, end = project_offsets(view, position, position + len("example.org"))
        self.assertEqual(self.TEXT[start:end], "billing at example.org")

    def test_text_with_no_spoken_form_projects_to_itself(self) -> None:
        view = normalise_spoken_forms("Turn the kitchen lights off.")
        self.assertFalse(view.changed)
        self.assertEqual(project_offsets(view, 5, 8), (5, 8))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

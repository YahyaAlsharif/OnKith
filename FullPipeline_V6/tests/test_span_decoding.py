"""Regression cover for the token -> character span decoder.

Every case here failed before the DeBERTa Model V2 forensic audit. Two families
of defect are covered.

SentencePiece offsets (DeBERTa-v2/v3) include the whitespace in front of a
token; the whole-word expansion used to walk left out of that whitespace and
swallow the preceding word, so a one-character ``AGE`` of ``"9"`` decoded as
``"by 9"``.

Entities that carry punctuation inside them - emails, URLs, IPv4 addresses,
phone numbers, dates - used to shatter into one span per alphanumeric run,
because words only merged across pure whitespace and because punctuation tokens
expanded into the words on both sides and produced overlapping duplicates.
``MFP31@aol.com`` decoded as six spans and matched none of them.
"""

from __future__ import annotations

import unittest

from integrated_pipeline.span_merger import (
    consolidate_predictions_to_words,
    derive_word_ids,
)


def sentencepiece_offsets(text: str, pieces: list[str]) -> list[tuple[int, int]]:
    """Offsets in the DeBERTa style: a token owns the whitespace before it.

    ``pieces`` are consumed in order and must reconstruct ``text`` exactly, so a
    typo in a fixture fails loudly instead of silently shifting every offset.
    """

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        start = cursor
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        body = text[cursor : cursor + len(piece.lstrip())]
        if body != piece.lstrip():
            raise AssertionError(f"fixture {piece!r} does not match {body!r} at {cursor}")
        cursor += len(body)
        offsets.append((start if piece[:1].isspace() else cursor - len(body), cursor))
    if cursor != len(text):
        raise AssertionError(f"fixture consumed {cursor} of {len(text)} characters")
    return offsets


def decode(text, pieces, labels, *, confidences=None):
    """Run one fixture through the production decoder."""

    offsets = sentencepiece_offsets(text, pieces)
    result = consolidate_predictions_to_words(
        text,
        offsets=offsets,
        labels=labels,
        confidences=confidences or [0.99] * len(labels),
    )
    return [(span.entity_type, span.start_char, span.end_char, span.text)
            for span in result.spans]


class LeadingWhitespaceOffsetTests(unittest.TestCase):
    """A token offset that opens on whitespace must not reach backwards."""

    def test_one_character_age_does_not_absorb_the_previous_word(self):
        text = "Accompanied by 9 for the trip."
        spans = decode(
            text,
            ["Accompanied", " by", " 9", " for", " the", " trip", "."],
            ["O", "O", "B-AGE", "O", "O", "O", "O"],
        )
        self.assertEqual(spans, [("AGE", 15, 16, "9")])

    def test_single_letter_sex_token_keeps_its_own_boundary(self):
        text = "Recorded sex F on the form."
        spans = decode(
            text,
            ["Recorded", " sex", " F", " on", " the", " form", "."],
            ["O", "O", "B-SEX", "O", "O", "O", "O"],
        )
        self.assertEqual(spans, [("SEX", 13, 14, "F")])

    def test_given_name_after_a_space_starts_at_the_name(self):
        text = "Contact Tamar today."
        spans = decode(
            text,
            ["Contact", " Tamar", " today", "."],
            ["O", "B-GIVENNAME", "O", "O"],
        )
        self.assertEqual(spans, [("GIVENNAME", 8, 13, "Tamar")])

    def test_whitespace_only_token_seeds_no_span(self):
        text = "a  b"
        result = consolidate_predictions_to_words(
            text,
            offsets=[(0, 1), (1, 3), (3, 4)],
            labels=["B-GIVENNAME", "I-GIVENNAME", "I-GIVENNAME"],
            confidences=[0.9, 0.9, 0.9],
        )
        for span in result.spans:
            self.assertFalse(span.text.startswith(" "), span.text)
            self.assertFalse(span.text.endswith(" "), span.text)


class PunctuationInternalEntityTests(unittest.TestCase):
    """Entities whose characters include punctuation must stay whole."""

    def test_email_decodes_to_one_exact_span(self):
        text = "Reach me at MFP31@aol.com today."
        spans = decode(
            text,
            ["Reach", " me", " at", " MFP31", "@", "aol", ".", "com", " today", "."],
            ["O", "O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL",
             "O", "O"],
        )
        self.assertEqual(spans, [("EMAIL", 12, 25, "MFP31@aol.com")])

    def test_url_decodes_to_one_exact_span(self):
        text = "See https://www.example.com/help now."
        spans = decode(
            text,
            ["See", " https", "://", "www", ".", "example", ".", "com", "/", "help",
             " now", "."],
            ["O", "B-URL", "I-URL", "I-URL", "I-URL", "I-URL", "I-URL", "I-URL",
             "I-URL", "I-URL", "O", "O"],
        )
        self.assertEqual(spans, [("URL", 4, 32, "https://www.example.com/help")])

    def test_ipv4_decodes_to_one_exact_span(self):
        text = "Host 192.168.1.100 responded."
        spans = decode(
            text,
            ["Host", " 192", ".", "168", ".", "1", ".", "100", " responded", "."],
            ["O", "B-IPV4", "I-IPV4", "I-IPV4", "I-IPV4", "I-IPV4", "I-IPV4",
             "I-IPV4", "O", "O"],
        )
        self.assertEqual(spans, [("IPV4", 5, 18, "192.168.1.100")])

    def test_telephone_number_decodes_to_one_exact_span(self):
        text = "Call at +659-785-716-8859 please."
        spans = decode(
            text,
            ["Call", " at", " +", "659", "-", "785", "-", "716", "-", "8859",
             " please", "."],
            ["O", "O", "B-TELEPHONENUM", "I-TELEPHONENUM", "I-TELEPHONENUM",
             "I-TELEPHONENUM", "I-TELEPHONENUM", "I-TELEPHONENUM",
             "I-TELEPHONENUM", "I-TELEPHONENUM", "O", "O"],
        )
        self.assertEqual(spans, [("TELEPHONENUM", 8, 25, "+659-785-716-8859")])

    def test_numeric_date_decodes_to_one_exact_span(self):
        text = "Born 28/11/1980 in Riyadh."
        spans = decode(
            text,
            ["Born", " 28", "/", "11", "/", "1980", " in", " Riyadh", "."],
            ["O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "I-DATE", "O", "B-CITY",
             "O"],
        )
        self.assertEqual(spans[0], ("DATE", 5, 15, "28/11/1980"))

    def test_written_date_merges_across_a_comma(self):
        text = "Dated May 13th, 2023 exactly."
        spans = decode(
            text,
            ["Dated", " May", " 13th", ",", " 2023", " exactly", "."],
            ["O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O", "O"],
        )
        self.assertEqual(spans, [("DATE", 6, 20, "May 13th, 2023")])

    def test_unlabelled_punctuation_still_joins_its_neighbours(self):
        """The tagger leaving the ``@`` as O must not split the address."""

        text = "Mail MFP31@aol.com now."
        spans = decode(
            text,
            ["Mail", " MFP31", "@", "aol", ".", "com", " now", "."],
            ["O", "B-EMAIL", "O", "I-EMAIL", "O", "I-EMAIL", "O", "O"],
        )
        self.assertEqual(spans, [("EMAIL", 5, 18, "MFP31@aol.com")])


class OverlapAndDuplicationTests(unittest.TestCase):
    """Punctuation must not invent a group spanning both of its neighbours."""

    def test_punctuation_token_is_its_own_word_group(self):
        text = "MFP31@aol.com"
        word_ids = derive_word_ids(text, [(0, 5), (5, 6), (6, 9), (9, 10), (10, 13)])
        self.assertEqual(len(set(word_ids)), 5)

    def test_no_two_decoded_spans_overlap(self):
        text = "Reach me at MFP31@aol.com or 192.168.1.100 today."
        spans = decode(
            text,
            ["Reach", " me", " at", " MFP31", "@", "aol", ".", "com", " or", " 192",
             ".", "168", ".", "1", ".", "100", " today", "."],
            ["O", "O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL",
             "O", "B-IPV4", "I-IPV4", "I-IPV4", "I-IPV4", "I-IPV4", "I-IPV4",
             "I-IPV4", "O", "O"],
        )
        self.assertEqual(
            spans,
            [("EMAIL", 12, 25, "MFP31@aol.com"), ("IPV4", 29, 42, "192.168.1.100")],
        )
        for earlier, later in zip(spans, spans[1:]):
            self.assertLessEqual(earlier[2], later[1])

    def test_one_gold_entity_yields_exactly_one_span(self):
        text = "Reach me at MFP31@aol.com today."
        spans = decode(
            text,
            ["Reach", " me", " at", " MFP31", "@", "aol", ".", "com", " today", "."],
            ["O", "O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL",
             "O", "O"],
        )
        self.assertEqual(len(spans), 1)


class SeparateEntityTests(unittest.TestCase):
    """The repair must not fuse two genuinely distinct entities."""

    def test_two_cities_in_a_list_stay_separate(self):
        text = "Offices in London, Paris and Cairo."
        spans = decode(
            text,
            ["Offices", " in", " London", ",", " Paris", " and", " Cairo", "."],
            ["O", "O", "B-CITY", "O", "B-CITY", "O", "B-CITY", "O"],
        )
        self.assertEqual(
            spans,
            [("CITY", 11, 17, "London"), ("CITY", 19, 24, "Paris"),
             ("CITY", 29, 34, "Cairo")],
        )

    def test_two_emails_separated_by_a_comma_stay_separate(self):
        text = "Write a@b.com, c@d.com now."
        spans = decode(
            text,
            ["Write", " a", "@", "b", ".", "com", ",", " c", "@", "d", ".", "com",
             " now", "."],
            ["O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "O",
             "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "O", "O"],
        )
        self.assertEqual(
            spans,
            [("EMAIL", 6, 13, "a@b.com"), ("EMAIL", 15, 22, "c@d.com")],
        )

    def test_multi_word_name_still_merges_across_whitespace(self):
        text = "Signed by Van Dijk."
        spans = decode(
            text,
            ["Signed", " by", " Van", " Dijk", "."],
            ["O", "O", "B-SURNAME", "I-SURNAME", "O"],
        )
        self.assertEqual(spans, [("SURNAME", 10, 18, "Van Dijk")])


class WordPieceParityTests(unittest.TestCase):
    """WordPiece offsets never open on whitespace; both must decode alike."""

    def test_wordpiece_and_sentencepiece_agree_on_an_email(self):
        text = "Reach me at MFP31@aol.com today."
        sentencepiece = decode(
            text,
            ["Reach", " me", " at", " MFP31", "@", "aol", ".", "com", " today", "."],
            ["O", "O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL",
             "O", "O"],
        )
        wordpiece_offsets = [
            (0, 5), (6, 8), (9, 11), (12, 17), (17, 18), (18, 21), (21, 22),
            (22, 25), (26, 31), (31, 32),
        ]
        result = consolidate_predictions_to_words(
            text,
            offsets=wordpiece_offsets,
            labels=["O", "O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL", "I-EMAIL",
                    "I-EMAIL", "O", "O"],
            confidences=[0.99] * 10,
        )
        wordpiece = [(span.entity_type, span.start_char, span.end_char, span.text)
                     for span in result.spans]
        self.assertEqual(sentencepiece, wordpiece)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from integrated_pipeline.transcript_merger import (
    EmptyTranscriptError,
    merge_segment_transcripts,
    normalize_overlap_text,
)


class TranscriptMergerTests(unittest.TestCase):
    def test_exact_normalizer(self) -> None:
        self.assertEqual(normalize_overlap_text("  Don’t `STOP`—Now!  "), "dont stop now")
        self.assertEqual(normalize_overlap_text("café + ID_42"), "caf id 42")

    def test_largest_overlap_up_to_fourteen(self) -> None:
        shared = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
        result = merge_segment_transcripts(
            [f"start {shared}", f"{shared} finish"]
        )
        self.assertEqual(result.text, f"start {shared} finish")
        self.assertEqual(result.deduplicated_word_count, 14)

    def test_one_word_overlap_has_no_length_guard(self) -> None:
        result = merge_segment_transcripts(["we end with the", "the next part"])
        self.assertEqual(result.text, "we end with the next part")
        self.assertEqual(result.deduplicated_word_count, 1)

    def test_punctuation_case_apostrophe_and_blank(self) -> None:
        result = merge_segment_transcripts(
            ["We CAN’T stop,", "can't STOP now", "   "]
        )
        self.assertEqual(result.text, "We CAN’T stop, now")
        self.assertEqual(result.blank_segment_indices, (2,))
        self.assertEqual(result.deduplicated_word_count, 2)

    def test_all_blank_is_failure(self) -> None:
        with self.assertRaises(EmptyTranscriptError):
            merge_segment_transcripts(["", "  "])


if __name__ == "__main__":
    unittest.main()

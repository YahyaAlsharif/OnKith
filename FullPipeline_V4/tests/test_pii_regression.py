from __future__ import annotations

import unittest

from integrated_pipeline.pipeline import BLOCKED_MARKER, FailClosedMasker
from integrated_pipeline.schemas import PIIResult
from integrated_pipeline.span_merger import (
    SpanCandidate,
    consolidate_predictions_to_words,
    merge_pii_spans,
)
from integrated_pipeline.structured_pii import StructuredPIIRecognizer


class PIIRegressionTests(unittest.TestCase):
    def test_structured_recognizers_and_numbered_placeholders_are_preserved(self):
        text = (
            "Email a@example.com, then b@example.com; "
            "my phone number is 555 123 4567."
        )
        structured = [
            span.as_candidate()
            for span in StructuredPIIRecognizer().find_spans(text)
        ]
        result = merge_pii_spans(text, [], structured)
        self.assertIn("[EMAIL_1]", result.masked_text)
        self.assertIn("[EMAIL_2]", result.masked_text)
        self.assertIn("[TELEPHONENUM_1]", result.masked_text)
        self.assertNotIn("a@example.com", result.masked_text)
        self.assertNotIn("555 123 4567", result.masked_text)

    def test_whole_word_consolidation_and_structured_priority_are_preserved(self):
        text = "Mohammed@example.com"
        consolidated = consolidate_predictions_to_words(
            text,
            offsets=[(0, 3), (3, 8)],
            labels=["B-GIVENNAME", "I-GIVENNAME"],
            confidences=[0.9, 0.8],
            word_ids=[0, 0],
        )
        self.assertEqual(consolidated.spans[0].text, "Mohammed")
        structured = SpanCandidate(
            entity_type="EMAIL",
            start_char=0,
            end_char=len(text),
            text=text,
            source="structured:email",
            confidence=1.0,
        )
        merged = merge_pii_spans(text, consolidated.spans, [structured])
        self.assertEqual(merged.masked_text, "[EMAIL_1]")
        self.assertEqual(merged.chosen_spans, [structured])

    def test_fail_closed_never_releases_raw_text(self):
        class Broken:
            def mask_text(self, text: str) -> PIIResult:
                raise RuntimeError(text)

        raw = "Mohammed secret 123456"
        result = FailClosedMasker(Broken()).mask_text(raw)
        self.assertFalse(result.safe_to_release)
        self.assertEqual(result.masked_text, BLOCKED_MARKER)
        self.assertNotIn(raw, result.error or "")


if __name__ == "__main__":
    unittest.main()

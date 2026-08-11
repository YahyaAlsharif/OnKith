from __future__ import annotations

import sys
import unittest
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from privacy_eval.adapters import StructuredOnlyMasker


class ProductionComponentTests(unittest.TestCase):
    def test_frozen_structured_email_masks_unseen_domain(self):
        text = "send to person@unseen-domain.technology now"
        result = StructuredOnlyMasker(include_supplemental=False).mask_text(text)
        self.assertEqual(result.entities[0]["entity_type"], "EMAIL")
        self.assertNotIn("person@", result.masked_text)

    def test_supplemental_spoken_number_is_separate_from_frozen_recognizer(self):
        text = "phone number is a zero five five a six seven three four four two one"
        frozen = StructuredOnlyMasker(include_supplemental=False).mask_text(text)
        supplemented = StructuredOnlyMasker(include_supplemental=True).mask_text(text)
        self.assertEqual(frozen.entities, [])
        self.assertTrue(any(item["entity_type"] == "TELEPHONENUM" for item in supplemented.entities))

    def test_hard_negative_short_number_stays_unmasked(self):
        text = "the meeting starts in five minutes"
        result = StructuredOnlyMasker(include_supplemental=True).mask_text(text)
        self.assertEqual(result.entities, [])


if __name__ == "__main__":
    unittest.main()

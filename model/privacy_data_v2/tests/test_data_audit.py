from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_audit import (  # noqa: E402
    audit_records,
    extract_row_entity_types,
    historical_multilabel_split,
)


class DataAuditTests(unittest.TestCase):
    def test_audit_reports_empty_rows_rare_labels_domains_and_case(self) -> None:
        records = [
            {"source_text": "clean", "privacy_mask": [], "region": "GB", "source_dataset": "x"},
            {
                "source_text": "Ana ana@example.org",
                "privacy_mask": [
                    {"label": "GIVENNAME", "start": 0, "end": 3, "value": "Ana"},
                    {"label": "EMAIL", "start": 4, "end": 19, "value": "ana@example.org"},
                ],
                "region": "GB",
                "source_dataset": "x",
            },
        ]
        report = audit_records(records)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["empty_mask_rows"], 1)
        self.assertEqual(report["email_domains"], {"example.org": 1})
        self.assertEqual(report["name_case"]["GIVENNAME"]["title"], 1)
        self.assertIn("EMAIL", report["rare_labels"]["singleton"])

    def test_historical_split_is_complete_disjoint_and_deterministic(self) -> None:
        masks = [
            [{"label": "A"}],
            [{"label": "A"}],
            [{"label": "B"}],
            [{"label": "B"}],
            [{"label": "A"}, {"label": "B"}],
            [],
            [],
            [{"label": "A"}],
            [{"label": "B"}],
            [{"label": "A"}],
        ]
        labels_by_row, labels, indicator = extract_row_entity_types(masks)
        self.assertEqual(labels, ["A", "B"])
        first = historical_multilabel_split(
            indicator, labels_by_row, validation_fraction=0.2, seed=42
        )
        second = historical_multilabel_split(
            indicator, labels_by_row, validation_fraction=0.2, seed=42
        )
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(set(first[0]) | set(first[1]), set(range(len(masks))))


if __name__ == "__main__":
    unittest.main()

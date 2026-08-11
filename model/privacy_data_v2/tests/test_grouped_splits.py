from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_splits import (  # noqa: E402
    audit_factor_overlap,
    canonical_template,
    split_by_factor,
)


def row(uid: int, value: str, domain: str, template: str = "Contact {value}") -> dict:
    email = f"{value}@{domain}"
    text = template.format(value=email)
    start = text.index(email)
    return {
        "uid": uid,
        "source_text": text,
        "privacy_mask": [
            {"label": "EMAIL", "start": start, "end": start + len(email), "value": email}
        ],
        "region": "synthetic",
        "source_dataset": "unit",
        "partition": "train_candidate",
    }


class GroupedSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            row(1, "a", "example.com"),
            row(2, "b", "example.com"),
            row(3, "c", "example.org"),
            row(4, "d", "example.net", "Please email {value}"),
            row(5, "e", "example.edu", "Please email {value}"),
            row(6, "f", "example.travel", "Send to {value}"),
        ]

    def test_domain_components_never_cross_splits(self) -> None:
        train, holdout, audit = split_by_factor(
            self.records,
            factor="email_domain",
            holdout_fraction=0.34,
            seed=42,
        )
        self.assertTrue(train)
        self.assertTrue(holdout)
        self.assertTrue(audit["is_disjoint"])
        self.assertEqual(audit["overlap_count"], 0)

    def test_template_components_never_cross_splits(self) -> None:
        train, holdout, _ = split_by_factor(
            self.records,
            factor="canonical_template",
            holdout_fraction=0.34,
            seed=42,
        )
        audit = audit_factor_overlap(
            self.records, train, holdout, factor="canonical_template"
        )
        self.assertTrue(audit["is_disjoint"])

    def test_split_is_deterministic(self) -> None:
        first = split_by_factor(
            self.records, factor="email_domain", holdout_fraction=0.34, seed=7
        )
        second = split_by_factor(
            self.records, factor="email_domain", holdout_fraction=0.34, seed=7
        )
        self.assertEqual(first, second)

    def test_frozen_final_cannot_enter_split(self) -> None:
        self.records[0]["partition"] = "ood_final"
        with self.assertRaisesRegex(ValueError, "Frozen OOD-final"):
            split_by_factor(
                self.records, factor="email_domain", holdout_fraction=0.2, seed=42
            )

    def test_canonical_template_replaces_only_the_span(self) -> None:
        self.assertEqual(canonical_template(self.records[0]), "contact {email}")


if __name__ == "__main__":
    unittest.main()

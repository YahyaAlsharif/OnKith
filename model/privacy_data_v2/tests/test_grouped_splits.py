from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_splits import (  # noqa: E402
    LOW_INFORMATION_VALUE_LABELS,
    audit_factor_overlap,
    canonical_template,
    component_diagnostics,
    email_domain_factors,
    is_public_email_domain,
    open_entity_value_factors,
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
        # Deliberately not example.com or gmail.com: those are public/reserved
        # names and no longer group anything, so a fixture built from them would
        # test nothing. See PUBLIC_EMAIL_DOMAINS.
        self.records = [
            row(1, "a", "northwind-savings.com"),
            row(2, "b", "northwind-savings.com"),
            row(3, "c", "meridian-trust.org"),
            row(4, "d", "swiftflow-logistics.net", "Please email {value}"),
            row(5, "e", "harborview-clinic.edu", "Please email {value}"),
            row(6, "f", "verdant-bio.travel", "Send to {value}"),
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

    def test_a_real_domain_still_groups_its_rows(self) -> None:
        """The narrowing must not turn the factor off entirely."""
        self.assertEqual(
            email_domain_factors(self.records[0]), {"northwind-savings.com"}
        )
        self.assertEqual(
            email_domain_factors(self.records[0]), email_domain_factors(self.records[1])
        )


class LowInformationKeyTests(unittest.TestCase):
    """Keys that connect unrelated rows are narrowed; real signals are not."""

    def email_row(self, address: str) -> dict:
        text = f"write to {address} please"
        start = text.index(address)
        return {
            "uid": address, "source_text": text,
            "privacy_mask": [{"label": "EMAIL", "start": start,
                              "end": start + len(address), "value": address}],
            "source_dataset": "unit", "partition": "candidate",
        }

    def value_row(self, label: str, value: str) -> dict:
        text = f"the record says {value} exactly"
        start = text.index(value)
        return {
            "uid": f"{label}:{value}", "source_text": text,
            "privacy_mask": [{"label": label, "start": start,
                              "end": start + len(value), "value": value}],
            "source_dataset": "unit", "partition": "candidate",
        }

    def test_public_providers_are_recognised(self) -> None:
        for domain in ("gmail.com", "GMAIL.COM", "icloud.com", "outlook.com",
                       "example.com", "example.co.uk", "example.ie", "test.org"):
            with self.subTest(domain=domain):
                self.assertTrue(is_public_email_domain(domain))

    def test_generator_specific_domains_are_not_public(self) -> None:
        for domain in ("swiftflow-logistics.com", "northwind-savings.com",
                       "domain7.example", "harborview-clinic.edu"):
            with self.subTest(domain=domain):
                self.assertFalse(is_public_email_domain(domain))

    def test_public_mail_domains_do_not_connect_rows(self) -> None:
        first = self.email_row("ana@gmail.com")
        second = self.email_row("omar@gmail.com")
        self.assertEqual(email_domain_factors(first), set())
        self.assertEqual(email_domain_factors(second), set())

    def test_the_whole_address_is_still_grouped_by_entity_value(self) -> None:
        """Narrowing the domain key must not stop the address itself grouping."""
        row_one = self.email_row("ana@gmail.com")
        self.assertEqual(
            open_entity_value_factors(row_one), {"EMAIL:ana@gmail.com"}
        )
        self.assertNotEqual(
            open_entity_value_factors(row_one),
            open_entity_value_factors(self.email_row("omar@gmail.com")),
        )

    def test_dates_and_building_numbers_do_not_connect_rows(self) -> None:
        for label in sorted(LOW_INFORMATION_VALUE_LABELS):
            with self.subTest(label=label):
                self.assertEqual(open_entity_value_factors(self.value_row(label, "12")), set())

    def test_names_are_still_enforced(self) -> None:
        """The largest source of connectivity is a real constraint and stays."""
        for label in ("GIVENNAME", "SURNAME", "EMAIL", "TAXNUM", "STREET"):
            with self.subTest(label=label):
                self.assertNotIn(label, LOW_INFORMATION_VALUE_LABELS)
                self.assertEqual(
                    open_entity_value_factors(self.value_row(label, "emily")),
                    {f"{label}:emily"},
                )


class ComponentDiagnosticsTests(unittest.TestCase):
    def records(self) -> list[dict]:
        rows = []
        for index in range(30):
            # Every row shares one date, so `open_entity_value` would weld the
            # pool together if DATE were still a grouping key.
            text = f"on 2024-07-15 contact person{index} about it"
            rows.append({
                "uid": f"r{index}", "source_text": text,
                "privacy_mask": [
                    {"label": "DATE", "start": 3, "end": 13, "value": "2024-07-15"},
                    {"label": "GIVENNAME", "start": text.index(f"person{index}"),
                     "end": text.index(f"person{index}") + len(f"person{index}"),
                     "value": f"person{index}"},
                ],
                "source_dataset": "unit", "partition": "candidate",
            })
        return rows

    def test_diagnostics_name_the_connecting_factor(self) -> None:
        report = component_diagnostics(self.records(), ["open_entity_value", "canonical_template"])
        self.assertEqual(report["pool_rows"], 30)
        self.assertIn("open_entity_value", report["per_factor"])
        self.assertIn("leave_one_out", report)
        self.assertGreater(report["distinct_keys"], 0)
        self.assertTrue(report["top_connector_keys"])

    def test_a_shared_date_no_longer_forms_one_component(self) -> None:
        report = component_diagnostics(self.records(), ["open_entity_value"])
        self.assertEqual(report["per_factor"]["open_entity_value"]["component_count"], 30)

    def test_an_empty_pool_is_handled(self) -> None:
        report = component_diagnostics([], ["open_entity_value"])
        self.assertEqual(report["pool_rows"], 0)


if __name__ == "__main__":
    unittest.main()

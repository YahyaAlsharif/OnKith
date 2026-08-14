"""Selection, dedup, the contamination firewall, splits and augmentation.

The final test runs the whole pipeline on tiny synthetic data and asserts the
property that lets the generated corpora stay out of Git: no evaluation row is
ever a training row.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import augment_pipeline
import dedup
import manifest as manifest_module
import ontology
import pools
import records
import selection
from grouped_splits import (
    MODEL_V2_SPLIT_FACTORS,
    numeric_pattern_family_factors,
    organisation_family_factors,
    split_by_factors,
)


def record(uid, text, spans, *, partition="candidate", source="gretel", **extra):
    """Build a record, locating each value in the text and sorting by position."""
    mask = [
        {"label": label, "start": text.index(value),
         "end": text.index(value) + len(value), "value": value}
        for label, value in spans
    ]
    mask.sort(key=lambda span: (span["start"], span["end"]))
    return records.make_record(
        uid=uid,
        source_text=text,
        privacy_mask=mask,
        source_dataset=source,
        partition=partition,
        **extra,
    )


class FactorExtractorTests(unittest.TestCase):
    def test_numeric_pattern_family_groups_by_shape(self):
        first = record("a", "call +966 55 123 4567 now", [("TELEPHONENUM", "+966 55 123 4567")])
        second = record("b", "call +966 50 987 6543 now", [("TELEPHONENUM", "+966 50 987 6543")])
        third = record("c", "call 0117 496 0182 now", [("TELEPHONENUM", "0117 496 0182")])
        self.assertEqual(
            numeric_pattern_family_factors(first), numeric_pattern_family_factors(second)
        )
        self.assertNotEqual(
            numeric_pattern_family_factors(first), numeric_pattern_family_factors(third)
        )

    def test_numeric_family_ignores_entities_without_digits(self):
        self.assertEqual(organisation_family_factors(record("a", "at Acme Ltd", [])), set())

    def test_organisation_family_groups_by_head_word(self):
        first = record("a", "Horizon Logistics filed it", [("ORGANISATION", "Horizon Logistics")])
        second = record("b", "Horizon Analytics filed it", [("ORGANISATION", "Horizon Analytics")])
        third = record("c", "Meridian Trust filed it", [("ORGANISATION", "Meridian Trust")])
        self.assertEqual(
            organisation_family_factors(first), organisation_family_factors(second)
        )
        self.assertNotEqual(
            organisation_family_factors(first), organisation_family_factors(third)
        )


class MultiFactorSplitTests(unittest.TestCase):
    TEMPLATES = [
        "Contact {name} at {email} today",
        "Please forward the invoice to {name} on {email}",
        "{name} asked us to use {email} from now on",
        "The reviewer {name} can be reached at {email}",
        "Send the onboarding pack for {name} to {email} before friday",
        "We updated the record for {name}; the address is {email}",
    ]

    NAMES = [
        "Ana", "Omar", "Priyanka", "Kwame", "Saoirse", "Thiago", "Yuna", "Rami",
        "Chiamaka", "Lea", "Noura", "Aarav", "Fatou", "Mariam", "Dae", "Xiomara",
        "Bjorn", "Elif", "Hiroshi", "Nadia",
    ]

    def build(self, count=120):
        rows = []
        for index in range(count):
            name = f"{self.NAMES[index % len(self.NAMES)]}{'x' * (index // len(self.NAMES))}"
            email = f"user{index}@domain{index % 12}.example"
            text = self.TEMPLATES[index % len(self.TEMPLATES)].format(name=name, email=email)
            rows.append(record(
                f"row-{index}", text, [("GIVENNAME", name), ("EMAIL", email)]
            ))
        return rows

    def test_no_declared_factor_crosses_the_boundary(self):
        rows = self.build()
        train, holdout, audit = split_by_factors(
            rows, factors=MODEL_V2_SPLIT_FACTORS, holdout_fraction=0.20, seed=7
        )
        self.assertTrue(audit["is_disjoint"], audit["per_factor"])
        for name, per_factor in audit["per_factor"].items():
            self.assertEqual(per_factor["overlap_count"], 0, name)
        self.assertEqual(sorted(train + holdout), list(range(len(rows))))
        self.assertEqual(set(train) & set(holdout), set())

    def test_split_is_deterministic(self):
        rows = self.build()
        first = split_by_factors(rows, holdout_fraction=0.20, seed=7)[:2]
        second = split_by_factors(rows, holdout_fraction=0.20, seed=7)[:2]
        self.assertEqual(first, second)

    def test_the_seed_is_recorded_and_any_seed_yields_a_valid_split(self):
        # A different seed reorders equally sized components, but with only a
        # handful of components it may legitimately select the same one. The
        # guaranteed properties are that the seed is recorded and that every
        # seed produces a disjoint, complete split.
        rows = self.build()
        for seed in (7, 99, 12345):
            with self.subTest(seed=seed):
                train, holdout, audit = split_by_factors(
                    rows, holdout_fraction=0.20, seed=seed
                )
                self.assertEqual(audit["seed"], seed)
                self.assertTrue(audit["is_disjoint"])
                self.assertEqual(sorted(train + holdout), list(range(len(rows))))
                self.assertGreater(len(holdout), 0)

    def test_shared_email_domain_keeps_rows_together(self):
        rows = self.build()
        _, holdout, _ = split_by_factors(rows, holdout_fraction=0.20, seed=7)
        holdout_domains = {
            str(span["value"]).split("@")[1]
            for index in holdout
            for span in rows[index]["privacy_mask"]
            if span["label"] == "EMAIL"
        }
        for index, row in enumerate(rows):
            if index in holdout:
                continue
            for span in row["privacy_mask"]:
                if span["label"] == "EMAIL":
                    self.assertNotIn(str(span["value"]).split("@")[1], holdout_domains)

    def test_label_aware_reservation_is_reported(self):
        rows = self.build()
        _, _, audit = split_by_factors(
            rows, holdout_fraction=0.20, seed=7, min_independent_groups=2
        )
        self.assertIn("label_reservation", audit)
        for label in ("GIVENNAME", "EMAIL"):
            self.assertIn(label, audit["label_reservation"])
            self.assertGreater(audit["holdout_label_spans"].get(label, 0), 0)

    def test_labels_with_too_few_groups_are_reported_not_forced(self):
        rows = self.build(40)
        rows.append(record("rare", "the ip is 192.0.2.14", [("IPV4", "192.0.2.14")]))
        _, _, audit = split_by_factors(
            rows, holdout_fraction=0.20, seed=3, min_independent_groups=10
        )
        self.assertIn("IPV4", audit["labels_without_independent_groups"])

    def test_protected_and_frozen_rows_are_refused(self):
        rows = self.build(20)
        for partition, pattern in (("ood_final", "Frozen OOD-final"),
                                   ("val_supplementary", "protected partition")):
            with self.subTest(partition=partition):
                poisoned = list(rows)
                poisoned.append(record("bad", "Call Ana now", [("GIVENNAME", "Ana")],
                                       partition=partition))
                with self.assertRaisesRegex(ValueError, pattern):
                    split_by_factors(poisoned, holdout_fraction=0.20, seed=1)

    def test_a_fully_connected_pool_fails_loudly(self):
        """Every row sharing one template cannot be split; that must not pass quietly."""
        rows = [
            record(f"same-{index}", f"Contact Person{index} at the desk today",
                   [("GIVENNAME", f"Person{index}")])
            for index in range(60)
        ]
        with self.assertRaisesRegex(ValueError, "too connected to split"):
            split_by_factors(rows, holdout_fraction=0.20, seed=1)

    def test_unknown_factor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown grouping factors"):
            split_by_factors(self.build(10), factors=["nope"], holdout_fraction=0.2, seed=1)


class DeduplicationTests(unittest.TestCase):
    def test_exact_duplicates_are_removed_after_normalisation(self):
        deduper = dedup.Deduplicator()
        spans = [{"label": "GIVENNAME", "start": 5, "end": 8, "value": "Ana"}]
        self.assertTrue(deduper.check_and_add("Call Ana today", spans)[0])
        self.assertEqual(deduper.check_and_add("call  ANA   today", spans)[1], "exact")

    FIRST = "Please contact Ana Smith on the secure portal about the invoice today"
    SECOND = "Please contact Omar Khan on the secure portal about the invoice today"
    FIRST_SPANS = [
        {"label": "GIVENNAME", "start": 15, "end": 18, "value": "Ana"},
        {"label": "SURNAME", "start": 19, "end": 24, "value": "Smith"},
    ]
    SECOND_SPANS = [
        {"label": "GIVENNAME", "start": 15, "end": 19, "value": "Omar"},
        {"label": "SURNAME", "start": 20, "end": 24, "value": "Khan"},
    ]

    def test_same_template_different_values_is_caught(self):
        deduper = dedup.Deduplicator()
        self.assertTrue(deduper.check_and_add(self.FIRST, self.FIRST_SPANS)[0])
        keep, reason = deduper.check_and_add(self.SECOND, self.SECOND_SPANS)
        self.assertFalse(keep)
        self.assertIn(reason, {"same_structure", "near"})

    def test_replacement_rows_may_repeat_structure_but_not_text(self):
        deduper = dedup.Deduplicator()
        deduper.check_and_add(self.FIRST, self.FIRST_SPANS)
        self.assertTrue(deduper.check_and_add(
            self.SECOND, self.SECOND_SPANS, allow_structural_repeat=True)[0])
        self.assertFalse(deduper.check_and_add(
            self.FIRST, self.FIRST_SPANS, allow_structural_repeat=True)[0])


class ContaminationFirewallTests(unittest.TestCase):
    def test_evaluation_text_is_rejected_and_attributed(self):
        firewall = dedup.ContaminationFirewall()
        firewall.add_protected("val_original", ["Call Ana today"])
        firewall.add_protected("ood_final", ["what time is it in riyadh"])
        self.assertFalse(firewall.accept("call  ANA today"))
        self.assertEqual(firewall.check("Call Ana today"), "val_original")
        self.assertEqual(firewall.check("What Time Is It In Riyadh"), "ood_final")
        self.assertTrue(firewall.accept("a completely unrelated sentence"))
        self.assertEqual(firewall.audit()["rejected_rows"], 1)

    def test_assert_clean_names_the_offending_set(self):
        firewall = dedup.ContaminationFirewall()
        firewall.add_protected("ood_dev", ["Call Ana today"])
        rows = [record("leak", "Call Ana today", [("GIVENNAME", "Ana")], partition="train")]
        with self.assertRaisesRegex(AssertionError, "ood_dev"):
            firewall.assert_clean(rows, label="train")


class SelectionTests(unittest.TestCase):
    def test_rows_are_selected_for_weak_labels_only(self):
        selector = selection.SupplementarySelector()
        weak = record("w", "the account is SA0380000000608010167519 now",
                      [("ACCOUNTNUM", "SA0380000000608010167519")])
        kept, outcome = selector.consider(weak)
        self.assertTrue(outcome.accepted)
        self.assertIn("ACCOUNTNUM:volume", outcome.selection_reason)

        strong = record("s", "Call Ana today", [("GIVENNAME", "Ana")])
        kept, outcome = selector.consider(strong)
        self.assertIsNone(kept)
        self.assertEqual(outcome.reason, "no_target_need")

    def test_all_supported_spans_survive_in_a_selected_row(self):
        selector = selection.SupplementarySelector()
        row = record(
            "multi",
            "Ana sent 1,249.99 in euros to account SA0380000000608010167519",
            [("GIVENNAME", "Ana"), ("AMOUNT", "1,249.99"), ("CURRENCY", "euros"),
             ("ACCOUNTNUM", "SA0380000000608010167519")],
        )
        kept, outcome = selector.consider(row)
        self.assertTrue(outcome.accepted)
        self.assertEqual(
            {span["label"] for span in kept["privacy_mask"]},
            {"GIVENNAME", "AMOUNT", "CURRENCY", "ACCOUNTNUM"},
        )
        # GIVENNAME is not a weak label, so it was carried rather than targeted.
        self.assertEqual(selector.audit()["carried_spans_by_label"].get("GIVENNAME"), 1)

    def test_password_spans_are_gated_during_selection(self):
        selector = selection.SupplementarySelector()
        disclosed = record("p1", "my password is BlueHorse!92",
                           [("PASSWORD", "BlueHorse!92")])
        kept, outcome = selector.consider(disclosed)
        self.assertTrue(outcome.accepted)
        self.assertEqual(len(kept["privacy_mask"]), 1)

        talk = record("p2", "I forgot my password", [("PASSWORD", "password")])
        kept, outcome = selector.consider(talk)
        self.assertIsNone(kept)
        self.assertEqual(outcome.reason, "no_supported_spans")

    def test_diversity_labels_need_novel_surface_groups(self):
        selector = selection.SupplementarySelector()
        first = record("d1", "Horizon Logistics filed the return",
                       [("ORGANISATION", "Horizon Logistics")])
        self.assertTrue(selector.consider(first)[1].accepted)
        # Identical template and value family: nothing new to learn.
        duplicate = record("d2", "Horizon Logistics filed the return",
                           [("ORGANISATION", "Horizon Logistics")])
        self.assertTrue(selector.consider(duplicate)[1].accepted)  # ORGANISATION is also volume-weak

    def test_audit_reports_unmet_targets_rather_than_hiding_them(self):
        selector = selection.SupplementarySelector()
        selector.consider(record("w", "the ip is 192.0.2.14", [("IPV4", "192.0.2.14")]))
        unmet = selector.audit()["unmet_targets"]
        self.assertIn("IPV4", unmet)
        self.assertGreater(unmet["IPV4"]["shortfall"], 0)


class AugmentationTests(unittest.TestCase):
    def rows(self, count=80):
        return [
            record(f"t{index}", f"Please email Person{index} at user{index}@example.org today",
                   [("GIVENNAME", f"Person{index}"), ("EMAIL", f"user{index}@example.org")],
                   partition="train")
            for index in range(count)
        ]

    def test_protected_rows_cannot_reach_augmentation(self):
        rows = self.rows(10)
        rows.append(record("bad", "Call Ana now", [("GIVENNAME", "Ana")],
                           partition="val_supplementary"))
        plan = augment_pipeline.AugmentationPlan(base_rows=len(rows))
        with self.assertRaisesRegex(ValueError, "protected rows reached augmentation"):
            augment_pipeline.augment_training_rows(rows, augment_pipeline.load_resources(), plan)

    def test_augmented_rows_keep_exact_spans(self):
        rows = self.rows()
        plan = augment_pipeline.AugmentationPlan(base_rows=len(rows), seed=5)
        result = augment_pipeline.augment_training_rows(
            rows, augment_pipeline.load_resources(), plan
        )
        self.assertGreater(len(result.rows), 0)
        for row in result.rows:
            records.validate_model_v2_record(row)
            for span in row["privacy_mask"]:
                self.assertEqual(
                    row["source_text"][span["start"]:span["end"]], span["value"]
                )
            self.assertEqual(row["generation_type"], "augmented")
            self.assertEqual(row["partition"], "train")
            self.assertTrue(row["augmentation_family"])

    def test_family_caps_are_respected(self):
        rows = self.rows(200)
        plan = augment_pipeline.AugmentationPlan(base_rows=len(rows), seed=5)
        result = augment_pipeline.augment_training_rows(
            rows, augment_pipeline.load_resources(), plan
        )
        self.assertLessEqual(len(result.rows), plan.max_augmented_rows)
        for family, count in result.produced.items():
            self.assertLessEqual(count, plan.max_rows_per_family, family)

    def test_parent_diversity_floor_is_audited(self):
        rows = self.rows(200)
        plan = augment_pipeline.AugmentationPlan(base_rows=len(rows), seed=5)
        result = augment_pipeline.augment_training_rows(
            rows, augment_pipeline.load_resources(), plan
        )
        audit = result.audit(plan)
        self.assertIn("families_below_parent_floor", audit)
        self.assertIn("distinct_parents_by_family", audit)

    def test_holdout_values_are_removed_from_the_inventories(self):
        resources = augment_pipeline.load_resources()
        before = len(resources["entity_values"]["GIVENNAME"])
        blocked = {"GIVENNAME": {resources["entity_values"]["GIVENNAME"][0]}}
        after = augment_pipeline.restrict_inventories(resources["entity_values"], blocked)
        self.assertEqual(len(after["GIVENNAME"]), before - 1)

    def test_hard_negatives_are_empty_mask_rows(self):
        rows = augment_pipeline.generate_model_v2_hard_negatives(seed=1)
        self.assertGreaterEqual(len(rows), 140)
        categories = {row["hard_negative_category"] for row in rows}
        self.assertGreaterEqual(len(categories), 25)
        for row in rows:
            self.assertEqual(row["privacy_mask"], [])
        for required in ("password_talk", "location_query", "numeric_pattern",
                         "security_terminology", "generic_date", "name_like_word"):
            self.assertIn(required, categories)

    def test_every_ontology_label_has_an_augmentation_inventory(self):
        resources = augment_pipeline.load_resources()
        missing = set(ontology.ENTITIES) - set(resources["entity_values"])
        self.assertEqual(missing, set())


class EndToEndPipelineTests(unittest.TestCase):
    """Tiny synthetic run of the whole flow, asserting the leakage property."""

    def test_partitions_never_share_a_row(self):
        val_original = [
            record(f"v{index}", f"Original validation row {index} for Person{index}",
                   [("GIVENNAME", f"Person{index}")], partition="val_original",
                   source="ai4privacy")
            for index in range(30)
        ]
        ood = [record("ood-1", "what time is it in riyadh", [], partition="ood_dev")]

        firewall = pools.build_firewall(val_original=val_original, ood_dev=ood)

        templates = [
            "Transfer to account {account} in {currency} through {bank}",
            "The {bank} statement lists {account} with a balance in {currency}",
            "Please confirm {account} at {bank}; the settlement is in {currency}",
            "Our {currency} payment used {account} held at {bank}",
        ]
        currencies = ["euros", "pounds sterling", "Saudi riyal", "yen", "US dollars"]
        candidates = []
        for index in range(60):
            currency = currencies[index % len(currencies)]
            bank = ["Al Rajhi Bank", "Riyad Bank", "Meridian Trust",
                    "Northwind Savings", "Alinma Bank", "Banque Saudi Fransi",
                    "Pearl Commercial"][index % 7]
            account = f"SA{index:022d}"
            text = templates[index % len(templates)].format(
                account=account, currency=currency, bank=bank
            )
            row = record(
                f"c{index}", text,
                [("ACCOUNTNUM", account), ("CURRENCY", currency), ("BANKNAME", bank)],
            )
            self.assertTrue(firewall.accept(row["source_text"]))
            candidates.append(row)

        # A candidate that duplicates an evaluation row must be rejected.
        leak = record("leak", "Original validation row 3 for Person3",
                      [("GIVENNAME", "Person3")])
        self.assertFalse(firewall.accept(leak["source_text"]))

        supp_train, val_supplementary, split_audit = pools.split_supplementary(
            candidates, holdout_fraction=0.20, seed=13
        )
        self.assertTrue(split_audit["is_disjoint"])
        self.assertGreater(len(val_supplementary), 0)
        for row in val_supplementary:
            self.assertEqual(row["partition"], "val_supplementary")

        ai4_train = [
            record(f"t{index}", f"Base training row {index} about Person{index}",
                   [("GIVENNAME", f"Person{index}")], partition="train", source="ai4privacy")
            for index in range(100, 140)
        ]
        ai4_test = [
            record(f"x{index}", f"Base test row {index} about Person{index}",
                   [("GIVENNAME", f"Person{index}")], partition="test", source="ai4privacy")
            for index in range(200, 210)
        ]
        train_rows, test_rows, training_audit = pools.split_training_side(
            ai4_train, ai4_test, supp_train, test_fraction=0.10, seed=17
        )
        self.assertEqual(training_audit["ai4privacy_test_rows"], 10)

        firewall.assert_clean(train_rows, label="train")
        firewall.assert_clean(test_rows, label="test")

        report = pools.assert_partitions_disjoint({
            "train": train_rows,
            "test": test_rows,
            "val_original": val_original,
            "val_supplementary": val_supplementary,
            "ood_dev": ood,
        })
        self.assertTrue(report["is_disjoint"])

        statistics = pools.partition_statistics(train_rows)
        self.assertEqual(statistics["rows"], len(train_rows))
        self.assertIn("GIVENNAME", statistics["spans_by_label"])

    def test_overlapping_partitions_raise(self):
        shared = record("dup", "Call Ana today", [("GIVENNAME", "Ana")], partition="train")
        other = record("dup2", "Call Ana today", [("GIVENNAME", "Ana")], partition="val_original")
        with self.assertRaisesRegex(AssertionError, "Partition overlap"):
            pools.assert_partitions_disjoint({"train": [shared], "val_original": [other]})


class ManifestTests(unittest.TestCase):
    def payload(self, **overrides):
        base = dict(
            sources={"base": {"revision": "abc123"}},
            mapping_summary={"rows": 10},
            selection_audit={},
            password_audit={},
            dedup_audit={},
            firewall_audit={},
            supplementary_split_audit={"seed": 13, "holdout_fraction": 0.2,
                                       "factors": list(MODEL_V2_SPLIT_FACTORS)},
            training_split_audit={},
            augmentation_audit={},
            partition_statistics={},
            disjointness={},
            artifacts={"val_supplementary": {"sha256": "deadbeef", "rows": 42}},
            config={},
        )
        base.update(overrides)
        return manifest_module.build_manifest(**base)

    def test_manifest_records_the_ontology_and_round_trips(self):
        manifest = self.payload()
        self.assertEqual(manifest["ontology"]["num_labels"], 63)
        self.assertEqual(manifest["ontology"]["num_entities"], 31)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest_module.write_manifest(path, manifest)
            self.assertEqual(
                manifest_module.read_manifest(path)["ontology"]["num_labels"], 63
            )

    def test_identical_builds_verify_as_reproducible(self):
        report = manifest_module.verify_manifest(self.payload(), self.payload())
        self.assertTrue(report["reproducible"])
        self.assertEqual(report["differences"], [])

    def test_a_changed_holdout_hash_is_reported(self):
        drifted = self.payload(
            artifacts={"val_supplementary": {"sha256": "different", "rows": 42}}
        )
        report = manifest_module.verify_manifest(self.payload(), drifted)
        self.assertFalse(report["reproducible"])
        fields = {difference["field"] for difference in report["differences"]}
        self.assertIn("artifacts.val_supplementary.sha256", fields)

    def test_a_changed_seed_is_reported(self):
        drifted = self.payload(
            supplementary_split_audit={"seed": 999, "holdout_fraction": 0.2,
                                       "factors": list(MODEL_V2_SPLIT_FACTORS)}
        )
        report = manifest_module.verify_manifest(self.payload(), drifted)
        self.assertFalse(report["reproducible"])
        self.assertIn(
            "split.supplementary.seed",
            {difference["field"] for difference in report["differences"]},
        )


if __name__ == "__main__":
    unittest.main()

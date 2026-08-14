"""The canonical Model V2 ontology, source mapping and record contract."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import label_mapping
import ontology
import records
from label_mapping import mapping_decision


class OntologyTests(unittest.TestCase):
    def test_thirty_one_entities_and_sixty_three_labels(self):
        self.assertEqual(len(ontology.ENTITIES), 31)
        self.assertEqual(ontology.NUM_LABELS, 63)
        self.assertEqual(len(ontology.BIO_LABELS), 63)

    def test_derivation_from_the_tinybert_head(self):
        expected = (set(ontology.TINYBERT_ENTITIES) - {"ALLERGIES", "USERNAME", "WEIGHT"}) | {
            "PASSWORD"
        }
        self.assertEqual(set(ontology.ENTITIES), expected)

    def test_removed_labels_are_absent(self):
        self.assertEqual(set(ontology.REMOVED_ENTITIES), {"ALLERGIES", "USERNAME", "WEIGHT"})
        self.assertEqual(set(ontology.ABSENT_ENTITIES), {"HEIGHT"})
        for label in (*ontology.REMOVED_ENTITIES, *ontology.ABSENT_ENTITIES):
            with self.subTest(label=label):
                self.assertNotIn(label, ontology.ENTITIES)
                self.assertNotIn(f"B-{label}", ontology.BIO_LABELS)
                self.assertNotIn(f"I-{label}", ontology.BIO_LABELS)

    def test_password_is_the_only_addition(self):
        self.assertEqual(set(ontology.ADDED_ENTITIES), {"PASSWORD"})
        self.assertIn("PASSWORD", ontology.ENTITIES)
        added = set(ontology.ENTITIES) - set(ontology.TINYBERT_ENTITIES)
        self.assertEqual(added, {"PASSWORD"})

    def test_height_was_never_in_the_tinybert_head(self):
        # This is why removing three labels nets 33 - 3 = 30, not 33 - 4.
        self.assertNotIn("HEIGHT", ontology.TINYBERT_ENTITIES)
        self.assertEqual(len(ontology.TINYBERT_ENTITIES), 33)

    def test_bio_ordering_matches_the_historical_convention(self):
        self.assertEqual(ontology.BIO_LABELS[0], "O")
        self.assertEqual(list(ontology.ENTITIES), sorted(ontology.ENTITIES))
        for index, entity in enumerate(ontology.ENTITIES):
            self.assertEqual(ontology.BIO_LABELS[1 + index * 2], f"B-{entity}")
            self.assertEqual(ontology.BIO_LABELS[2 + index * 2], f"I-{entity}")

    def test_label_map_is_a_bijection_over_a_contiguous_range(self):
        self.assertEqual(set(ontology.LABEL2ID.values()), set(range(63)))
        for label, index in ontology.LABEL2ID.items():
            self.assertEqual(ontology.ID2LABEL[index], label)

    def test_shared_with_tinybert_excludes_password(self):
        self.assertEqual(len(ontology.SHARED_WITH_TINYBERT), 30)
        self.assertNotIn("PASSWORD", ontology.SHARED_WITH_TINYBERT)
        self.assertNotIn("WEIGHT", ontology.SHARED_WITH_TINYBERT)

    def test_weak_labels_come_from_the_measured_baseline(self):
        for label in ontology.WEAK_LABELS:
            self.assertLess(ontology.BASELINE_TRAIN_SPAN_COUNTS[label], 100)
        # PASSWORD has zero AI4Privacy English support, so it must be weak.
        self.assertIn("PASSWORD", ontology.WEAK_LABELS)
        self.assertEqual(ontology.BASELINE_TRAIN_SPAN_COUNTS["PASSWORD"], 0)

    def test_committed_label_map_matches_the_module(self):
        payload = ontology.label_map_payload()
        self.assertEqual(payload["num_entities"], 31)
        self.assertEqual(payload["num_labels"], 63)
        self.assertEqual(payload["entities"], list(ontology.ENTITIES))

        committed = (
            Path(__file__).resolve().parents[2]
            / "DeBERTa-v3-xsmall"
            / "config"
            / "model_v2_label_map.json"
        )
        if committed.exists():
            stored = json.loads(committed.read_text(encoding="utf-8"))
            self.assertEqual(stored["entities"], list(ontology.ENTITIES))
            self.assertEqual(stored["num_labels"], 63)


class LabelMappingTests(unittest.TestCase):
    def test_no_mapping_targets_a_label_outside_the_ontology(self):
        supported = set(ontology.ENTITIES)
        for row in label_mapping.mapping_table():
            if row["status"] == "mapped":
                self.assertIn(row["onkith_label"], supported, row)

    def test_every_ontology_label_has_at_least_one_source(self):
        mapped = {
            row["onkith_label"]
            for row in label_mapping.mapping_table()
            if row["status"] == "mapped"
        }
        self.assertEqual(set(ontology.ENTITIES) - mapped, set())

    def test_removed_labels_are_rejected_from_every_source(self):
        for source in label_mapping.SOURCES:
            for native in ("USERNAME", "user_name", "ALLERGIES", "allergies", "HEIGHT", "height",
                           "WEIGHT", "weight"):
                with self.subTest(source=source, native=native):
                    decision = mapping_decision(source, native)
                    self.assertIsNone(decision.target)
                    self.assertEqual(decision.status, "dropped")

    def test_unknown_labels_are_never_guessed(self):
        decision = mapping_decision("gretel", "some_label_nobody_declared")
        self.assertIsNone(decision.target)
        self.assertIn("never guessed", decision.reason)

    def test_every_decision_carries_a_reason(self):
        for row in label_mapping.mapping_table():
            self.assertTrue(row["reason"].strip(), row)

    def test_password_is_always_medium_confidence_and_span_gated(self):
        for source in ("gretel", "nemotron", "kiji", "ai4privacy"):
            for native in ("password", "PASSWORD"):
                with self.subTest(source=source, native=native):
                    decision = mapping_decision(source, native)
                    self.assertEqual(decision.target, "PASSWORD")
                    self.assertEqual(decision.confidence, "medium")
        gated = [
            row for row in label_mapping.mapping_table()
            if row["onkith_label"] == "PASSWORD"
        ]
        self.assertTrue(gated)
        for row in gated:
            self.assertEqual(row["requires_span_gate"], "password_rules")

    def test_ambiguous_concepts_stay_rejected(self):
        for native in ("name", "full_name", "address", "state", "nationality", "ipv6",
                       "routing_number", "swift_bic", "cvv", "api_key", "customer_id",
                       "medical_record_number", "license_plate", "date_time"):
            with self.subTest(native=native):
                self.assertIsNone(mapping_decision("gretel", native).target)

    def test_medium_confidence_mappings_are_flagged_for_review(self):
        for native in ("iban", "national_id", "date_of_birth"):
            self.assertEqual(mapping_decision("gretel", native).confidence, "medium")
        self.assertEqual(mapping_decision("gretel", "first_name").confidence, "high")

    def test_mapping_csv_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.csv"
            summary = label_mapping.write_mapping_csv(path)
            self.assertTrue(path.exists())
            self.assertEqual(summary["rows"], len(label_mapping.mapping_table()))
            self.assertGreater(summary["mapped"], 0)
            self.assertEqual(summary["distinct_targets"], 31)


class RecordContractTests(unittest.TestCase):
    def make(self, **overrides):
        payload = {
            "uid": "unit-1",
            "source_text": "Email Ana at ana@example.com",
            "privacy_mask": [
                {"label": "GIVENNAME", "start": 6, "end": 9, "value": "Ana"},
                {"label": "EMAIL", "start": 13, "end": 28, "value": "ana@example.com"},
            ],
            "source_dataset": "gretel",
            "partition": "candidate",
        }
        payload.update(overrides)
        return records.make_record(**payload)

    def test_valid_record_passes(self):
        record = self.make()
        records.validate_model_v2_record(record)
        self.assertEqual(records.labels_of(record), {"GIVENNAME", "EMAIL"})

    def test_out_of_ontology_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the Model V2 ontology"):
            self.make(privacy_mask=[{"label": "WEIGHT", "start": 6, "end": 9, "value": "Ana"}])

    def test_span_must_match_the_text_exactly(self):
        with self.assertRaises(ValueError):
            self.make(privacy_mask=[{"label": "GIVENNAME", "start": 0, "end": 3, "value": "Ana"}])

    def test_protected_partition_cannot_hold_augmented_rows(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            self.make(partition="val_original", generation_type="augmented")

    def test_protected_partitions_are_shared_with_the_augmentation_guard(self):
        import augmentation

        self.assertIs(records.PROTECTED_PARTITIONS, augmentation.PROTECTED_PARTITIONS)
        for partition in ("test", "val_original", "val_supplementary", "ood_dev", "ood_final"):
            self.assertIn(partition, records.PROTECTED_PARTITIONS)
        self.assertNotIn("train", records.PROTECTED_PARTITIONS)

    def test_builder_row_adapter_ignores_the_builder_split(self):
        row = {
            "id": "saudi-mena:abc",
            "text": "Call Omar at +966 55 123 4567",
            "spans": json.dumps([
                {"label": "GIVENNAME", "start": 5, "end": 9, "value": "Omar"},
                {"label": "TELEPHONENUM", "start": 13, "end": 29, "value": "+966 55 123 4567"},
            ]),
            "final_split": "validation",
            "source_dataset": "Saudi/MENA augmentation",
            "generation_type": "synthetic",
            "entity_source": "curated",
        }
        record = records.from_builder_row(row, partition="candidate", source_dataset="saudi_mena")
        self.assertEqual(record["partition"], "candidate")
        self.assertEqual(record["builder_final_split"], "validation")
        self.assertEqual(len(record["privacy_mask"]), 2)

    def test_jsonl_round_trip_is_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            summary = records.write_jsonl(path, [self.make(), self.make(uid="unit-2")])
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["spans"], 4)
            self.assertEqual(len(summary["sha256"]), 64)
            self.assertEqual(len(records.load_jsonl(path)), 2)

    def test_fingerprint_normalises_whitespace_and_case(self):
        self.assertEqual(
            records.text_fingerprint("Call  ANA today"),
            records.text_fingerprint("call ana   today"),
        )


if __name__ == "__main__":
    unittest.main()

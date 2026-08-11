from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
import sys

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from privacy_eval.fixtures import CATEGORY_ALLOCATION, generate_partition, write_partition
from privacy_eval.schema import load_cases

ROOT = EVALUATION_ROOT
FIXTURES = ROOT / "fixtures"


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = load_cases(FIXTURES / "ood_dev.jsonl")
        cls.final = load_cases(FIXTURES / "ood_final.jsonl")

    def test_exact_size_and_primary_allocation(self):
        for cases in (self.dev, self.final):
            self.assertEqual(len(cases), 360)
            self.assertEqual(Counter(case.category for case in cases), Counter(CATEGORY_ALLOCATION))

    def test_every_span_is_exactly_source_aligned(self):
        for case in [*self.dev, *self.final]:
            case.validate()

    def test_partitions_use_disjoint_ids_templates_and_entity_values(self):
        self.assertFalse({case.case_id for case in self.dev} & {case.case_id for case in self.final})
        self.assertFalse({case.template_id for case in self.dev} & {case.template_id for case in self.final})
        dev_values = {entity.text.casefold() for case in self.dev for entity in case.entities if len(entity.text) > 1}
        final_values = {entity.text.casefold() for case in self.final for entity in case.entities if len(entity.text) > 1}
        self.assertFalse(dev_values & final_values)

    def test_manifests_match_fixture_bytes_and_freeze_policy(self):
        for partition in ("dev", "final"):
            fixture = FIXTURES / f"ood_{partition}.jsonl"
            manifest = json.loads((FIXTURES / f"ood_{partition}.manifest.json").read_text(encoding="utf-8"))
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            self.assertEqual(manifest["fixture_sha256"], digest)
            self.assertEqual(manifest["category_allocation"], CATEGORY_ALLOCATION)
            self.assertEqual(manifest["frozen"], partition == "final")

    def test_generator_is_deterministic(self):
        self.assertEqual(
            [case.to_dict() for case in generate_partition("dev")],
            [case.to_dict() for case in self.dev],
        )
        self.assertEqual(
            [case.to_dict() for case in generate_partition("final")],
            [case.to_dict() for case in self.final],
        )

    def test_frozen_final_requires_explicit_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_partition(output, "final")
            with self.assertRaises(FileExistsError):
                write_partition(output, "final")

    def test_name_groups_include_controlled_case_minimal_pairs(self):
        for cases in (self.dev, self.final):
            groups = defaultdict(set)
            for case in cases:
                if case.category == "names":
                    groups[case.pair_id].add(case.subcategory)
            self.assertEqual(len(groups), 12)
            for variants in groups.values():
                self.assertTrue({"standard", "lowercase", "uppercase"} <= variants)

    def test_speech_and_asr_are_cross_cutting(self):
        for cases in (self.dev, self.final):
            speech_categories = {case.category for case in cases if "speech_like" in case.tags}
            self.assertTrue({"names", "email", "numeric_structured", "contextual_multi_entity", "hard_negatives"} <= speech_categories)

    def test_email_domains_are_not_limited_to_training_domains(self):
        familiar = {"gmail.com", "outlook.com", "yahoo.com", "hotmail.com", "aol.com", "tutanota.com", "protonmail.com"}
        for cases in (self.dev, self.final):
            addresses = [entity.text.lower() for case in cases for entity in case.entities if entity.entity_type == "EMAIL" and "@" in entity.text]
            self.assertTrue(addresses)
            self.assertFalse(any(address.rsplit("@", 1)[1] in familiar for address in addresses))

    def test_long_inputs_are_really_multi_window(self):
        for cases in (self.dev, self.final):
            long_cases = [case for case in cases if case.category == "long_window"]
            self.assertEqual(len(long_cases), 24)
            self.assertTrue(all(len(case.text.split()) > 700 for case in long_cases))
            self.assertTrue(any("near_beginning" in case.tags for case in long_cases))
            self.assertTrue(any("near_end" in case.tags for case in long_cases))
            self.assertTrue(any("window_overlap_probe" in case.tags for case in long_cases))

    def test_packaged_tokenizer_audit_covers_every_long_entity(self):
        audit = json.loads((ROOT / "baselines" / "tinybert_long_window_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["case_count"], 48)
        self.assertGreater(audit["minimum_full_token_count"], 512)
        self.assertGreaterEqual(audit["minimum_window_count"], 2)
        self.assertTrue(audit["all_entities_covered"])


if __name__ == "__main__":
    unittest.main()

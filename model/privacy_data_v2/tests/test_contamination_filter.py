"""Regression tests for the cross-partition contamination filter.

The failure these cover is real and was found by the firewall in a production
run: AI4Privacy English ships zero-length ``source_text`` rows in *both* its
official train and its official validation split, so the pinned project-train
split carried rows that were byte-identical to rows in ``val_original``.

The rules being locked down here:

* the protected evaluation row always survives;
* the training-side copy is the one that goes;
* every removal is auditable;
* the final firewall still raises if anything slips past.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dedup
import manifest as manifest_module
import pools
import records


def record(uid, text, spans=(), *, partition="candidate", source="ai4privacy", **extra):
    mask = [
        {"label": label, "start": text.index(value),
         "end": text.index(value) + len(value), "value": value}
        for label, value in spans
    ]
    mask.sort(key=lambda span: (span["start"], span["end"]))
    return records.make_record(
        uid=uid, source_text=text, privacy_mask=mask,
        source_dataset=source, partition=partition, **extra,
    )


def firewall_with(val_original=(), val_supplementary=(), ood_dev=(), ood_final=()):
    return pools.build_firewall(
        val_original=val_original, val_supplementary=val_supplementary,
        ood_dev=ood_dev, ood_final=ood_final,
    )


class ProtectedDuplicateInTrainTests(unittest.TestCase):
    """A training row that duplicates val_original: the training row goes."""

    def setUp(self):
        self.protected = [
            record("ai4privacy:24794024", "", partition="val_original"),
            record("v1", "Original validation row for Person1",
                   [("GIVENNAME", "Person1")], partition="val_original"),
        ]
        self.firewall = firewall_with(val_original=self.protected)
        self.candidates = [
            record("ai4privacy:24645970", "", partition="train"),
            record("ai4privacy:24645983", "", partition="train"),
            record("clean-1", "A genuinely different training row", partition="train"),
        ]

    def test_the_training_copy_is_removed_and_the_protected_row_is_untouched(self):
        cleaned, audit = pools.filter_contamination(
            self.firewall, {"ai4privacy_train": self.candidates}, stage="pre_split"
        )
        surviving = [row["uid"] for row in cleaned["ai4privacy_train"]]
        self.assertEqual(surviving, ["clean-1"])
        # val_original is never rewritten, shortened or reordered.
        self.assertEqual([row["uid"] for row in self.protected],
                         ["ai4privacy:24794024", "v1"])
        self.assertEqual(len(self.protected), 2)

    def test_every_removal_is_audited_with_both_sides_named(self):
        _, audit = pools.filter_contamination(
            self.firewall, {"ai4privacy_train": self.candidates}, stage="pre_split"
        )
        self.assertEqual(audit["removed_rows_total"], 2)
        self.assertEqual(audit["removed_rows_by_pool"]["ai4privacy_train"], 2)
        self.assertEqual(
            audit["collisions_by_candidate_and_protected_set"]["ai4privacy_train"]["val_original"], 2
        )
        self.assertEqual(audit["version"], pools.CONTAMINATION_FILTER_VERSION)
        self.assertEqual(audit["stage"], "pre_split")
        removed = {entry["uid"]: entry for entry in audit["removals"]}
        self.assertEqual(set(removed), {"ai4privacy:24645970", "ai4privacy:24645983"})
        for entry in removed.values():
            self.assertEqual(entry["protected_set"], "val_original")
            self.assertEqual(entry["protected_uid"], "ai4privacy:24794024")
            self.assertEqual(entry["source_dataset"], "ai4privacy")
        self.assertEqual(audit["unique_protected_rows_hit"], 1)
        self.assertIn("ai4privacy_train -> val_original [ai4privacy]",
                      audit["collisions_by_source_dataset"])

    def test_the_firewall_passes_after_filtering(self):
        cleaned, _ = pools.filter_contamination(
            self.firewall, {"ai4privacy_train": self.candidates}, stage="pre_split"
        )
        self.firewall.assert_clean(cleaned["ai4privacy_train"], label="train")
        remaining = pools.assert_no_protected_overlap(
            self.firewall, {"train": cleaned["ai4privacy_train"]}
        )
        self.assertEqual(sum(remaining.values()), 0)

    def test_the_unfiltered_pool_still_fails_the_firewall(self):
        """Proof the test data really is contaminated before filtering."""
        with self.assertRaisesRegex(AssertionError, "val_original"):
            self.firewall.assert_clean(self.candidates, label="train")


class ProtectedDuplicateInHistoricalTestTests(unittest.TestCase):
    """The pinned 16,374-row comparison set is not sacred; integrity is."""

    def test_a_contaminated_test_row_is_removed_and_the_count_is_restated(self):
        protected = [record("ai4privacy:24794024", "", partition="val_original")]
        firewall = firewall_with(val_original=protected)
        historical_test = [record("ai4privacy:24645941", "", partition="test")] + [
            record(f"t{index}", f"Historical test row {index}", partition="test")
            for index in range(5)
        ]
        cleaned, audit = pools.filter_contamination(
            firewall, {"ai4privacy_test": historical_test}, stage="pre_split"
        )
        self.assertEqual(len(historical_test), 6)
        self.assertEqual(len(cleaned["ai4privacy_test"]), 5)
        self.assertEqual(audit["removed_rows_by_pool"]["ai4privacy_test"], 1)
        self.assertNotIn(
            "ai4privacy:24645941", [row["uid"] for row in cleaned["ai4privacy_test"]]
        )
        # The protected row survives untouched.
        self.assertEqual(len(protected), 1)
        firewall.assert_clean(cleaned["ai4privacy_test"], label="test")

    def test_the_comparison_subset_shrinks_rather_than_keeping_a_dirty_row(self):
        protected = [record("p", "shared text", partition="val_original")]
        firewall = firewall_with(val_original=protected)
        historical = [record("dirty", "shared text", partition="test"),
                      record("clean", "unique text", partition="test")]
        cleaned, audit = pools.filter_contamination(
            firewall, {"ai4privacy_test": historical}, stage="pre_split"
        )
        historical_rows = len(historical)
        clean_rows = len(cleaned["ai4privacy_test"])
        self.assertLess(clean_rows, historical_rows)
        self.assertEqual(historical_rows - clean_rows,
                         audit["removed_rows_by_pool"]["ai4privacy_test"])


class SupplementaryCollisionTests(unittest.TestCase):
    def test_a_supplementary_row_colliding_with_the_holdout_is_rejected(self):
        holdout = [record("h1", "The transfer used account SA0380000000608010167519",
                          [("ACCOUNTNUM", "SA0380000000608010167519")],
                          partition="val_supplementary", source="gretel")]
        firewall = firewall_with(val_supplementary=holdout)
        candidates = [
            record("s1", "the transfer used   ACCOUNT SA0380000000608010167519",
                   [("ACCOUNTNUM", "SA0380000000608010167519")], source="gretel"),
            record("s2", "A different supplementary row", source="gretel"),
        ]
        cleaned, audit = pools.filter_contamination(
            firewall, {"supplementary_training_side": candidates}, stage="pre_split"
        )
        # Normalisation makes the first row the same text as the holdout row.
        self.assertEqual([row["uid"] for row in cleaned["supplementary_training_side"]], ["s2"])
        self.assertEqual(
            audit["collisions_by_candidate_and_protected_set"]
            ["supplementary_training_side"]["val_supplementary"],
            1,
        )
        self.assertEqual(len(holdout), 1)
        self.assertEqual(holdout[0]["uid"], "h1")

    def test_ood_sets_are_protected_too(self):
        firewall = firewall_with(
            ood_dev=[record("d1", "what time is it in riyadh", partition="ood_dev")],
            ood_final=[record("f1", "read me the wifi password", partition="ood_final")],
        )
        candidates = [
            record("a", "What Time Is It In Riyadh", source="kiji"),
            record("b", "Read me the WiFi password", source="kiji"),
            record("c", "an unrelated sentence entirely", source="kiji"),
        ]
        cleaned, audit = pools.filter_contamination(
            firewall, {"supplementary_training_side": candidates}, stage="pre_split"
        )
        self.assertEqual([row["uid"] for row in cleaned["supplementary_training_side"]], ["c"])
        matrix = audit["collisions_by_candidate_and_protected_set"]["supplementary_training_side"]
        self.assertEqual(matrix["ood_dev"], 1)
        self.assertEqual(matrix["ood_final"], 1)


class DeterminismTests(unittest.TestCase):
    def pools_and_firewall(self):
        protected = [record(f"v{index}", f"protected row {index}", partition="val_original")
                     for index in range(20)]
        firewall = firewall_with(val_original=protected)
        candidates = {
            "ai4privacy_train": [
                record(f"t{index}", f"protected row {index}" if index % 5 == 0
                       else f"training row {index}", partition="train")
                for index in range(40)
            ],
            "supplementary_training_side": [
                record(f"s{index}", f"supplementary row {index}", source="gretel")
                for index in range(15)
            ],
        }
        return firewall, candidates

    def test_the_same_input_produces_the_same_cleaned_rows_and_audit(self):
        first_cleaned, first_audit = pools.filter_contamination(
            *self.pools_and_firewall(), stage="pre_split"
        )
        second_cleaned, second_audit = pools.filter_contamination(
            *self.pools_and_firewall(), stage="pre_split"
        )
        self.assertEqual(first_cleaned, second_cleaned)
        self.assertEqual(first_audit, second_audit)

    def test_survivors_keep_their_input_order(self):
        firewall, candidates = self.pools_and_firewall()
        cleaned, _ = pools.filter_contamination(firewall, candidates, stage="pre_split")
        expected = [row["uid"] for row in candidates["ai4privacy_train"]
                    if firewall.check(row["source_text"]) is None]
        self.assertEqual([row["uid"] for row in cleaned["ai4privacy_train"]], expected)

    def test_pools_are_swept_in_a_fixed_order_regardless_of_caller_order(self):
        firewall, candidates = self.pools_and_firewall()
        reversed_input = dict(reversed(list(candidates.items())))
        _, audit = pools.filter_contamination(firewall, reversed_input, stage="pre_split")
        self.assertEqual(
            list(audit["removed_rows_by_pool"]),
            ["ai4privacy_train", "supplementary_training_side"],
        )

    def test_the_written_artifact_hash_is_stable_across_identical_builds(self):
        import tempfile

        digests = []
        for _ in range(2):
            firewall, candidates = self.pools_and_firewall()
            cleaned, _ = pools.filter_contamination(firewall, candidates, stage="pre_split")
            rows = [dict(row, partition="train") for row in cleaned["ai4privacy_train"]]
            with tempfile.TemporaryDirectory() as directory:
                artifact = records.write_jsonl(Path(directory) / "train.jsonl", rows)
                digests.append(artifact["sha256"])
        self.assertEqual(digests[0], digests[1])


class CacheInvalidationTests(unittest.TestCase):
    def test_a_manifest_without_a_pipeline_version_is_stale(self):
        self.assertEqual(manifest_module.cache_pipeline_version({}), 0)
        self.assertFalse(manifest_module.cache_is_current({}))
        self.assertFalse(manifest_module.cache_is_current(None))

    def test_a_pre_filter_pipeline_version_cannot_be_reused(self):
        old = {"data_pipeline_version": manifest_module.DATA_PIPELINE_VERSION - 1}
        self.assertFalse(manifest_module.cache_is_current(old))
        with self.assertRaises(manifest_module.StaleCacheError):
            manifest_module.assert_cache_current(old, path="manifest.json")

    def test_the_stale_message_says_what_changed_and_that_nothing_must_be_deleted(self):
        message = manifest_module.describe_stale_cache({"data_pipeline_version": 2},
                                                       path="manifest.json")
        self.assertIn("contamination filter", message)
        self.assertIn("manifest.json", message)
        self.assertIn("by hand", message)

    def test_a_current_manifest_is_accepted(self):
        current = {"data_pipeline_version": manifest_module.DATA_PIPELINE_VERSION}
        self.assertTrue(manifest_module.cache_is_current(current))
        manifest_module.assert_cache_current(current)

    def test_the_pipeline_version_is_a_critical_manifest_field(self):
        self.assertIn("data_pipeline_version", manifest_module.CRITICAL_FIELDS)
        self.assertIn("contamination_filter.version", manifest_module.CRITICAL_FIELDS)

    def test_a_pipeline_version_change_is_reported_by_verify_manifest(self):
        committed = {"data_pipeline_version": 3, "contamination_filter": {"version": 1}}
        regenerated = {"data_pipeline_version": 2, "contamination_filter": {"version": 1}}
        report = manifest_module.verify_manifest(committed, regenerated)
        self.assertFalse(report["reproducible"])
        self.assertIn("data_pipeline_version",
                      {difference["field"] for difference in report["differences"]})


class NoBypassTests(unittest.TestCase):
    """Filtering must not be able to replace the assertion."""

    def test_a_row_injected_after_filtering_still_raises(self):
        protected = [record("p", "protected row zero", partition="val_original")]
        firewall = firewall_with(val_original=protected)
        candidates = [record("c", "an ordinary training row", partition="train")]
        cleaned, _ = pools.filter_contamination(
            firewall, {"ai4privacy_train": candidates}, stage="pre_split"
        )
        firewall.assert_clean(cleaned["ai4privacy_train"], label="train")

        injected = cleaned["ai4privacy_train"] + [
            record("injected", "Protected  Row   Zero", partition="train")
        ]
        with self.assertRaisesRegex(AssertionError, "injected"):
            firewall.assert_clean(injected, label="train")
        with self.assertRaisesRegex(AssertionError, "val_original"):
            pools.assert_no_protected_overlap(firewall, {"train": injected})

    def test_assert_no_protected_overlap_names_the_pool_that_failed(self):
        firewall = firewall_with(
            ood_dev=[record("d", "what time is it in riyadh", partition="ood_dev")]
        )
        dirty = [record("x", "what time is it in riyadh", partition="test")]
        with self.assertRaisesRegex(AssertionError, "test row"):
            pools.assert_no_protected_overlap(firewall, {"test": dirty})


class AuditRenderingTests(unittest.TestCase):
    def test_the_printed_audit_states_the_removals_and_the_stage(self):
        firewall = firewall_with(
            val_original=[record("p", "shared row", partition="val_original")]
        )
        _, audit = pools.filter_contamination(
            firewall,
            {"ai4privacy_train": [record("a", "shared row"), record("b", "other row")]},
            stage="pre_split",
        )
        rendered = pools.render_contamination_audit(audit)
        self.assertIn("cross-partition contamination audit", rendered)
        self.assertIn("ai4privacy_train", rendered)
        self.assertIn("val_original: 1", rendered)
        self.assertIn("pre_split", rendered)

    def test_an_empty_sweep_renders_zeroes_rather_than_nothing(self):
        firewall = firewall_with(
            val_original=[record("p", "unrelated", partition="val_original")]
        )
        _, audit = pools.filter_contamination(
            firewall, {"ai4privacy_train": [record("a", "clean row")]}, stage="pre_write"
        )
        self.assertEqual(audit["removed_rows_total"], 0)
        self.assertIn("val_original: 0", pools.render_contamination_audit(audit))


class FirewallProvenanceTests(unittest.TestCase):
    def test_the_firewall_remembers_which_protected_row_a_text_belongs_to(self):
        firewall = dedup.ContaminationFirewall()
        firewall.add_protected("val_original", ["Call Ana today"], uids=["v-1"])
        self.assertEqual(firewall.locate("call  ANA today"), ("val_original", "v-1"))
        self.assertEqual(firewall.check("call  ANA today"), "val_original")
        self.assertIsNone(firewall.locate("something else"))

    def test_texts_without_uids_still_register(self):
        firewall = dedup.ContaminationFirewall()
        firewall.add_protected("ood_final", ["read me the wifi password"])
        self.assertEqual(firewall.locate("Read me the WiFi password"), ("ood_final", None))

    def test_the_first_protected_set_to_claim_a_text_keeps_it(self):
        firewall = dedup.ContaminationFirewall()
        firewall.add_protected("val_original", ["duplicated across sets"], uids=["v-1"])
        firewall.add_protected("ood_dev", ["duplicated across sets"], uids=["d-1"])
        self.assertEqual(firewall.locate("duplicated across sets"), ("val_original", "v-1"))


if __name__ == "__main__":
    unittest.main()

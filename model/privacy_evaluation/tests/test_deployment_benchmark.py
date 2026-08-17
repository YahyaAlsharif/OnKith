"""The committed CPU-benchmark subset must be identical on every device.

These tests never need the source corpus. They read the committed fixture, which
is exactly what a Raspberry Pi will have, and they check the two properties that
make a cross-device comparison meaningful: the bytes are the ones the manifest
describes, and selection is a pure function of the seed.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from privacy_eval import deployment_benchmark as bench


class CommittedSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.manifest = bench.load_subset()

    def test_the_fixture_matches_its_manifest(self) -> None:
        """load_subset re-hashes; this states the expectation the Pi depends on."""

        self.assertEqual(
            bench.sha256_file(bench.SUBSET_PATH), self.manifest["fixture_sha256"]
        )
        self.assertEqual(len(self.rows), self.manifest["coverage"]["rows"])

    def test_the_fixture_has_unix_line_endings(self) -> None:
        """A CRLF checkout would change the hash and break every cross-device claim."""

        self.assertNotIn(b"\r\n", bench.SUBSET_PATH.read_bytes())

    def test_row_ids_are_unique_and_content_addressed(self) -> None:
        ids = [row["row_id"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        for row in self.rows:
            self.assertEqual(bench.row_id(row), row["row_id"])
        self.assertEqual(
            bench.sha256_bytes("\n".join(ids).encode("utf-8")),
            self.manifest["row_ids_sha256"],
        )

    def test_rows_are_stored_in_row_id_order(self) -> None:
        """Order is the fixture's contract: a smoke prefix must be the same prefix."""

        ids = [row["row_id"] for row in self.rows]
        self.assertEqual(ids, sorted(ids))

    def test_every_span_is_exactly_source_aligned(self) -> None:
        for row in self.rows:
            for span in row["privacy_mask"]:
                self.assertEqual(
                    row["source_text"][span["start"] : span["end"]], span["value"]
                )

    def test_the_subset_is_about_two_thousand_rows(self) -> None:
        self.assertGreaterEqual(len(self.rows), 1900)
        self.assertLessEqual(len(self.rows), 2100)

    def test_hard_negatives_and_rare_labels_are_present(self) -> None:
        self.assertGreater(self.manifest["coverage"]["rows_without_pii"], 0)
        rows_by_entity = self.manifest["coverage"]["rows_by_entity"]
        # Every entity type the source view contains at all is represented.
        self.assertGreaterEqual(len(rows_by_entity), 30)
        for rare in ("BANKNAME", "PASSWORD", "TIMEZONE", "ACCOUNTNUM"):
            self.assertIn(rare, rows_by_entity)

    def test_common_labels_keep_their_natural_frequency(self) -> None:
        """Rare-label floors must not turn the subset into a balanced set."""

        rows_by_entity = self.manifest["coverage"]["rows_by_entity"]
        self.assertGreater(rows_by_entity["DATE"], 10 * rows_by_entity["ACCOUNTNUM"])

    def test_the_source_is_held_out_and_not_the_frozen_fixture(self) -> None:
        source = self.manifest["source"]["path"]
        self.assertEqual(self.manifest["source"]["view"], "test")
        self.assertIn("model_v2_test.jsonl", source)
        self.assertNotIn("ood_final", source)
        self.assertNotIn("train", Path(source).name)

    def test_take_is_a_deterministic_prefix(self) -> None:
        first = bench.take(self.rows, 25)
        self.assertEqual(len(first), 25)
        self.assertEqual([row["row_id"] for row in first], [row["row_id"] for row in self.rows[:25]])


class SelectionTests(unittest.TestCase):
    """Selection behaviour, exercised on a synthetic corpus rather than the real one."""

    @staticmethod
    def corpus() -> list[dict]:
        records = []
        for index in range(400):
            label = "DATE" if index % 2 else "CITY"
            text = f"row {index} mentions Riyadh on 2026-01-01 and more filler text"
            records.append(
                {
                    "uid": f"synthetic:{index}",
                    "source_text": text,
                    "source_dataset": "synthetic",
                    "generation_type": "original",
                    "privacy_mask": [
                        {"start": 20, "end": 26, "label": label, "value": text[20:26]}
                    ],
                }
            )
        for index in range(3):
            text = f"benign row {index} with nothing sensitive in it at all"
            records.append(
                {
                    "uid": f"synthetic:benign-{index}",
                    "source_text": text,
                    "source_dataset": "synthetic",
                    "generation_type": "original",
                    "privacy_mask": [],
                }
            )
        rare = "The transfer used CURRENCY marker QAR for the settlement leg"
        records.append(
            {
                "uid": "synthetic:rare",
                "source_text": rare,
                "source_dataset": "synthetic",
                "generation_type": "original",
                "privacy_mask": [{"start": 34, "end": 37, "label": "CURRENCY", "value": "QAR"}],
            }
        )
        return records

    def test_selection_is_reproducible(self) -> None:
        records = self.corpus()
        first = bench.select(records, target_rows=50, seed=7)
        second = bench.select(records, target_rows=50, seed=7)
        self.assertEqual(first, second)

    def test_a_different_seed_selects_differently(self) -> None:
        records = self.corpus()
        self.assertNotEqual(
            [row["row_id"] for row in bench.select(records, target_rows=50, seed=7)],
            [row["row_id"] for row in bench.select(records, target_rows=50, seed=8)],
        )

    def test_selection_does_not_depend_on_corpus_order(self) -> None:
        records = self.corpus()
        shuffled = list(reversed(records))
        self.assertEqual(
            bench.select(records, target_rows=50, seed=7),
            bench.select(shuffled, target_rows=50, seed=7),
        )

    def test_hard_negatives_and_the_rare_label_always_survive(self) -> None:
        rows = bench.select(self.corpus(), target_rows=20, seed=7)
        self.assertEqual(sum(1 for row in rows if not row["privacy_mask"]), 3)
        labels = Counter(
            span["label"] for row in rows for span in row["privacy_mask"]
        )
        self.assertEqual(labels["CURRENCY"], 1)

    def test_build_writes_a_verifiable_pair(self) -> None:
        records = self.corpus()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / "source.jsonl"
            with source.open("w", encoding="utf-8", newline="\n") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
            subset, manifest_path = out / "subset.jsonl", out / "subset.manifest.json"
            manifest = bench.build(
                source_path=source,
                subset_path=subset,
                manifest_path=manifest_path,
                target_rows=40,
                seed=7,
            )
            self.assertEqual(manifest["coverage"]["rows"], 40)
            rows, loaded = bench.load_subset(
                subset_path=subset, manifest_path=manifest_path
            )
            self.assertEqual(len(rows), 40)
            self.assertEqual(loaded["fixture_sha256"], manifest["fixture_sha256"])

            subset.write_text(subset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(bench.SubsetRejected):
                bench.load_subset(subset_path=subset, manifest_path=manifest_path)


if __name__ == "__main__":
    unittest.main()

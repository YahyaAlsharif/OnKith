"""Oracle ceiling for the Model V2 decode path.

The forensic audit of the DeBERTa-v3-xsmall run found that typed entity F1 was
being capped by the decoder rather than by the model: feeding the *gold* labels
through ``repair_bio`` -> ``consolidate_predictions_to_words`` ->
``evaluate_spans`` scored 0.2500, so no model could ever score higher.

These tests pin that ceiling back where it belongs. A flawless tagger must score
approximately 1.0. Any future change that shatters spans, shifts boundaries or
emits duplicates drops this number long before anyone spends a GPU-day noticing
it in a training run.

No model is loaded and no artifact is written.
"""

from __future__ import annotations

import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for extra in (REPO_ROOT / "FullPipeline_V4", REPO_ROOT / "model" / "privacy_evaluation"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from integrated_pipeline.span_merger import consolidate_predictions_to_words
from privacy_eval.dataset_metrics import evaluate_spans
from privacy_eval.model_v2_adapter import repair_bio

MODEL_DIR = REPO_ROOT / "model" / "DeBERTa-v3-xsmall"
TEST_JSONL = MODEL_DIR / "artifacts" / "model_v2" / "data" / "model_v2_test.jsonl"
MODEL_CACHE = MODEL_DIR / "artifacts" / "model_v2" / "cache"
BASE_MODEL = "microsoft/deberta-v3-xsmall"
BASE_MODEL_REVISION = "4b419818330868dff6a60ad3e6b1c730f8b8c0c6"
CONTEXT_LIMIT = 512
OVERFLOW_STRIDE = 128


def align_bio(text, spans, offsets):
    """The notebook's section 12 alignment, kept in step with `encode_records`."""

    labels = []
    previous_entity = None
    for start, end in offsets:
        if start == end:
            labels.append("O")
            previous_entity = None
            continue
        tag = "O"
        for span in spans:
            if start >= int(span["start"]) and end <= int(span["end"]):
                tag = f"{'B' if start == int(span['start']) else 'I'}-{span['label']}"
                break
            if start < int(span["end"]) and end > int(span["start"]):
                tag = f"{'B' if start <= int(span['start']) else 'I'}-{span['label']}"
                break
        if tag.startswith("I-") and previous_entity != tag[2:]:
            tag = "B-" + tag[2:]
        labels.append(tag)
        previous_entity = None if tag == "O" else tag[2:]
    return labels


def decode_gold(text, spans, offsets):
    """Everything a perfect tagger would hand the decoder, and nothing more."""

    labels = align_bio(text, spans, offsets)
    keep = [index for index, (start, end) in enumerate(offsets) if start != end]
    kept_offsets = [tuple(offsets[index]) for index in keep]
    repaired, _ = repair_bio([labels[index] for index in keep])
    consolidated = consolidate_predictions_to_words(
        text, kept_offsets, repaired, [1.0] * len(keep)
    )
    return [{"label": span.entity_type, "start": span.start_char, "end": span.end_char}
            for span in consolidated.spans]


def sentencepiece_offsets(text):
    """Offsets in the DeBERTa style, without needing the real tokenizer.

    Words and punctuation runs are split apart, and every token that follows
    whitespace absorbs that whitespace into its offset, which is the behaviour
    that broke the decoder.
    """

    offsets = []
    position = 0
    while position < len(text):
        start = position
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        if text[position].isalnum():
            while position < len(text) and text[position].isalnum():
                position += 1
        else:
            position += 1
        offsets.append((start, position))
    return offsets


class HermeticOracleTests(unittest.TestCase):
    """Runs everywhere: no tokenizer download, no dataset on disk."""

    records = [
        {
            "source_text": "Reach MFP31@aol.com or call +659-785-716-8859 by 9 today.",
            "privacy_mask": [
                {"label": "EMAIL", "start": 6, "end": 19},
                {"label": "TELEPHONENUM", "start": 28, "end": 45},
                {"label": "AGE", "start": 49, "end": 50},
            ],
        },
        {
            "source_text": "Host 192.168.1.100 served https://www.example.com/help on 28/11/1980.",
            "privacy_mask": [
                {"label": "IPV4", "start": 5, "end": 18},
                {"label": "URL", "start": 26, "end": 54},
                {"label": "DATE", "start": 58, "end": 68},
            ],
        },
        {
            "source_text": "Van Dijk moved to Riyadh on May 13th, 2023.",
            "privacy_mask": [
                {"label": "SURNAME", "start": 0, "end": 8},
                {"label": "CITY", "start": 18, "end": 24},
                {"label": "DATE", "start": 28, "end": 42},
            ],
        },
        {
            "source_text": "No personal details appear anywhere in this sentence.",
            "privacy_mask": [],
        },
    ]

    def _predict(self):
        return [decode_gold(record["source_text"], record["privacy_mask"],
                            sentencepiece_offsets(record["source_text"]))
                for record in self.records]

    def test_gold_labels_decode_to_a_perfect_typed_f1(self):
        metrics = evaluate_spans(self.records, self._predict())
        self.assertEqual(metrics["typed_entity"]["f1"], 1.0)

    def test_no_span_is_over_or_under_generated(self):
        metrics = evaluate_spans(self.records, self._predict())
        self.assertEqual(
            metrics["typed_entity"]["predicted"], metrics["typed_entity"]["support"]
        )

    def test_nothing_outside_a_gold_span_is_masked(self):
        metrics = evaluate_spans(self.records, self._predict())
        self.assertEqual(metrics["non_private_overmasking_rate"], 0.0)
        self.assertEqual(metrics["private_character_recall"], 1.0)

    def test_the_hard_negative_row_stays_clean(self):
        metrics = evaluate_spans(self.records, self._predict())
        self.assertEqual(metrics["hard_negative_false_positive_rate"], 0.0)


class RealTokenizerOracleTests(unittest.TestCase):
    """The same ceiling, measured with the real tokenizer on real test rows.

    Skipped when the gitignored corpus or the cached tokenizer is absent.
    """

    #: Measured at 0.9966 over 2,000 rows once the decoder was repaired. The
    #: shortfall is a handful of phone formats and window-truncated spans, not a
    #: systematic defect. The gate sits below that so it fails on regressions,
    #: not on noise.
    MINIMUM_TYPED_F1 = 0.99
    ROWS = 400

    @classmethod
    def setUpClass(cls):
        if not TEST_JSONL.exists():
            raise unittest.SkipTest(f"corpus not built: {TEST_JSONL}")
        try:
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"transformers unavailable: {error}")
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL, revision=BASE_MODEL_REVISION, use_fast=True,
                cache_dir=str(MODEL_CACHE), local_files_only=True,
            )
        except Exception as error:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"tokenizer not cached locally: {error}")
        cls.records = []
        with TEST_JSONL.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= cls.ROWS:
                    break
                cls.records.append(json.loads(line))

    def _predict(self):
        predictions = []
        for record in self.records:
            text = record["source_text"]
            encoded = self.tokenizer(
                text, truncation=True, max_length=CONTEXT_LIMIT,
                stride=OVERFLOW_STRIDE, return_overflowing_tokens=True,
                return_offsets_mapping=True, padding=False,
                return_token_type_ids=False,
            )
            # Overlapping windows see the same character twice; keep one offset.
            merged = {}
            for window in encoded["offset_mapping"]:
                for start, end in window:
                    if start != end:
                        merged[(int(start), int(end))] = True
            offsets = sorted(merged)
            predictions.append(decode_gold(text, record["privacy_mask"], offsets))
        return predictions

    def test_gold_labels_decode_to_near_perfect_typed_f1(self):
        metrics = evaluate_spans(self.records, self._predict())
        self.assertGreaterEqual(
            metrics["typed_entity"]["f1"], self.MINIMUM_TYPED_F1,
            f"decoder ceiling regressed to {metrics['typed_entity']['f1']:.4f}",
        )

    def test_span_count_tracks_the_gold_count(self):
        metrics = evaluate_spans(self.records, self._predict())
        typed = metrics["typed_entity"]
        ratio = typed["predicted"] / max(1, typed["support"])
        self.assertLess(ratio, 1.02, f"decoder emits {ratio:.2f}x the gold spans")
        self.assertGreater(ratio, 0.98, f"decoder emits {ratio:.2f}x the gold spans")

    def test_punctuation_bearing_labels_are_recoverable(self):
        """EMAIL scored exactly zero true positives before the repair."""

        metrics = evaluate_spans(self.records, self._predict())
        per_label = metrics["per_label"]
        for label in ("EMAIL", "TELEPHONENUM", "DATE", "AGE"):
            with self.subTest(label=label):
                if label not in per_label or not per_label[label]["support"]:
                    self.skipTest(f"{label} absent from this sample")
                self.assertGreater(per_label[label]["recall"], 0.9)

    def test_overmasking_stays_negligible(self):
        metrics = evaluate_spans(self.records, self._predict())
        self.assertLess(metrics["non_private_overmasking_rate"], 0.01)


if __name__ == "__main__":
    unittest.main()

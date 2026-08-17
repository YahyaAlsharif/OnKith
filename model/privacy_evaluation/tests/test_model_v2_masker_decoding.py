"""Regression cover for span decoding inside ``ModelV2OnnxMasker`` itself.

``test_span_decoding_oracle`` proves the shared decoder is correct, but it calls
``consolidate_predictions_to_words`` directly. The production ONNX masker sat
one layer above it and passed ``word_ids=`` taken from the tokenizer, so the
oracle stayed green while every ONNX evaluation decoded badly.

DeBERTa-v3 is SentencePiece, and its pre-tokenizer splits on whitespace only.
``Walusiak,`` and ``23G@tutanota.com.`` are therefore *one* word id each,
punctuation included. ``consolidate_predictions_to_words`` expands a sensitive
word to its full group bounds, so those ids appended the trailing comma or
period to almost every predicted entity. The span became a superset of the gold
span: character recall stayed perfect, exact-boundary typed F1 collapsed. Model
V2 typed F1 measured through this masker read 0.5901 where the identical logits
decoded through the training-side path read 0.9620.

These tests drive ``mask_text`` end to end with the real Model V2 tokenizer and
a stub session that emits gold labels, so a perfect tagger must produce exactly
the gold spans. No ONNX graph is loaded and no artifact is written.
"""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
for extra in (ROOT, REPO_ROOT / "FullPipeline_V3"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from privacy_eval.dataset_metrics import evaluate_spans  # noqa: E402
from privacy_eval.model_v2_adapter import ModelV2OnnxMasker  # noqa: E402

TOKENIZER_DIR = REPO_ROOT / "model" / "DeBERTa-v3-xsmall" / "artifacts" / "model_v2" / "tokenizer"
LABEL_MAP = REPO_ROOT / "model" / "DeBERTa-v3-xsmall" / "config" / "model_v2_label_map.json"

def make_case(name, text, *entities):
    """Locate each entity in ``text`` so no fixture offset is ever hand-counted."""

    spans = []
    for label, value in entities:
        start = text.index(value)
        if text.count(value) != 1:
            raise AssertionError(f"{value!r} is ambiguous in {text!r}")
        spans.append({"label": label, "start": start, "end": start + len(value)})
    return {"name": name, "source_text": text, "privacy_mask": spans}


#: Every case pairs an entity with the punctuation that used to be swallowed.
CASES = [
    make_case(
        "trailing_comma_and_period",
        "Reach Lautrim Walusiak, 641 Duxton Road, Singapore.",
        ("GIVENNAME", "Lautrim"),
        ("SURNAME", "Walusiak"),
        ("BUILDINGNUM", "641"),
        ("STREET", "Duxton Road"),
        ("CITY", "Singapore"),
    ),
    make_case("email_then_period", "Write to MFP31@aol.com.", ("EMAIL", "MFP31@aol.com")),
    make_case(
        "url_then_comma",
        "See https://www.example.com/help, then reply.",
        ("URL", "https://www.example.com/help"),
    ),
    make_case(
        "phone_then_comma",
        "Call +659-785-716-8859, then wait.",
        ("TELEPHONENUM", "+659-785-716-8859"),
    ),
    make_case("date_then_period", "He was born on 28/11/1980.", ("DATE", "28/11/1980")),
    make_case(
        "short_age_then_comma",
        "She is 9, and moved to Riyadh.",
        ("AGE", "9"),
        ("CITY", "Riyadh"),
    ),
    make_case("zipcode_then_period", "The postcode is 188116.", ("ZIPCODE", "188116")),
    make_case(
        "hard_negative",
        "No personal details appear anywhere in this sentence.",
    ),
]


def align_bio(text, spans, offsets):
    """Label every token from the gold spans - what a flawless tagger emits.

    Matches ``test_span_decoding_oracle.align_bio``: a SentencePiece token owns
    the whitespace in front of it, so a token may start before its entity and
    still belong to it.
    """

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


class GoldLabelSession:
    """An ONNX session stand-in that emits one fixed label per token position."""

    def __init__(self, label_ids, num_labels):
        self._label_ids = list(label_ids)
        self._num_labels = int(num_labels)

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, output_names, feeds):
        rows, length = np.asarray(feeds["input_ids"]).shape
        if rows != 1 or length != len(self._label_ids):
            raise AssertionError(
                f"fixture expects one unpadded window of {len(self._label_ids)} "
                f"tokens, got {rows}x{length}"
            )
        logits = np.full((rows, length, self._num_labels), -8.0, dtype=np.float32)
        for position, label_id in enumerate(self._label_ids):
            logits[0, position, label_id] = 8.0
        return [logits]


class MaskerSpanDecodingTests(unittest.TestCase):
    """A perfect tagger must decode to exactly the gold spans, punctuation aside."""

    @classmethod
    def setUpClass(cls):
        if not (TOKENIZER_DIR / "tokenizer.json").is_file():
            raise unittest.SkipTest(f"tokenizer not present: {TOKENIZER_DIR}")
        if not LABEL_MAP.is_file():
            raise unittest.SkipTest(f"label map not generated yet: {LABEL_MAP}")
        try:
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"transformers unavailable: {error}")
        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER_DIR), local_files_only=True, use_fast=True
        )
        payload = json.loads(LABEL_MAP.read_text(encoding="utf-8"))
        cls.id2label = {int(key): str(value) for key, value in payload["id2label"].items()}
        cls.label2id = {label: index for index, label in cls.id2label.items()}

    def _encode(self, text):
        return self.tokenizer(
            text, truncation=False, padding=False, return_attention_mask=True,
            return_offsets_mapping=True, return_token_type_ids=False,
        )

    def _predict(self, case):
        """Decode ``case`` through the real masker fed with gold labels."""

        text = case["source_text"]
        encoded = self._encode(text)
        offsets = [tuple(map(int, item)) for item in encoded["offset_mapping"]]
        labels = align_bio(text, case["privacy_mask"], offsets)
        unknown = sorted({label for label in labels if label not in self.label2id})
        self.assertEqual(unknown, [], f"fixture uses labels outside the ontology: {unknown}")

        masker = ModelV2OnnxMasker("unused.onnx", TOKENIZER_DIR, self.id2label)
        masker._tokenizer = self.tokenizer
        masker._session = GoldLabelSession(
            [self.label2id[label] for label in labels], len(self.id2label)
        )
        result = masker.mask_text(text)
        self.assertEqual(result.raw_model_output["window_count"], 1)
        return result

    def _spans(self, case):
        return sorted(
            (entity["start"], entity["end"], entity["entity_type"])
            for entity in self._predict(case).entities
        )

    def _gold(self, case):
        return sorted(
            (int(span["start"]), int(span["end"]), str(span["label"]))
            for span in case["privacy_mask"]
        )

    def _case(self, name):
        return next(case for case in CASES if case["name"] == name)

    # -- the defect itself --------------------------------------------------

    def test_the_tokenizer_really_merges_trailing_punctuation(self):
        """Without this the other tests would pass vacuously.

        If a future tokenizer starts splitting punctuation, the hazard is gone
        and this test says so out loud rather than leaving dead cover behind.
        """

        text = "Reach Lautrim Walusiak, 641 Duxton Road."
        surname_end = text.index("Walusiak") + len("Walusiak")
        encoded = self._encode(text)
        word_ids = encoded.word_ids()
        offsets = [tuple(map(int, item)) for item in encoded["offset_mapping"]]
        comma = next(index for index, off in enumerate(offsets)
                     if text[off[0]:off[1]] == ",")
        last_of_surname = next(index for index, off in enumerate(offsets)
                               if off[1] == surname_end and off[0] != off[1])
        self.assertEqual(
            word_ids[comma], word_ids[last_of_surname],
            "SentencePiece no longer merges a trailing comma into the preceding word; "
            "the masker regression these tests guard can no longer occur.",
        )

    def test_an_entity_never_swallows_the_following_punctuation(self):
        """The exact regression: same word id, but the comma stays outside."""

        case = self._case("trailing_comma_and_period")
        text = case["source_text"]
        for start, end, entity_type in self._spans(case):
            with self.subTest(entity=entity_type):
                self.assertNotIn(
                    text[end - 1], ",.;:",
                    f"{entity_type} span {text[start:end]!r} swallowed its punctuation",
                )

    # -- per-entity-family exact-boundary cover -----------------------------

    def test_trailing_comma_and_period_decode_exactly(self):
        case = self._case("trailing_comma_and_period")
        self.assertEqual(self._spans(case), self._gold(case))

    def test_email_decodes_as_one_exact_span(self):
        case = self._case("email_then_period")
        self.assertEqual(self._spans(case), self._gold(case))

    def test_url_decodes_as_one_exact_span(self):
        case = self._case("url_then_comma")
        self.assertEqual(self._spans(case), self._gold(case))

    def test_phone_decodes_as_one_exact_span(self):
        case = self._case("phone_then_comma")
        self.assertEqual(self._spans(case), self._gold(case))

    def test_date_decodes_as_one_exact_span(self):
        case = self._case("date_then_period")
        self.assertEqual(self._spans(case), self._gold(case))

    def test_a_single_character_age_decodes_exactly(self):
        case = self._case("short_age_then_comma")
        self.assertEqual(self._spans(case), self._gold(case))

    def test_zipcode_decodes_exactly(self):
        case = self._case("zipcode_then_period")
        self.assertEqual(self._spans(case), self._gold(case))

    def test_a_hard_negative_stays_clean(self):
        case = self._case("hard_negative")
        self.assertEqual(self._spans(case), [])

    # -- aggregate ----------------------------------------------------------

    def test_gold_labels_decode_to_a_perfect_typed_f1(self):
        predictions = [
            [
                {"label": entity_type, "start": start, "end": end}
                for start, end, entity_type in self._spans(case)
            ]
            for case in CASES
        ]
        metrics = evaluate_spans(CASES, predictions)
        self.assertEqual(metrics["typed_entity"]["f1"], 1.0)
        self.assertEqual(metrics["private_character_recall"], 1.0)
        self.assertEqual(metrics["non_private_overmasking_rate"], 0.0)
        self.assertEqual(metrics["hard_negative_false_positive_rate"], 0.0)

    def test_the_masked_text_leaves_punctuation_in_place(self):
        result = self._predict(self._case("email_then_period"))
        self.assertEqual(result.masked_text, "Write to [EMAIL_1].")


class MaskerDecodingContractTests(unittest.TestCase):
    """The masker must not reintroduce tokenizer word ids."""

    SOURCE = ROOT / "privacy_eval" / "model_v2_adapter.py"

    def test_no_tokenizer_word_ids_reach_the_shared_decoder(self):
        tree = ast.parse(self.SOURCE.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "consolidate_predictions_to_words"
        ]
        self.assertEqual(len(calls), 1, "expected exactly one consolidation call")
        self.assertNotIn(
            "word_ids", [keyword.arg for keyword in calls[0].keywords],
            "the masker is passing word ids again; SentencePiece ids merge "
            "punctuation into the preceding word and corrupt span boundaries",
        )

    def test_the_masker_never_reads_tokenizer_word_ids(self):
        self.assertNotIn(".word_ids()", self.SOURCE.read_text(encoding="utf-8"))

    def test_the_production_wordpiece_adapter_keeps_its_own_word_ids(self):
        """The fix is Model V2 only; WordPiece word ids are correct as they are."""

        source = (
            REPO_ROOT / "FullPipeline_V3" / "integrated_pipeline" / "pii_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertIn("word_ids=", source)


if __name__ == "__main__":
    unittest.main()

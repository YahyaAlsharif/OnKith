"""The audio benchmark's fixtures, metrics and result schema.

None of these tests loads a model. They read the committed WAV clips - exactly
what a Raspberry Pi will have - and exercise the pure functions the benchmark
draws its conclusions from: word error rate, entity alignment across a
recognizer's mistakes, failure attribution, and the shape of what gets written
to disk. A test that needed the 700 MB Canary artifact would never run in CI,
so the model-dependent paths are covered by construction instead: the runner
reports the recognizer as unavailable and still completes.
"""

from __future__ import annotations

import json
import sys
import unittest
import wave
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from privacy_eval import audio_benchmark as audio
from privacy_eval import benchmark_common as common


class ManifestTests(unittest.TestCase):
    """The manifest is the contract between the workstation and the Pi."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = audio.load_manifest()
        cls.clips = cls.manifest["clips"]

    def test_the_benchmark_stays_small_enough_for_a_raspberry_pi(self) -> None:
        self.assertLessEqual(len(self.clips), 50)
        self.assertGreaterEqual(len(self.clips), 20)
        self.assertLess(self.manifest["totals"]["total_bytes"], 24 * 1024 ** 2)

    def test_every_clip_matches_its_committed_hash(self) -> None:
        for clip in self.clips:
            with self.subTest(clip=clip["clip_id"]):
                path = audio.verify_clip(clip)
                self.assertEqual(path.stat().st_size, clip["bytes"])

    def test_a_rewritten_clip_is_rejected(self) -> None:
        tampered = dict(self.clips[0])
        tampered["sha256"] = "0" * 64
        with self.assertRaises(audio.FixtureRejected):
            audio.verify_clip(tampered)

    def test_clip_ids_are_unique(self) -> None:
        ids = [clip["clip_id"] for clip in self.clips]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_clip_is_16_khz_mono_16_bit(self) -> None:
        """The pipeline refuses anything else, and Canary refuses to resample."""

        for clip in self.clips:
            path = audio.FIXTURE_DIR / clip["audio_path"]
            with wave.open(str(path), "rb") as stream:
                self.assertEqual(stream.getframerate(), common.SAMPLE_RATE)
                self.assertEqual(stream.getnchannels(), 1)
                self.assertEqual(stream.getsampwidth(), 2)
                frames = stream.getnframes()
            self.assertAlmostEqual(
                frames / common.SAMPLE_RATE, clip["duration_seconds"], places=2
            )

    def test_every_expected_span_is_source_aligned(self) -> None:
        for clip in self.clips:
            reference = clip["reference_transcript"]
            for span in clip["expected_private_spans"]:
                with self.subTest(clip=clip["clip_id"], label=span["label"]):
                    self.assertEqual(
                        reference[span["start"]:span["end"]], span["text"]
                    )

    def test_every_label_is_in_the_model_v2_ontology(self) -> None:
        label_map = json.loads(
            (common.PRODUCTION_ROOT / "models" / "deberta_pii" / "config"
             / "model_v2_label_map.json").read_text(encoding="utf-8")
        )
        allowed = set(label_map["entities"])
        self.assertEqual(self.manifest["ontology_version"],
                         label_map["ontology_version"])
        for clip in self.clips:
            for span in clip["expected_private_spans"]:
                self.assertIn(span["label"], allowed)

    def test_the_benchmark_covers_several_duration_buckets_and_both_subsets(self) -> None:
        buckets = self.manifest["totals"]["duration_buckets"]
        self.assertGreaterEqual(len(buckets), 3)
        self.assertEqual({clip["subset"] for clip in self.clips},
                         {"synthetic_privacy", "public"})

    def test_hard_negatives_exist_so_false_positives_have_a_denominator(self) -> None:
        self.assertGreaterEqual(self.manifest["totals"]["hard_negative_clips"], 5)

    def test_the_synthetic_subset_is_labelled_as_synthetic_everywhere(self) -> None:
        """Reporting must never present generated audio as natural distribution."""

        subset = self.manifest["subsets"]["synthetic_privacy"]
        self.assertEqual(subset["reporting_label"], "synthetic privacy speech benchmark")
        self.assertIn("fictional", subset["privacy_notice"])
        for clip in self.clips:
            if clip["subset"] == "synthetic_privacy":
                self.assertEqual(clip["source"], "synthetic")

    def test_the_public_subset_records_its_licence_and_citation(self) -> None:
        subset = self.manifest["subsets"]["public"]
        self.assertEqual(subset["license"], "CC BY 4.0")
        self.assertIn("Panayotov", subset["citation"])
        self.assertIn("dev-clean", subset["split"])

    def test_no_clip_carries_a_real_contact_detail(self) -> None:
        """Every telephone number and domain must be a reserved fictional one."""

        for clip in self.clips:
            reference = clip["reference_transcript"].casefold()
            for span in clip["expected_private_spans"]:
                if span["label"] in ("EMAIL", "URL"):
                    self.assertIn("example", span["text"].casefold())
                if span["label"] == "TELEPHONENUM":
                    self.assertTrue(span["text"].casefold().startswith("five five five"))
            self.assertNotIn("@", reference)  # spoken form only, never a live address


class AudioReadingTests(unittest.TestCase):
    def test_a_clip_reads_back_as_normalised_float_audio(self) -> None:
        manifest = audio.load_manifest()
        clip = manifest["clips"][0]
        samples = audio.read_wav(audio.FIXTURE_DIR / clip["audio_path"])
        self.assertEqual(samples.dtype.name, "float32")
        self.assertLessEqual(float(abs(samples).max()), 1.0)
        self.assertAlmostEqual(
            len(samples) / common.SAMPLE_RATE, clip["duration_seconds"], places=2
        )


class NormalisationTests(unittest.TestCase):
    def test_casing_and_punctuation_are_removed_for_comparison(self) -> None:
        self.assertEqual(audio.normalise_text("My name is Alex Carter."),
                         "my name is alex carter")

    def test_apostrophes_are_deleted_rather_than_split(self) -> None:
        """LibriSpeech writes QUILTER'S; a recognizer may write quilters."""

        self.assertEqual(audio.normalise_text("MISTER QUILTER'S"), "mister quilters")

    def test_the_uncased_control_preserves_every_offset(self) -> None:
        text = "Call Alex Carter on 555-0147, please."
        uncased = audio.uncased_unpunctuated(text)
        self.assertEqual(len(uncased), len(text))
        self.assertEqual(uncased[5:16], text[5:16].casefold())
        self.assertNotIn(",", uncased)
        self.assertEqual(uncased, uncased.casefold())


class WordErrorRateTests(unittest.TestCase):
    def test_a_perfect_transcript_scores_zero(self) -> None:
        result = audio.word_error_rate("the quick brown fox", "The quick brown fox.")
        self.assertEqual(result["wer"], 0.0)
        self.assertEqual(result["errors"], 0)

    def test_one_substitution_in_four_words(self) -> None:
        result = audio.word_error_rate("the quick brown fox", "the quick brown box")
        self.assertEqual(result["substitutions"], 1)
        self.assertAlmostEqual(result["wer"], 0.25)

    def test_deletions_and_insertions_are_counted_separately(self) -> None:
        result = audio.word_error_rate("one two three four", "one two three four five")
        self.assertEqual(result["insertions"], 1)
        self.assertEqual(result["deletions"], 0)
        result = audio.word_error_rate("one two three four", "one two four")
        self.assertEqual(result["deletions"], 1)

    def test_an_empty_reference_does_not_divide_by_zero(self) -> None:
        self.assertEqual(audio.word_error_rate("", "")["wer"], 0.0)
        self.assertEqual(audio.word_error_rate("", "spurious")["wer"], 1.0)


class EntityAlignmentTests(unittest.TestCase):
    """Following a gold entity into a transcript the recognizer altered."""

    REFERENCE = "Call Alex Carter on five five five zero one four seven."
    SPAN = {"label": "GIVENNAME", "text": "Alex Carter",
            "start": REFERENCE.index("Alex Carter"),
            "end": REFERENCE.index("Alex Carter") + len("Alex Carter")}

    def test_an_intact_entity_is_fully_preserved(self) -> None:
        hypothesis = "call alex carter on five five five zero one four seven"
        location = audio.locate_entity(self.REFERENCE, self.SPAN, hypothesis)
        self.assertEqual(location["preservation_ratio"], 1.0)
        self.assertEqual(location["hypothesis_surface"], "alex carter")
        self.assertEqual(location["alignment"], "matched")

    def test_a_half_recognised_entity_reports_a_partial_ratio(self) -> None:
        hypothesis = "call alex carbon on five five five zero one four seven"
        location = audio.locate_entity(self.REFERENCE, self.SPAN, hypothesis)
        self.assertAlmostEqual(location["preservation_ratio"], 0.5)

    def test_a_wholly_misrecognised_entity_still_reports_where_it_went(self) -> None:
        """The substituted words are what the privacy model actually had to mask."""

        hypothesis = "call alexis cartwright on five five five zero one four seven"
        location = audio.locate_entity(self.REFERENCE, self.SPAN, hypothesis)
        self.assertEqual(location["preservation_ratio"], 0.0)
        self.assertEqual(location["alignment"], "substituted")
        self.assertIn("alexis", location["hypothesis_surface"])

    def test_a_dropped_entity_reports_no_surface(self) -> None:
        hypothesis = "call on five five five zero one four seven"
        location = audio.locate_entity(self.REFERENCE, self.SPAN, hypothesis)
        self.assertEqual(location["preserved_tokens"], 0)

    def test_alignment_is_not_confused_by_a_repeated_word(self) -> None:
        span = {"label": "SURNAME", "text": "Carter",
                "start": self.REFERENCE.index("Carter"),
                "end": self.REFERENCE.index("Carter") + len("Carter")}
        hypothesis = "call alex carter carter on five five five zero one four seven"
        location = audio.locate_entity(self.REFERENCE, span, hypothesis)
        self.assertEqual(location["preservation_ratio"], 1.0)
        self.assertEqual(location["hypothesis_surface"], "carter")

    def test_a_gold_span_inside_a_digit_run_locates_the_whole_numeral(self) -> None:
        """The fold makes a read-aloud number one token, and it is indivisible.

        "five five five" is the first three digits of one seven-digit number.
        Once both sides are folded there is no such thing as three sevenths of a
        numeral, and the region the privacy model had to mask is the whole run.
        """

        span = {"label": "TELEPHONENUM", "text": "five five five",
                "start": self.REFERENCE.index("five five five"),
                "end": self.REFERENCE.index("five five five") + len("five five five")}
        hypothesis = "call alex carter on five five five zero one four seven"
        location = audio.locate_entity(self.REFERENCE, span, hypothesis)
        self.assertEqual(location["preservation_ratio"], 1.0)
        self.assertEqual(location["hypothesis_surface"],
                         "five five five zero one four seven")

    def test_a_spoken_number_is_found_where_the_recognizer_wrote_a_numeral(self) -> None:
        """The V4 defect this fixes: Whisper writes "5550147", Canary wrote words."""

        span = {"label": "TELEPHONENUM", "text": "five five five zero one four seven",
                "start": self.REFERENCE.index("five five five"),
                "end": self.REFERENCE.index("five five five")
                + len("five five five zero one four seven")}
        location = audio.locate_entity(self.REFERENCE, span, "Call Alex Carter on 5550147.")
        self.assertEqual(location["preservation_ratio"], 1.0)
        self.assertEqual(location["hypothesis_surface"], "5550147")

    def test_sentence_punctuation_is_not_part_of_the_entity(self) -> None:
        """A masker leaves the full stop standing, and that is not a leak."""

        reference = "My name is Alex Carter."
        span = {"label": "SURNAME", "text": "Carter",
                "start": reference.index("Carter"),
                "end": reference.index("Carter") + len("Carter")}
        location = audio.locate_entity(reference, span, "My name is Alex Carter.")
        self.assertEqual(location["hypothesis_surface"], "Carter")

    def test_punctuation_inside_an_entity_is_kept(self) -> None:
        """Only the edges are trimmed: the hyphen in a number carries the number."""

        reference = "Call me back on five five five, zero one four seven."
        span = {"label": "TELEPHONENUM", "text": "five five five, zero one four seven",
                "start": reference.index("five five five"),
                "end": reference.index("five five five")
                + len("five five five, zero one four seven")}
        location = audio.locate_entity(reference, span, "Call me back on 555-0147.")
        self.assertEqual(location["hypothesis_surface"], "555-0147")


class FailureAttributionTests(unittest.TestCase):
    """The distinction the report depends on: whose fault was the leak."""

    @staticmethod
    def location(ratio: float) -> dict:
        return {"preservation_ratio": ratio}

    def test_a_preserved_and_fully_masked_entity_is_protected(self) -> None:
        self.assertEqual(audio.classify_entity(self.location(1.0), 1.0), "masked")

    def test_a_preserved_unmasked_entity_is_a_privacy_model_failure(self) -> None:
        self.assertEqual(audio.classify_entity(self.location(1.0), 0.0),
                         "leaked_privacy_model")

    def test_partial_masking_of_a_preserved_entity_is_reported_separately(self) -> None:
        self.assertEqual(audio.classify_entity(self.location(1.0), 0.5),
                         "partially_masked")

    def test_an_entity_the_recogniser_destroyed_is_attributed_to_stt(self) -> None:
        self.assertEqual(audio.classify_entity(self.location(0.0), 0.0),
                         "stt_altered_leaked")

    def test_a_destroyed_entity_masked_anyway_is_recorded_as_such(self) -> None:
        """Surface changed, protection held. That is a result, not a failure."""

        self.assertEqual(audio.classify_entity(self.location(0.0), 1.0),
                         "stt_altered_masked")

    def test_a_trailing_space_does_not_turn_a_mask_into_a_leak(self) -> None:
        self.assertEqual(audio.classify_entity(self.location(1.0), 0.9995), "masked")


class MaskCoverageTests(unittest.TestCase):
    """Coverage counts informative characters, never whitespace."""

    TEXT = "at ten past eleven today"

    def spans(self, *ranges) -> set[int]:
        covered: set[int] = set()
        for start, end in ranges:
            covered.update(range(start, end))
        return covered

    def test_one_span_over_the_whole_entity_is_full_coverage(self) -> None:
        start, end = self.TEXT.index("ten"), self.TEXT.index(" today")
        coverage = audio.mask_coverage(self.TEXT, start, end, self.spans((start, end)))
        self.assertEqual(coverage, 1.0)

    def test_two_adjacent_spans_still_count_as_full_coverage(self) -> None:
        """"ten past" as TIME and "eleven" as DATE leave a space between them."""

        start, end = self.TEXT.index("ten"), self.TEXT.index(" today")
        split = self.TEXT.index("eleven")
        covered = self.spans((start, split - 1), (split, end))
        self.assertEqual(audio.mask_coverage(self.TEXT, start, end, covered), 1.0)

    def test_a_missed_word_is_a_real_partial(self) -> None:
        start, end = self.TEXT.index("ten"), self.TEXT.index(" today")
        covered = self.spans((self.TEXT.index("eleven"), end))
        coverage = audio.mask_coverage(self.TEXT, start, end, covered)
        self.assertGreater(coverage, 0.0)
        self.assertLess(coverage, audio.FULL_MASK_THRESHOLD)

    def test_nothing_masked_is_zero(self) -> None:
        start, end = self.TEXT.index("ten"), self.TEXT.index(" today")
        self.assertEqual(audio.mask_coverage(self.TEXT, start, end, set()), 0.0)

    def test_an_absent_region_is_zero_rather_than_an_error(self) -> None:
        self.assertEqual(audio.mask_coverage(self.TEXT, None, None, {1, 2}), 0.0)
        self.assertEqual(audio.mask_coverage(self.TEXT, 5, 5, {1, 2}), 0.0)


class ComparableArmTests(unittest.TestCase):
    """The metric that makes "reference text vs audio pipeline" a fair figure.

    ``evaluate_spans`` scores a control arm against gold offsets and counts
    every gold character, spaces included. The end-to-end arm cannot use it at
    all - recognized text has no gold offsets - so it measures each entity's
    own characters, whitespace excluded, against the spans the model produced.
    Plotting one against the other compares two different quantities.
    ``_comparable_arm`` re-scores a control arm under the end-to-end rule, and
    these tests pin that rule down.
    """

    arm = staticmethod(audio.AudioBenchmark._comparable_arm)

    @staticmethod
    def clip(reference: str, *entities: tuple) -> dict:
        spans = []
        for text, label in entities:
            start = reference.index(text)
            spans.append({"start": start, "end": start + len(text),
                          "label": label, "text": text})
        return {"reference_transcript": reference, "expected_private_spans": spans}

    @staticmethod
    def cover(text: str, *fragments: str, label: str = "SURNAME") -> list:
        return [{"start": text.index(f), "end": text.index(f) + len(f),
                 "label": label}
                for f in fragments]

    def test_a_fully_covered_entity_is_recalled_and_leaks_nothing(self) -> None:
        reference = "call john smith at noon"
        clip = self.clip(reference, ("john smith", "SURNAME"))
        result = self.arm([(clip, self.cover(reference, "john smith"))])
        self.assertEqual(result["expected_entities"], 1)
        self.assertEqual(result["entities_fully_masked"], 1)
        self.assertEqual(result["entity_recall"], 1.0)
        self.assertEqual(result["character_recall"], 1.0)
        self.assertEqual(result["character_leakage"], 0.0)

    def test_two_adjacent_spans_over_one_entity_still_count_as_covered(self) -> None:
        """The whole reason this definition exists: no mask covers a space."""

        reference = "call john smith at noon"
        clip = self.clip(reference, ("john smith", "SURNAME"))
        split = self.cover(reference, "john", "smith")
        result = self.arm([(clip, split)])
        self.assertEqual(result["character_recall"], 1.0)
        self.assertEqual(result["entities_fully_masked"], 1)

    def test_an_unmasked_entity_leaks_all_of_its_characters(self) -> None:
        reference = "call john smith at noon"
        clip = self.clip(reference, ("john smith", "SURNAME"))
        result = self.arm([(clip, [])])
        self.assertEqual(result["entity_recall"], 0.0)
        self.assertEqual(result["character_recall"], 0.0)
        self.assertEqual(result["character_leakage"], 1.0)

    def test_a_half_masked_entity_is_a_partial_never_a_full_mask(self) -> None:
        reference = "call john smith at noon"
        clip = self.clip(reference, ("john smith", "SURNAME"))
        result = self.arm([(clip, self.cover(reference, "john"))])
        self.assertEqual(result["entities_fully_masked"], 0)
        self.assertEqual(result["entity_recall"], 0.0)
        self.assertAlmostEqual(result["character_recall"], 4 / 9)
        self.assertAlmostEqual(result["character_leakage"], 5 / 9)

    def test_character_recall_is_weighted_by_characters_not_by_entity(self) -> None:
        """A long leaked entity must cost more than a short one."""

        reference = "reference nine six three two five one and tim"
        clip = self.clip(reference, ("nine six three two five one", "TELEPHONENUM"),
                         ("tim", "GIVENNAME"))
        masked_short = self.arm([(clip, self.cover(reference, "tim"))])
        masked_long = self.arm(
            [(clip, self.cover(reference, "nine six three two five one"))]
        )
        self.assertEqual(masked_short["entity_recall"], masked_long["entity_recall"])
        self.assertLess(masked_short["character_recall"],
                        masked_long["character_recall"])
        # "nine six three two five one" is 22 informative characters, "tim" 3.
        self.assertAlmostEqual(masked_short["character_recall"], 3 / 25)
        self.assertAlmostEqual(masked_long["character_recall"], 22 / 25)

    def test_an_arm_with_no_scored_clip_reports_zeros_rather_than_dividing(self) -> None:
        result = self.arm([])
        self.assertEqual(result["expected_entities"], 0)
        self.assertEqual(result["entity_recall"], 0.0)
        self.assertEqual(result["character_recall"], 0.0)
        self.assertEqual(result["character_leakage"], 0.0)

    def test_a_clip_without_entities_contributes_no_denominator(self) -> None:
        reference = "turn the kitchen lights off"
        benign = {"reference_transcript": reference, "expected_private_spans": []}
        self.assertEqual(self.arm([(benign, [])])["expected_entities"], 0)

    def test_the_definition_travels_with_the_numbers(self) -> None:
        """Anyone reading the JSON must see which denominator produced it."""

        result = self.arm([])
        self.assertIn("whitespace excluded", result["definition"])
        self.assertIn("comparable", result["definition"])

    def test_it_scores_a_control_arm_exactly_as_the_end_to_end_arm_is_scored(self) -> None:
        """The property the arm-comparison figure rests on.

        Feed the end-to-end path a "recognizer" that returned the reference
        transcript verbatim. Both arms then describe the same masking of the
        same text, so every number they report has to agree exactly. If they
        disagree, the figure is comparing two different quantities again.
        """

        reference = "call john smith at ten past about the visa"
        clip = self.clip(reference, ("john smith", "SURNAME"),
                         ("ten past", "TIME"), ("visa", "ORGANISATION"))
        spans = self.cover(reference, "john smith", "vis")  # one full, one partial
        comparable = self.arm([(clip, spans)])

        covered = audio._covered_characters(spans)
        entities = []
        for span in clip["expected_private_spans"]:
            location = audio.locate_entity(reference, span, reference)
            coverage = audio.mask_coverage(
                reference, location["hypothesis_char_start"],
                location["hypothesis_char_end"], covered,
            )
            entities.append({"outcome": audio.classify_entity(location, coverage),
                             "mask_coverage": coverage, **location})

        weights = [audio._informative_characters(e["hypothesis_surface"])
                   for e in entities]
        end_to_end_recall = (
            sum(e["mask_coverage"] * w for e, w in zip(entities, weights))
            / sum(weights)
        )
        masked = sum(1 for e in entities if e["outcome"] == "masked")

        self.assertAlmostEqual(comparable["character_recall"], end_to_end_recall)
        self.assertEqual(comparable["entities_fully_masked"], masked)
        self.assertAlmostEqual(comparable["entity_recall"], masked / len(entities))

    def test_it_differs_from_evaluate_spans_where_evaluate_spans_cannot_follow(self) -> None:
        """Not a duplicate of the existing metric - a different denominator.

        ``evaluate_spans`` counts the space inside "john smith" as a leaked
        private character when the model emits two adjacent spans. That is the
        exact disagreement that made the old figure wrong, so it is worth a
        test rather than a comment.
        """

        from privacy_eval.dataset_metrics import evaluate_spans

        reference = "call john smith at noon"
        clip = self.clip(reference, ("john smith", "SURNAME"))
        split = self.cover(reference, "john", "smith")
        gold = [{"source_text": reference, "privacy_mask": [
            {"start": s["start"], "end": s["end"], "label": s["label"], "value": s["text"]}
            for s in clip["expected_private_spans"]]}]

        standard = evaluate_spans(gold, [split])
        comparable = self.arm([(clip, split)])
        self.assertLess(standard["private_character_recall"], 1.0)
        self.assertEqual(comparable["character_recall"], 1.0)


class PrivacyScoringScopeTests(unittest.TestCase):
    """Only annotated clips may produce a privacy number.

    LibriSpeech is read prose. It contains person names - Linnell, Mason, Polly
    - and no gold annotation for them, so the model detects real entities there
    that nothing in the fixture credits. Counting those detections as false
    positives measures the missing annotation, not the model.
    """

    @staticmethod
    def benchmark(clips, results) -> audio.AudioBenchmark:
        runner = object.__new__(audio.AudioBenchmark)
        runner.run_id = "test-run"
        runner.clips = clips
        runner.clip_results = results
        runner.control_metrics = {}
        runner.privacy_scored_clips = []
        runner.privacy_metrics = {}
        runner.stt_available = True
        runner.completed = {}
        return runner

    @staticmethod
    def public_clip() -> dict:
        return {"clip_id": "public_001", "subset": "public",
                "privacy_annotated": False,
                "reference_transcript": "linnells pictures are a sort of up guards",
                "expected_private_spans": []}

    @staticmethod
    def synthetic_clip() -> dict:
        reference = "turn the kitchen lights off"
        return {"clip_id": "benign_001", "subset": "synthetic_privacy",
                "privacy_annotated": True, "is_hard_negative": True,
                "reference_transcript": reference, "expected_private_spans": []}

    @staticmethod
    def result(clip, *, predicted: int, spans) -> dict:
        return {"clip_id": clip["clip_id"], "expected_entities": 0,
                "predicted_entities": predicted, "entities": [],
                "_control_spans": {"reference_text": spans,
                                   "reference_uncased": spans}}

    def test_a_detection_in_unannotated_prose_is_not_a_false_positive(self) -> None:
        public = self.public_clip()
        detection = [{"start": 0, "end": 8, "label": "SURNAME"}]  # "linnells"
        runner = self.benchmark(
            [public], [self.result(public, predicted=1, spans=detection)]
        )
        runner.summarise_privacy()
        end_to_end = runner.privacy_metrics["end_to_end"]
        self.assertEqual(end_to_end["hard_negative_clips"], 0)
        self.assertEqual(end_to_end["hard_negative_false_positive_clips"], 0)
        self.assertEqual(end_to_end["hard_negative_false_positive_rate"], 0.0)

    def test_an_annotated_hard_negative_still_counts_against_the_model(self) -> None:
        """The fix must not quietly delete the false-positive denominator."""

        synthetic = self.synthetic_clip()
        detection = [{"start": 9, "end": 16, "label": "ORGANISATION"}]
        runner = self.benchmark(
            [synthetic], [self.result(synthetic, predicted=1, spans=detection)]
        )
        runner.summarise_privacy()
        end_to_end = runner.privacy_metrics["end_to_end"]
        self.assertEqual(end_to_end["hard_negative_clips"], 1)
        self.assertEqual(end_to_end["hard_negative_false_positive_clips"], 1)
        self.assertEqual(end_to_end["hard_negative_false_positive_rate"], 1.0)

    def test_unannotated_clips_are_kept_out_of_the_control_arms(self) -> None:
        public, synthetic = self.public_clip(), self.synthetic_clip()
        runner = self.benchmark(
            [public, synthetic],
            [self.result(public, predicted=1, spans=[{"start": 0, "end": 8,
                                                      "label": "SURNAME"}]),
             self.result(synthetic, predicted=0, spans=[])],
        )
        runner.summarise_privacy()
        self.assertEqual(runner.privacy_scored_clips, ["benign_001"])
        for arm in ("reference_text", "reference_uncased"):
            with self.subTest(arm=arm):
                metrics = runner.control_metrics[arm]
                # One row scored, not two: the prose clip never reaches the arm,
                # so its unannotated "linnells" detection cannot be charged to
                # the model as a hard-negative false positive.
                self.assertEqual(metrics["rows"], 1)
                self.assertEqual(metrics["hard_negative_rows"], 1)
                self.assertEqual(metrics["hard_negative_false_positive_rate"], 0.0)
                self.assertEqual(metrics["typed_entity"]["predicted"], 0)
                self.assertIn("comparable", metrics)

    def test_the_result_file_says_which_clips_were_scored_for_privacy(self) -> None:
        """A reader must be able to check the exclusion, not take it on trust."""

        public, synthetic = self.public_clip(), self.synthetic_clip()
        runner = self.benchmark(
            [public, synthetic],
            [self.result(public, predicted=1, spans=[]),
             self.result(synthetic, predicted=0, spans=[])],
        )
        runner.summarise_privacy()
        self.assertIn("LibriSpeech", runner.privacy_metrics["privacy_scoring_note"])
        self.assertNotIn("public_001", runner.privacy_metrics["privacy_scored_clips"])


class PostprocessingAmplificationTests(unittest.TestCase):
    """The spoken-digit finding, kept measurable so it cannot vanish quietly.

    Canary runs with ``pnc: no``, so "call me back on five five five zero one
    four seven" reaches the privacy model as words. When the model misses it,
    the display renderer - which runs after masking, by design - rewrites the
    surviving words into ``5550147``. The leak is the model's; the compact
    form is postprocessing's. The benchmark reports both, separately.
    """

    def test_a_number_the_masker_left_standing_is_reported_when_rendered(self) -> None:
        runs = audio.reconstructed_digit_runs(
            "call me back on five five five zero one four seven",
            "call me back on 5550147",
        )
        self.assertEqual([r["value"] for r in runs], ["5550147"])
        self.assertEqual(runs[0]["digits"], 7)

    def test_a_masked_number_produces_nothing_to_rebuild(self) -> None:
        """The placeholder has no digit words left, so the renderer is inert."""

        self.assertEqual(
            audio.reconstructed_digit_runs(
                "call me back on [TELEPHONENUM_1]",
                "call me back on [TELEPHONENUM_1]",
            ),
            [],
        )

    def test_digits_the_recogniser_already_produced_are_not_postprocessing(self) -> None:
        """Only what the render step *added* counts; otherwise every leak double-counts."""

        self.assertEqual(
            audio.reconstructed_digit_runs("the code is 5550147", "the code is 5550147"),
            [],
        )

    def test_an_everyday_quantity_is_below_identifier_scale(self) -> None:
        self.assertEqual(
            audio.reconstructed_digit_runs("wait thirty seconds", "wait 30 seconds"), []
        )
        self.assertEqual(
            audio.reconstructed_digit_runs("call back in twenty five minutes",
                                           "call back in 25 minutes"), [],
        )

    def test_the_threshold_follows_the_deployments_own_identifier_scale(self) -> None:
        common.ensure_import_paths()
        from integrated_pipeline.structured_pii import StructuredPIIConfig

        self.assertEqual(audio.IDENTIFIER_DIGITS,
                         StructuredPIIConfig().phone_min_digits)

    def test_a_card_length_run_is_caught_as_well_as_a_phone_length_one(self) -> None:
        runs = audio.reconstructed_digit_runs(
            "the card ends with four one one two six six eight three",
            "the card ends with 41126683",
        )
        self.assertEqual([r["digits"] for r in runs], [8])

    def test_several_rebuilt_numbers_in_one_clip_are_all_reported(self) -> None:
        runs = audio.reconstructed_digit_runs(
            "one two three four five six seven then eight seven six five four three two",
            "1234567 then 8765432",
        )
        self.assertEqual([r["value"] for r in runs], ["1234567", "8765432"])

    def test_the_run_reports_where_in_the_released_text_it_appears(self) -> None:
        """A reader must be able to point at the character, not just the value."""

        final = "call me back on 5550147"
        run = audio.reconstructed_digit_runs(
            "call me back on five five five zero one four seven", final)[0]
        self.assertEqual(final[run["start"]:run["end"]], run["value"])

    def test_the_production_renderer_really_does_rebuild_the_number(self) -> None:
        """Against the deployment's own code, not a string the test invented.

        This is the finding itself, pinned. The renderer is behaving exactly as
        specified - rendering spoken numerals is what it is for - so the test
        asserts the behaviour rather than forbidding it. If the deployment ever
        changes here, this fails and the reported limitation gets revisited
        instead of silently going stale.
        """

        common.ensure_import_paths()
        from app.digit_normalizer import normalize_spoken_digits

        leaked = normalize_spoken_digits(
            "call me back on five five five zero one four seven"
        )
        self.assertEqual(leaked, "call me back on 5550147")
        self.assertTrue(audio.reconstructed_digit_runs(
            "call me back on five five five zero one four seven", leaked))

        # And the same utterance, once the privacy model has done its job.
        masked = "call me back on [TELEPHONENUM_1]"
        self.assertEqual(normalize_spoken_digits(masked), masked)
        self.assertEqual(audio.reconstructed_digit_runs(masked,
                                                        normalize_spoken_digits(masked)),
                         [])


class AmplificationReportingTests(unittest.TestCase):
    """It has to reach the result files, not just the detector."""

    REFERENCE = "call me back on five five five zero one four seven"

    def runner(self, *, masked: bool) -> audio.AudioBenchmark:
        start = self.REFERENCE.index("five")
        clip = {"clip_id": "privacy_003", "subset": "synthetic_privacy",
                "privacy_annotated": True, "is_hard_negative": False,
                "reference_transcript": self.REFERENCE,
                "expected_private_spans": [
                    {"start": start, "end": len(self.REFERENCE),
                     "label": "TELEPHONENUM",
                     "text": self.REFERENCE[start:]}]}
        spans = ([{"start": start, "end": len(self.REFERENCE),
                   "label": "TELEPHONENUM"}] if masked else [])
        covered = audio._covered_characters(spans)
        entities = []
        for span in clip["expected_private_spans"]:
            location = audio.locate_entity(self.REFERENCE, span, self.REFERENCE)
            coverage = audio.mask_coverage(
                self.REFERENCE, location["hypothesis_char_start"],
                location["hypothesis_char_end"], covered)
            entities.append({"clip_id": clip["clip_id"], "label": span["label"],
                             "reference_text": span["text"],
                             "outcome": audio.classify_entity(location, coverage),
                             "mask_coverage": coverage, **location})
        final = ("call me back on [TELEPHONENUM_1]" if masked
                 else "call me back on 5550147")
        masked_text = ("call me back on [TELEPHONENUM_1]" if masked
                       else self.REFERENCE)
        record = {"clip_id": clip["clip_id"], "subset": clip["subset"],
                  "expected_entities": 1, "predicted_entities": len(spans),
                  "entities": entities, "privacy_input_text": self.REFERENCE,
                  "masked_text": masked_text, "final_masked_output": final,
                  "reconstructed_identifiers": audio.reconstructed_digit_runs(
                      masked_text, final),
                  "_control_spans": {"reference_text": spans,
                                     "reference_uncased": spans}}

        runner = object.__new__(audio.AudioBenchmark)
        runner.run_id = "test-run"
        runner.clips = [clip]
        runner.clip_results = [record]
        runner.control_metrics = {}
        runner.privacy_scored_clips = []
        runner.privacy_metrics = {}
        runner.stt_available = True
        runner.completed = {}
        runner.summarise_privacy()
        return runner

    def test_a_rebuilt_identifier_reaches_the_privacy_metrics(self) -> None:
        runner = self.runner(masked=False)
        block = runner.privacy_metrics["postprocessing_amplification"]
        self.assertEqual(block["clips_affected"], 1)
        self.assertEqual(block["identifiers_reconstructed"], 1)
        case = block["cases"][0]
        self.assertEqual(case["clip_id"], "privacy_003")
        self.assertEqual(case["rendered_value"], "5550147")
        self.assertEqual(case["unmasked_entity_labels"], ["TELEPHONENUM"])

    def test_masking_the_number_removes_the_amplification_entirely(self) -> None:
        """Where the model does its job, postprocessing has nothing to rebuild."""

        block = self.runner(masked=True).privacy_metrics["postprocessing_amplification"]
        self.assertEqual(block["identifiers_reconstructed"], 0)
        self.assertEqual(block["cases"], [])

    def test_it_is_reported_beside_the_other_causes_without_inflating_them(self) -> None:
        """One leak, counted once. The rebuild is a property of that leak."""

        attribution = self.runner(masked=False).privacy_metrics["failure_attribution"]
        self.assertEqual(attribution["postprocessing_amplified"], 1)
        self.assertEqual(attribution["privacy_model_caused"], 1)
        self.assertEqual(attribution["stt_caused"], 0)
        self.assertEqual(attribution["protected"], 0)
        self.assertIn("not a fourth kind of leak", attribution["interpretation"])

    def test_the_mechanism_and_its_blast_radius_are_written_down(self) -> None:
        block = self.runner(masked=False).privacy_metrics["postprocessing_amplification"]
        self.assertIn("after masking", block["mechanism"])
        self.assertIn("Display only", block["scope"])


class ArmParityTests(unittest.TestCase):
    """End to end against the control arm, through the real summariser.

    ``ComparableArmTests`` pins the definition; this pins the aggregation that
    the figure actually plots. A perfect recognizer returns the reference
    transcript verbatim, so the end-to-end arm and the reference-text arm are
    then describing the same masking of the same characters and every number
    they publish must agree exactly. Weighting one aggregate by raw region
    length and the other by informative characters would break this, which is
    precisely the mistake it exists to catch.
    """

    REFERENCE = "call john smith at ten past about the visa"

    @classmethod
    def clip(cls) -> dict:
        spans = []
        for text, label in (("john smith", "SURNAME"), ("ten past", "TIME"),
                            ("visa", "ORGANISATION")):
            start = cls.REFERENCE.index(text)
            spans.append({"start": start, "end": start + len(text),
                          "label": label, "text": text})
        return {"clip_id": "privacy_001", "subset": "synthetic_privacy",
                "privacy_annotated": True, "is_hard_negative": False,
                "reference_transcript": cls.REFERENCE,
                "expected_private_spans": spans}

    @classmethod
    def record(cls, clip: dict, predicted: list) -> dict:
        """One clip result as ``process_clip`` would build it for perfect STT."""

        covered = audio._covered_characters(predicted)
        entities = []
        for span in clip["expected_private_spans"]:
            location = audio.locate_entity(cls.REFERENCE, span, cls.REFERENCE)
            coverage = audio.mask_coverage(
                cls.REFERENCE, location["hypothesis_char_start"],
                location["hypothesis_char_end"], covered,
            )
            entities.append({
                "clip_id": clip["clip_id"], "label": span["label"],
                "reference_text": span["text"],
                "outcome": audio.classify_entity(location, coverage),
                "mask_coverage": coverage, **location,
            })
        return {"clip_id": clip["clip_id"], "expected_entities": len(entities),
                "predicted_entities": len(predicted), "entities": entities,
                "_control_spans": {"reference_text": predicted,
                                   "reference_uncased": predicted}}

    def summarise(self, predicted: list) -> audio.AudioBenchmark:
        runner = object.__new__(audio.AudioBenchmark)
        runner.run_id = "test-run"
        clip = self.clip()
        runner.clips = [clip]
        runner.clip_results = [self.record(clip, predicted)]
        runner.control_metrics = {}
        runner.privacy_scored_clips = []
        runner.privacy_metrics = {}
        runner.stt_available = True
        runner.completed = {}
        runner.summarise_privacy()
        return runner

    def spans(self, *fragments: str) -> list:
        return [{"start": self.REFERENCE.index(f),
                 "end": self.REFERENCE.index(f) + len(f), "label": "SURNAME"}
                for f in fragments]

    def assert_arms_agree(self, predicted: list) -> None:
        runner = self.summarise(predicted)
        end_to_end = runner.privacy_metrics["end_to_end"]
        comparable = runner.control_metrics["reference_text"]["comparable"]
        self.assertAlmostEqual(end_to_end["character_recall_after_stt"],
                               comparable["character_recall"])
        self.assertAlmostEqual(end_to_end["character_leakage_after_stt"],
                               comparable["character_leakage"])
        self.assertAlmostEqual(end_to_end["entity_recall_after_stt"],
                               comparable["entity_recall"])
        self.assertEqual(end_to_end["entities_fully_masked"],
                         comparable["entities_fully_masked"])
        self.assertEqual(end_to_end["expected_entities"],
                         comparable["expected_entities"])

    def test_the_two_arms_agree_when_every_entity_is_masked(self) -> None:
        self.assert_arms_agree(self.spans("john smith", "ten past", "visa"))

    def test_the_two_arms_agree_when_nothing_is_masked(self) -> None:
        self.assert_arms_agree([])

    def test_the_two_arms_agree_on_a_mixture_of_full_and_partial_masking(self) -> None:
        """The case where a length-weighted aggregate can disagree."""

        self.assert_arms_agree(self.spans("john smith", "vis"))

    def test_the_two_arms_agree_when_a_multi_word_entity_is_split(self) -> None:
        """Multi-word entities are where a whitespace-inclusive weight bites."""

        self.assert_arms_agree(self.spans("john", "smith", "ten", "past"))

    def test_a_leaked_multi_word_entity_costs_its_informative_characters(self) -> None:
        """Not a tautology: the shared value is the one both arms should report."""

        runner = self.summarise(self.spans("visa"))
        # "john smith" (9) and "ten past" (7) leak, "visa" (4) is covered.
        self.assertAlmostEqual(
            runner.privacy_metrics["end_to_end"]["character_recall_after_stt"],
            4 / 20,
        )


class PrivacyAnnotationManifestTests(unittest.TestCase):
    """The corrected fixture semantics, checked against the committed manifest."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = audio.load_manifest()
        cls.clips = cls.manifest["clips"]

    def test_every_clip_declares_whether_it_is_annotated_for_privacy(self) -> None:
        for clip in self.clips:
            with self.subTest(clip=clip["clip_id"]):
                self.assertIn(clip["privacy_annotated"], (True, False))

    def test_the_public_subset_is_not_annotated_and_says_why(self) -> None:
        subset = self.manifest["subsets"]["public"]
        self.assertFalse(subset["privacy_annotated"])
        self.assertIn("names", subset["privacy_notice"])
        for clip in self.clips:
            if clip["subset"] == "public":
                with self.subTest(clip=clip["clip_id"]):
                    self.assertFalse(clip["privacy_annotated"])
                    self.assertEqual(clip["expected_private_spans"], [])
                    self.assertFalse(clip["is_hard_negative"])

    def test_the_synthetic_subset_carries_the_privacy_annotations(self) -> None:
        self.assertTrue(self.manifest["subsets"]["synthetic_privacy"]["privacy_annotated"])
        for clip in self.clips:
            if clip["subset"] == "synthetic_privacy":
                with self.subTest(clip=clip["clip_id"]):
                    self.assertTrue(clip["privacy_annotated"])

    def test_every_hard_negative_is_annotated_so_it_has_a_denominator(self) -> None:
        negatives = [c for c in self.clips if c["is_hard_negative"]]
        self.assertGreaterEqual(len(negatives), 5)
        for clip in negatives:
            with self.subTest(clip=clip["clip_id"]):
                self.assertTrue(clip["privacy_annotated"])
                self.assertEqual(clip["expected_private_spans"], [])

    def test_the_totals_count_annotated_clips_separately_from_all_clips(self) -> None:
        totals = self.manifest["totals"]
        annotated = sum(1 for c in self.clips if c["privacy_annotated"])
        self.assertEqual(totals["privacy_annotated_clips"], annotated)
        self.assertLess(totals["privacy_annotated_clips"], totals["clips"])

    def test_the_public_subset_still_carries_reference_transcripts_for_wer(self) -> None:
        """Excluded from privacy scoring, not from the benchmark."""

        public = [c for c in self.clips if c["subset"] == "public"]
        self.assertGreaterEqual(len(public), 10)
        for clip in public:
            with self.subTest(clip=clip["clip_id"]):
                self.assertTrue(clip["reference_transcript"].strip())
                self.assertGreater(clip["duration_seconds"], 0.0)


class RealTimeFactorTests(unittest.TestCase):
    def test_rtf_below_one_means_faster_than_real_time(self) -> None:
        summary = common.summarise([0.5, 0.25, 0.75], unit_scale=1.0)
        self.assertAlmostEqual(summary["mean_ms"], 0.5)
        self.assertAlmostEqual(summary["p50_ms"], 0.5)
        self.assertLess(summary["mean_ms"], 1.0)

    def test_percentiles_interpolate_between_order_statistics(self) -> None:
        summary = common.summarise(range(1, 101), unit_scale=1.0)
        self.assertAlmostEqual(summary["p50_ms"], 50.5)
        self.assertAlmostEqual(summary["p95_ms"], 95.05)
        self.assertEqual(summary["count"], 100)

    def test_an_empty_sample_reports_a_count_and_nothing_else(self) -> None:
        self.assertEqual(common.summarise([]), {"count": 0})


class ClipSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = audio.load_manifest()

    def options(self, **kwargs) -> audio.AudioBenchmarkOptions:
        return audio.AudioBenchmarkOptions(**kwargs)

    def test_the_default_run_scores_every_clip(self) -> None:
        chosen = audio.select_clips(self.manifest, self.options())
        self.assertEqual(len(chosen), len(self.manifest["clips"]))

    def test_smoke_mode_is_small_deterministic_and_representative(self) -> None:
        first = audio.select_clips(self.manifest, self.options(smoke_test=True))
        second = audio.select_clips(self.manifest, self.options(smoke_test=True))
        self.assertEqual([c["clip_id"] for c in first], [c["clip_id"] for c in second])
        self.assertLessEqual(len(first), 5)
        subsets = {clip["subset"] for clip in first}
        self.assertEqual(subsets, {"synthetic_privacy", "public"})
        self.assertTrue(any(clip["expected_private_spans"] for clip in first))
        self.assertTrue(any(not clip["expected_private_spans"] for clip in first))

    def test_a_single_clip_can_be_selected_for_debugging(self) -> None:
        chosen = audio.select_clips(self.manifest, self.options(clips=("privacy_001",)))
        self.assertEqual([clip["clip_id"] for clip in chosen], ["privacy_001"])

    def test_an_unknown_clip_id_is_an_error_rather_than_an_empty_run(self) -> None:
        with self.assertRaises(audio.FixtureRejected):
            audio.select_clips(self.manifest, self.options(clips=("nope",)))

    def test_a_subset_filter_selects_only_that_subset(self) -> None:
        chosen = audio.select_clips(self.manifest, self.options(subset="public"))
        self.assertTrue(chosen)
        self.assertEqual({clip["subset"] for clip in chosen}, {"public"})


class ResultSchemaTests(unittest.TestCase):
    def test_the_clip_csv_columns_cover_every_stage_the_report_needs(self) -> None:
        required = {
            "clip_id", "audio_duration_s", "vad_latency_ms", "stt_latency_ms",
            "normalization_latency_ms", "privacy_latency_ms",
            "postprocessing_latency_ms", "end_to_end_latency_ms", "peak_rss_mb",
            "wer", "stt_real_time_factor", "pipeline_real_time_factor",
        }
        self.assertTrue(required.issubset(set(audio.CLIP_CSV_FIELDS)))

    def test_the_run_id_marks_an_audio_run_apart_from_a_text_run(self) -> None:
        run_id = common.make_run_id(common.utc_now(), "full", "pi5", prefix="audio")
        self.assertIn("_audio_", run_id)
        self.assertTrue(run_id.endswith("_pi5"))

    def test_the_run_directory_writes_lf_regardless_of_platform(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            run = common.RunDirectory("unit-test", Path(directory))
            run.json("payload.json", {"a": 1})
            run.jsonl("rows.jsonl", [{"b": 2}, {"b": 3}])
            run.markdown("summary.md", ["# title", "", "body"])
            for name in ("payload.json", "rows.jsonl", "summary.md"):
                self.assertNotIn(b"\r\n", (run.path / name).read_bytes())
            self.assertEqual(len(run.written), 3)

    def test_an_empty_table_writes_no_file_at_all(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            run = common.RunDirectory("unit-test", Path(directory))
            self.assertIsNone(run.csv("empty.csv", [], ["a"]))
            self.assertIsNone(run.jsonl("empty.jsonl", []))
            self.assertEqual(run.written, [])


class UnavailableRecogniserTests(unittest.TestCase):
    def test_the_help_text_names_the_exact_install_command(self) -> None:
        """A teammate on the Pi must not have to guess what is missing."""

        help_text = audio.STT_UNAVAILABLE_HELP
        self.assertIn("pip install", help_text)
        self.assertIn("nemo_toolkit", help_text)
        self.assertIn("requirements-", help_text)

    def test_the_help_names_the_pipeline_under_measurement(self) -> None:
        """A requirements file from the wrong package installs the wrong stack."""

        self.assertIn(
            common.PRODUCTION_ROOT.name, audio.STT_UNAVAILABLE_HELP
        )

    def test_the_help_says_how_to_get_the_pinned_checkpoint(self) -> None:
        """Unlike Whisper's, Parakeet's weights are pinned by hash.

        The recognizer refuses to start on a checkpoint that is not the exact
        459,243,520 bytes it was measured on, so "pip install" alone leaves a
        teammate stuck one step short.
        """

        self.assertIn("cache_parakeet_checkpoint", audio.STT_UNAVAILABLE_HELP)

    def test_the_help_never_points_at_a_superseded_recognizer(self) -> None:
        """Canary lives in the archive and Whisper lives in FullPipeline_V4.

        Both have frozen result folders. Pointing a stuck teammate at either
        installer would invite a rerun of a historical baseline under a run id
        that says V5.
        """

        help_text = audio.STT_UNAVAILABLE_HELP.lower()
        self.assertNotIn("acquire_canary", audio.STT_UNAVAILABLE_HELP)
        self.assertNotIn("faster-whisper", help_text)


if __name__ == "__main__":
    unittest.main()

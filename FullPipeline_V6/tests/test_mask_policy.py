"""Regression tests for the configurable per-label masking policy.

The policy separates what the model *detects* from what OnKith *masks*. These
tests pin both halves of that contract: a disabled label must survive into the
released text while still being reported, and nothing short of an explicit
boolean ``false`` may ever suppress masking.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from integrated_pipeline.mask_policy import MASK_EVERYTHING, MaskPolicy, resolve
from integrated_pipeline.span_merger import SpanCandidate, merge_pii_spans

TEXT = "what time is it in Riyadh and email ana@example.com"
CITY_SPAN = SpanCandidate("CITY", 19, 25, "Riyadh", "model_whole_word", 0.9)
EMAIL_SPAN = SpanCandidate("EMAIL", 36, 51, "ana@example.com", "structured:email", 1.0)
FULLY_MASKED = "what time is it in [CITY_1] and email [EMAIL_1]"


def merge(policy):
    return merge_pii_spans(TEXT, [CITY_SPAN], [EMAIL_SPAN], mask_policy=policy)


class DefaultBehaviourTests(unittest.TestCase):
    """An existing deployment that knows nothing about mask_policy is unchanged."""

    def test_no_policy_argument_masks_everything(self):
        merged = merge_pii_spans(TEXT, [CITY_SPAN], [EMAIL_SPAN])
        self.assertEqual(merged.masked_text, FULLY_MASKED)
        self.assertTrue(all(entity["masked"] for entity in merged.entities))
        self.assertEqual(merged.policy_exempt_span_count, 0)

    def test_none_and_empty_policy_match_the_default(self):
        for policy in (None, {}):
            with self.subTest(policy=policy):
                self.assertEqual(merge(policy).masked_text, FULLY_MASKED)

    def test_absent_label_defaults_to_masked(self):
        merged = merge({"EMAIL": True})
        self.assertEqual(merged.masked_text, FULLY_MASKED)

    def test_unknown_future_label_defaults_to_masked(self):
        self.assertTrue(MaskPolicy.from_config({"CITY": False}).should_mask("PASSWORD"))
        self.assertTrue(MASK_EVERYTHING.should_mask("SOME_LABEL_INVENTED_LATER"))


class EnabledLabelTests(unittest.TestCase):
    def test_explicitly_enabled_label_is_masked_normally(self):
        merged = merge({"CITY": True})
        self.assertEqual(merged.masked_text, FULLY_MASKED)
        city = next(e for e in merged.entities if e["entity_type"] == "CITY")
        self.assertTrue(city["masked"])
        self.assertEqual(city["placeholder"], "[CITY_1]")


class DisabledLabelTests(unittest.TestCase):
    def test_disabled_label_stays_visible_but_remains_detected(self):
        merged = merge({"CITY": False})
        self.assertEqual(merged.masked_text, "what time is it in Riyadh and email [EMAIL_1]")

        city = next(e for e in merged.entities if e["entity_type"] == "CITY")
        self.assertFalse(city["masked"])
        self.assertIsNone(city["placeholder"])
        # Detection is fully preserved: type, offsets and surface text.
        self.assertEqual((city["start"], city["end"], city["text"]), (19, 25, "Riyadh"))
        self.assertEqual(merged.policy_exempt_span_count, 1)

    def test_disabled_label_is_recorded_in_the_span_trace(self):
        merged = merge({"CITY": False})
        reasons = [d.reason for d in merged.decisions if d.span.entity_type == "CITY"]
        self.assertTrue(any("mask_policy_disabled" in reason for reason in reasons))

    def test_placeholder_numbering_only_counts_masked_spans(self):
        text = "Riyadh and Jeddah and ana@example.com"
        cities = [
            SpanCandidate("CITY", 0, 6, "Riyadh", "model_whole_word", 0.9),
            SpanCandidate("CITY", 11, 17, "Jeddah", "model_whole_word", 0.9),
        ]
        email = [SpanCandidate("EMAIL", 22, 37, "ana@example.com", "structured:email", 1.0)]
        self.assertEqual(
            merge_pii_spans(text, cities, email, mask_policy={"CITY": False}).masked_text,
            "Riyadh and Jeddah and [EMAIL_1]",
        )
        self.assertEqual(
            merge_pii_spans(text, cities, email).masked_text,
            "[CITY_1] and [CITY_2] and [EMAIL_1]",
        )


class MixedPolicyTests(unittest.TestCase):
    def test_enabled_and_disabled_labels_coexist(self):
        merged = merge({"EMAIL": True, "CITY": False})
        self.assertEqual(merged.masked_text, "what time is it in Riyadh and email [EMAIL_1]")
        by_type = {e["entity_type"]: e["masked"] for e in merged.entities}
        self.assertEqual(by_type, {"CITY": False, "EMAIL": True})


class FailClosedTests(unittest.TestCase):
    """Configuration mistakes must never become a privacy bypass."""

    def test_non_boolean_values_still_mask(self):
        for value in ("false", "no", 0, 1, None, [], {}, 0.0):
            with self.subTest(value=value):
                merged = merge({"CITY": value})
                self.assertEqual(merged.masked_text, FULLY_MASKED)
                self.assertTrue(merged.mask_policy["errors"])

    def test_non_mapping_policy_still_masks(self):
        for value in ("not-a-dict", ["CITY"], 42, 3.5):
            with self.subTest(value=value):
                merged = merge(value)
                self.assertEqual(merged.masked_text, FULLY_MASKED)
                self.assertTrue(merged.mask_policy["errors"])

    def test_invalid_keys_are_ignored_not_obeyed(self):
        policy = MaskPolicy.from_config({"": False, "   ": False})
        self.assertEqual(policy.disabled_labels, ())
        self.assertEqual(len(policy.errors), 2)

    def test_truthy_zero_is_not_false(self):
        # `isinstance(False, int)` pitfalls: 0 must not disable masking.
        self.assertTrue(MaskPolicy.from_config({"CITY": 0}).should_mask("CITY"))
        self.assertFalse(MaskPolicy.from_config({"CITY": False}).should_mask("CITY"))

    def test_non_string_entity_type_is_masked(self):
        self.assertTrue(MaskPolicy.from_config({"CITY": False}).should_mask(None))
        self.assertTrue(MaskPolicy.from_config({"CITY": False}).should_mask(7))

    def test_resolve_accepts_policy_mapping_or_none(self):
        self.assertIs(resolve(MASK_EVERYTHING), MASK_EVERYTHING)
        self.assertIs(resolve(None), MASK_EVERYTHING)
        self.assertEqual(resolve({"CITY": False}).disabled_labels, ("CITY",))


class ModelIndependenceTests(unittest.TestCase):
    """The mechanism survived the Model V1 (TinyBERT) to Model V2 (DeBERTa) swap."""

    def test_policy_operates_on_labels_not_model_internals(self):
        merged = merge_pii_spans(
            "the password is BlueHorse!92",
            [SpanCandidate("PASSWORD", 16, 28, "BlueHorse!92", "deberta_whole_word", 0.99)],
            [],
            mask_policy={"PASSWORD": False},
        )
        self.assertEqual(merged.masked_text, "the password is BlueHorse!92")
        self.assertFalse(merged.entities[0]["masked"])

    def test_policy_applies_to_structured_recognizer_spans_too(self):
        merged = merge({"EMAIL": False})
        email = next(e for e in merged.entities if e["entity_type"] == "EMAIL")
        self.assertFalse(email["masked"])
        self.assertIn("ana@example.com", merged.masked_text)

    def test_no_label_specific_branching(self):
        source = Path(__file__).resolve().parents[1] / "integrated_pipeline" / "mask_policy.py"
        text = source.read_text(encoding="utf-8")
        for label in ("CITY", "COUNTRY", "GIVENNAME", "EMAIL", "DATE"):
            self.assertNotIn(f'"{label}"', text)
            self.assertNotIn(f"'{label}'", text)


class DeploymentConfigTests(unittest.TestCase):
    def test_live_config_declares_an_empty_default_policy(self):
        config = json.loads(
            (Path(__file__).resolve().parents[1] / "config" / "live_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["pii"]["mask_policy"], {})
        self.assertIn("mask_policy_note", config["pii"])

    def test_pipeline_config_exposes_the_policy(self):
        from integrated_pipeline.config import PipelineConfig

        config = PipelineConfig()
        self.assertEqual(config.mask_policy, {})
        self.assertEqual(config.unmasked_labels, ())
        self.assertTrue(MaskPolicy.from_pipeline_config(config).masks_everything)

    def test_policy_is_not_part_of_the_pinned_validate_contract(self):
        from integrated_pipeline.config import PipelineConfig

        config = PipelineConfig()
        config.data["pii"]["mask_policy"] = {"CITY": False}
        config.validate()  # operator-tunable, so this must not raise
        self.assertEqual(config.unmasked_labels, ("CITY",))


class SupplementalMaskerTests(unittest.TestCase):
    """The supplemental wrapper re-merges, so it must honour the same policy."""

    def test_wrapper_inherits_policy_from_the_inner_masker(self):
        from integrated_pipeline.supplemental_pii import SupplementalPIIMasker

        class InnerMasker:
            mask_policy = MaskPolicy.from_config({"CITY": False})

            def mask_text(self, text):
                raise AssertionError("not called in this test")

        wrapper = SupplementalPIIMasker(InnerMasker())
        self.assertEqual(wrapper.mask_policy.disabled_labels, ("CITY",))

    def test_wrapper_without_a_policy_aware_inner_masker_masks_everything(self):
        from integrated_pipeline.supplemental_pii import SupplementalPIIMasker

        class LegacyMasker:
            def mask_text(self, text):
                raise AssertionError("not called in this test")

        wrapper = SupplementalPIIMasker(LegacyMasker())
        self.assertIsNone(wrapper.mask_policy)
        self.assertTrue(resolve(wrapper.mask_policy).masks_everything)


if __name__ == "__main__":
    unittest.main()

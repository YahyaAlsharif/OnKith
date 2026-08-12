from __future__ import annotations

import unittest

from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.pii_adapter import PIIMaskerAdapter
from integrated_pipeline.pipeline import BLOCKED_MARKER, FailClosedMasker
from integrated_pipeline.schemas import PIIResult
from integrated_pipeline.span_merger import (
    SpanCandidate,
    consolidate_predictions_to_words,
    merge_pii_spans,
)
from integrated_pipeline.structured_pii import StructuredPIIRecognizer


class PIIRegressionTests(unittest.TestCase):
    def test_structured_recognizers_and_numbered_placeholders_are_preserved(self):
        text = (
            "Email a@example.com, then b@example.com; "
            "my phone number is 555 123 4567."
        )
        structured = [
            span.as_candidate()
            for span in StructuredPIIRecognizer().find_spans(text)
        ]
        result = merge_pii_spans(text, [], structured)
        self.assertIn("[EMAIL_1]", result.masked_text)
        self.assertIn("[EMAIL_2]", result.masked_text)
        self.assertIn("[TELEPHONENUM_1]", result.masked_text)
        self.assertNotIn("a@example.com", result.masked_text)
        self.assertNotIn("555 123 4567", result.masked_text)

    def test_whole_word_consolidation_and_structured_priority_are_preserved(self):
        text = "Mohammed@example.com"
        consolidated = consolidate_predictions_to_words(
            text,
            offsets=[(0, 3), (3, 8)],
            labels=["B-GIVENNAME", "I-GIVENNAME"],
            confidences=[0.9, 0.8],
            word_ids=[0, 0],
        )
        self.assertEqual(consolidated.spans[0].text, "Mohammed")
        structured = SpanCandidate(
            entity_type="EMAIL",
            start_char=0,
            end_char=len(text),
            text=text,
            source="structured:email",
            confidence=1.0,
        )
        merged = merge_pii_spans(text, consolidated.spans, [structured])
        self.assertEqual(merged.masked_text, "[EMAIL_1]")
        self.assertEqual(merged.chosen_spans, [structured])

    def test_fail_closed_never_releases_raw_text(self):
        class Broken:
            def mask_text(self, text: str) -> PIIResult:
                raise RuntimeError(text)

        raw = "Mohammed secret 123456"
        result = FailClosedMasker(Broken()).mask_text(raw)
        self.assertFalse(result.safe_to_release)
        self.assertEqual(result.masked_text, BLOCKED_MARKER)
        self.assertNotIn(raw, result.error or "")


class ReleasedEntityTests(unittest.TestCase):
    """A place name is not a private detail, and hiding it costs the device.

    Asked the time in Saudi Arabia, the model was handed "what time is it
    right now in [CITY_1] Arabia" and said, correctly, that it did not
    know. Riyadh scores 0.996 and Jeddah scores 1.0, so no threshold
    reaches this. The classifier is not wrong; the policy was.
    """

    def test_the_shipped_policy_releases_places_and_nothing_else(self):
        released = set(PipelineConfig().never_masked_entities)
        self.assertEqual(released, {"CITY", "COUNTRY", "TIMEZONE"})
        # A street, a building number and a postcode are somebody's home.
        for still_private in ("STREET", "BUILDINGNUM", "ZIPCODE", "GIVENNAME"):
            with self.subTest(entity=still_private):
                self.assertNotIn(still_private, released)

    def test_a_released_label_never_becomes_a_placeholder(self):
        adapter = PIIMaskerAdapter(PipelineConfig())
        adapter.load()
        masked = adapter.mask_text("what time is it in Riyadh now").masked_text
        self.assertEqual(masked, "what time is it in Riyadh now")

    def test_an_address_is_still_taken_apart(self):
        # Only the city survives, and a city on its own identifies nobody.
        adapter = PIIMaskerAdapter(PipelineConfig())
        adapter.load()
        masked = adapter.mask_text(
            "I live at 42 King Fahd Road, Riyadh 11564"
        ).masked_text
        self.assertIn("Riyadh", masked)
        for gone in ("42", "King Fahd Road", "11564"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, masked)

    def test_the_policy_is_config_not_code(self):
        config = PipelineConfig()
        config.data["pii"]["never_mask"] = ["  city  "]
        self.assertEqual(config.never_masked_entities, ("CITY",))
        config.data["pii"]["never_mask"] = []
        self.assertEqual(config.never_masked_entities, ())


if __name__ == "__main__":
    unittest.main()


class TheDevicesOwnNameTest(unittest.TestCase):
    """TinyBERT read Onkith as a person and redacted it, every utterance.

    Seen on the Pi: the transcript said "[GIVENNAME_1], turn the light
    off" every single time. Correct of the classifier, as far as it
    knows, and wrong for this device: the name is how it was addressed,
    not somebody's personal detail.

    It also silently killed the wake word strip. That runs on the masked
    text, by design, so the raw transcript never reaches a network model,
    and by then the name was a placeholder and there was nothing to
    remove. The assistant got the placeholder and its reply was
    rehydrated on the way out, so the model saying "users like you,
    [GIVENNAME_1]" said the name out loud.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PipelineConfig()
        cls.masker = FailClosedMasker(PIIMaskerAdapter(cls.config))

    def masked(self, text: str) -> str:
        return self.masker.mask_text(text).masked_text

    def test_the_device_keeps_its_own_name(self) -> None:
        for name in self.config.wake.get("names", []):
            spoken = name[:1].upper() + name[1:]
            with self.subTest(name=spoken):
                self.assertIn(spoken, self.masked(f"{spoken}, turn the light off"))

    def test_a_person_is_still_masked_beside_it(self) -> None:
        # The whole point of the device, in one sentence: its own name
        # released, everything about the person kept back.
        out = self.masked("Onkith, call Turki on 0501234567")
        self.assertIn("Onkith", out)
        self.assertNotIn("Turki", out)
        self.assertNotIn("0501234567", out)

    def test_a_longer_name_containing_it_is_still_masked(self) -> None:
        # Only an exact whole-span match is released. A person whose name
        # merely contains the device's is still a person.
        self.assertNotIn("Alshuaibi", self.masked("my name is Onkith Alshuaibi"))

    def test_a_name_that_only_sounds_like_it_is_still_masked(self) -> None:
        # Ankith is a name people have. It is in the wake `never` list so
        # it cannot wake the device, and it is not in the release list, so
        # it stays masked like any other person.
        self.assertNotIn("Ankith", self.masked("someone named Ankith called me"))

    def test_the_release_list_follows_the_configured_names(self) -> None:
        # Written out beside the names rather than derived from them, a
        # renamed device would go back to masking itself.
        config = PipelineConfig()
        config.data["wake"] = {"names": ["jarvis"], "spellings": [], "never": []}
        self.assertIn("jarvis", config.never_masked_words)
        self.assertNotIn("onkith", config.never_masked_words)

    def test_the_wake_word_strip_works_again(self) -> None:
        from app.onkith_music import (
            WakeWords,
            configure_wake_words,
            without_wake_word,
            wake_words,
        )

        previous = wake_words()
        configure_wake_words(WakeWords.from_config(self.config.wake))
        try:
            handed_over = without_wake_word(self.masked("Onkith, what is your purpose?"))
            self.assertEqual(handed_over, "what is your purpose?")
        finally:
            configure_wake_words(previous)

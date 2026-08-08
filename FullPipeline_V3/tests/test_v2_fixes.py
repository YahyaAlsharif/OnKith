"""Regression tests for the V2 fix pass.

Grouped to match the reported issues: preview repetition during silence,
endpoint handling, worker ordering, number grammar, flicker beside unstable
text, the confirmed plaintext PII leak, canonical-vs-display separation, and
per-utterance label reset.

Every test here is a pure unit/mock test. Nothing loads Canary, and the PII
group uses the real recognizer classes but not the ONNX model, so the whole
file runs on x86 with no hardware.
"""

from __future__ import annotations

import threading
import unittest

from app.digit_normalizer import normalize_spoken_digits
from app.live_preview import BLUR_TOKEN, REDACTED_LABEL, LivePreview
from integrated_pipeline.pipeline import BLOCKED_MARKER, FailClosedMasker
from integrated_pipeline.supplemental_pii import (
    SupplementalPIIMasker,
    SupplementalPIIRecognizer,
)
from tests.test_live_preview import FakeMasker, build

DIGIT_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine",
)


# ---------------------------------------------------------------- repetition


class PreviewRepetitionTests(unittest.TestCase):
    def test_blank_segments_after_a_render_emit_nothing(self) -> None:
        preview, renders = build()
        preview.add_segment("hello everybody this is a test", unique_seconds=8.512)
        self.assertEqual(len(renders), 1)
        for _ in range(5):
            preview.add_segment("", unique_seconds=8.512)
        self.assertEqual(len(renders), 1, "silence must not repaint")
        self.assertEqual(preview.blank_segment_count, 5)

    def test_blank_segments_never_re_run_the_masker(self) -> None:
        masker = FakeMasker()
        preview, _ = build(masker)
        preview.add_segment("hello everybody this is a test", unique_seconds=8.512)
        calls = len(masker.calls)
        for _ in range(4):
            preview.add_segment("   ", unique_seconds=8.512)
        self.assertEqual(len(masker.calls), calls, "unchanged buffer re-masked")

    def test_identical_render_is_suppressed_and_counted(self) -> None:
        preview, renders = build()
        preview.add_segment("alpha bravo charlie delta echo", unique_seconds=8.512)
        # Force an identical repaint through the same buffer.
        preview.add_segment("", unique_seconds=8.512, force=True)
        before = preview.duplicate_suppressed_count
        with preview._lock:
            preview._last_signature = (
                preview._generation,
                renders[-1].line,
                renders[-1].blurred_words,
                renders[-1].degraded,
                renders[-1].redacted,
            )
        preview.add_segment("", unique_seconds=8.512, force=True)
        self.assertEqual(len(renders), 1)
        self.assertGreaterEqual(preview.duplicate_suppressed_count, before)

    def test_a_genuinely_changed_render_still_emits(self) -> None:
        preview, renders = build()
        preview.add_segment("alpha bravo charlie delta echo", unique_seconds=8.512)
        preview.add_segment("foxtrot golf hotel india juliet", unique_seconds=8.512)
        self.assertEqual(len(renders), 2)
        self.assertNotEqual(renders[0].line, renders[1].line)

    def test_a_failed_callback_does_not_permanently_suppress_a_retry(self) -> None:
        state = {"fail": True}
        seen: list[str] = []

        def on_render(render):
            if state["fail"]:
                raise RuntimeError("terminal not ready")
            seen.append(render.line)

        preview = LivePreview(FakeMasker(), on_render, minimum_cycle_seconds=0.0)
        preview.add_segment("alpha bravo charlie delta echo", unique_seconds=8.512)
        self.assertEqual(seen, [])
        # The refused draw must not be recorded as "already shown", otherwise
        # an identical retry would be suppressed forever.
        self.assertIsNone(preview._last_signature)

        # Once the display recovers, drawing resumes.
        state["fail"] = False
        preview.add_segment("foxtrot golf hotel india", unique_seconds=8.512)
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(preview._last_signature)

    def test_a_successful_render_records_its_signature(self) -> None:
        preview, _ = build()
        preview.add_segment("alpha bravo charlie delta echo", unique_seconds=8.512)
        self.assertIsNotNone(preview._last_signature)


# ------------------------------------------------------------------ endpoint


class EndpointTests(unittest.TestCase):
    def test_blank_transcript_does_not_release_the_unstable_tail(self) -> None:
        preview, renders = build()
        preview.add_segment(
            "my number is 05511 and the rest is coming", unique_seconds=8.512
        )
        blurred_before = renders[-1].blurred_words
        preview.add_segment("", unique_seconds=8.512)
        # No new render at all, and the buffered tail is untouched.
        self.assertEqual(len(renders), 1)
        self.assertGreater(blurred_before, 0)
        self.assertTrue(preview._segments[-1].unstable_words > 0)

    def test_blank_transcript_does_not_finalize_the_utterance(self) -> None:
        preview, _ = build()
        preview.add_segment("hello everybody this is a test", unique_seconds=8.512)
        generation = preview._generation
        preview.add_segment("", unique_seconds=8.512)
        self.assertEqual(preview._generation, generation)
        self.assertEqual(preview.finalize_count, 0)
        self.assertTrue(preview._segments)

    def test_finalize_clears_the_buffer_and_bumps_the_generation(self) -> None:
        preview, _ = build()
        preview.add_segment("hello everybody this is a test", unique_seconds=8.512)
        preview.finalize_utterance()
        self.assertEqual(preview.finalize_count, 1)
        self.assertEqual(preview._segments, [])

    def test_a_dangling_cue_stays_blurred(self) -> None:
        preview, renders = build()
        preview.add_segment("my phone number is", unique_seconds=8.512)
        self.assertIn(BLUR_TOKEN, renders[-1].line)
        self.assertNotIn("phone number is 0", renders[-1].line)


# --------------------------------------------------------------- concurrency


class WorkerOrderingTests(unittest.TestCase):
    def test_a_queued_segment_cannot_render_after_finalize(self) -> None:
        preview, renders = build()
        preview.add_segment("hello everybody this is a test", unique_seconds=8.512)
        generation = preview._generation
        preview.finalize_utterance()
        # Replay the queued work exactly as the worker lane would.
        preview.add_segment(
            "stale words from the closed utterance",
            unique_seconds=8.512,
            generation=generation,
        )
        self.assertEqual(len(renders), 1, "stale preview printed after final")
        self.assertEqual(preview.stale_event_count, 1)

    def test_a_closed_utterance_rejects_every_stale_generation(self) -> None:
        preview, renders = build()
        preview.add_segment("hello everybody this is a test", unique_seconds=8.512)
        preview.finalize_utterance()
        preview.finalize_utterance()
        for stale in range(preview._generation):
            preview.add_segment("late", unique_seconds=8.512, generation=stale)
        self.assertEqual(len(renders), 1)

    def test_an_identical_first_line_in_a_new_utterance_still_prints(self) -> None:
        preview, renders = build()
        text = "alpha bravo charlie delta echo"
        preview.add_segment(text, unique_seconds=8.512)
        preview.finalize_utterance()
        preview.add_segment(text, unique_seconds=8.512)
        self.assertEqual(len(renders), 2)
        self.assertEqual(renders[0].line, renders[1].line)

    def test_concurrent_finalize_and_submit_cannot_corrupt_state(self) -> None:
        preview, _ = build()
        errors: list[BaseException] = []

        def add() -> None:
            try:
                for index in range(40):
                    preview.add_segment(f"word{index} and more", unique_seconds=8.512)
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        def finalize() -> None:
            try:
                for _ in range(40):
                    preview.finalize_utterance()
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=add), threading.Thread(target=finalize)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertIsInstance(preview.audit(), dict)

    def test_close_is_idempotent(self) -> None:
        preview, _ = build()
        preview.add_segment("hello everybody this is a test", unique_seconds=8.512)
        preview.close()
        preview.close()
        self.assertEqual(preview._segments, [])
        self.assertTrue(preview._closed)


# ------------------------------------------------------------------- grammar


class NumberGrammarTests(unittest.TestCase):
    CASES = (
        ("twenty minutes", "20 minutes"),
        ("thirty seconds", "30 seconds"),
        ("I am forty years old", "I am 40 years old"),
        ("twenty five", "25"),
        ("thirty two", "32"),
        ("ninety nine", "99"),
        ("zero five five one one", "05511"),
        ("the code is four seven two nine", "the code is 4729"),
        ("thirty twenty", "30 20"),
        ("my code is thirty zero one two", "my code is 30012"),
        ("three thirty pm", "3:30 pm"),
        ("meeting at one thirty", "meeting at 1:30"),
        ("one of the doctors said two things", "one of the doctors said two things"),
        ("I have one question", "I have one question"),
        ("double four", "44"),
        ("triple seven", "777"),
        ("double oh seven", "007"),
        ("double click", "double click"),
        ("oh really", "oh really"),
        ("five oh five", "505"),
        ("room double four seven", "room 447"),
    )

    def test_every_specified_case(self) -> None:
        for source, expected in self.CASES:
            with self.subTest(source=source):
                self.assertEqual(normalize_spoken_digits(source), expected)

    def test_no_sequence_is_left_partly_spelled(self) -> None:
        for source, _ in self.CASES:
            with self.subTest(source=source):
                rendered = normalize_spoken_digits(source)
                if any(character.isdigit() for character in rendered):
                    leftover = [
                        word
                        for word in rendered.lower().split()
                        if word.strip(".,:") in DIGIT_WORDS
                    ]
                    self.assertEqual(leftover, [], f"partly spelled: {rendered!r}")

    def test_idempotent(self) -> None:
        for source, _ in self.CASES:
            with self.subTest(source=source):
                once = normalize_spoken_digits(source)
                self.assertEqual(normalize_spoken_digits(once), once)

    def test_placeholders_stay_opaque(self) -> None:
        text = "my phone number is [TELEPHONENUM_1] thanks"
        self.assertEqual(normalize_spoken_digits(text), text)


# ------------------------------------------------------------------- flicker


class FlickerTests(unittest.TestCase):
    def test_a_number_touching_the_blur_token_stays_words(self) -> None:
        text = f"zero five five {BLUR_TOKEN}"
        self.assertEqual(normalize_spoken_digits(text), text)

    def test_the_same_number_converts_once_the_blur_is_gone(self) -> None:
        self.assertEqual(
            normalize_spoken_digits("zero five five one one"), "05511"
        )

    def test_a_number_touching_the_redaction_marker_stays_words(self) -> None:
        text = f"four seven two {REDACTED_LABEL}"
        self.assertEqual(normalize_spoken_digits(text), text)

    def test_a_number_next_to_an_ordinary_placeholder_still_converts(self) -> None:
        self.assertEqual(
            normalize_spoken_digits("[GIVENNAME_1] four seven two nine"),
            "[GIVENNAME_1] 4729",
        )

    def test_normalisation_never_moves_the_blur_decision(self) -> None:
        text = "my account number is zero one two three four five six seven"
        preview_a, renders_a = build(FakeMasker(), blur_cap_words=5)
        preview_a.add_segment(text, unique_seconds=8.512)
        preview_b, renders_b = build(FakeMasker(), blur_cap_words=5)
        preview_b.add_segment(text, unique_seconds=8.512)
        self.assertEqual(renders_a[-1].blurred_words, renders_b[-1].blurred_words)
        self.assertEqual(renders_a[-1].redacted, renders_b[-1].redacted)


# ----------------------------------------------------------------------- PII


class StubMasker:
    """Stands in for TinyBERT: finds nothing, releases everything."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def mask_text(self, text: str):
        self.calls.append(text)

        class Result:
            pass

        result = Result()
        result.original_text = text
        result.masked_text = text
        result.latency_seconds = 0.0
        result.entities = []
        result.safe_to_release = True
        result.privacy_status = "masked"
        result.error = None
        result.raw_model_output = {}
        return result


def _plaintext_digits(text: str) -> list[str]:
    return [word for word in text.lower().split() if word.strip(".,") in DIGIT_WORDS]


class SupplementalPIITests(unittest.TestCase):
    def setUp(self) -> None:
        self.masker = SupplementalPIIMasker(StubMasker())

    def test_noisy_filler_phone_number_is_masked(self) -> None:
        text = "phone number is a zero five five a six seven three four four two one"
        masked = self.masker.mask_text(text).masked_text
        self.assertEqual(_plaintext_digits(masked), [])
        self.assertIn("[TELEPHONENUM_1]", masked)

    def test_bank_id_is_masked(self) -> None:
        text = "bank id is four five three three two one one one six five"
        masked = self.masker.mask_text(text).masked_text
        self.assertEqual(_plaintext_digits(masked), [])
        self.assertIn("[IDCARDNUM_1]", masked)

    def test_account_id_is_masked(self) -> None:
        text = "account id is nine eight seven six five four three two"
        masked = self.masker.mask_text(text).masked_text
        self.assertEqual(_plaintext_digits(masked), [])
        self.assertIn("[ACCOUNTNUM_1]", masked)

    def test_a_long_run_is_protected_without_any_cue(self) -> None:
        text = "just saying four five three three two one one one six five now"
        masked = self.masker.mask_text(text).masked_text
        self.assertEqual(_plaintext_digits(masked), [])

    def test_arbitrary_words_are_not_skipped_inside_a_sequence(self) -> None:
        # "banana" is not an allowed filler, so the sequence stops there and
        # the remaining digits are too few for a phone number.
        text = "phone number is zero five banana five one one nine"
        masked = self.masker.mask_text(text).masked_text
        self.assertNotIn("[TELEPHONENUM_1]", masked)

    def test_filler_budget_is_bounded(self) -> None:
        recognizer = SupplementalPIIRecognizer(max_fillers=2)
        text = "phone number is a uh um er zero five five one one nine five"
        spans = recognizer.find_spans(text)
        phone = [span for span in spans if span.entity_type == "TELEPHONENUM"]
        self.assertEqual(phone, [], "filler budget was not enforced")

    def test_normal_quantities_are_not_over_masked(self) -> None:
        for text in (
            "I have two questions",
            "room two zero five",
            "the meeting starts in five minutes",
            "one of the doctors said two things",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.masker.mask_text(text).masked_text, text)

    def test_a_correct_specific_label_is_never_downgraded(self) -> None:
        # A cued phone number must stay TELEPHONENUM; the cue-less fallback
        # must not relabel it as the vaguer NUMERICID.
        text = "phone number is zero five five one one nine five seven one nine"
        masked = self.masker.mask_text(text).masked_text
        self.assertIn("[TELEPHONENUM_1]", masked)
        self.assertNotIn("NUMERICID", masked)

    def test_masker_remains_fail_closed_on_exception(self) -> None:
        class Boom:
            def mask_text(self, text: str):
                raise RuntimeError("model gone")

        result = FailClosedMasker(SupplementalPIIMasker(Boom())).mask_text(
            "phone number is zero five five one one nine five"
        )
        self.assertFalse(result.safe_to_release)
        self.assertEqual(result.masked_text, BLOCKED_MARKER)
        self.assertEqual(_plaintext_digits(result.error or ""), [])

    def test_a_blocked_inner_result_is_passed_through_untouched(self) -> None:
        class Blocked:
            def mask_text(self, text: str):
                class Result:
                    masked_text = BLOCKED_MARKER
                    entities: list = []
                    safe_to_release = False
                    raw_model_output: dict = {}
                    error = None

                return Result()

        result = SupplementalPIIMasker(Blocked()).mask_text("phone number is one")
        self.assertEqual(result.masked_text, BLOCKED_MARKER)


# ------------------------------------------------------- canonical / display


class CanonicalDisplayTests(unittest.TestCase):
    def test_normalisation_never_reaches_the_masker(self) -> None:
        masker = FakeMasker()
        preview, _ = build(masker)
        preview.add_segment(
            "my account number is four seven two nine", unique_seconds=8.512
        )
        self.assertIn("four seven two nine", masker.calls[-1])
        self.assertNotIn("4729", masker.calls[-1])

    def test_display_and_canonical_differ_only_in_rendering(self) -> None:
        canonical = "the code is thirty zero one two"
        display = normalize_spoken_digits(canonical)
        self.assertEqual(canonical, "the code is thirty zero one two")
        self.assertEqual(display, "the code is 30012")

    def test_audit_output_carries_no_transcript_text(self) -> None:
        preview, _ = build()
        preview.add_segment(
            "my phone number is 05511 95719 thanks", unique_seconds=8.512
        )
        blob = repr(preview.audit())
        for leak in ("05511", "95719", "phone", "rayyan"):
            self.assertNotIn(leak, blob)


# -------------------------------------------------------------------- labels


class RegistryResetTests(unittest.TestCase):
    def test_registry_resets_at_utterance_finalization(self) -> None:
        preview, _ = build()
        preview.add_segment("rayyan is here today and talking", unique_seconds=8.512)
        self.assertGreater(preview._registry.size, 0)
        preview.finalize_utterance()
        self.assertEqual(preview._registry.size, 0)

    def test_a_new_utterance_starts_numbering_at_one(self) -> None:
        preview, renders = build()
        preview.add_segment("rayyan is here today and talking", unique_seconds=8.512)
        preview.finalize_utterance()
        preview.add_segment("rayyan is here today and talking", unique_seconds=8.512)
        self.assertIn("[GIVENNAME_1]", renders[-1].line)
        self.assertNotIn("[GIVENNAME_2]", renders[-1].line)

    def test_a_late_event_cannot_repopulate_a_cleared_registry(self) -> None:
        preview, _ = build()
        preview.add_segment("rayyan is here today and talking", unique_seconds=8.512)
        generation = preview._generation
        preview.finalize_utterance()
        preview.add_segment(
            "rayyan again from the old utterance",
            unique_seconds=8.512,
            generation=generation,
        )
        self.assertEqual(preview._registry.size, 0)

    def test_the_salt_is_regenerated_so_labels_are_not_linkable(self) -> None:
        preview, _ = build()
        preview.add_segment("rayyan is here today and talking", unique_seconds=8.512)
        before = preview._registry._salt
        preview.finalize_utterance()
        self.assertNotEqual(preview._registry._salt, before)


if __name__ == "__main__":
    unittest.main()

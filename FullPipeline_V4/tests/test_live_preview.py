"""Step 2: rolling buffer, whole-buffer re-mask, repaint with a blurred tail."""

from __future__ import annotations

import unittest

import math

from app.live_preview import (
    ADJACENT_OVERLAP_SECONDS,
    BLUR_CAP_WORDS,
    BLUR_TOKEN,
    CUE_LOOKBACK_WORDS,
    MINIMUM_WINDOW_SECONDS,
    REDACTED_LABEL,
    LivePreview,
    PreviewRender,
    _LabelRegistry,
)


class FakeMasker:
    """Masks a fixed vocabulary, and only when it sees a complete value.

    Mirrors the real failure this layer exists to prevent: a phone number is
    recognised only once every digit group is present in the same string.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def mask_text(self, text: str):
        self.calls.append(text)
        entities = []
        for surface, entity_type in (
            ("05511 95719", "TELEPHONENUM"),
            ("abdulla.alhulays@gmail.com", "EMAIL"),
            ("rayyan", "GIVENNAME"),
        ):
            start = text.find(surface)
            while start != -1:
                entities.append(
                    {
                        "start": start,
                        "end": start + len(surface),
                        "entity_type": entity_type,
                        "text": surface,
                        "placeholder": f"[{entity_type}_1]",
                    }
                )
                start = text.find(surface, start + len(surface))
        entities.sort(key=lambda item: item["start"])
        masked = []
        cursor = 0
        for entity in entities:
            masked.append(text[cursor : entity["start"]])
            masked.append(entity["placeholder"])
            cursor = entity["end"]
        masked.append(text[cursor:])

        class Result:
            pass

        result = Result()
        result.masked_text = "".join(masked)
        result.entities = entities
        result.safe_to_release = True
        result.error = None
        return result


class RaisingMasker:
    def mask_text(self, text: str):
        raise RuntimeError("model unavailable")


class BlockedMasker:
    def mask_text(self, text: str):
        class Result:
            masked_text = "[BLOCKED: PII MASKING FAILED]"
            entities: list = []
            safe_to_release = False
            error = "masking operation failed"

        return Result()


def build(masker=None, **kwargs):
    renders: list[PreviewRender] = []
    preview = LivePreview(
        masker or FakeMasker(),
        renders.append,
        minimum_cycle_seconds=0.0,
        **kwargs,
    )
    return preview, renders


class BufferAndRepaintTests(unittest.TestCase):
    def test_tail_is_blurred_and_head_is_shown(self) -> None:
        preview, renders = build()
        render = preview.add_segment(
            "hello there this is a normal sentence about nothing",
            unique_seconds=8.512,
        )
        self.assertIsNotNone(render)
        self.assertIn(BLUR_TOKEN, render.line)
        self.assertTrue(render.line.startswith("hello there"))
        self.assertGreater(render.blurred_words, 0)
        # The blurred words must not appear as text anywhere in the line.
        self.assertNotIn("nothing", render.line)
        self.assertEqual(len(renders), 1)

    def test_masker_always_sees_the_whole_buffer(self) -> None:
        masker = FakeMasker()
        preview, _ = build(masker)
        preview.add_segment("my number is 05511", unique_seconds=8.512)
        preview.add_segment("95719 call me back", unique_seconds=8.512)
        # The second call must contain both halves, not just the new segment.
        self.assertIn("05511 95719", masker.calls[-1])

    def test_phone_split_across_a_boundary_never_prints_raw_digits(self) -> None:
        preview, renders = build()
        preview.add_segment("my number is 05511", unique_seconds=8.512)
        preview.add_segment("95719 call me back tomorrow", unique_seconds=8.512)
        for render in renders:
            self.assertNotIn("05511", render.line)
            self.assertNotIn("95719", render.line)
        # Once both halves are buffered the masker produces one phone label.
        self.assertIn("[TELEPHONENUM_1]", renders[-1].line)

    def test_email_split_across_a_boundary_never_prints_raw_address(self) -> None:
        # The 0.704 s overlap means segment two re-transcribes the words that
        # ended segment one, so the address arrives whole in the second segment
        # even though it ended the first. See the note in _raw_buffer.
        preview, renders = build()
        preview.add_segment("write to abdulla.alhulays@gmail.com", unique_seconds=8.512)
        preview.add_segment(
            "abdulla.alhulays@gmail.com any time you like", unique_seconds=8.512
        )
        for render in renders:
            self.assertNotIn("alhulays", render.line)
            self.assertNotIn("gmail", render.line)
        self.assertIn("[EMAIL_1]", renders[-1].line)

    def test_overlap_words_are_not_printed_twice(self) -> None:
        # Plain concatenation would render "05511 95719 05511 95719".
        preview, renders = build()
        preview.add_segment(
            "please write down my number is 05511 95719", unique_seconds=8.512
        )
        preview.add_segment(
            "05511 95719 and call me back tomorrow", unique_seconds=8.512
        )
        self.assertEqual(renders[-1].line.count("[TELEPHONENUM_1]"), 1)

    def test_repaint_replaces_rather_than_appends(self) -> None:
        masker = FakeMasker()
        preview, renders = build(masker)
        preview.add_segment("my name is rayyan", unique_seconds=8.512)
        first = renders[-1].line
        preview.add_segment("and i live nearby", unique_seconds=8.512)
        second = renders[-1].line
        # The second line is a fresh render of the whole buffer, so the earlier
        # content reappears in it rather than being appended to by the caller.
        self.assertIn("[GIVENNAME_1]", second)
        self.assertNotEqual(first, second)
        self.assertEqual(masker.calls[-1].count("rayyan"), 1)

    def test_entity_straddling_the_cut_is_blurred_whole(self) -> None:
        preview, renders = build()
        # The phone number sits at the very end, inside the unstable tail.
        preview.add_segment(
            "please write this down my number is 05511 95719",
            unique_seconds=8.512,
        )
        line = renders[-1].line
        self.assertNotIn("05511", line)
        self.assertNotIn("95719", line)
        self.assertIn(BLUR_TOKEN, line)

    def test_buffer_stays_bounded_across_a_long_session(self) -> None:
        # Eviction is stability-based, not a time slice, so the buffer can sit
        # a little above the window while a region settles. What matters is
        # that it stops growing.
        preview, _ = build(window_seconds=MINIMUM_WINDOW_SECONDS)
        for index in range(20):
            preview.add_segment(
                f"we talked about topic number {index} for a little while",
                unique_seconds=8.512,
            )
        total_added = 20 * 8.512
        self.assertGreater(preview.evicted_segment_count, 0)
        self.assertLess(preview.buffered_seconds, total_added / 2)
        self.assertGreaterEqual(len(preview._segments), 2)

    def test_window_floor_is_enforced(self) -> None:
        preview, _ = build(window_seconds=1.0)
        self.assertGreaterEqual(preview.window_seconds, MINIMUM_WINDOW_SECONDS)

    def test_unstable_word_count_tracks_the_overlap_duration(self) -> None:
        preview, _ = build()
        # 24 words over 8.512 s is a fast talker, so more words fall inside the
        # 0.704 s overlap than for a slow one.
        fast = preview._unstable_words_for(" ".join(["word"] * 24), 8.512)
        slow = preview._unstable_words_for(" ".join(["word"] * 8), 8.512)
        self.assertGreater(fast, slow)
        self.assertGreaterEqual(slow, 1)
        self.assertLess(ADJACENT_OVERLAP_SECONDS, 1.0)


class FailureIsolationTests(unittest.TestCase):
    def test_masker_exception_blurs_everything(self) -> None:
        preview, renders = build(RaisingMasker())
        render = preview.add_segment("my number is 05511 95719", unique_seconds=8.512)
        self.assertTrue(render.degraded)
        self.assertEqual(render.line, BLUR_TOKEN)
        self.assertNotIn("05511", renders[-1].line)

    def test_blocked_mask_blurs_everything(self) -> None:
        preview, renders = build(BlockedMasker())
        render = preview.add_segment("my name is rayyan", unique_seconds=8.512)
        # An empty entity list from a blocked result must not read as clean.
        self.assertTrue(render.degraded)
        self.assertEqual(render.line, BLUR_TOKEN)
        self.assertNotIn("rayyan", renders[-1].line)

    def test_render_callback_exception_does_not_propagate(self) -> None:
        def explode(render):
            raise RuntimeError("terminal is gone")

        preview = LivePreview(FakeMasker(), explode, minimum_cycle_seconds=0.0)
        preview.add_segment("hello there friend", unique_seconds=8.512)  # must not raise

    def test_close_wipes_the_buffer_and_stops_rendering(self) -> None:
        preview, renders = build()
        preview.add_segment("my name is rayyan", unique_seconds=8.512)
        preview.close()
        self.assertEqual(preview._segments, [])
        self.assertIsNone(preview.add_segment("more speech", unique_seconds=8.512))
        self.assertEqual(len(renders), 1)

    def test_finalize_clears_the_buffer(self) -> None:
        preview, _ = build()
        preview.add_segment("my name is rayyan", unique_seconds=8.512)
        preview.finalize_utterance()
        self.assertEqual(preview.buffered_seconds, 0.0)

    def test_audit_never_exposes_buffered_text(self) -> None:
        preview, _ = build()
        preview.add_segment("my name is rayyan", unique_seconds=8.512)
        blob = repr(preview.audit())
        self.assertNotIn("rayyan", blob)

    def test_minimum_cycle_interval_skips_repeated_repaints(self) -> None:
        renders: list[PreviewRender] = []
        clock = iter([0.0, 0.1, 0.2, 10.0])
        preview = LivePreview(
            FakeMasker(),
            renders.append,
            minimum_cycle_seconds=1.0,
            clock=lambda: next(clock),
        )
        self.assertIsNotNone(preview.add_segment("first part here", unique_seconds=8.512))
        self.assertIsNone(preview.add_segment("second part here", unique_seconds=8.512))
        self.assertIsNone(preview.add_segment("third part here", unique_seconds=8.512))
        self.assertIsNotNone(preview.add_segment("fourth part here", unique_seconds=8.512))
        self.assertEqual(preview.skipped_cycle_count, 2)


class SelectiveBlurTests(unittest.TestCase):
    """Step 3: blurring becomes selective rather than always-the-tail."""

    def test_clean_speech_blurs_only_the_overlap_words(self) -> None:
        preview, renders = build()
        text = "we talked about the weather and then walked to the shop together"
        render = preview.add_segment(text, unique_seconds=8.512)
        words = text.split()
        # Everything except the trailing overlap words is printed as-is.
        self.assertTrue(render.line.startswith(" ".join(words[:6])))
        self.assertLessEqual(render.blurred_words, 4)
        self.assertGreater(render.blurred_words, 0)

    def test_entity_touching_the_end_blurs_from_its_start(self) -> None:
        preview, renders = build()
        # The name is the final word, so it may still be growing.
        preview.add_segment("please say hello to rayyan", unique_seconds=8.512)
        line = renders[-1].line
        self.assertNotIn("rayyan", line)
        self.assertNotIn("[GIVENNAME_1]", line)
        self.assertTrue(line.startswith("please say hello"))

    def test_entity_well_before_the_end_is_labelled_not_blurred(self) -> None:
        preview, renders = build()
        preview.add_segment(
            "rayyan went to the market early this morning to buy some fresh bread",
            unique_seconds=8.512,
        )
        line = renders[-1].line
        # The name is settled, so it shows as a label rather than a blur.
        self.assertIn("[GIVENNAME_1]", line)
        self.assertNotIn("rayyan", line)

    def test_phone_cue_with_no_digits_yet_blurs_from_the_cue(self) -> None:
        preview, renders = build()
        preview.add_segment(
            "hold on a second let me find it my phone number is", unique_seconds=8.512
        )
        line = renders[-1].line
        self.assertNotIn("phone number", line)
        self.assertTrue(line.startswith("hold on a second"))
        self.assertIn(BLUR_TOKEN, line)

    def test_email_cue_with_no_value_yet_blurs_from_the_cue(self) -> None:
        preview, renders = build()
        preview.add_segment(
            "sure thing i will send it over my email address is", unique_seconds=8.512
        )
        line = renders[-1].line
        self.assertNotIn("email address", line)
        self.assertTrue(line.startswith("sure thing"))

    def test_account_and_zip_and_id_cues_are_watched(self) -> None:
        for phrase in (
            "let me check that my account number is",
            "the delivery postal code is",
            "for the form my id number is",
        ):
            with self.subTest(phrase=phrase):
                preview, renders = build()
                preview.add_segment(phrase, unique_seconds=8.512)
                self.assertIn(BLUR_TOKEN, renders[-1].line)

    def test_a_fulfilled_cue_does_not_blur(self) -> None:
        preview, renders = build()
        preview.add_segment(
            "my number is 05511 95719 and that is the best way to reach me quickly",
            unique_seconds=8.512,
        )
        line = renders[-1].line
        # The value arrived and was labelled, so the cue is satisfied.
        self.assertIn("[TELEPHONENUM_1]", line)
        self.assertIn("my number is", line)

    def test_an_old_unfulfilled_cue_does_not_grey_out_the_buffer(self) -> None:
        preview, renders = build()
        preview.add_segment(
            "my phone number is well actually never mind about that let us talk "
            "instead about the weather and the plans we made for the weekend ahead",
            unique_seconds=8.512,
        )
        line = renders[-1].line
        # The speaker moved on; blurring from that cue would grey out everything.
        self.assertIn("my phone number is", line)

    def test_earlier_candidate_wins_over_the_overlap_floor(self) -> None:
        preview, _ = build()
        raw = "hi there my phone number is"
        overlap = preview._overlap_start(raw, 1)
        cue = preview._unfulfilled_cue_start(raw, [])
        combined = preview._risky_start(raw, 1, [])
        self.assertIsNotNone(cue)
        # The cue starts well before the overlap floor, so it must win.
        self.assertLess(cue, overlap)
        self.assertEqual(combined, cue)

    def test_entity_touching_end_wins_over_the_overlap_floor(self) -> None:
        preview, _ = build()
        raw = "please pass this message along to rayyan"
        entities = [
            {"start": raw.index("rayyan"), "end": len(raw), "entity_type": "GIVENNAME"}
        ]
        overlap = preview._overlap_start(raw, 1)
        entity = preview._entity_touching_end(raw, entities)
        combined = preview._risky_start(raw, 1, entities)
        self.assertIsNotNone(entity)
        self.assertLessEqual(entity, overlap)
        self.assertEqual(combined, entity)

    def test_malformed_entity_span_blurs_everything(self) -> None:
        preview, _ = build()
        raw = "some ordinary words here"
        self.assertEqual(preview._entity_touching_end(raw, [{"start": "x"}]), 0)
        self.assertEqual(preview._unfulfilled_cue_start(raw, [{"start": None}]), 0)

    def test_selective_blur_still_never_prints_a_split_phone(self) -> None:
        preview, renders = build()
        preview.add_segment("my number is 05511", unique_seconds=8.512)
        preview.add_segment("05511 95719 thanks a lot for that", unique_seconds=8.512)
        for render in renders:
            self.assertNotIn("05511", render.line)
            self.assertNotIn("95719", render.line)
        self.assertIn("[TELEPHONENUM_1]", renders[-1].line)


class RoundingTests(unittest.TestCase):
    def test_overlap_word_count_rounds_up_never_to_nearest(self) -> None:
        preview, _ = build()
        # Pick a rate where the exact value has a small fractional part: a
        # round-to-nearest would drop it to the lower integer and print a word
        # the next segment may still change.
        seconds = 8.512
        for word_count in range(1, 60):
            exact = ADJACENT_OVERLAP_SECONDS * (word_count / seconds)
            expected = max(1, min(word_count, math.ceil(exact)))
            actual = preview._unstable_words_for(
                " ".join(["w"] * word_count), seconds
            )
            self.assertEqual(actual, expected, f"word_count={word_count}")
            self.assertGreaterEqual(actual, exact)

    def test_blur_cap_and_cue_lookback_are_separate_constants(self) -> None:
        preview, _ = build(blur_cap_words=3, cue_lookback_words=40)
        self.assertEqual(preview.blur_cap_words, 3)
        self.assertEqual(preview.cue_lookback_words, 40)
        self.assertIsInstance(BLUR_CAP_WORDS, int)
        self.assertIsInstance(CUE_LOOKBACK_WORDS, int)


class RegistryTests(unittest.TestCase):
    def test_same_surface_form_keeps_its_label(self) -> None:
        registry = _LabelRegistry()
        first = registry.label_for("GIVENNAME", "rayyan")
        second = registry.label_for("GIVENNAME", "Rayyan ")
        self.assertEqual(first, second)
        self.assertEqual(first, "[GIVENNAME_1]")

    def test_different_surfaces_get_different_numbers(self) -> None:
        registry = _LabelRegistry()
        self.assertEqual(registry.label_for("GIVENNAME", "rayyan"), "[GIVENNAME_1]")
        self.assertEqual(registry.label_for("GIVENNAME", "abdulla"), "[GIVENNAME_2]")
        self.assertEqual(registry.label_for("EMAIL", "gmail dot com"), "[EMAIL_1]")

    def test_registry_never_stores_the_surface_form(self) -> None:
        registry = _LabelRegistry()
        registry.label_for("GIVENNAME", "rayyan")
        blob = repr(vars(registry))
        self.assertNotIn("rayyan", blob)

    def test_clear_discards_salt_and_mappings(self) -> None:
        registry = _LabelRegistry()
        registry.label_for("GIVENNAME", "rayyan")
        self.assertEqual(registry.size, 1)
        registry.clear()
        self.assertEqual(registry.size, 0)

    def test_labels_are_stable_across_repaints(self) -> None:
        preview, renders = build()
        preview.add_segment("rayyan came by the office earlier today", unique_seconds=8.512)
        preview.add_segment("and rayyan said he would return soon enough", unique_seconds=8.512)
        for render in renders:
            if "GIVENNAME" in render.line:
                self.assertIn("[GIVENNAME_1]", render.line)
                self.assertNotIn("[GIVENNAME_2]", render.line)

    def test_labels_are_stable_across_eviction(self) -> None:
        preview, renders = build(window_seconds=MINIMUM_WINDOW_SECONDS)
        # Push enough segments that the earliest ones are evicted, then bring
        # the same person back. The label must not renumber.
        preview.add_segment("rayyan opened the shop this morning", unique_seconds=8.512)
        for index in range(5):
            preview.add_segment(
                f"then we discussed topic number {index} for a while", unique_seconds=8.512
            )
        self.assertGreater(preview.evicted_segment_count, 0)
        preview.add_segment("rayyan closed the shop again tonight", unique_seconds=8.512)
        final = renders[-1].line
        if "GIVENNAME" in final:
            self.assertIn("[GIVENNAME_1]", final)


class EvictionTests(unittest.TestCase):
    def test_unstable_region_is_kept_even_when_older_than_the_window(self) -> None:
        preview, _ = build(window_seconds=MINIMUM_WINDOW_SECONDS)
        # A cue with no value keeps the risky start at the very beginning, so
        # nothing after it is clean and nothing may be evicted.
        preview.add_segment("my phone number is", unique_seconds=8.512)
        for index in range(6):
            preview.add_segment(f"still looking {index}", unique_seconds=8.512)
        self.assertEqual(preview.evicted_segment_count, 0)
        self.assertGreater(preview.buffered_seconds, MINIMUM_WINDOW_SECONDS)

    def test_a_region_needs_two_identical_repaints_before_eviction(self) -> None:
        preview, _ = build(window_seconds=MINIMUM_WINDOW_SECONDS)
        preview.add_segment("the first thing we discussed was travel", unique_seconds=8.512)
        preview.add_segment("the second thing we discussed was food", unique_seconds=8.512)
        # One repaint has happened for the first region; not yet evictable.
        self.assertEqual(preview.evicted_segment_count, 0)
        for index in range(4):
            preview.add_segment(f"more talk number {index} here", unique_seconds=8.512)
        self.assertGreater(preview.evicted_segment_count, 0)

    def test_eviction_never_drops_the_only_segment(self) -> None:
        preview, _ = build(window_seconds=MINIMUM_WINDOW_SECONDS)
        preview.add_segment("a single very long stretch of ordinary speech", unique_seconds=90.0)
        self.assertEqual(len(preview._segments), 1)


class BlurCapTests(unittest.TestCase):
    def test_long_digit_run_is_redacted_rather_than_stalling(self) -> None:
        preview, renders = build(blur_cap_words=5)
        # A cue with no value blurs from the cue, and the run keeps growing as
        # digits arrive. Past the cap it must print something masked.
        preview.add_segment(
            "my account number is zero one two three four five six seven eight nine",
            unique_seconds=8.512,
        )
        render = renders[-1]
        self.assertTrue(render.redacted)
        self.assertIn(REDACTED_LABEL, render.line)
        self.assertNotIn(BLUR_TOKEN, render.line)
        # The digits themselves never appear.
        for digit_word in ("zero", "seven", "nine"):
            self.assertNotIn(digit_word, render.line)

    def test_short_blur_run_uses_the_placeholder_not_redaction(self) -> None:
        preview, renders = build(blur_cap_words=20)
        preview.add_segment(
            "we walked along the river and talked about the weather", unique_seconds=8.512
        )
        render = renders[-1]
        self.assertFalse(render.redacted)
        self.assertIn(BLUR_TOKEN, render.line)
        self.assertNotIn(REDACTED_LABEL, render.line)

    def test_redaction_marker_is_distinct_from_entity_labels(self) -> None:
        # A real label is [TYPE_N]; the redaction marker must not be mistaken
        # for a detection.
        self.assertTrue(REDACTED_LABEL.startswith("[["))
        self.assertNotRegex(REDACTED_LABEL, r"^\[[A-Z]+_\d+\]$")


if __name__ == "__main__":
    unittest.main()

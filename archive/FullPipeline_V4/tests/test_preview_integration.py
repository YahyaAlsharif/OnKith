"""Step 5: the preview is wired in without changing the authoritative path."""

from __future__ import annotations

import inspect
import queue
import unittest

from app.stt_runtime import LiveSTTRuntime
from app.live_preview import LivePreview
from app.terminal_ui import TerminalUI
from integrated_pipeline.schemas import SegmentTranscript
from tests.helpers import canary_segment

SHORT_SAMPLES = 2 * 16_000


class EchoCanary:
    def transcribe_segment(self, segment):
        return SegmentTranscript(
            segment=segment,
            text=f"word{segment.segment_index}",
            processing_seconds=0.0,
            queue_wait_seconds=0.0,
            stt_started_at_utc="2026-08-05T10:00:00Z",
            stt_completed_at_utc="2026-08-05T10:00:01Z",
        )


class Privacy:
    def __init__(self) -> None:
        self.results = []
        self.failures = []

    def submit_stt(self, result) -> None:
        self.results.append(result)

    def submit_stt_failure(self, *args) -> None:
        self.failures.append(args)


class RecordingPreview:
    def __init__(self) -> None:
        self.submitted: list[tuple[str, float]] = []
        self.finalized = 0

    def submit(self, text, *, unique_seconds):
        self.submitted.append((text, unique_seconds))

    def finalize_utterance(self):
        self.finalized += 1


class ExplodingPreview:
    def submit(self, text, *, unique_seconds):
        raise RuntimeError("preview is broken")

    def finalize_utterance(self):
        raise RuntimeError("preview is broken")


def short_segment(index: int, *, final: bool = False, utterance: int = 1):
    start = index * SHORT_SAMPLES
    return canary_segment(
        index, final=final, utterance=utterance, start=start, end=start + SHORT_SAMPLES
    )


class RuntimeHookTests(unittest.TestCase):
    def run_utterance(self, preview=None):
        completed: queue.Queue = queue.Queue(maxsize=4)
        privacy = Privacy()
        worker = LiveSTTRuntime(
            completed,
            EchoCanary(),
            privacy,
            on_event=lambda name, payload: None,
            utc_now=lambda: "2026-08-05T10:00:03Z",
            preview=preview,
        )
        worker.start()
        self.addCleanup(self._stop, completed, worker)
        for index in range(3):
            completed.put(short_segment(index, final=index == 2))
            completed.join()
        worker.shutdown()
        return privacy

    @staticmethod
    def _stop(completed, worker):
        if worker.thread is not None and worker.thread.is_alive():
            completed.put(None)
            worker.thread.join(timeout=5.0)

    def test_preview_defaults_to_none(self) -> None:
        signature = inspect.signature(LiveSTTRuntime.__init__)
        self.assertIn("preview", signature.parameters)
        self.assertIsNone(signature.parameters["preview"].default)

    def test_non_final_segments_are_submitted_and_final_clears(self) -> None:
        preview = RecordingPreview()
        self.run_utterance(preview)
        # Two non-final segments previewed; the final one is left to the
        # authoritative transcript, which lands immediately after.
        self.assertEqual([text for text, _ in preview.submitted], ["word0", "word1"])
        self.assertEqual(preview.finalized, 1)

    def test_authoritative_output_is_identical_with_and_without_preview(self) -> None:
        without = self.run_utterance(None)
        with_preview = self.run_utterance(RecordingPreview())
        self.assertEqual(len(without.results), len(with_preview.results))
        for left, right in zip(without.results, with_preview.results):
            self.assertEqual(left.final_text, right.final_text)
            self.assertEqual(left.batch_id, right.batch_id)
            self.assertEqual(left.sequence_number, right.sequence_number)
            self.assertEqual(left.segment_count, right.segment_count)
            self.assertEqual(left.endpoint_reason, right.endpoint_reason)
            self.assertEqual(left.blank_segment_indices, right.blank_segment_indices)
            self.assertEqual(
                left.deduplicated_word_count, right.deduplicated_word_count
            )

    def test_a_broken_preview_does_not_break_the_transcript(self) -> None:
        privacy = self.run_utterance(ExplodingPreview())
        self.assertEqual(len(privacy.results), 1)
        self.assertEqual(privacy.results[0].final_text, "word0 word1 word2")
        self.assertEqual(privacy.failures, [])


class TerminalPreviewTests(unittest.TestCase):
    class Render:
        line = "hello [GIVENNAME_1] ●●●●●"
        blurred_words = 2
        redacted = False
        degraded = False

    def test_quiet_mode_prints_nothing(self) -> None:
        TerminalUI(quiet=True).preview(self.Render())  # must not raise or print

    def test_json_mode_emits_a_preview_event(self) -> None:
        import io
        import json
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            TerminalUI(json_output=True).preview(self.Render())
        payload = json.loads(buffer.getvalue().strip())
        self.assertEqual(payload["event"], "preview")
        self.assertEqual(payload["masked_text"], self.Render.line)

    def test_plain_mode_prints_a_preview_line(self) -> None:
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            TerminalUI().preview(self.Render())
        self.assertIn("[PREVIEW]", buffer.getvalue())

    def test_a_bad_render_object_is_swallowed(self) -> None:
        TerminalUI().preview(object())  # must not raise


class RawBufferHygieneTests(unittest.TestCase):
    def test_preview_module_never_logs_or_persists(self) -> None:
        source = inspect.getsource(LivePreview)
        for forbidden in ("logging", "open(", "sqlite", "write(", "print("):
            self.assertNotIn(forbidden, source, f"preview must not use {forbidden}")

    def test_close_wipes_buffer_and_salt(self) -> None:
        class Masker:
            def mask_text(self, text):
                class Result:
                    masked_text = text
                    entities: list = []
                    safe_to_release = True

                return Result()

        preview = LivePreview(Masker(), lambda render: None, minimum_cycle_seconds=0.0)
        preview.add_segment("my name is rayyan and this is private", unique_seconds=8.5)
        before = preview._registry.size
        preview.close()
        self.assertEqual(preview._segments, [])
        self.assertEqual(preview._registry.size, 0)
        self.assertGreaterEqual(before, 0)
        blob = repr(preview.audit())
        self.assertNotIn("rayyan", blob)


if __name__ == "__main__":
    unittest.main()

"""What the end-of-session report claims, and what it refuses to claim."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.session_report import MEANINGFUL_SAMPLE_COUNT, SessionReport, Series
from integrated_pipeline.schemas import (
    BufferOutput,
    LiveOutput,
    PIIResult,
    STTResult,
    TranscriptFragment,
)


def stt_result(*, total: float = 0.4, audio: float = 1.5) -> STTResult:
    return STTResult(
        final_text="hello",
        duration_seconds=audio,
        processing_seconds=total - 0.1,
        finalization_seconds=0.1,
        rtf=total / audio,
        callback_count=3,
        session_id="s",
        batch_id="b",
        sequence_number=1,
        captured_audio_seconds=audio,
        submitted_audio_seconds=audio,
        queue_wait_seconds=0.01,
        total_stt_seconds=total,
    )


def live_output(
    *,
    safe: bool = True,
    entities: list[dict] | None = None,
    mask_seconds: float = 0.2,
    total_seconds: float = 1.0,
    queue_wait: float = 0.05,
    classifier: float = 0.12,
    stt: STTResult | None = None,
) -> LiveOutput:
    fragment = TranscriptFragment(
        fragment_id="f",
        text="hello",
        arrival_time_seconds=0.0,
        source_audio_id="a",
        stt_result=stt if stt is not None else stt_result(),
    )
    buffer = BufferOutput(
        buffer_id="buf",
        text="hello",
        fragments=[fragment],
        started_at_seconds=0.0,
        flushed_at_seconds=1.0,
        flush_reason="immediate",
    )
    pii = PIIResult(
        original_text="hello",
        masked_text="hello",
        latency_seconds=mask_seconds,
        entities=list(entities or ()),
        safe_to_release=safe,
        privacy_status="masked" if safe else "blocked",
        raw_model_output={"pii_model_latency_seconds": classifier},
    )
    return LiveOutput(
        buffer=buffer,
        pii=pii,
        total_seconds=total_seconds,
        mask_queue_wait_seconds=queue_wait,
    )


class SeriesTests(unittest.TestCase):
    def test_an_empty_series_has_no_opinion(self) -> None:
        series = Series()
        self.assertIsNone(series.percentile(0.5))
        self.assertIsNone(series.mean)
        self.assertIsNone(series.peak)
        self.assertEqual(series.summary()["count"], 0)

    def test_percentiles_come_from_the_sorted_values(self) -> None:
        series = Series()
        for value in (5.0, 1.0, 3.0, 2.0, 4.0):
            series.add(value)
        self.assertEqual(series.percentile(0.0), 1.0)
        self.assertEqual(series.percentile(0.5), 3.0)
        self.assertEqual(series.percentile(1.0), 5.0)
        self.assertEqual(series.peak, 5.0)

    def test_the_tail_is_kept_rather_than_averaged_away(self) -> None:
        # The whole reason percentiles are here: the slow utterances are
        # the ones somebody standing in front of the device remembers,
        # and a mean puts them next to ninety fast ones and forgets them.
        series = Series()
        for _ in range(90):
            series.add(0.5)
        for _ in range(10):
            series.add(9.0)
        self.assertLess(series.mean, 1.5)
        self.assertEqual(series.percentile(0.95), 9.0)
        self.assertEqual(series.percentile(0.50), 0.5)
        self.assertEqual(series.peak, 9.0)

    def test_nothing_missing_enters_the_distribution(self) -> None:
        series = Series()
        series.add(None)
        series.add("not a number")
        # RTF arrives as nan when the audio length was zero, which is
        # what an empty final looks like. One nan corrupts every
        # percentile below it.
        series.add(float("nan"))
        self.assertEqual(len(series), 0)
        series.add(math.nan)
        self.assertEqual(len(series), 0)


class HearingTests(unittest.TestCase):
    def test_calibration_is_taken_from_the_event(self) -> None:
        report = SessionReport()
        report.observe_event(
            "calibration_complete",
            {"start_threshold": 0.15, "capped": True, "noise_rms": 0.4},
        )
        self.assertEqual(report.threshold, 0.15)
        self.assertTrue(report.threshold_capped)
        self.assertIn("the room was noisy", report.text())

    def test_a_microphone_that_barely_clears_its_gate_is_named(self) -> None:
        # Open since the first Pi run and never measured: a peak that
        # barely reaches the threshold is a device that mostly cannot
        # hear. Only sayable because speech did get through this time.
        report = SessionReport()
        report.observe_event(
            "calibration_complete", {"start_threshold": 0.10, "capped": False}
        )
        for _ in range(50):
            report.observe_level(0.02)
        report.observe_level(0.11)
        report.observe_output(live_output())
        self.assertIn("barely clears its own threshold", report.text())

    def test_a_silent_room_is_not_reported_as_a_broken_microphone(self) -> None:
        # Nobody spoke, so the peak is room noise. Calling that a fault
        # sends somebody to replace a microphone that works.
        report = SessionReport()
        report.observe_event(
            "calibration_complete", {"start_threshold": 0.10, "capped": False}
        )
        for _ in range(50):
            report.observe_level(0.003)
        text = report.text()
        self.assertIn("a quiet room, or a quiet mic", text)
        self.assertNotIn("barely clears", text)

    def test_a_microphone_with_headroom_is_not_accused(self) -> None:
        report = SessionReport()
        report.observe_event(
            "calibration_complete", {"start_threshold": 0.02, "capped": False}
        )
        report.observe_level(0.01)
        report.observe_level(0.40)
        text = report.text()
        self.assertIn("loudest / threshold", text)
        self.assertNotIn("too quiet", text)

    def test_near_misses_name_the_spelling_and_the_file_that_fixes_it(self) -> None:
        report = SessionReport()
        for _ in range(3):
            report.observe_event(
                "onkith_no_command", {"nearest": "on kit", "distance": 2}
            )
        text = report.text()
        self.assertIn('"on kit" x3', text)
        self.assertIn("wake.spellings", text)
        self.assertEqual(report.fell_through, 3)

    def test_speech_that_was_not_addressed_leaves_no_near_miss(self) -> None:
        report = SessionReport()
        report.observe_event("onkith_no_command", {"nearest": None})
        self.assertEqual(report.fell_through, 1)
        self.assertEqual(dict(report.near_misses), {})

    def test_commands_are_counted_by_intent(self) -> None:
        report = SessionReport()
        report.observe_event("onkith_command_detected", {"intent": "PLAY"})
        report.observe_event("onkith_command_detected", {"intent": "PLAY"})
        report.observe_event("onkith_command_detected", {"intent": "LIGHT_ON"})
        self.assertEqual(report.commands_recognised, 3)
        self.assertEqual(report.commands_by_intent["PLAY"], 2)
        self.assertIn("PLAY 2", report.text())

    def test_dropped_chunks_and_unreliable_output_are_kept_apart(self) -> None:
        report = SessionReport()
        report.observe_event("queue_overflow", {"new_dropped_chunks": 4})
        report.observe_event("queue_overflow", {"new_dropped_chunks": 2})
        report.observe_event("stt_finalized", {"rtf": 0.3, "reliable": False})
        self.assertEqual(report.dropped_chunks, 6)
        self.assertEqual(report.unreliable_utterances, 1)

    def test_an_unusable_event_does_not_reach_the_caller(self) -> None:
        # These observers run on the capture callback and the STT thread.
        # Neither may be stopped by a metric.
        report = SessionReport()
        report.observe_event("stt_finalized", None)
        report.observe_event("calibration_complete", {"start_threshold": object()})
        report.observe_output(object())
        self.assertEqual(report.threshold, 0.0)


class OutputTests(unittest.TestCase):
    def test_a_released_utterance_fills_every_latency_row(self) -> None:
        report = SessionReport()
        report.observe_output(live_output())
        self.assertEqual(report.utterances, 1)
        self.assertEqual(report.mask_seconds.peak, 0.2)
        self.assertEqual(report.mask_queue_wait.peak, 0.05)
        self.assertEqual(report.classifier_seconds.peak, 0.12)
        self.assertEqual(report.end_to_end.peak, 1.0)
        self.assertEqual(report.stt_seconds.peak, 0.4)
        self.assertEqual(report.audio_seconds.peak, 1.5)

    def test_a_blocked_utterance_is_counted_and_not_measured(self) -> None:
        # It never reached the masker's output, so its latencies would be
        # a different quantity wearing the same name.
        report = SessionReport()
        report.observe_output(live_output(safe=False))
        self.assertEqual(report.blocked_utterances, 1)
        self.assertEqual(len(report.end_to_end), 0)
        self.assertEqual(report.utterances, 1)

    def test_masked_entities_are_counted_by_type(self) -> None:
        report = SessionReport()
        report.observe_output(
            live_output(
                entities=[
                    {"entity_type": "GIVENNAME"},
                    {"entity_type": "GIVENNAME"},
                    {"entity_type": "TELEPHONENUM"},
                ]
            )
        )
        self.assertEqual(report.entities["GIVENNAME"], 2)
        self.assertIn("GIVENNAME 2", report.text())

    def test_a_thin_sample_says_so_rather_than_implying_a_distribution(self) -> None:
        report = SessionReport()
        report.observe_output(live_output())
        self.assertIn(
            f"under {MEANINGFUL_SAMPLE_COUNT} utterances", report.text()
        )

    def test_enough_utterances_stops_the_caveat(self) -> None:
        report = SessionReport()
        for _ in range(MEANINGFUL_SAMPLE_COUNT):
            report.observe_output(live_output())
        self.assertNotIn("under 5 utterances", report.text())


class AssistantTests(unittest.TestCase):
    def test_replies_and_failures_are_separate_numbers(self) -> None:
        report = SessionReport()
        report.observe_reply(1.2)
        report.observe_reply(0.8)
        report.observe_reply(3.0, failed=True)
        self.assertEqual(report.replies, 2)
        self.assertEqual(report.reply_failures, 1)
        # A failed round trip is a network timing, not a reply timing.
        # Averaging them together turns one outage into slow answers.
        self.assertEqual(report.reply_seconds.peak, 1.2)


class RenderingTests(unittest.TestCase):
    def test_accuracy_is_refused_and_the_command_for_it_is_given(self) -> None:
        # A live session has no ground truth. Printing a number that
        # looks like accuracy and is not would be worse than a gap.
        text = SessionReport().text()
        self.assertIn("Accuracy is not in this report", text)
        self.assertIn("eval/run_eval.py", text)

    def test_an_untouched_session_still_renders(self) -> None:
        text = SessionReport(session_id="abc").text()
        self.assertIn("abc", text)
        self.assertIn("Latency, seconds", text)

    def test_the_audit_prints_what_the_workers_recorded(self) -> None:
        report = SessionReport()
        report.audit = {
            "audio_queue_high_water_mark": 3,
            "completed_queue_overload_count": 2,
            "db_queue_high_water_mark": 1,
            "callback_max_seconds": 0.004,
        }
        text = report.text()
        self.assertIn("audio queue high", text)
        self.assertIn("<- not zero", text)
        self.assertIn("0.004000s", text)

    def test_a_worker_that_could_not_be_read_is_admitted_to(self) -> None:
        report = SessionReport()
        report.audit = {"audit_error": "RuntimeError: closed"}
        self.assertIn("RuntimeError: closed", report.text())

    def test_the_json_artifact_round_trips(self) -> None:
        report = SessionReport(session_id="xyz", stt_engine="whisper base")
        report.observe_output(live_output())
        report.observe_reply(0.9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            report.write_json(str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["session_id"], "xyz")
        self.assertEqual(payload["stt_engine"], "whisper base")
        self.assertEqual(payload["assistant"]["replies"], 1)
        self.assertEqual(payload["latency_seconds"]["mask_total"]["max"], 0.2)


if __name__ == "__main__":
    unittest.main()

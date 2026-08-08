from __future__ import annotations

import queue
import inspect
import unittest

from app.live_runtime import LivePrivacyRuntime, LiveSTTRuntime
from integrated_pipeline.schemas import CanaryUtteranceEnd
from tests.helpers import canary_segment, segment_transcript


class FakeCanary:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.segments = []

    def transcribe_segment(self, segment):
        self.segments.append(segment)
        return segment_transcript(segment, self.texts.pop(0))


class FailingCanary:
    def transcribe_segment(self, segment):
        raise RuntimeError("model inference failed")


class FakePrivacy:
    def __init__(self) -> None:
        self.results = []
        self.failures = []

    def submit_stt(self, result) -> None:
        self.results.append(result)

    def submit_stt_failure(self, *args) -> None:
        self.failures.append(args)


class FakeMusic:
    def __init__(self) -> None:
        self.plays = []
        self.closes = 0

    def submit_play(self, text: str) -> None:
        self.plays.append(text)

    def submit_close(self) -> None:
        self.closes += 1


def runtime(texts: list[str], privacy: FakePrivacy, music=None):
    completed = queue.Queue(maxsize=4)
    events = []
    worker = LiveSTTRuntime(
        completed,
        FakeCanary(texts),
        privacy,
        on_event=lambda event, payload: events.append((event, payload)),
        music_worker=music,
        utc_now=lambda: "2026-08-05T10:00:03Z",
    )
    worker.start()
    return completed, worker, events


class LiveCanaryRuntimeTests(unittest.TestCase):
    def test_live_stage5_does_not_instantiate_legacy_timer(self) -> None:
        self.assertNotIn(
            "WallClockBufferWorker(", inspect.getsource(LivePrivacyRuntime)
        )

    def test_nonfinal_is_held_and_one_final_result_is_merged(self) -> None:
        privacy = FakePrivacy()
        completed, worker, events = runtime(["hello the", "the world"], privacy)
        first = canary_segment(0, final=False, start=0, end=141_824)
        final = canary_segment(1, final=True, start=130_560, end=180_000)
        completed.put(first)
        completed.join()
        self.assertEqual(privacy.results, [])
        completed.put(final)
        completed.join()
        worker.shutdown()
        self.assertEqual(len(privacy.results), 1)
        result = privacy.results[0]
        self.assertEqual(result.final_text, "hello the world")
        self.assertEqual(result.segment_count, 2)
        self.assertEqual(result.deduplicated_word_count, 1)
        self.assertEqual(result.callback_count, 0)
        self.assertAlmostEqual(result.submitted_audio_seconds, (141_824 + 49_440) / 16_000)
        self.assertAlmostEqual(result.unique_submitted_audio_seconds, 180_000 / 16_000)
        names = [event for event, payload in events]
        self.assertEqual(names.count("stt_finalized"), 1)
        self.assertIn("stt_segment_complete", names)

    def test_blank_segment_warns_but_other_text_survives(self) -> None:
        privacy = FakePrivacy()
        completed, worker, events = runtime(["", "usable text"], privacy)
        completed.put(canary_segment(0, final=False, start=0, end=141_824))
        completed.put(canary_segment(1, final=True, start=130_560, end=160_000))
        worker.shutdown()
        self.assertEqual(privacy.results[0].final_text, "usable text")
        self.assertEqual(privacy.results[0].blank_segment_indices, (0,))
        self.assertIn("blank_segment_transcript", [item[0] for item in events])

    def test_all_blank_uses_failure_storage_path(self) -> None:
        privacy = FakePrivacy()
        completed, worker, _ = runtime(["", "  "], privacy)
        completed.put(canary_segment(0, final=False, start=0, end=141_824))
        completed.put(canary_segment(1, final=True, start=130_560, end=160_000))
        worker.shutdown()
        self.assertEqual(privacy.results, [])
        self.assertEqual(len(privacy.failures), 1)
        self.assertIn("empty_transcript", privacy.failures[0][-1])

    def test_onkith_runs_once_only_after_final_merge(self) -> None:
        privacy = FakePrivacy()
        music = FakeMusic()
        completed, worker, events = runtime(
            ["normal words Onkith play", "play believer"], privacy, music
        )
        completed.put(canary_segment(0, final=False, start=0, end=141_824))
        completed.join()
        self.assertEqual(music.plays, [])
        completed.put(canary_segment(1, final=True, start=130_560, end=160_000))
        worker.shutdown()
        self.assertEqual(music.plays, ["play believer"])
        self.assertEqual(len(privacy.results), 1)
        self.assertEqual(privacy.results[0].final_text, "normal words")
        self.assertEqual(
            [item[0] for item in events].count("onkith_command_detected"), 1
        )

    def test_command_only_skips_stage5(self) -> None:
        privacy = FakePrivacy()
        music = FakeMusic()
        completed, worker, _ = runtime(["Onkith stop the music"], privacy, music)
        completed.put(canary_segment(0, final=True))
        worker.shutdown()
        self.assertEqual(music.closes, 1)
        self.assertEqual(privacy.results, [])

    def test_shutdown_control_finalizes_without_second_inference(self) -> None:
        privacy = FakePrivacy()
        completed, worker, _ = runtime(["already decoded"], privacy)
        first = canary_segment(0, final=False, start=0, end=141_824)
        completed.put(first)
        completed.put(
            CanaryUtteranceEnd(
                session_id=first.session_id,
                utterance_id=first.utterance_id,
                utterance_sequence_number=first.utterance_sequence_number,
                decision_sample=141_824,
                endpoint_reason="shutdown",
                reliable=False,
                dropped_audio_chunks=0,
                recording_closed_monotonic=first.recording_closed_monotonic,
                audio_ended_monotonic=first.audio_ended_monotonic,
                audio_ended_at_utc=first.audio_ended_at_utc,
                utterance_start_sample=first.utterance_start_sample,
            )
        )
        worker.shutdown()
        self.assertEqual(len(privacy.results), 1)
        self.assertEqual(privacy.results[0].endpoint_reason, "shutdown")
        self.assertFalse(privacy.results[0].reliable)

    def test_out_of_order_segments_fail_clearly(self) -> None:
        privacy = FakePrivacy()
        completed, worker, _ = runtime(["text"], privacy)
        completed.put(canary_segment(1, final=True))
        completed.join()
        with self.assertRaisesRegex(RuntimeError, "STT worker failure"):
            worker.shutdown()

    def test_inference_error_uses_failure_storage_path(self) -> None:
        privacy = FakePrivacy()
        completed = queue.Queue(maxsize=4)
        worker = LiveSTTRuntime(
            completed,
            FailingCanary(),
            privacy,
            on_event=lambda event, payload: None,
            utc_now=lambda: "2026-08-05T10:00:03Z",
        )
        worker.start()
        completed.put(canary_segment(0, final=True))
        worker.shutdown()
        self.assertEqual(privacy.results, [])
        self.assertEqual(len(privacy.failures), 1)
        self.assertIn("model inference failed", privacy.failures[0][-1])


if __name__ == "__main__":
    unittest.main()

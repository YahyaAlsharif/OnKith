from __future__ import annotations

import queue
import threading
import time
import unittest

from app.live_runtime import LiveSTTRuntime
from app.omit_ai_live import _shutdown_pipeline
from tests.helpers import canary_segment, segment_transcript


class Privacy:
    def __init__(self) -> None:
        self.results = []

    def submit_stt(self, result) -> None:
        self.results.append(result)

    def submit_stt_failure(self, *args) -> None:
        raise AssertionError(args)


class BlockingCanary:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe_segment(self, segment):
        self.started.set()
        self.release.wait(timeout=3)
        return segment_transcript(segment, "drained text")


class ShutdownTests(unittest.TestCase):
    def test_stt_shutdown_waits_for_inflight_canary(self) -> None:
        completed = queue.Queue(maxsize=4)
        canary = BlockingCanary()
        privacy = Privacy()
        runtime = LiveSTTRuntime(
            completed,
            canary,
            privacy,
            on_event=lambda event, payload: None,
            utc_now=lambda: "2026-08-05T10:00:03Z",
        )
        runtime.start()
        completed.put(canary_segment(0, final=True))
        self.assertTrue(canary.started.wait(timeout=1))
        done = threading.Event()

        def shutdown() -> None:
            runtime.shutdown()
            done.set()

        thread = threading.Thread(target=shutdown)
        thread.start()
        time.sleep(0.05)
        self.assertFalse(done.is_set())
        canary.release.set()
        thread.join(timeout=2)
        self.assertTrue(done.is_set())
        self.assertEqual(privacy.results[0].final_text, "drained text")

    def test_pipeline_shutdown_order_is_drain_first(self) -> None:
        calls = []

        class Item:
            def __init__(self, name):
                self.name = name

            def stop(self): calls.append("audio.stop")
            def shutdown(self, *args, **kwargs): calls.append(f"{self.name}.shutdown")
            def drain(self): calls.append("storage.drain")
            def finish_session(self, *args): calls.append("storage.finish")
            def query_session(self, *args): calls.append("storage.query"); return []

        class UI:
            def session_transcripts(self, rows): calls.append("ui.session")

        errors = []
        _shutdown_pipeline(
            session_id="session",
            ui=UI(),
            audio=Item("audio"),
            stt=Item("stt"),
            music=Item("music"),
            privacy=Item("privacy"),
            storage=Item("storage"),
            errors=errors,
        )
        self.assertEqual(errors, [])
        self.assertEqual(
            calls,
            [
                "audio.stop",
                "stt.shutdown",
                "music.shutdown",
                "privacy.shutdown",
                "storage.drain",
                "storage.finish",
                "storage.drain",
                "storage.query",
                "ui.session",
                "storage.shutdown",
            ],
        )


if __name__ == "__main__":
    unittest.main()

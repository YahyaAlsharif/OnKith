from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from pathlib import Path

from app.live_runtime import LivePrivacyRuntime, LiveSTTRuntime
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.pipeline import FailClosedMasker
from integrated_pipeline.schemas import PIIResult
from integrated_pipeline.sqlite_storage import SQLiteTranscriptStore

from tests.helpers import recording, stt_result


class Masker:
    def mask_text(self, text: str) -> PIIResult:
        return PIIResult(
            original_text=text,
            masked_text="[GIVENNAME_1] spoke",
            latency_seconds=0.001,
            entities=[
                {
                    "entity_type": "GIVENNAME",
                    "placeholder": "[GIVENNAME_1]",
                }
            ],
            safe_to_release=True,
            privacy_status="masked",
        )


class Moonshine:
    def transcribe_recording(self, item, partial_callback=None):
        return stt_result(item, "Mohammed spoke")


class ShutdownTests(unittest.TestCase):
    def test_last_pending_recording_drains_through_stt_mask_and_database(self):
        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteTranscriptStore(
                Path(directory) / "session_transcript.db"
            )
            store.start()
            store.start_session(
                "session-test", "2026-07-29T10:00:00Z"
            )
            outputs = []
            privacy = LivePrivacyRuntime(
                config,
                FailClosedMasker(Masker()),
                store,
                on_event=lambda *args: None,
                on_output=outputs.append,
            )
            completed = queue.Queue(maxsize=4)
            stt = LiveSTTRuntime(
                completed,
                Moonshine(),
                privacy,
                on_event=lambda *args: None,
            )
            privacy.start()
            stt.start()
            completed.put(recording(1, endpoint_reason="shutdown"))
            stt.shutdown()
            privacy.shutdown()
            store.drain()
            store.finish_session(
                "session-test", "2026-07-29T10:00:10Z", "complete"
            )
            store.drain()
            rows = store.query_session("session-test")
            store.shutdown()

            self.assertEqual(len(outputs), 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].status, "complete")
            self.assertEqual(rows[0].raw_transcript, "Mohammed spoke")
            self.assertEqual(rows[0].masked_transcript, "[GIVENNAME_1] spoke")
            self.assertEqual(completed.unfinished_tasks, 0)
            self.assertEqual(privacy.mask_queue.unfinished_tasks, 0)
            self.assertEqual(store.operation_queue.unfinished_tasks, 0)
            self.assertFalse(stt.thread.is_alive())
            self.assertFalse(privacy.thread.is_alive())
            self.assertFalse(store.thread.is_alive())

    def test_database_operation_queued_at_shutdown_is_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteTranscriptStore(
                Path(directory) / "session_transcript.db"
            )
            store.start()
            store.start_session(
                "session-test", "2026-07-29T10:00:00Z"
            )
            item = recording(1)
            store.insert_raw(stt_result(item, "final pending transcript"))
            store.drain()
            rows = store.query_session("session-test")
            store.shutdown()
            self.assertEqual(rows[0].raw_transcript, "final pending transcript")

    def test_shutdown_waits_for_pending_stt_then_preserves_result(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingMoonshine:
            def transcribe_recording(self, item, partial_callback=None):
                started.set()
                release.wait(timeout=2.0)
                return stt_result(item, "pending STT survives")

        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteTranscriptStore(
                Path(directory) / "session_transcript.db"
            )
            store.start()
            store.start_session(
                "session-test", "2026-07-29T10:00:00Z"
            )
            privacy = LivePrivacyRuntime(
                config,
                FailClosedMasker(Masker()),
                store,
                on_event=lambda *args: None,
                on_output=lambda output: None,
            )
            completed = queue.Queue(maxsize=4)
            stt = LiveSTTRuntime(
                completed,
                BlockingMoonshine(),
                privacy,
                on_event=lambda *args: None,
            )
            privacy.start()
            stt.start()
            completed.put(recording(1))
            self.assertTrue(started.wait(timeout=1.0))
            shutdown = threading.Thread(target=stt.shutdown)
            shutdown.start()
            self.assertTrue(shutdown.is_alive())
            release.set()
            shutdown.join(timeout=2.0)
            privacy.shutdown()
            store.drain()
            rows = store.query_session("session-test")
            store.shutdown()
            self.assertFalse(shutdown.is_alive())
            self.assertEqual(rows[0].raw_transcript, "pending STT survives")
            self.assertEqual(rows[0].status, "complete")

    def test_shutdown_waits_for_pending_mask_then_stores_completion(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingMasker:
            def mask_text(self, text: str) -> PIIResult:
                started.set()
                release.wait(timeout=2.0)
                return Masker().mask_text(text)

        config = PipelineConfig()
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteTranscriptStore(
                Path(directory) / "session_transcript.db"
            )
            store.start()
            store.start_session(
                "session-test", "2026-07-29T10:00:00Z"
            )
            item = recording(1)
            result = stt_result(item, "pending mask survives")
            privacy = LivePrivacyRuntime(
                config,
                FailClosedMasker(BlockingMasker()),
                store,
                on_event=lambda *args: None,
                on_output=lambda output: None,
            )
            privacy.start()
            privacy.submit_stt(result)
            self.assertTrue(started.wait(timeout=1.0))
            shutdown = threading.Thread(target=privacy.shutdown)
            shutdown.start()
            self.assertTrue(shutdown.is_alive())
            release.set()
            shutdown.join(timeout=2.0)
            store.drain()
            rows = store.query_session("session-test")
            store.shutdown()
            self.assertFalse(shutdown.is_alive())
            self.assertEqual(rows[0].status, "complete")
            self.assertEqual(rows[0].masked_transcript, "[GIVENNAME_1] spoke")


if __name__ == "__main__":
    unittest.main()

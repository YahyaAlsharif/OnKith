from __future__ import annotations

import inspect
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path

from app.live_runtime import (
    LivePrivacyRuntime,
    LiveSTTRuntime,
    deduplicate_boundary,
)
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.pipeline import BLOCKED_MARKER, FailClosedMasker
from integrated_pipeline.schemas import PIIResult
from integrated_pipeline.sqlite_storage import (
    InsertRawTranscript,
    SQLiteTranscriptStore,
)

from tests.helpers import recording, stt_result


class FakeStorage:
    def __init__(self, order: list[str] | None = None):
        self.raw = []
        self.completed = []
        self.failed = []
        self.stt_failed = []
        self.order = order

    def insert_raw(self, result):
        if self.order is not None:
            self.order.append("raw")
        self.raw.append(result)

    def insert_stt_failure(self, *args):
        self.stt_failed.append(args)

    def complete_masked(self, operation):
        if self.order is not None:
            self.order.append("database")
        self.completed.append(operation)

    def mark_mask_failed(self, operation):
        if self.order is not None:
            self.order.append("database")
        self.failed.append(operation)


class FakeMasker:
    def mask_text(self, text: str) -> PIIResult:
        return PIIResult(
            original_text=text,
            masked_text=text.replace("secret", "[IDCARDNUM_1]"),
            latency_seconds=0.001,
            entities=[],
            safe_to_release=True,
            privacy_status="masked",
        )


class BrokenMasker:
    def mask_text(self, text: str) -> PIIResult:
        raise RuntimeError(f"must not release {text}")


class FakeMoonshine:
    def __init__(self, texts: list[str], delay: float = 0.0):
        self.texts = iter(texts)
        self.delay = delay
        self.calls = []

    def transcribe_recording(self, item, partial_callback=None):
        self.calls.append((item, threading.current_thread().name))
        if self.delay:
            time.sleep(self.delay)
        text = next(self.texts)
        if partial_callback is not None:
            partial_callback(text)
        return stt_result(item, text)


class PrivacyAndSTTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PipelineConfig()
        self.events = []

    def test_boundary_dedup_is_conservative(self) -> None:
        text, count = deduplicate_boundary(
            "the exact boundary words",
            "Boundary words continue here",
        )
        self.assertEqual((text, count), ("continue here", 2))
        unchanged, count = deduplicate_boundary(
            "nothing matches", "the current transcript"
        )
        self.assertEqual((unchanged, count), ("the current transcript", 0))
        common, count = deduplicate_boundary("ends with the", "the next part")
        self.assertEqual((common, count), ("the next part", 0))

    def test_final_text_reaches_mask_queue_immediately_without_buffer_thread(self):
        storage = FakeStorage()
        outputs = []
        privacy = LivePrivacyRuntime(
            self.config,
            FailClosedMasker(FakeMasker()),
            storage,
            on_event=lambda *args: self.events.append(args),
            on_output=outputs.append,
        )
        privacy.start()
        item = recording(1)
        started = time.perf_counter()
        privacy.submit_stt(stt_result(item, "a secret value"))
        privacy.mask_queue.join()
        elapsed = time.perf_counter() - started
        privacy.shutdown()
        self.assertLess(elapsed, 1.0)
        self.assertFalse(hasattr(privacy, "buffer_worker"))
        self.assertFalse(
            any(
                thread.name == "omit-buffer-timer"
                for thread in threading.enumerate()
            )
        )
        self.assertEqual(outputs[0].buffer.flush_reason, "finalized_recording")
        self.assertEqual(outputs[0].buffer.buffer_waiting_seconds, 0.0)
        self.assertEqual(len(storage.raw), 1)
        self.assertEqual(len(storage.completed), 1)

    def test_stage5_materializes_batch_and_owns_both_fanout_identities(self):
        storage = FakeStorage()
        privacy = LivePrivacyRuntime(
            self.config,
            FailClosedMasker(FakeMasker()),
            storage,
            on_event=lambda *args: None,
            on_output=lambda output: None,
            clock=lambda: 123.0,
        )
        result = stt_result(recording(7), "same raw transcript")

        # No worker is running: submit_stt must only materialize and enqueue.
        privacy.submit_stt(result)

        self.assertEqual(storage.raw, [result])
        task = privacy.mask_queue.get_nowait()
        self.assertEqual(task.buffer.session_id, result.session_id)
        self.assertEqual(task.buffer.batch_id, result.batch_id)
        self.assertEqual(task.buffer.sequence_number, result.sequence_number)
        self.assertEqual(task.buffer.text, result.final_text)
        self.assertEqual(task.buffer.buffer_id, result.batch_id)
        self.assertEqual(task.buffer.flush_reason, "finalized_recording")
        self.assertEqual(task.buffer.buffer_waiting_seconds, 0.0)
        self.assertEqual(task.buffer.started_at_seconds, 123.0)
        self.assertEqual(task.buffer.flushed_at_seconds, 123.0)
        self.assertIs(task.buffer.fragments[0].stt_result, result)
        self.assertEqual(task.enqueued_monotonic, 123.0)
        privacy.mask_queue.task_done()

    def test_display_occurs_before_database_completion_enqueue(self) -> None:
        order = []
        storage = FakeStorage(order)
        privacy = LivePrivacyRuntime(
            self.config,
            FailClosedMasker(FakeMasker()),
            storage,
            on_event=lambda *args: None,
            on_output=lambda output: order.append("display"),
        )
        privacy.start()
        privacy.submit_stt(stt_result(recording(1), "safe text"))
        privacy.shutdown()
        self.assertEqual(order, ["raw", "display", "database"])
        self.assertEqual(
            storage.raw[0].batch_id,
            storage.completed[0].batch_id,
        )

    def test_fail_closed_path_never_stores_raw_as_masked(self) -> None:
        storage = FakeStorage()
        outputs = []
        raw = "secret raw value"
        privacy = LivePrivacyRuntime(
            self.config,
            FailClosedMasker(BrokenMasker()),
            storage,
            on_event=lambda *args: None,
            on_output=outputs.append,
        )
        privacy.start()
        privacy.submit_stt(stt_result(recording(1), raw))
        privacy.shutdown()
        self.assertFalse(outputs[0].pii.safe_to_release)
        self.assertEqual(outputs[0].pii.masked_text, BLOCKED_MARKER)
        self.assertEqual(len(storage.failed), 1)
        self.assertNotIn(raw, repr(storage.failed[0]))

    def test_stt_submits_only_to_stage5_and_deduplicates(self) -> None:
        class Privacy:
            def __init__(self):
                self.results = []

            def submit_stt(self, result):
                self.results.append(result)

        completed = queue.Queue()
        privacy = Privacy()
        runtime = LiveSTTRuntime(
            completed,
            FakeMoonshine(
                ["we cross the boundary words", "boundary words continue"]
            ),
            privacy,
            on_event=lambda *args: None,
        )
        runtime.start()
        completed.put(recording(1, endpoint_reason="maximum_utterance"))
        completed.put(recording(2, overlap=True))
        runtime.shutdown()
        self.assertEqual(
            [result.final_text for result in privacy.results],
            ["we cross the boundary words", "continue"],
        )
        self.assertNotIn(
            "storage",
            inspect.signature(LiveSTTRuntime.__init__).parameters,
        )
        self.assertNotIn(
            "self.storage",
            inspect.getsource(LiveSTTRuntime),
        )

    def test_slow_stt_does_not_block_recording_producer(self) -> None:
        completed = queue.Queue(maxsize=4)

        class Privacy:
            def __init__(self):
                self.results = []

            def submit_stt(self, result):
                self.results.append(result)

        privacy = Privacy()
        runtime = LiveSTTRuntime(
            completed,
            FakeMoonshine(["one", "two", "three"], delay=0.08),
            privacy,
            on_event=lambda *args: None,
        )
        runtime.start()
        started = time.perf_counter()
        for sequence in range(1, 4):
            completed.put_nowait(recording(sequence))
        producer_elapsed = time.perf_counter() - started
        runtime.shutdown()
        self.assertLess(producer_elapsed, 0.03)
        self.assertEqual(len(privacy.results), 3)

    def test_slow_classifier_does_not_block_stt_submission(self) -> None:
        mask_started = threading.Event()
        release_mask = threading.Event()

        class BlockingMasker:
            def mask_text(self, text: str) -> PIIResult:
                mask_started.set()
                release_mask.wait(timeout=2.0)
                return FakeMasker().mask_text(text)

        storage = FakeStorage()
        privacy = LivePrivacyRuntime(
            self.config,
            FailClosedMasker(BlockingMasker()),
            storage,
            on_event=lambda *args: None,
            on_output=lambda output: None,
        )
        completed = queue.Queue(maxsize=4)
        runtime = LiveSTTRuntime(
            completed,
            FakeMoonshine(["one", "two", "three"]),
            privacy,
            on_event=lambda *args: None,
        )
        privacy.start()
        runtime.start()
        for sequence in range(1, 4):
            completed.put_nowait(recording(sequence))
        self.assertTrue(mask_started.wait(timeout=1.0))
        started = time.perf_counter()
        runtime.shutdown()
        elapsed = time.perf_counter() - started
        try:
            self.assertLess(elapsed, 0.5)
            self.assertEqual(
                [item.final_text for item in storage.raw],
                ["one", "two", "three"],
            )
            self.assertEqual(privacy.mask_queue.unfinished_tasks, 3)
        finally:
            release_mask.set()
            privacy.shutdown()

    def test_slow_sqlite_writer_does_not_block_masking_or_display(self) -> None:
        raw_write_started = threading.Event()
        release_raw_write = threading.Event()
        displayed = threading.Event()

        class BlockingRawStore(SQLiteTranscriptStore):
            def _execute(self, connection, operation):
                if isinstance(operation, InsertRawTranscript):
                    raw_write_started.set()
                    release_raw_write.wait(timeout=2.0)
                return super()._execute(connection, operation)

        with tempfile.TemporaryDirectory() as directory:
            store = BlockingRawStore(
                Path(directory) / "session_transcript.db"
            )
            store.start()
            store.start_session(
                "session-test", "2026-07-29T10:00:00Z"
            )
            store.drain()
            privacy = LivePrivacyRuntime(
                self.config,
                FailClosedMasker(FakeMasker()),
                store,
                on_event=lambda *args: None,
                on_output=lambda output: displayed.set(),
            )
            privacy.start()
            try:
                started = time.perf_counter()
                privacy.submit_stt(
                    stt_result(recording(1), "a secret value")
                )
                submit_elapsed = time.perf_counter() - started
                self.assertLess(submit_elapsed, 0.5)
                self.assertTrue(raw_write_started.wait(timeout=1.0))
                self.assertTrue(displayed.wait(timeout=1.0))
                self.assertFalse(release_raw_write.is_set())
            finally:
                release_raw_write.set()
                privacy.shutdown()
                store.drain()
                rows = store.query_session("session-test")
                store.shutdown()
            self.assertEqual(rows[0].status, "complete")
            self.assertEqual(
                rows[0].masked_transcript, "a [IDCARDNUM_1] value"
            )

    def test_live_worker_threads_are_non_daemon_and_stop(self) -> None:
        storage = FakeStorage()
        privacy = LivePrivacyRuntime(
            self.config,
            FailClosedMasker(FakeMasker()),
            storage,
            on_event=lambda *args: None,
            on_output=lambda output: None,
        )
        privacy.start()
        self.assertFalse(privacy.thread.daemon)
        privacy.shutdown()
        self.assertFalse(privacy.thread.is_alive())
        self.assertEqual(privacy.mask_queue.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()

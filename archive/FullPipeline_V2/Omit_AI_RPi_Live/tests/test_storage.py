from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from integrated_pipeline.sqlite_storage import (
    CompleteMaskedTranscript,
    SQLiteTranscriptStore,
    StorageWriterError,
)

from tests.helpers import recording, stt_result


class SQLiteStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "session_transcript.db"
        self.store = SQLiteTranscriptStore(self.path)
        self.store.start()
        self.store.start_session(
            "session-test",
            "2026-07-29T10:00:00Z",
            model_identity="medium-streaming-cache",
        )

    def tearDown(self) -> None:
        if self.store.thread is not None and self.store.thread.is_alive():
            try:
                self.store.shutdown()
            except StorageWriterError:
                pass
        self.temp.cleanup()

    def insert(self, sequence: int, text: str = "raw text"):
        item = recording(sequence)
        result = stt_result(item, text)
        self.store.insert_raw(result)
        return result

    def test_wal_normal_single_connection_owner_and_waiting_row(self) -> None:
        self.insert(1)
        self.store.drain()
        rows = self.store.query_session("session-test")
        self.assertEqual(self.store.journal_mode, "wal")
        self.assertEqual(self.store.synchronous_level, 1)
        self.assertEqual(
            self.store.writer_thread_id, self.store.connection_thread_id
        )
        self.assertNotEqual(
            self.store.connection_thread_id, threading.get_ident()
        )
        self.assertEqual(rows[0].status, "waiting_for_mask")
        self.assertIsNone(rows[0].masked_transcript)
        self.assertIsNone(rows[0].entities)

    def test_completion_updates_same_batch_and_entities_round_trip(self) -> None:
        result = self.insert(1, "my email is a@example.com")
        entities = [
            {
                "entity_type": "EMAIL",
                "start": 12,
                "end": 25,
                "placeholder": "[EMAIL_1]",
            }
        ]
        self.store.complete_masked(
            CompleteMaskedTranscript(
                session_id=result.session_id,
                batch_id=result.batch_id,
                masked_transcript="my email is [EMAIL_1]",
                entities=entities,
                mask_started_at="2026-07-29T10:00:02Z",
                mask_completed_at="2026-07-29T10:00:03Z",
                mask_latency_seconds=0.2,
                total_latency_seconds=1.2,
            )
        )
        self.store.drain()
        row = self.store.query_session("session-test")[0]
        self.assertEqual(row.status, "complete")
        self.assertEqual(row.batch_id, result.batch_id)
        self.assertEqual(row.entities, entities)
        self.assertEqual(row.masked_transcript, "my email is [EMAIL_1]")

    def test_append_preserves_fragment_order_and_pairing(self) -> None:
        result = self.insert(1, "first")
        self.store.append_raw(
            result.session_id,
            result.batch_id,
            "second",
            "2026-07-29T10:00:03Z",
        )
        self.store.append_raw(
            result.session_id,
            result.batch_id,
            "third",
            "2026-07-29T10:00:04Z",
        )
        self.store.drain()
        row = self.store.query_session("session-test")[0]
        self.assertEqual(row.raw_transcript, "first second third")
        self.assertEqual(row.batch_id, result.batch_id)

    def test_query_is_ordered_by_sequence_even_if_inserts_are_not(self) -> None:
        self.insert(2, "second")
        self.insert(1, "first")
        self.store.drain()
        rows = self.store.query_session("session-test")
        self.assertEqual([row.sequence_number for row in rows], [1, 2])
        self.assertEqual(
            [row.raw_transcript for row in rows], ["first", "second"]
        )

    def test_reopen_preserves_prior_session(self) -> None:
        self.insert(1, "persisted")
        self.store.finish_session(
            "session-test", "2026-07-29T10:01:00Z", "complete"
        )
        self.store.shutdown()
        reopened = SQLiteTranscriptStore(self.path)
        reopened.start()
        rows = reopened.query_session("session-test")
        reopened.shutdown()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].raw_transcript, "persisted")

    def test_concurrent_producers_do_not_corrupt_session_order(self) -> None:
        threads = []

        def producer(sequence: int) -> None:
            self.insert(sequence, f"text-{sequence}")

        for sequence in range(1, 21):
            thread = threading.Thread(target=producer, args=(sequence,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        self.store.drain()
        rows = self.store.query_session("session-test")
        self.assertEqual(
            [row.sequence_number for row in rows], list(range(1, 21))
        )

    def test_writer_exception_surfaces_at_drain_and_shutdown(self) -> None:
        self.insert(1, "first")
        self.insert(1, "duplicate primary key")
        with self.assertRaises(StorageWriterError):
            self.store.drain()
        with self.assertRaises(StorageWriterError):
            self.store.shutdown()

    def test_connection_factory_runs_on_writer_thread_only(self) -> None:
        self.store.shutdown()
        ids = []

        def factory(path: str):
            ids.append(threading.get_ident())
            return sqlite3.connect(path)

        isolated = SQLiteTranscriptStore(
            Path(self.temp.name) / "owner.db",
            connection_factory=factory,
        )
        isolated.start()
        isolated.shutdown()
        self.assertEqual(ids, [isolated.writer_thread_id])
        self.assertNotEqual(ids[0], threading.get_ident())


if __name__ == "__main__":
    unittest.main()

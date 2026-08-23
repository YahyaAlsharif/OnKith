"""End-to-end SQLite ordering for safety-split logical utterances.

`query_session` orders by `sequence_number`, so a safety-split continuation
must take the next normal chronological number rather than one from a reserved
high range. These tests drive the real STT runtime into a real SQLite database
and assert the stored order matches the spoken order.
"""

from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path

from app.stt_runtime import LiveSTTRuntime
from integrated_pipeline.config import MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS
from integrated_pipeline.schemas import SegmentTranscript
from integrated_pipeline.sqlite_storage import SQLiteTranscriptStore
from tests.helpers import canary_segment

SHORT_SAMPLES = 2 * 16_000
SESSION = "session-test"


def utc_now_iso() -> str:
    return "2026-08-05T10:00:03Z"


class EchoCanary:
    def transcribe_segment(self, segment):
        return SegmentTranscript(
            segment=segment,
            text=f"word{segment.segment_index}",
            processing_seconds=0.01,
            queue_wait_seconds=0.0,
            stt_started_at_utc=utc_now_iso(),
            stt_completed_at_utc=utc_now_iso(),
        )


class StoringPrivacy:
    """Minimal Stage 5 stand-in: persist raw rows exactly like production."""

    def __init__(self, store: SQLiteTranscriptStore) -> None:
        self.store = store
        self.results = []
        self.failures = []

    def submit_stt(self, result) -> None:
        self.results.append(result)
        self.store.insert_raw(result)

    def submit_stt_failure(self, recording, started, completed, error) -> None:
        self.failures.append((recording, error))
        self.store.insert_stt_failure(recording, started, completed, error)


def short_segment(index: int, *, final: bool = False, utterance: int = 1):
    start = index * SHORT_SAMPLES
    return canary_segment(
        index,
        final=final,
        utterance=utterance,
        start=start,
        end=start + SHORT_SAMPLES,
    )


class SafetySplitOrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SQLiteTranscriptStore(
            Path(self.directory.name) / "session_transcript.db"
        )
        self.store.start()
        self.addCleanup(self._close_store)
        self.store.start_session(SESSION, utc_now_iso(), model_identity="test")
        self.privacy = StoringPrivacy(self.store)
        self.completed: queue.Queue = queue.Queue(maxsize=4)
        self.worker = LiveSTTRuntime(
            self.completed,
            EchoCanary(),
            self.privacy,
            on_event=lambda event, payload: None,
            utc_now=utc_now_iso,
        )
        self.worker.start()
        self.addCleanup(self._stop_worker)

    def _close_store(self) -> None:
        try:
            self.store.shutdown(checkpoint=False)
        except Exception:
            pass

    def _stop_worker(self) -> None:
        if self.worker.thread is not None and self.worker.thread.is_alive():
            self.completed.put(None)
            self.worker.thread.join(timeout=5.0)

    def feed(self, segments) -> None:
        for segment in segments:
            self.completed.put(segment)
            self.completed.join()

    def rows(self):
        self.store.drain()
        return self.store.query_session(SESSION)

    def test_continuation_precedes_later_independent_utterances(self) -> None:
        # Utterance 1 is long enough to trip the safety split.
        self.feed(
            [
                short_segment(index)
                for index in range(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS)
            ]
        )
        self.assertEqual(self.worker.safety_split_count, 1)
        # Its continuation finishes normally...
        self.feed([short_segment(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS, final=True)])
        # ...then two later, independent utterances are spoken.
        self.feed([short_segment(0, final=True, utterance=2)])
        self.feed([short_segment(0, final=True, utterance=3)])
        self.worker.shutdown()

        rows = self.rows()
        self.assertEqual(len(rows), 4)
        sequences = [row.sequence_number for row in rows]
        # query_session orders by sequence_number; the result must be the
        # order things were actually spoken.
        self.assertEqual(sequences, [1, 2, 3, 4])
        self.assertEqual(sequences, sorted(sequences))

        batch_ids = [row.batch_id for row in rows]
        self.assertEqual(
            batch_ids,
            [
                "session-test-utterance-000001",
                "session-test-utterance-000001-part-000002",
                "session-test-utterance-000002",
                "session-test-utterance-000003",
            ],
        )
        # Both parts of utterance 1 come before every later utterance.
        part_positions = [
            index
            for index, row in enumerate(rows)
            if row.logical_utterance_id == "session-test-utterance-000001"
        ]
        later_positions = [
            index
            for index, row in enumerate(rows)
            if row.logical_utterance_id != "session-test-utterance-000001"
        ]
        self.assertEqual(part_positions, [0, 1])
        self.assertTrue(max(part_positions) < min(later_positions))

    def test_no_reserved_high_sequence_numbers_are_used(self) -> None:
        self.feed(
            [
                short_segment(index)
                for index in range(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS)
            ]
        )
        self.feed([short_segment(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS, final=True)])
        self.feed([short_segment(0, final=True, utterance=2)])
        self.worker.shutdown()
        rows = self.rows()
        for row in rows:
            self.assertLess(
                row.sequence_number,
                1_000,
                "sequence numbers must stay in the normal chronological range",
            )
        self.assertEqual([row.sequence_number for row in rows], [1, 2, 3])

    def test_split_relationship_metadata_is_persisted(self) -> None:
        self.feed(
            [
                short_segment(index)
                for index in range(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS)
            ]
        )
        self.feed([short_segment(MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS, final=True)])
        self.feed([short_segment(0, final=True, utterance=2)])
        self.worker.shutdown()
        rows = self.rows()
        self.assertEqual(
            [row.logical_utterance_id for row in rows],
            [
                "session-test-utterance-000001",
                "session-test-utterance-000001",
                "session-test-utterance-000002",
            ],
        )
        self.assertEqual([row.continuation_part_index for row in rows], [1, 1 + 1, 1])
        # The relationship is recoverable purely from metadata.
        grouped: dict[str, list[int]] = {}
        for row in rows:
            grouped.setdefault(str(row.logical_utterance_id), []).append(
                int(row.continuation_part_index or 0)
            )
        self.assertEqual(grouped["session-test-utterance-000001"], [1, 2])
        self.assertEqual(grouped["session-test-utterance-000002"], [1])

    def test_ordinary_session_rows_are_unchanged(self) -> None:
        for utterance in (1, 2, 3):
            self.feed([short_segment(0, final=True, utterance=utterance)])
        self.worker.shutdown()
        rows = self.rows()
        self.assertEqual([row.sequence_number for row in rows], [1, 2, 3])
        self.assertEqual(
            [row.batch_id for row in rows],
            [
                "session-test-utterance-000001",
                "session-test-utterance-000002",
                "session-test-utterance-000003",
            ],
        )
        for row in rows:
            self.assertEqual(row.continuation_part_index, 1)
            self.assertEqual(row.logical_utterance_id, row.batch_id)
            self.assertEqual(row.status, "waiting_for_mask")

    def test_three_way_split_stays_ordered(self) -> None:
        # Enough segments for two safety splits, then a later utterance.
        total = 2 * MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS + 1
        self.feed(
            [
                short_segment(index, final=index == total - 1)
                for index in range(total)
            ]
        )
        self.feed([short_segment(0, final=True, utterance=2)])
        self.worker.shutdown()
        rows = self.rows()
        self.assertEqual(self.worker.safety_split_count, 2)
        self.assertEqual([row.sequence_number for row in rows], [1, 2, 3, 4])
        self.assertEqual(
            [row.continuation_part_index for row in rows], [1, 2, 3, 1]
        )
        self.assertEqual(
            [row.logical_utterance_id for row in rows],
            ["session-test-utterance-000001"] * 3
            + ["session-test-utterance-000002"],
        )
        # No text lost across two splits.
        words = " ".join(row.raw_transcript for row in rows[:3]).split()
        self.assertEqual(words, [f"word{index}" for index in range(total)])


class MigrationTests(unittest.TestCase):
    def test_existing_database_without_the_new_columns_is_migrated(self) -> None:
        import sqlite3

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session_transcript.db"
            # Build the pre-change schema by hand, without the two columns.
            legacy = sqlite3.connect(path)
            legacy.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
                    stopped_at TEXT, model_identity TEXT,
                    device_information TEXT, final_status TEXT NOT NULL
                );
                CREATE TABLE transcript_batches (
                    session_id TEXT NOT NULL, batch_id TEXT PRIMARY KEY,
                    sequence_number INTEGER NOT NULL,
                    raw_transcript TEXT NOT NULL, masked_transcript TEXT,
                    entities_json TEXT, status TEXT NOT NULL,
                    endpoint_reason TEXT, reliable INTEGER NOT NULL,
                    dropped_audio_chunks INTEGER NOT NULL DEFAULT 0,
                    audio_started_at TEXT, audio_ended_at TEXT,
                    stt_started_at TEXT, stt_completed_at TEXT,
                    mask_started_at TEXT, mask_completed_at TEXT,
                    captured_audio_seconds REAL, submitted_audio_seconds REAL,
                    stt_queue_wait_seconds REAL, stt_processing_seconds REAL,
                    stt_finalization_seconds REAL, stt_rtf REAL,
                    mask_latency_seconds REAL, total_latency_seconds REAL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    error TEXT,
                    UNIQUE(session_id, sequence_number),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );
                """
            )
            legacy.commit()
            legacy.close()

            store = SQLiteTranscriptStore(path)
            store.start()
            columns = None
            try:
                store.start_session("s", utc_now_iso())
                store.drain()
                probe = sqlite3.connect(path)
                columns = {
                    str(row[1])
                    for row in probe.execute(
                        "PRAGMA table_info(transcript_batches)"
                    ).fetchall()
                }
                probe.close()
            finally:
                store.shutdown(checkpoint=False)
            self.assertIn("logical_utterance_id", columns)
            self.assertIn("continuation_part_index", columns)


if __name__ == "__main__":
    unittest.main()

"""Single-owner asynchronous SQLite persistence for transcript batches."""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .schemas import RecordedUtterance, STTResult


class StorageWriterError(RuntimeError):
    """Raised outside the writer after an asynchronous database failure."""


@dataclass(frozen=True)
class StartSession:
    session_id: str
    started_at: str
    model_identity: str
    device_information: str


@dataclass(frozen=True)
class InsertRawTranscript:
    result: STTResult


@dataclass(frozen=True)
class AppendRawTranscript:
    session_id: str
    batch_id: str
    raw_fragment: str
    updated_at: str


@dataclass(frozen=True)
class InsertSTTFailure:
    recording: RecordedUtterance
    stt_started_at: str
    stt_completed_at: str
    error: str


@dataclass(frozen=True)
class CompleteMaskedTranscript:
    session_id: str
    batch_id: str
    masked_transcript: str
    entities: list[dict[str, Any]]
    mask_started_at: str
    mask_completed_at: str
    mask_latency_seconds: float
    total_latency_seconds: float


@dataclass(frozen=True)
class MarkMaskFailed:
    session_id: str
    batch_id: str
    mask_started_at: str
    mask_completed_at: str
    mask_latency_seconds: float
    total_latency_seconds: float
    error: str


@dataclass(frozen=True)
class FinishSession:
    session_id: str
    stopped_at: str
    final_status: str


@dataclass(frozen=True)
class QuerySession:
    session_id: str
    response: queue.Queue[Any]


@dataclass(frozen=True)
class Checkpoint:
    response: queue.Queue[Any]


@dataclass(frozen=True)
class StopWriter:
    pass


StorageOperation = (
    StartSession
    | InsertRawTranscript
    | AppendRawTranscript
    | InsertSTTFailure
    | CompleteMaskedTranscript
    | MarkMaskFailed
    | FinishSession
    | QuerySession
    | Checkpoint
    | StopWriter
)


@dataclass(frozen=True)
class QueuedStorageOperation:
    operation: StorageOperation
    enqueued_monotonic: float


@dataclass(frozen=True)
class StoredTranscriptBatch:
    session_id: str
    batch_id: str
    sequence_number: int
    raw_transcript: str
    masked_transcript: str | None
    entities: list[dict[str, Any]] | None
    status: str
    endpoint_reason: str | None
    reliable: bool
    dropped_audio_chunks: int
    audio_started_at: str | None
    audio_ended_at: str | None
    stt_started_at: str | None
    stt_completed_at: str | None
    mask_started_at: str | None
    mask_completed_at: str | None
    captured_audio_seconds: float | None
    submitted_audio_seconds: float | None
    stt_queue_wait_seconds: float | None
    stt_processing_seconds: float | None
    stt_finalization_seconds: float | None
    stt_rtf: float | None
    mask_latency_seconds: float | None
    total_latency_seconds: float | None
    created_at: str
    updated_at: str
    error: str | None
    # Safety-split relationship. sequence_number stays chronological, so these
    # identify which logical utterance a row belongs to and which part it is.
    logical_utterance_id: str | None = None
    continuation_part_index: int | None = None


def sanitize_error(error: str, *, limit: int = 300) -> str:
    """Store a bounded diagnostic without multiline or transcript payloads."""

    return " ".join(str(error).split())[:limit] or "operation failed"


class SQLiteTranscriptStore:
    """One writer thread owns the only connection for the database lifetime."""

    def __init__(
        self,
        path: Path,
        *,
        busy_timeout_ms: int = 5000,
        clock: Callable[[], float] = time.monotonic,
        connection_factory: Callable[..., sqlite3.Connection] = sqlite3.connect,
    ):
        self.path = Path(path).expanduser().resolve()
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.clock = clock
        self.connection_factory = connection_factory
        self.operation_queue: queue.Queue[QueuedStorageOperation] = queue.Queue()
        self.thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._error_lock = threading.Lock()
        self._first_error: str | None = None
        self._stopping = False
        self.writer_thread_id: int | None = None
        self.connection_thread_id: int | None = None
        self.queue_high_water_mark = 0
        self.operation_count = 0
        self.rejected_operation_count = 0
        self.total_queue_wait_seconds = 0.0
        self.maximum_queue_wait_seconds = 0.0
        self.total_write_seconds = 0.0
        self.maximum_write_seconds = 0.0
        self.journal_mode = ""
        self.synchronous_level: int | None = None

    @property
    def first_error(self) -> str | None:
        with self._error_lock:
            return self._first_error

    def _set_error(self, exc: BaseException) -> None:
        message = sanitize_error(f"{type(exc).__name__}: {exc}")
        with self._error_lock:
            if self._first_error is None:
                self._first_error = message

    def _raise_if_failed(self) -> None:
        error = self.first_error
        if error is not None:
            raise StorageWriterError(f"SQLite writer failed: {error}")

    def start(self) -> None:
        if self.thread is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(
            target=self._writer_loop,
            name="omit-sqlite-writer",
            daemon=False,
        )
        self.thread.start()
        if not self._ready.wait(timeout=10.0):
            raise StorageWriterError("SQLite writer did not become ready")
        self._raise_if_failed()

    def enqueue(self, operation: StorageOperation) -> None:
        if self.thread is None:
            raise StorageWriterError("SQLite writer has not started")
        if self._stopping:
            raise StorageWriterError("SQLite writer is stopping")
        if self.first_error is not None:
            self.rejected_operation_count += 1
            self._raise_if_failed()
        item = QueuedStorageOperation(operation, self.clock())
        self.operation_queue.put(item)
        self.queue_high_water_mark = max(
            self.queue_high_water_mark, self.operation_queue.qsize()
        )

    def start_session(
        self,
        session_id: str,
        started_at: str,
        *,
        model_identity: str = "",
        device_information: str = "",
    ) -> None:
        self.enqueue(
            StartSession(
                session_id=session_id,
                started_at=started_at,
                model_identity=model_identity,
                device_information=device_information,
            )
        )

    def insert_raw(self, result: STTResult) -> None:
        self.enqueue(InsertRawTranscript(result))

    def append_raw(
        self, session_id: str, batch_id: str, raw_fragment: str, updated_at: str
    ) -> None:
        self.enqueue(
            AppendRawTranscript(session_id, batch_id, raw_fragment, updated_at)
        )

    def insert_stt_failure(
        self,
        recording: RecordedUtterance,
        stt_started_at: str,
        stt_completed_at: str,
        error: str,
    ) -> None:
        self.enqueue(
            InsertSTTFailure(
                recording,
                stt_started_at,
                stt_completed_at,
                sanitize_error(error),
            )
        )

    def complete_masked(self, operation: CompleteMaskedTranscript) -> None:
        self.enqueue(operation)

    def mark_mask_failed(self, operation: MarkMaskFailed) -> None:
        self.enqueue(operation)

    def finish_session(
        self, session_id: str, stopped_at: str, final_status: str
    ) -> None:
        self.enqueue(FinishSession(session_id, stopped_at, final_status))

    def drain(self) -> None:
        self.operation_queue.join()
        self._raise_if_failed()

    def query_session(self, session_id: str) -> list[StoredTranscriptBatch]:
        response: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.enqueue(QuerySession(session_id, response))
        value = response.get(timeout=10.0)
        if isinstance(value, BaseException):
            raise StorageWriterError(str(value))
        return value

    def checkpoint(self) -> None:
        response: queue.Queue[Any] = queue.Queue(maxsize=1)
        self.enqueue(Checkpoint(response))
        value = response.get(timeout=10.0)
        if isinstance(value, BaseException):
            raise StorageWriterError(str(value))

    def shutdown(self, *, checkpoint: bool = True) -> None:
        if self.thread is None or self._closed.is_set():
            return
        pending_error: BaseException | None = None
        try:
            self.drain()
            if checkpoint:
                self.checkpoint()
                self.drain()
        except BaseException as exc:
            pending_error = exc
        self._stopping = True
        self.operation_queue.put(
            QueuedStorageOperation(StopWriter(), self.clock())
        )
        self.operation_queue.join()
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            raise StorageWriterError("SQLite writer did not stop")
        if pending_error is not None:
            raise pending_error
        self._raise_if_failed()

    def audit(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "writer_thread_id": self.writer_thread_id,
            "connection_thread_id": self.connection_thread_id,
            "journal_mode": self.journal_mode,
            "synchronous_level": self.synchronous_level,
            "operation_count": self.operation_count,
            "queue_high_water_mark": self.queue_high_water_mark,
            "rejected_operation_count": self.rejected_operation_count,
            "total_queue_wait_seconds": self.total_queue_wait_seconds,
            "maximum_queue_wait_seconds": self.maximum_queue_wait_seconds,
            "total_write_seconds": self.total_write_seconds,
            "maximum_write_seconds": self.maximum_write_seconds,
            "first_error": self.first_error,
        }

    def _writer_loop(self) -> None:
        connection: sqlite3.Connection | None = None
        self.writer_thread_id = threading.get_ident()
        try:
            connection = self.connection_factory(str(self.path))
            self.connection_thread_id = threading.get_ident()
            connection.row_factory = sqlite3.Row
            self.journal_mode = str(
                connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            ).lower()
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
            self.synchronous_level = int(
                connection.execute("PRAGMA synchronous").fetchone()[0]
            )
            self._create_schema(connection)
            connection.commit()
            self._protect_files()
        except BaseException as exc:
            self._set_error(exc)
            self._ready.set()
            if connection is not None:
                connection.close()
            self._closed.set()
            return
        self._ready.set()

        while True:
            item = self.operation_queue.get()
            should_stop = isinstance(item.operation, StopWriter)
            try:
                queue_wait = max(
                    0.0, self.clock() - item.enqueued_monotonic
                )
                self.total_queue_wait_seconds += queue_wait
                self.maximum_queue_wait_seconds = max(
                    self.maximum_queue_wait_seconds, queue_wait
                )
                if should_stop:
                    break
                if self.first_error is not None:
                    self._respond_to_rejected_query(item.operation)
                    continue
                started = time.perf_counter()
                try:
                    self._execute(connection, item.operation)
                    connection.commit()
                    self.operation_count += 1
                except BaseException as exc:
                    connection.rollback()
                    self._set_error(exc)
                    self._respond_to_failed_query(item.operation, exc)
                finally:
                    elapsed = time.perf_counter() - started
                    self.total_write_seconds += elapsed
                    self.maximum_write_seconds = max(
                        self.maximum_write_seconds, elapsed
                    )
            finally:
                self.operation_queue.task_done()
            if should_stop:
                break

        try:
            connection.commit()
            self._protect_files()
        except BaseException as exc:
            self._set_error(exc)
        finally:
            connection.close()
            self._closed.set()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                stopped_at TEXT,
                model_identity TEXT,
                device_information TEXT,
                final_status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transcript_batches (
                session_id TEXT NOT NULL,
                batch_id TEXT PRIMARY KEY,
                sequence_number INTEGER NOT NULL,
                raw_transcript TEXT NOT NULL,
                masked_transcript TEXT,
                entities_json TEXT,
                status TEXT NOT NULL,
                endpoint_reason TEXT,
                reliable INTEGER NOT NULL,
                dropped_audio_chunks INTEGER NOT NULL DEFAULT 0,
                audio_started_at TEXT,
                audio_ended_at TEXT,
                stt_started_at TEXT,
                stt_completed_at TEXT,
                mask_started_at TEXT,
                mask_completed_at TEXT,
                captured_audio_seconds REAL,
                submitted_audio_seconds REAL,
                stt_queue_wait_seconds REAL,
                stt_processing_seconds REAL,
                stt_finalization_seconds REAL,
                stt_rtf REAL,
                mask_latency_seconds REAL,
                total_latency_seconds REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT,
                logical_utterance_id TEXT,
                continuation_part_index INTEGER,
                UNIQUE(session_id, sequence_number),
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            """
        )
        # Additive migration for databases created before the safety-split
        # metadata existed. sequence_number stays strictly chronological, so
        # the split relationship lives in these two nullable columns instead.
        existing = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(transcript_batches)"
            ).fetchall()
        }
        for column, declaration in (
            ("logical_utterance_id", "TEXT"),
            ("continuation_part_index", "INTEGER"),
        ):
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE transcript_batches ADD COLUMN {column} {declaration}"
                )

    def _execute(
        self, connection: sqlite3.Connection, operation: StorageOperation
    ) -> None:
        if threading.get_ident() != self.connection_thread_id:
            raise RuntimeError("SQLite connection accessed outside its owner thread")
        if isinstance(operation, StartSession):
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, started_at, model_identity,
                    device_information, final_status
                ) VALUES (?, ?, ?, ?, 'running')
                """,
                (
                    operation.session_id,
                    operation.started_at,
                    operation.model_identity,
                    operation.device_information,
                ),
            )
            return
        if isinstance(operation, InsertRawTranscript):
            result = operation.result
            connection.execute(
                """
                INSERT INTO transcript_batches (
                    session_id, batch_id, sequence_number, raw_transcript,
                    masked_transcript, entities_json, status, endpoint_reason,
                    reliable, dropped_audio_chunks, audio_started_at,
                    audio_ended_at, stt_started_at, stt_completed_at,
                    captured_audio_seconds, submitted_audio_seconds,
                    stt_queue_wait_seconds, stt_processing_seconds,
                    stt_finalization_seconds, stt_rtf, created_at, updated_at,
                    logical_utterance_id, continuation_part_index
                ) VALUES (
                    ?, ?, ?, ?, NULL, NULL, 'waiting_for_mask', ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    result.session_id,
                    result.batch_id,
                    result.sequence_number,
                    result.final_text,
                    result.endpoint_reason,
                    int(result.reliable),
                    result.dropped_audio_chunks,
                    result.audio_started_at_utc,
                    result.audio_ended_at_utc,
                    result.stt_started_at_utc,
                    result.stt_completed_at_utc,
                    result.captured_audio_seconds,
                    result.submitted_audio_seconds,
                    result.queue_wait_seconds,
                    result.processing_seconds,
                    result.finalization_seconds,
                    result.rtf,
                    result.stt_completed_at_utc,
                    result.stt_completed_at_utc,
                    result.logical_utterance_id or result.batch_id,
                    result.continuation_part_index,
                ),
            )
            return
        if isinstance(operation, AppendRawTranscript):
            cursor = connection.execute(
                """
                UPDATE transcript_batches
                SET raw_transcript = CASE
                    WHEN raw_transcript = '' THEN ?
                    ELSE raw_transcript || ' ' || ?
                END,
                    updated_at = ?
                WHERE session_id = ? AND batch_id = ?
                """,
                (
                    operation.raw_fragment,
                    operation.raw_fragment,
                    operation.updated_at,
                    operation.session_id,
                    operation.batch_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Raw append target does not exist")
            return
        if isinstance(operation, InsertSTTFailure):
            recording = operation.recording
            connection.execute(
                """
                INSERT INTO transcript_batches (
                    session_id, batch_id, sequence_number, raw_transcript,
                    status, endpoint_reason, reliable, dropped_audio_chunks,
                    audio_started_at, audio_ended_at, stt_started_at,
                    stt_completed_at, captured_audio_seconds,
                    submitted_audio_seconds, created_at, updated_at, error,
                    logical_utterance_id, continuation_part_index
                ) VALUES (
                    ?, ?, ?, '', 'stt_failed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?
                )
                """,
                (
                    recording.session_id,
                    recording.batch_id,
                    recording.sequence_number,
                    recording.endpoint_reason,
                    int(recording.reliable),
                    recording.dropped_audio_chunks,
                    recording.audio_started_at_utc,
                    recording.audio_ended_at_utc,
                    operation.stt_started_at,
                    operation.stt_completed_at,
                    recording.captured_audio_seconds,
                    recording.submitted_audio_seconds,
                    operation.stt_completed_at,
                    operation.stt_completed_at,
                    sanitize_error(operation.error),
                    recording.logical_utterance_id or recording.batch_id,
                    recording.continuation_part_index,
                ),
            )
            return
        if isinstance(operation, CompleteMaskedTranscript):
            cursor = connection.execute(
                """
                UPDATE transcript_batches
                SET masked_transcript = ?, entities_json = ?, status = 'complete',
                    mask_started_at = ?, mask_completed_at = ?,
                    mask_latency_seconds = ?, total_latency_seconds = ?,
                    updated_at = ?, error = NULL
                WHERE session_id = ? AND batch_id = ?
                """,
                (
                    operation.masked_transcript,
                    json.dumps(operation.entities, ensure_ascii=False),
                    operation.mask_started_at,
                    operation.mask_completed_at,
                    operation.mask_latency_seconds,
                    operation.total_latency_seconds,
                    operation.mask_completed_at,
                    operation.session_id,
                    operation.batch_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Mask completion target does not exist")
            return
        if isinstance(operation, MarkMaskFailed):
            cursor = connection.execute(
                """
                UPDATE transcript_batches
                SET masked_transcript = NULL, entities_json = NULL,
                    status = 'mask_failed', mask_started_at = ?,
                    mask_completed_at = ?, mask_latency_seconds = ?,
                    total_latency_seconds = ?, updated_at = ?, error = ?
                WHERE session_id = ? AND batch_id = ?
                """,
                (
                    operation.mask_started_at,
                    operation.mask_completed_at,
                    operation.mask_latency_seconds,
                    operation.total_latency_seconds,
                    operation.mask_completed_at,
                    sanitize_error(operation.error),
                    operation.session_id,
                    operation.batch_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Mask failure target does not exist")
            return
        if isinstance(operation, FinishSession):
            cursor = connection.execute(
                """
                UPDATE sessions
                SET stopped_at = ?, final_status = ?
                WHERE session_id = ?
                """,
                (
                    operation.stopped_at,
                    operation.final_status,
                    operation.session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Session completion target does not exist")
            return
        if isinstance(operation, QuerySession):
            rows = connection.execute(
                """
                SELECT * FROM transcript_batches
                WHERE session_id = ?
                ORDER BY sequence_number
                """,
                (operation.session_id,),
            ).fetchall()
            operation.response.put([self._row_to_record(row) for row in rows])
            return
        if isinstance(operation, Checkpoint):
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            operation.response.put(True)
            return
        raise TypeError(f"Unsupported SQLite operation: {type(operation).__name__}")

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> StoredTranscriptBatch:
        entities = (
            json.loads(row["entities_json"])
            if row["entities_json"] is not None
            else None
        )
        return StoredTranscriptBatch(
            session_id=row["session_id"],
            batch_id=row["batch_id"],
            sequence_number=row["sequence_number"],
            raw_transcript=row["raw_transcript"],
            masked_transcript=row["masked_transcript"],
            entities=entities,
            status=row["status"],
            endpoint_reason=row["endpoint_reason"],
            reliable=bool(row["reliable"]),
            dropped_audio_chunks=row["dropped_audio_chunks"],
            audio_started_at=row["audio_started_at"],
            audio_ended_at=row["audio_ended_at"],
            stt_started_at=row["stt_started_at"],
            stt_completed_at=row["stt_completed_at"],
            mask_started_at=row["mask_started_at"],
            mask_completed_at=row["mask_completed_at"],
            captured_audio_seconds=row["captured_audio_seconds"],
            submitted_audio_seconds=row["submitted_audio_seconds"],
            stt_queue_wait_seconds=row["stt_queue_wait_seconds"],
            stt_processing_seconds=row["stt_processing_seconds"],
            stt_finalization_seconds=row["stt_finalization_seconds"],
            stt_rtf=row["stt_rtf"],
            mask_latency_seconds=row["mask_latency_seconds"],
            total_latency_seconds=row["total_latency_seconds"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error=row["error"],
            logical_utterance_id=row["logical_utterance_id"],
            continuation_part_index=row["continuation_part_index"],
        )

    @staticmethod
    def _respond_to_failed_query(
        operation: StorageOperation, exc: BaseException
    ) -> None:
        if isinstance(operation, (QuerySession, Checkpoint)):
            operation.response.put(exc)

    def _respond_to_rejected_query(self, operation: StorageOperation) -> None:
        if isinstance(operation, (QuerySession, Checkpoint)):
            operation.response.put(
                StorageWriterError(self.first_error or "SQLite writer failed")
            )

    def _protect_files(self) -> None:
        if os.name == "nt":
            return
        for path in (
            self.path,
            Path(f"{self.path}-wal"),
            Path(f"{self.path}-shm"),
        ):
            if path.exists():
                os.chmod(path, 0o600)

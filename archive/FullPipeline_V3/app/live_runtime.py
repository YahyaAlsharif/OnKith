"""Serialized Canary STT grouping plus preserved immediate PII workers."""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from app.canary_stt_runtime import LiveSTTRuntime, deduplicate_boundary
from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.inactivity_buffer import InactivityTranscriptBuffer
from integrated_pipeline.pipeline import FailClosedMasker
from integrated_pipeline.schemas import (
    BufferOutput,
    LiveOutput,
    MaskTask,
    RecordedUtterance,
    STTResult,
    TranscriptFragment,
)
from integrated_pipeline.sqlite_storage import (
    CompleteMaskedTranscript,
    MarkMaskFailed,
    SQLiteTranscriptStore,
    StorageWriterError,
    sanitize_error,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class WallClockBufferWorker:
    """Legacy compatibility worker; the live runtime never instantiates it."""

    def __init__(
        self,
        config: PipelineConfig,
        on_flush: Callable[[BufferOutput], None],
        *,
        clock: Callable[[], float] = time.monotonic,
        run_thread: bool = True,
    ):
        self.clock = clock
        self.on_flush = on_flush
        self.buffer = InactivityTranscriptBuffer(
            inactivity_seconds=config.buffer_inactivity_seconds,
            maximum_tokens=config.maximum_buffer_tokens,
            maximum_duration_seconds=config.maximum_buffer_duration_seconds,
        )
        self.run_thread = run_thread
        self._condition = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.run_thread or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._timer_loop,
            name="omit-buffer-timer",
            daemon=False,
        )
        self._thread.start()

    def add_fragment(self, fragment: TranscriptFragment) -> list[BufferOutput]:
        with self._condition:
            outputs = self.buffer.add_fragment(fragment)
            self._condition.notify_all()
        for output in outputs:
            self.on_flush(output)
        return outputs

    def poll(self) -> BufferOutput | None:
        with self._condition:
            output = self.buffer.flush_due(self.clock())
        if output is not None:
            self.on_flush(output)
        return output

    def _timer_loop(self) -> None:
        while True:
            output: BufferOutput | None = None
            with self._condition:
                if self._stopping:
                    return
                deadline = self.buffer.next_deadline
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - self.clock()
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                output = self.buffer.flush_due(self.clock())
            if output is not None:
                self.on_flush(output)

    def shutdown(self) -> BufferOutput | None:
        with self._condition:
            output = self.buffer.flush(self.clock(), reason="shutdown")
            self._stopping = True
            self._condition.notify_all()
        if output is not None:
            self.on_flush(output)
        if self._thread is not None:
            self._thread.join()
        return output


class LivePrivacyRuntime:
    """Immediately mask each finalized recording on a separate worker."""

    def __init__(
        self,
        config: PipelineConfig,
        masker: FailClosedMasker,
        storage: SQLiteTranscriptStore,
        *,
        on_event: Callable[[str, dict[str, Any]], None],
        on_output: Callable[[LiveOutput], None],
        queue_overflow_count: Callable[[], int] = lambda: 0,
        clock: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], str] = utc_now_iso,
    ):
        if config.stage5["mode"] != "immediate":
            raise ValueError("The live Stage 5 mode must be immediate")
        self.config = config
        self.masker = masker
        self.storage = storage
        self.on_event = on_event
        self.on_output = on_output
        self.queue_overflow_count = queue_overflow_count
        self.clock = clock
        self.utc_now = utc_now
        self.mask_queue: queue.Queue[MaskTask | None] = queue.Queue()
        self._mask_queue = self.mask_queue
        self.thread: threading.Thread | None = None
        self._output_condition = threading.Condition()
        self._output_count = 0
        self.errors: list[str] = []
        self.queue_high_water_mark = 0
        self._shutdown_complete = False

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(
            target=self._mask_loop,
            name="omit-pii-worker",
            daemon=False,
        )
        self.thread.start()

    def submit_stt(self, result: STTResult) -> None:
        """Own one finalized transcript and fan it out without worker waits."""

        text = (result.final_text or "").strip()
        if not text:
            self.on_event(
                "empty_final_ignored", {"batch_id": result.batch_id}
            )
            return
        output, enqueued_monotonic = self._create_immediate_buffer_output(
            result, text
        )
        # The writer API only enqueues a lightweight FIFO operation. Stage 5
        # does not wait for the INSERT or its commit before masking.
        try:
            self.storage.insert_raw(result)
        except StorageWriterError as exc:
            self._database_error(exc)
        self.on_event(
            "stage5_submitted",
            {"batch_id": result.batch_id},
        )
        self.mask_queue.put(MaskTask(output, enqueued_monotonic))
        self.queue_high_water_mark = max(
            self.queue_high_water_mark, self.mask_queue.qsize()
        )

    def submit_stt_failure(
        self,
        recording: RecordedUtterance,
        stt_started_at: str,
        stt_completed_at: str,
        error: str,
    ) -> None:
        """Keep all transcript-storage submission owned by Stage 5."""

        try:
            self.storage.insert_stt_failure(
                recording,
                stt_started_at,
                stt_completed_at,
                error,
            )
        except StorageWriterError as exc:
            self._database_error(exc)

    def _create_immediate_buffer_output(
        self,
        result: STTResult,
        text: str,
    ) -> tuple[BufferOutput, float]:
        """Materialize the zero-wait Stage 5 batch before either fan-out."""

        now = self.clock()
        fragment = TranscriptFragment(
            fragment_id=f"{result.batch_id}-fragment-000001",
            text=text,
            arrival_time_seconds=now,
            source_audio_id=result.batch_id,
            stt_result=result,
            session_id=result.session_id,
            batch_id=result.batch_id,
            sequence_number=result.sequence_number,
        )
        output = BufferOutput(
            buffer_id=result.batch_id,
            text=text,
            fragments=[fragment],
            started_at_seconds=now,
            flushed_at_seconds=now,
            flush_reason="finalized_recording",
            session_id=result.session_id,
            batch_id=result.batch_id,
            sequence_number=result.sequence_number,
        )
        return output, now

    def _database_error(self, exc: BaseException) -> None:
        self.on_event(
            "database_error",
            {"error": sanitize_error(f"{type(exc).__name__}: {exc}")},
        )

    def _mask_loop(self) -> None:
        while True:
            task = self.mask_queue.get()
            try:
                if task is None:
                    return
                mask_started_monotonic = self.clock()
                mask_started_at = self.utc_now()
                self.on_event(
                    "masking_started",
                    {"batch_id": task.buffer.batch_id},
                )
                pii = self.masker.mask_text(task.buffer.text)
                mask_completed_at = self.utc_now()
                displayed_at_monotonic = self.clock()
                mask_queue_wait = max(
                    0.0,
                    mask_started_monotonic - task.enqueued_monotonic,
                )
                stt_result = task.buffer.fragments[0].stt_result
                audio_ended = (
                    stt_result.audio_ended_monotonic
                    if stt_result is not None
                    else task.enqueued_monotonic
                )
                total_latency = max(
                    0.0, displayed_at_monotonic - audio_ended
                )
                result = LiveOutput(
                    buffer=task.buffer,
                    pii=pii,
                    total_seconds=total_latency,
                    queue_overflow_count=self.queue_overflow_count(),
                    mask_started_at_utc=mask_started_at,
                    mask_completed_at_utc=mask_completed_at,
                    mask_queue_wait_seconds=mask_queue_wait,
                )
                try:
                    # Privacy-visible output must complete before its database
                    # completion operation is enqueued.
                    self.on_output(result)
                except Exception as exc:
                    message = sanitize_error(
                        f"{type(exc).__name__}: terminal output failed"
                    )
                    self.errors.append(
                        f"{task.buffer.batch_id}: {message}"
                    )
                    self.on_event(
                        "mask_output_error",
                        {
                            "batch_id": task.buffer.batch_id,
                            "error": message,
                        },
                    )
                    continue

                try:
                    if pii.safe_to_release:
                        self.storage.complete_masked(
                            CompleteMaskedTranscript(
                                session_id=task.buffer.session_id,
                                batch_id=task.buffer.batch_id,
                                masked_transcript=pii.masked_text,
                                entities=pii.entities,
                                mask_started_at=mask_started_at,
                                mask_completed_at=mask_completed_at,
                                mask_latency_seconds=pii.latency_seconds,
                                total_latency_seconds=total_latency,
                            )
                        )
                    else:
                        self.storage.mark_mask_failed(
                            MarkMaskFailed(
                                session_id=task.buffer.session_id,
                                batch_id=task.buffer.batch_id,
                                mask_started_at=mask_started_at,
                                mask_completed_at=mask_completed_at,
                                mask_latency_seconds=pii.latency_seconds,
                                total_latency_seconds=total_latency,
                                error=pii.error or "masking operation failed",
                            )
                        )
                except StorageWriterError as exc:
                    self._database_error(exc)
                with self._output_condition:
                    self._output_count += 1
                    self._output_condition.notify_all()
            except Exception as exc:
                batch_id = (
                    task.buffer.batch_id
                    if isinstance(task, MaskTask)
                    else "unknown"
                )
                message = sanitize_error(f"{type(exc).__name__}: {exc}")
                self.errors.append(f"{batch_id}: {message}")
                self.on_event(
                    "mask_output_error",
                    {"batch_id": batch_id, "error": message},
                )
            finally:
                self.mask_queue.task_done()

    def wait_for_output(
        self, previous_count: int = 0, timeout: float = 4.0
    ) -> bool:
        deadline = time.monotonic() + timeout
        with self._output_condition:
            while self._output_count <= previous_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._output_condition.wait(timeout=remaining)
            return True

    def shutdown(self) -> None:
        if self._shutdown_complete:
            if self.errors:
                raise RuntimeError(
                    "PII worker failure: " + "; ".join(self.errors)
                )
            return
        self.mask_queue.join()
        self.mask_queue.put(None)
        if self.thread is not None:
            self.thread.join()
        self._shutdown_complete = True
        if self.errors:
            raise RuntimeError(
                "PII worker failure: " + "; ".join(self.errors)
            )


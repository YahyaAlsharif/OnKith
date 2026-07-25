"""Real wall-clock transcript timer and independent PII worker."""

from __future__ import annotations

import itertools
import queue
import threading
import time
from typing import Any, Callable

from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.inactivity_buffer import InactivityTranscriptBuffer
from integrated_pipeline.pipeline import FailClosedMasker
from integrated_pipeline.schemas import (
    BufferOutput,
    LiveOutput,
    STTResult,
    TranscriptFragment,
)


class WallClockBufferWorker:
    """Flush a transcript on a real deadline even if no new text arrives."""

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
            daemon=True,
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
        """Deterministic timer step for an injected fake clock."""

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
            self._thread.join(timeout=2.0)
        return output


class LivePrivacyRuntime:
    """Never run TinyBERT on the audio worker or buffer timer thread."""

    def __init__(
        self,
        config: PipelineConfig,
        masker: FailClosedMasker,
        *,
        on_event: Callable[[str, dict[str, Any]], None],
        on_output: Callable[[LiveOutput], None],
        queue_overflow_count: Callable[[], int] = lambda: 0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.masker = masker
        self.on_event = on_event
        self.on_output = on_output
        self.queue_overflow_count = queue_overflow_count
        self.clock = clock
        self._sequence = itertools.count(1)
        self._mask_queue: queue.Queue[BufferOutput | None] = queue.Queue()
        self._mask_thread: threading.Thread | None = None
        self._output_condition = threading.Condition()
        self._output_count = 0
        self.buffer_worker = WallClockBufferWorker(
            config,
            self._enqueue_buffer,
            clock=clock,
            run_thread=True,
        )

    def start(self) -> None:
        if self._mask_thread is None:
            self._mask_thread = threading.Thread(
                target=self._mask_loop,
                name="omit-pii-worker",
                daemon=True,
            )
            self._mask_thread.start()
        self.buffer_worker.start()

    def submit_stt(self, result: STTResult) -> None:
        """Only finalized, nonempty Moonshine text can enter the buffer."""

        text = (result.final_text or "").strip()
        if not text:
            self.on_event("empty_final_ignored", {})
            return
        now = self.clock()
        sequence = next(self._sequence)
        fragment = TranscriptFragment(
            fragment_id=f"fragment-{sequence:06d}",
            text=text,
            arrival_time_seconds=now,
            source_audio_id=f"live-{sequence:06d}",
            stt_result=result,
        )
        self.buffer_worker.add_fragment(fragment)
        pending = self.buffer_worker.buffer.pending_fragment_count
        if pending:
            self.on_event(
                "buffer_waiting",
                {
                    "seconds": self.config.buffer_inactivity_seconds,
                    "pending_fragments": pending,
                },
            )

    def _enqueue_buffer(self, output: BufferOutput) -> None:
        self._mask_queue.put(output)

    def _mask_loop(self) -> None:
        while True:
            output = self._mask_queue.get()
            try:
                if output is None:
                    return
                pii = self.masker.mask_text(output.text)
                stt_seconds = sum(
                    (
                        fragment.stt_result.processing_seconds
                        + fragment.stt_result.finalization_seconds
                    )
                    for fragment in output.fragments
                    if fragment.stt_result is not None
                )
                result = LiveOutput(
                    buffer=output,
                    pii=pii,
                    total_seconds=(
                        stt_seconds
                        + output.buffer_waiting_seconds
                        + pii.latency_seconds
                    ),
                    queue_overflow_count=self.queue_overflow_count(),
                )
                self.on_output(result)
                with self._output_condition:
                    self._output_count += 1
                    self._output_condition.notify_all()
            finally:
                self._mask_queue.task_done()

    def wait_for_output(self, previous_count: int = 0, timeout: float = 4.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._output_condition:
            while self._output_count <= previous_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._output_condition.wait(timeout=remaining)
            return True

    def shutdown(self) -> None:
        self.buffer_worker.shutdown()
        self._mask_queue.join()
        self._mask_queue.put(None)
        if self._mask_thread is not None:
            self._mask_thread.join(timeout=5.0)

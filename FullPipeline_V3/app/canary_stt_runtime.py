"""Serialized Canary segment decoding, utterance grouping, and final release."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from app.onkith_music import OnkithMusicWorker, intercept_onkith_command
from app.wake_word_normalizer import CANONICAL_WAKE_WORD, normalize_wake_word
from integrated_pipeline.config import (
    MAXIMUM_LOGICAL_UTTERANCE_SECONDS,
    MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS,
)
from integrated_pipeline.schemas import (
    CanarySegment,
    CanaryUtteranceEnd,
    RecordedUtterance,
    SegmentSummary,
    STTResult,
)
from integrated_pipeline.sqlite_storage import sanitize_error
from integrated_pipeline.transcript_merger import (
    EmptyTranscriptError,
    merge_segment_transcripts,
)


def deduplicate_boundary(
    previous_text: str,
    current_text: str,
    *,
    maximum_words: int = 14,
) -> tuple[str, int]:
    """Compatibility helper backed by the exact 14-word merger."""

    merged = merge_segment_transcripts(
        [previous_text, current_text], maximum_overlap_words=maximum_words
    )
    overlap = merged.decisions[-1].overlap_words
    return " ".join((current_text or "").strip().split()[overlap:]), overlap


def _interval_union_samples(segments: list[SegmentSummary]) -> int:
    intervals = sorted(
        (item.segment_start_sample, item.segment_end_sample) for item in segments
    )
    if not intervals:
        return 0
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


@dataclass
class _Accumulator:
    """Audio-free state for one *logical downstream* utterance.

    ``summaries`` holds only scalars and text, so retention is
    O(metadata + text) regardless of how long the speaker talks. Segment audio
    is released as soon as its inference completes.

    A logical utterance normally maps one-to-one onto an acoustic utterance.
    The pathological-speech safety limit can close a logical utterance early;
    later segments of the same acoustic utterance then continue in a fresh
    accumulator with ``part_index`` incremented.

    ``downstream_sequence_number`` is allocated chronologically at the moment
    the logical utterance opens, so persisted rows always sort in the order
    they were spoken.
    """

    utterance_id: str
    sequence_number: int
    summaries: list[SegmentSummary]
    batch_id: str
    downstream_sequence_number: int
    expected_next_index: int = 0
    part_index: int = 1

    @property
    def logical_utterance_id(self) -> str:
        """The acoustic utterance every part of this split belongs to."""

        return self.utterance_id

    @property
    def segment_count(self) -> int:
        return len(self.summaries)

    @property
    def submitted_span_seconds(self) -> float:
        """Span from the first submitted sample to the latest submitted one."""

        if not self.summaries:
            return 0.0
        first, last = self.summaries[0], self.summaries[-1]
        return (
            last.segment_end_sample - first.segment_start_sample
        ) / float(last.sample_rate)

    def exceeds_safety_limits(self) -> str | None:
        """Return which bound tripped, or None while inside both limits."""

        if self.segment_count >= MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS:
            return "maximum_segments"
        if self.submitted_span_seconds >= MAXIMUM_LOGICAL_UTTERANCE_SECONDS:
            return "maximum_duration"
        return None


def continuation_batch_id(utterance_id: str, part_index: int) -> str:
    """Unique batch id for one part of a logical utterance.

    Part 1 keeps the untouched acoustic utterance id, so an ordinary utterance
    is stored exactly as before. Later parts are suffixed, satisfying the
    ``batch_id PRIMARY KEY`` constraint without disturbing sequencing.
    """

    if part_index <= 1:
        return utterance_id
    return f"{utterance_id}-part-{part_index:06d}"


class LiveSTTRuntime:
    """Decode each segment independently and release one final merged utterance."""

    def __init__(
        self,
        completed_segment_queue: queue.Queue[
            CanarySegment | CanaryUtteranceEnd | None
        ],
        canary: Any,
        privacy: Any,
        *,
        on_event: Callable[[str, dict[str, Any]], None],
        music_worker: OnkithMusicWorker | None = None,
        utc_now: Callable[[], str],
        preview: Any | None = None,
    ) -> None:
        self.completed_segment_queue = completed_segment_queue
        self.completed_recording_queue = completed_segment_queue
        self.canary = canary
        self.privacy = privacy
        self.on_event = on_event
        self.music_worker = music_worker
        self.utc_now = utc_now
        # Display-only live preview. None means the authoritative path behaves
        # exactly as it did before this existed; nothing below changes what is
        # merged, intercepted or stored.
        self.preview = preview
        self.thread: threading.Thread | None = None
        self.errors: list[str] = []
        self.failure_count = 0
        self.final_utterance_count = 0
        self.safety_split_count = 0
        # Monotonic chronological numbering for everything this session
        # persists. One logical utterance takes one number at the moment it
        # opens, so a safety-split continuation is numbered before any later
        # independent utterance and query_session returns them in spoken order.
        # With no safety split this reproduces the acoustic numbering exactly.
        self._downstream_sequence = 0
        self._accumulator: _Accumulator | None = None
        self._discard_utterance_id: str | None = None
        self._shutdown_complete = False

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(
                target=self._worker_loop,
                name="omit-canary-stt-worker",
                daemon=False,
            )
            self.thread.start()

    def _next_downstream_sequence(self) -> int:
        """Take the next chronological session sequence number."""

        self._downstream_sequence += 1
        return self._downstream_sequence

    def _validate_order(self, segment: CanarySegment) -> None:
        active = self._accumulator
        if active is None:
            if segment.segment_index != 0:
                raise RuntimeError("A new utterance must begin at segment index 0")
            self._accumulator = _Accumulator(
                utterance_id=segment.utterance_id,
                sequence_number=segment.utterance_sequence_number,
                summaries=[],
                batch_id=continuation_batch_id(segment.utterance_id, 1),
                downstream_sequence_number=self._next_downstream_sequence(),
                expected_next_index=0,
                part_index=1,
            )
            return
        if segment.utterance_id != active.utterance_id:
            raise RuntimeError("A new utterance arrived before the active one ended")
        expected = active.expected_next_index
        if segment.segment_index != expected:
            raise RuntimeError(
                f"Out-of-order Canary segment {segment.segment_index}; expected {expected}"
            )
        if segment.utterance_sequence_number != active.sequence_number:
            raise RuntimeError("Inconsistent Canary utterance sequence metadata")

    @staticmethod
    def _failure_recording(
        segments: list[SegmentSummary],
        end: CanaryUtteranceEnd | None = None,
        *,
        batch_id: str | None = None,
        sequence_number: int | None = None,
        logical_utterance_id: str = "",
        continuation_part_index: int = 1,
    ) -> RecordedUtterance:
        first, final = segments[0], segments[-1]
        decision_sample = (
            final.utterance_captured_end_sample if end is None else end.decision_sample
        )
        captured = max(
            0, decision_sample - first.utterance_start_sample
        )
        return RecordedUtterance(
            session_id=final.session_id,
            batch_id=final.utterance_id if batch_id is None else batch_id,
            sequence_number=(
                final.utterance_sequence_number
                if sequence_number is None
                else sequence_number
            ),
            # The exact submitted duration is known from scalar sample
            # intervals, so no audio buffer is reconstructed here.
            samples=np.empty(0, dtype=np.float32),
            submitted_audio_seconds_override=(
                _interval_union_samples(segments) / float(final.sample_rate)
            ),
            sample_rate=final.sample_rate,
            audio_started_monotonic=first.audio_started_monotonic,
            audio_ended_monotonic=(
                final.audio_ended_monotonic if end is None else end.audio_ended_monotonic
            ),
            audio_started_at_utc=first.audio_started_at_utc,
            audio_ended_at_utc=(
                final.audio_ended_at_utc if end is None else end.audio_ended_at_utc
            ),
            captured_at_utc=(
                final.audio_ended_at_utc if end is None else end.audio_ended_at_utc
            ),
            recording_closed_monotonic=(
                final.recording_closed_monotonic
                if end is None
                else end.recording_closed_monotonic
            ),
            endpoint_reason=final.endpoint_reason if end is None else end.endpoint_reason,
            reliable=(
                all(item.reliable for item in segments)
                and (True if end is None else end.reliable)
            ),
            dropped_audio_chunks=max(
                [item.dropped_audio_chunks for item in segments]
                + ([] if end is None else [end.dropped_audio_chunks])
            ),
            captured_audio_seconds=captured / final.sample_rate,
            logical_utterance_id=(
                logical_utterance_id or final.utterance_id
            ),
            continuation_part_index=continuation_part_index,
        )

    def _record_failure(
        self,
        segments: list[SegmentSummary],
        started_at: str,
        message: str,
        end: CanaryUtteranceEnd | None = None,
        *,
        active: _Accumulator,
    ) -> None:
        """Store one failed logical utterance under its own chronological row."""

        self.failure_count += 1
        self.on_event(
            "stt_error",
            {"batch_id": active.batch_id, "error": message},
        )
        self.privacy.submit_stt_failure(
            self._failure_recording(
                segments,
                end,
                batch_id=active.batch_id,
                sequence_number=active.downstream_sequence_number,
                logical_utterance_id=active.logical_utterance_id,
                continuation_part_index=active.part_index,
            ),
            started_at,
            self.utc_now(),
            message,
        )

    def _finalize(
        self,
        active: _Accumulator,
        end: CanaryUtteranceEnd | None = None,
        *,
        endpoint_reason_override: str | None = None,
    ) -> STTResult:
        started = time.perf_counter()
        segments = active.summaries
        merge = merge_segment_transcripts([item.text for item in segments])
        first, final = segments[0], segments[-1]
        processing = sum(item.processing_seconds for item in segments)
        submitted = sum(item.submitted_audio_seconds for item in segments)
        unique = _interval_union_samples(segments) / float(final.sample_rate)
        decision_sample = (
            final.utterance_captured_end_sample if end is None else end.decision_sample
        )
        captured = max(0, decision_sample - first.utterance_start_sample) / float(
            final.sample_rate
        )
        result = STTResult(
            final_text=merge.text,
            duration_seconds=unique,
            processing_seconds=processing,
            finalization_seconds=0.0,
            rtf=processing / unique if unique else float("nan"),
            callback_count=0,
            reliable=(
                all(item.reliable for item in segments)
                and (True if end is None else end.reliable)
            ),
            session_id=final.session_id,
            batch_id=active.batch_id,
            sequence_number=active.downstream_sequence_number,
            endpoint_reason=(
                endpoint_reason_override
                if endpoint_reason_override is not None
                else (final.endpoint_reason if end is None else end.endpoint_reason)
            ),
            dropped_audio_chunks=max(
                [item.dropped_audio_chunks for item in segments]
                + ([] if end is None else [end.dropped_audio_chunks])
            ),
            audio_started_monotonic=first.audio_started_monotonic,
            audio_ended_monotonic=(
                final.audio_ended_monotonic if end is None else end.audio_ended_monotonic
            ),
            recording_closed_monotonic=(
                final.recording_closed_monotonic
                if end is None
                else end.recording_closed_monotonic
            ),
            audio_started_at_utc=first.audio_started_at_utc,
            audio_ended_at_utc=(
                final.audio_ended_at_utc if end is None else end.audio_ended_at_utc
            ),
            stt_started_at_utc=segments[0].stt_started_at_utc,
            stt_completed_at_utc=self.utc_now(),
            captured_audio_seconds=captured,
            submitted_audio_seconds=submitted,
            unique_submitted_audio_seconds=unique,
            submitted_audio_rtf=(
                processing / submitted if submitted else float("nan")
            ),
            queue_wait_seconds=sum(item.queue_wait_seconds for item in segments),
            total_stt_seconds=0.0,
            deduplicated_word_count=merge.deduplicated_word_count,
            segment_count=len(segments),
            blank_segment_indices=merge.blank_segment_indices,
            logical_utterance_id=active.logical_utterance_id,
            continuation_part_index=active.part_index,
        )
        finalization = time.perf_counter() - started
        result = replace(
            result,
            finalization_seconds=finalization,
            total_stt_seconds=processing + finalization,
        )
        self.on_event(
            "transcript_merged",
            {
                "utterance_id": final.utterance_id,
                "segment_count": len(segments),
                "blank_segment_indices": list(merge.blank_segment_indices),
                "deduplicated_word_count": merge.deduplicated_word_count,
                "merge_decisions": [
                    {
                        "segment_index": item.segment_index,
                        "overlap_words": item.overlap_words,
                        "appended_words": item.appended_words,
                        "blank": item.blank,
                    }
                    for item in merge.decisions
                ],
            },
        )
        return result

    def _release_final(self, result: STTResult) -> None:
        # Repair a mis-transcribed wake word for the interceptor's copy only.
        # This has to happen here, before masking, because "aunt Keith" becomes
        # "aunt [GIVENNAME_1]" once the masker has run and the wake word is
        # then unrecoverable. `result` itself is untouched, so if no command is
        # found the speaker's original words are what continue downstream.
        wake_text, wake_variant = normalize_wake_word(result.final_text)
        if wake_variant is not None:
            self.on_event(
                "onkith_wake_word_normalized",
                # The variant and its canonical form only - never the
                # surrounding transcript.
                {"variant": wake_variant, "canonical": CANONICAL_WAKE_WORD},
            )
        interception = intercept_onkith_command(wake_text)
        forwarded = True
        if interception.command_detected:
            self.on_event(
                "onkith_command_detected",
                {
                    "wake_word": interception.matched_wake_word,
                    "command_text": interception.command_text,
                    "intent": interception.intent,
                },
            )
            if self.music_worker is not None:
                if interception.intent == "CLOSE":
                    self.music_worker.submit_close()
                else:
                    self.music_worker.submit_play(interception.command_text)
                self.on_event("onkith_command_queued", {"intent": interception.intent})
            remaining = interception.remaining_text.strip()
            if remaining:
                self.privacy.submit_stt(replace(result, final_text=remaining))
                self.on_event("onkith_remaining_forwarded", {"remaining_text": remaining})
            else:
                forwarded = False
                self.on_event("onkith_command_only", {"batch_id": result.batch_id})
        else:
            self.privacy.submit_stt(result)
            self.on_event("onkith_no_command", {"batch_id": result.batch_id})
        self.final_utterance_count += 1
        self.on_event(
            "stt_finalized",
            {
                "session_id": result.session_id,
                "batch_id": result.batch_id,
                "sequence_number": result.sequence_number,
                "endpoint_reason": result.endpoint_reason,
                "captured_audio_seconds": result.captured_audio_seconds,
                "submitted_audio_seconds": result.submitted_audio_seconds,
                "unique_submitted_audio_seconds": result.unique_submitted_audio_seconds,
                "queue_wait_seconds": result.queue_wait_seconds,
                "processing_seconds": result.processing_seconds,
                "finalization_seconds": result.finalization_seconds,
                "total_stt_seconds": result.total_stt_seconds,
                "rtf": result.rtf,
                "submitted_audio_rtf": result.submitted_audio_rtf,
                "callback_count": 0,
                "reliable": result.reliable,
                "deduplicated_word_count": result.deduplicated_word_count,
                "segment_count": result.segment_count,
                "blank_segment_indices": list(result.blank_segment_indices),
                "forwarded_to_stage5": forwarded,
            },
        )

    def _process_segment(self, segment: CanarySegment) -> None:
        if self._discard_utterance_id is not None:
            if segment.utterance_id != self._discard_utterance_id:
                raise RuntimeError("Unexpected utterance while discarding a failed one")
            if segment.is_final_segment_of_utterance:
                self._discard_utterance_id = None
            return
        self._validate_order(segment)
        assert self._accumulator is not None
        attempt_started = self.utc_now()
        try:
            transcript = self.canary.transcribe_segment(segment)
        except Exception as exc:
            message = sanitize_error(f"{type(exc).__name__}: {exc}")
            known = list(self._accumulator.summaries)
            known.append(SegmentSummary.from_segment(segment))
            self._record_failure(
                known, attempt_started, message, active=self._accumulator
            )
            self._accumulator = None
            if not segment.is_final_segment_of_utterance:
                self._discard_utterance_id = segment.utterance_id
            return
        # Inference is done: keep scalars and text only, and drop every
        # reference to the decoded audio so it can be collected immediately.
        summary = transcript.summarize()
        del transcript
        self._accumulator.summaries.append(summary)
        self._accumulator.expected_next_index = segment.segment_index + 1
        is_final = segment.is_final_segment_of_utterance
        utterance_id = segment.utterance_id
        segment_index = segment.segment_index
        del segment
        if not summary.text.strip():
            self.on_event(
                "blank_segment_transcript",
                {"utterance_id": utterance_id, "segment_index": segment_index},
            )
        if not is_final:
            self.on_event(
                "stt_segment_complete",
                {
                    "utterance_id": utterance_id,
                    "segment_index": segment_index,
                    "processing_seconds": summary.processing_seconds,
                    "downstream_released": False,
                },
            )
            self._notify_preview(summary)
            self._apply_safety_limit(self._accumulator)
            return
        active = self._accumulator
        # Seal the preview *before* the authoritative release, not after. This
        # is the real endpoint - it comes from the segmenter's final-segment
        # flag, never from a blank transcript - and sealing here invalidates
        # anything still queued so no preview line can appear after [MASKED].
        self._finish_preview()
        try:
            self._release_final(self._finalize(active))
        except EmptyTranscriptError as exc:
            self._record_failure(
                active.summaries,
                active.summaries[0].stt_started_at_utc,
                sanitize_error(f"empty_transcript: {exc}"),
                active=active,
            )
        finally:
            self._accumulator = None

    def _notify_preview(self, summary: SegmentSummary) -> None:
        """Hand a decoded segment to the display preview, if one is attached.

        Isolated on purpose: the preview owns its own worker thread, so this
        call returns immediately and never delays the next Canary inference,
        and any failure inside it is swallowed here.
        """

        if self.preview is None:
            return
        try:
            self.preview.submit(
                summary.text, unique_seconds=summary.submitted_audio_seconds
            )
        except Exception:
            pass

    def _finish_preview(self) -> None:
        """Clear the preview buffer once the authoritative transcript lands."""

        if self.preview is None:
            return
        try:
            self.preview.finalize_utterance()
        except Exception:
            pass

    def _apply_safety_limit(self, active: _Accumulator) -> None:
        """Close a pathological logical utterance at this segment boundary.

        The acoustic segmentation state machine is untouched: Canary is not
        reloaded, no decoder state crosses the boundary, and later segments of
        the same acoustic utterance simply continue in a new logical utterance.
        Onkith therefore runs once per safety-bounded logical utterance in this
        exceptional case instead of once per acoustic utterance.
        """

        reason = active.exceeds_safety_limits()
        if reason is None:
            return
        self.safety_split_count += 1
        self.on_event(
            "utterance_safety_split",
            {
                "utterance_id": active.utterance_id,
                "batch_id": active.batch_id,
                "limit": reason,
                "segment_count": active.segment_count,
                "submitted_span_seconds": active.submitted_span_seconds,
                "maximum_segments": MAXIMUM_LOGICAL_UTTERANCE_SEGMENTS,
                "maximum_seconds": MAXIMUM_LOGICAL_UTTERANCE_SECONDS,
                "next_part_index": active.part_index + 1,
            },
        )
        try:
            self._release_final(
                self._finalize(active, endpoint_reason_override="utterance_safety_split")
            )
        except EmptyTranscriptError as exc:
            # Every segment so far was blank; the existing failure path owns it
            # and no text is dropped silently.
            self._record_failure(
                active.summaries,
                active.summaries[0].stt_started_at_utc,
                sanitize_error(f"empty_transcript: {exc}"),
                active=active,
            )
        part_index = active.part_index + 1
        self._accumulator = _Accumulator(
            utterance_id=active.utterance_id,
            sequence_number=active.sequence_number,
            summaries=[],
            batch_id=continuation_batch_id(active.utterance_id, part_index),
            # The continuation takes the next chronological number now, before
            # any later independent utterance can claim one.
            downstream_sequence_number=self._next_downstream_sequence(),
            expected_next_index=active.expected_next_index,
            part_index=part_index,
        )

    def _process_end(self, end: CanaryUtteranceEnd) -> None:
        if self._discard_utterance_id is not None:
            if end.utterance_id != self._discard_utterance_id:
                raise RuntimeError("Unexpected utterance-end control while discarding")
            self._discard_utterance_id = None
            return
        active = self._accumulator
        if (
            active is not None
            and not active.summaries
            and active.part_index > 1
            and end.utterance_id == active.utterance_id
        ):
            # A safety split closed and released the previous part, and the
            # stream ended before any further segment arrived. There is nothing
            # left to merge and no text is lost.
            self.on_event(
                "utterance_end_after_safety_split",
                {"utterance_id": end.utterance_id, "reason": end.endpoint_reason},
            )
            self._accumulator = None
            return
        if active is None or not active.summaries:
            raise RuntimeError("Utterance-end control arrived without decoded segments")
        if (
            end.utterance_id != active.utterance_id
            or end.utterance_sequence_number != active.sequence_number
        ):
            raise RuntimeError("Utterance-end control identity is inconsistent")
        try:
            self._release_final(self._finalize(active, end))
        except EmptyTranscriptError as exc:
            self._record_failure(
                active.summaries,
                active.summaries[0].stt_started_at_utc,
                sanitize_error(f"empty_transcript: {exc}"),
                end,
                active=active,
            )
        finally:
            self._accumulator = None

    def _worker_loop(self) -> None:
        while True:
            segment = self.completed_segment_queue.get()
            try:
                if segment is None:
                    return
                identity = getattr(segment, "segment_id", segment.utterance_id)
                utterance_id = segment.utterance_id
                try:
                    if isinstance(segment, CanaryUtteranceEnd):
                        self._process_end(segment)
                    else:
                        self._process_segment(segment)
                except Exception as exc:
                    message = sanitize_error(f"{type(exc).__name__}: {exc}")
                    self.errors.append(f"{identity}: {message}")
                    self.on_event(
                        "stt_error", {"batch_id": utterance_id, "error": message}
                    )
            finally:
                # Drop this loop's own reference *before* releasing the queue
                # join, so that once a segment is accounted for as done its
                # audio is already unreachable and never held alive while the
                # worker waits for more work.
                segment = None
                self.completed_segment_queue.task_done()

    def shutdown(self) -> None:
        if self._shutdown_complete:
            if self.errors:
                raise RuntimeError("STT worker failure: " + "; ".join(self.errors))
            return
        self.completed_segment_queue.join()
        self.completed_segment_queue.put(None)
        if self.thread is not None:
            self.thread.join()
        self._shutdown_complete = True
        if self._accumulator is not None and self._accumulator.summaries:
            raise RuntimeError("STT shutdown found an unterminated utterance")
        # An empty continuation accumulator left by a safety split holds no
        # decoded text and is not an unterminated utterance.
        self._accumulator = None
        if self.errors:
            raise RuntimeError("STT worker failure: " + "; ".join(self.errors))



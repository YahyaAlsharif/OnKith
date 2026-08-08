"""Display-only masked preview of speech that is still being spoken.

The authoritative transcript is released only when an utterance ends, which
means a speaker who talks for two minutes sees nothing until they stop. This
module renders a live, masked view in the meantime.

It is a display layer and nothing else. It never writes to the database, never
reaches Onkith, and never changes the final transcript. If anything in here
fails, the failure is contained and the authoritative path continues untouched.

Why the whole buffer is re-masked every cycle
---------------------------------------------
Masking each segment on its own and appending the results is unsafe. A segment
ending with ``my number is 05511`` followed by one starting with ``95719``
contains no complete phone number on either side, so neither half is masked and
the raw digits are printed. The masker is not wrong; it is being shown half an
entity.

So the buffer keeps recent raw text, the masker runs over all of it each cycle,
and the visible line is repainted from that fresh result. On the second pass the
masker sees ``05511 95719`` as one string and produces one phone label.

Why the tail is blurred
-----------------------
The last words of the buffer are unstable: the next segment may complete them
into an entity, and the 0.704 s of overlap audio is transcribed again and may
literally change. Those words are rendered as a placeholder rather than their
text, so a half-finished entity is never printed. The next repaint replaces the
placeholder with the real label.
"""

from __future__ import annotations

import hashlib
import math
import os
import queue  # noqa: F401  (preview worker hand-off)
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.digit_normalizer import normalize_spoken_digits
from integrated_pipeline.config import (
    FORCED_OVERLAP_SAMPLES,
    SAMPLE_RATE,
    TRAILING_CONTEXT_SAMPLES,
)
from integrated_pipeline.transcript_merger import (
    EmptyTranscriptError,
    merge_segment_transcripts,
)

# --------------------------------------------------------------------------
# Private names borrowed from integrated_pipeline.structured_pii.
#
# The preview needs the cue patterns themselves, not the spans they produce.
# find_spans() only returns a span once the value is present, which is exactly
# the case this layer must catch early: "my phone number is" has arrived but
# the digits have not, so there is nothing yet for anything to detect.
#
# These are private names, so this is deliberate coupling kept in one place.
# If they move or are renamed, this block is the only thing to update.
# --------------------------------------------------------------------------
from integrated_pipeline.structured_pii import StructuredPIIRecognizer

_BORROWED_CUES: tuple[re.Pattern[str], ...] = (
    StructuredPIIRecognizer._PHONE_CUE,
    StructuredPIIRecognizer._ACCOUNT_CUE,
    StructuredPIIRecognizer._ZIP_CUE,
    StructuredPIIRecognizer._IDENTIFIER_CUE,
)

# How much recent raw speech the buffer keeps. Must span at least two segments
# so an entity split across one boundary is seen whole at least once.
PREVIEW_WINDOW_SECONDS = 30.0
MINIMUM_WINDOW_SECONDS = 17.8

# Longest run of words the preview will hold behind a placeholder. Past this it
# is redacted and printed instead, so someone reading a long number aloud does
# not stall the display forever.
BLUR_CAP_WORDS = 20

# How far back a cue is still considered pending. Separate from the blur cap:
# one bounds how much is hidden, this bounds how stale a cue may be before the
# speaker is assumed to have moved on without giving the value.
CUE_LOOKBACK_WORDS = 20

# Repaints are driven by segment completion, not a timer. This is only a floor,
# so a burst of very short segments cannot cause repeated masking work.
MINIMUM_CYCLE_SECONDS = 0.5

# Shown in place of unstable words. Deliberately not shaped like an entity
# label, so it does not read as a detection.
BLUR_TOKEN = "●●●●●"

# Used when the blur cap trips. Double brackets and upper case keep it visually
# distinct from real entity labels such as [GIVENNAME_1].
REDACTED_LABEL = "[[REDACTED]]"

# Audio that adjacent segments share, in seconds. Words inside this run are
# re-transcribed by the next segment and are always treated as unstable.
ADJACENT_OVERLAP_SECONDS = (
    FORCED_OVERLAP_SAMPLES + TRAILING_CONTEXT_SAMPLES
) / float(SAMPLE_RATE)

# Display-only email cue. Canary runs with pnc: no, so a spoken address arrives
# as "abdullah dot alhulez at gmail dot com" with no @ symbol at all, and
# _EMAIL_PATTERN in structured_pii.py never fires on speech. TinyBERT does
# recognise the address, so this cue exists only to blur the window between
# "my email address is" and the model having enough of the value to label it.
#
# It lives here, not in structured_pii.py, because it must not change what the
# authoritative masker detects. Shape mirrors _PHONE_CUE.
_EMAIL_CUE = re.compile(
    r"\b(?:e-?mail(?:\s+address)?|mail\s+address)\b"
    r"(?:\s*(?:is|equals|:|-)\s*)?",
    flags=re.IGNORECASE,
)

# Every cue the preview watches: the four borrowed ones plus the email cue.
_PREVIEW_CUES: tuple[re.Pattern[str], ...] = _BORROWED_CUES + (_EMAIL_CUE,)

# Fallback speaking rate, used only when a segment reports no usable duration.
_FALLBACK_WORDS_PER_SECOND = 3.0


@dataclass
class _BufferedSegment:
    """One decoded segment held in the rolling buffer.

    ``text`` is raw, unmasked transcript. It exists only in memory, is never
    logged or persisted, and is dropped when the buffer evicts or closes.
    """

    text: str
    unique_seconds: float
    unstable_words: int
    stable_fingerprint: str | None = None
    stable_repaints: int = 0
    # Latched once the region has survived two identical repaints. Evicting an
    # earlier region shifts this one's slice bounds, which changes its
    # fingerprint through no fault of its own; without the latch stability
    # would reset on every eviction and the buffer could never keep pace.
    stable: bool = False


@dataclass
class PreviewRender:
    """What the display should show. Carries no raw text."""

    line: str
    blurred_words: int
    entity_count: int
    buffered_seconds: float
    degraded: bool = False
    redacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _words(text: str) -> list[str]:
    return (text or "").split()


class _LabelRegistry:
    """Stable display labels keyed on a salted hash of the surface form.

    Raw surface forms are never stored. Keeping a plain lookup table would hold
    every piece of PII from the session in memory, which would defeat the point
    of bounding the buffer at all. The salt is per session and is discarded on
    close, so the hashes are useless afterwards.

    This exists because the authoritative masker numbers placeholders by
    position within a single call: ``[GIVENNAME_1]`` means "first given name in
    this string", not "rayyan". Across repaints, and especially across an
    eviction, that number moves. The preview relabels so a person keeps the same
    number for as long as they are on screen.
    """

    def __init__(self) -> None:
        self._salt = os.urandom(16)
        self._labels: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def _key(self, entity_type: str, surface: str) -> str:
        digest = hashlib.sha256()
        digest.update(self._salt)
        digest.update(entity_type.encode("utf-8", "replace"))
        digest.update(b"\x00")
        digest.update(surface.strip().lower().encode("utf-8", "replace"))
        return digest.hexdigest()

    def label_for(self, entity_type: str, surface: str) -> str:
        key = self._key(entity_type, surface)
        existing = self._labels.get(key)
        if existing is not None:
            return existing
        self._counts[entity_type] = self._counts.get(entity_type, 0) + 1
        label = f"[{entity_type}_{self._counts[entity_type]}]"
        self._labels[key] = label
        return label

    def clear(self) -> None:
        """Discard the salt and every mapping."""

        self._salt = os.urandom(16)
        self._labels.clear()
        self._counts.clear()

    @property
    def size(self) -> int:
        return len(self._labels)


class LivePreview:
    """Hold recent raw speech, re-mask it each cycle, and render a safe line.

    ``masker`` is any object with ``mask_text(str)`` returning a result that
    exposes ``masked_text``, ``entities`` and ``safe_to_release`` - in practice
    the same ``FailClosedMasker`` the authoritative path uses.

    ``on_render`` receives a :class:`PreviewRender` and does the actual
    printing.

    Call :meth:`start` to run repaints on a dedicated worker thread, which is
    what the live application does so neither Canary nor the privacy worker
    ever waits for display work. Without :meth:`start` the repaint happens
    inline, which is what the tests use.
    """

    def __init__(
        self,
        masker: Any,
        on_render: Callable[[PreviewRender], None],
        *,
        window_seconds: float = PREVIEW_WINDOW_SECONDS,
        blur_cap_words: int = BLUR_CAP_WORDS,
        cue_lookback_words: int = CUE_LOOKBACK_WORDS,
        minimum_cycle_seconds: float = MINIMUM_CYCLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.masker = masker
        self.on_render = on_render
        # A window shorter than two segments could hide the very boundary this
        # layer exists to cover, so the floor is enforced rather than trusted.
        self.window_seconds = max(float(window_seconds), MINIMUM_WINDOW_SECONDS)
        self.blur_cap_words = max(1, int(blur_cap_words))
        self.cue_lookback_words = max(1, int(cue_lookback_words))
        self.minimum_cycle_seconds = max(0.0, float(minimum_cycle_seconds))
        self.clock = clock

        self._lock = threading.Lock()
        self._segments: list[_BufferedSegment] = []
        self._registry = _LabelRegistry()
        self._last_cycle: float | None = None
        self._closed = False
        # Every queued item carries the generation it was created under. The
        # counter is bumped synchronously by finalize_utterance(), so anything
        # already in flight becomes stale the instant the utterance closes -
        # that is what stops a preview printing after the authoritative line.
        self._generation = 0
        self._last_signature: tuple[Any, ...] | None = None
        self._queue: queue.Queue[tuple[Any, ...] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.render_count = 0
        self.skipped_cycle_count = 0
        self.degraded_cycle_count = 0
        self.evicted_segment_count = 0
        self.redacted_render_count = 0
        self.duplicate_suppressed_count = 0
        self.stale_event_count = 0
        self.blank_segment_count = 0
        self.finalize_count = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Run repaints on a dedicated thread so no other lane waits."""

        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="omit-preview-worker",
            daemon=True,
        )
        self._thread.start()

    def _worker_loop(self) -> None:
        """One ordered lane for every state-changing preview operation."""

        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                kind = item[0]
                if kind == "segment":
                    _, generation, text, seconds = item
                    self.add_segment(
                        text, unique_seconds=seconds, generation=generation
                    )
                elif kind == "finalize":
                    self._apply_finalize()
            except Exception:
                # A broken event must not kill the lane; the next one still runs.
                pass

    def submit(self, text: str, *, unique_seconds: float) -> None:
        """Hand a segment to the preview without blocking the caller.

        Used from the STT worker: display work must never delay the next
        Canary call.
        """

        try:
            if self._closed:
                return
            if self._thread is None:
                self.add_segment(text, unique_seconds=unique_seconds)
                return
            with self._lock:
                generation = self._generation
            self._queue.put_nowait(("segment", generation, text, unique_seconds))
        except Exception:
            pass

    def _apply_finalize(self) -> None:
        """Ordered half of finalization: clear buffer, labels and salt."""

        with self._lock:
            self._segments.clear()
            self._last_cycle = None
            self._last_signature = None
            # A new utterance is a new person-numbering scope, and a new salt.
            # Without this the registry grows for the whole session and the
            # first name of a fresh utterance renders as [GIVENNAME_47].
            self._registry.clear()

    def finalize_utterance(self) -> None:
        """Close the current utterance once the authoritative line is released.

        The generation bump happens synchronously so that every segment already
        sitting in the queue is invalidated immediately, even if the worker is
        still busy. The buffer/registry wipe is then queued so it lands in order
        behind any event the worker had already started.
        """

        with self._lock:
            self._generation += 1
            self.finalize_count += 1
            self._last_signature = None
            closed = self._closed
        if closed or self._thread is None:
            self._apply_finalize()
            return
        try:
            self._queue.put_nowait(("finalize",))
        except Exception:
            self._apply_finalize()

    def close(self) -> None:
        """Session end: wipe every trace of raw speech and the salt."""

        with self._lock:
            self._segments.clear()
            self._last_cycle = None
            self._last_signature = None
            self._closed = True
            # Invalidate anything still queued so a late worker pass cannot
            # draw after shutdown has begun.
            self._generation += 1
            self._registry.clear()
        if self._thread is not None:
            try:
                self._queue.put_nowait(None)
                self._thread.join(timeout=2.0)
            except Exception:
                pass
            self._thread = None

    # -- buffer ------------------------------------------------------------

    @property
    def buffered_seconds(self) -> float:
        return sum(item.unique_seconds for item in self._segments)

    def _unstable_words_for(self, text: str, unique_seconds: float) -> int:
        """How many trailing words the next segment may re-transcribe.

        Derived from this segment's own speaking rate rather than assumed, so a
        fast talker gets more words blurred than a slow one for the same 0.704 s
        of shared audio.

        Rounds up, never to nearest: rounding down by one word would print a
        word the next segment may still change.
        """

        count = len(_words(text))
        if count == 0:
            return 0
        seconds = float(unique_seconds)
        rate = count / seconds if seconds > 0 else _FALLBACK_WORDS_PER_SECOND
        unstable = math.ceil(ADJACENT_OVERLAP_SECONDS * rate)
        return max(1, min(count, unstable))

    def _prefix_ends(self, texts: list[str]) -> list[int]:
        """Where each segment ends inside the merged buffer.

        The merger only ever appends non-overlapping words, so merging the
        first ``n`` segments gives a genuine prefix of the full buffer.
        """

        ends: list[int] = []
        for index in range(1, len(texts) + 1):
            try:
                ends.append(len(merge_segment_transcripts(texts[:index]).text))
            except EmptyTranscriptError:
                ends.append(0)
        return ends

    @staticmethod
    def _masked_slice(
        raw: str,
        entities: list[dict[str, Any]],
        start: int,
        end: int,
        labels: dict[int, str],
    ) -> str | None:
        """Masked rendering of one region, or None if it is not self-contained.

        Fingerprinting each region on its own content rather than on the whole
        prefix matters: a prefix fingerprint changes every time an earlier
        region is evicted, so nothing would ever accumulate two identical
        repaints and the buffer would grow without bound.

        Returns None when an entity straddles either edge, since the region
        cannot then be judged stable independently.
        """

        parts: list[str] = []
        cursor = start
        for entity in sorted(entities, key=lambda item: int(item["start"])):
            span_start = int(entity["start"])
            span_end = int(entity["end"])
            if span_end <= start or span_start >= end:
                continue
            if span_start < start or span_end > end:
                return None
            if span_start < cursor:
                continue
            parts.append(raw[cursor:span_start])
            parts.append(labels.get(span_start, REDACTED_LABEL))
            cursor = span_end
        parts.append(raw[cursor:end])
        return "".join(parts)

    def _evict(self, raw: str, entities: list[dict[str, Any]],
               labels: dict[int, str], risky_start: int) -> None:
        """Drop only regions that have settled.

        A region is evictable when its mask was byte-identical across two
        consecutive repaints and there is clean, non-risky text after it.
        Eviction on a raw time slice could cut an entity in half and freeze a
        leak into already-printed output, so an unstable region is kept even
        when it is older than the window. The blur cap bounds the worst case.
        """

        if len(self._segments) < 2:
            return
        texts = [item.text for item in self._segments]
        ends = self._prefix_ends(texts)
        evictable = 0
        for index in range(len(self._segments) - 1):
            end = ends[index]
            if end >= risky_start:
                # No clean text after this region yet, and nothing beyond this
                # point can be clean either.
                break
            segment = self._segments[index]
            if not segment.stable:
                start = 0 if index == 0 else ends[index - 1]
                fingerprint = self._masked_slice(raw, entities, start, end, labels)
                if fingerprint is None:
                    # An entity straddles this region's edge; still moving.
                    segment.stable_fingerprint = None
                    segment.stable_repaints = 0
                elif segment.stable_fingerprint == fingerprint:
                    segment.stable_repaints += 1
                    if segment.stable_repaints >= 2:
                        segment.stable = True
                else:
                    segment.stable_fingerprint = fingerprint
                    segment.stable_repaints = 1
            # Every region keeps accumulating stability each repaint, but
            # eviction stays contiguous from the front: region N can only go
            # once every region before it has gone.
            if segment.stable and evictable == index:
                evictable = index + 1
        if not evictable:
            return
        # Only trim once the window is actually exceeded; stability alone is
        # not a reason to forget recent context.
        while (
            evictable > 0
            and len(self._segments) > 1
            and self.buffered_seconds > self.window_seconds
        ):
            self._segments.pop(0)
            self.evicted_segment_count += 1
            evictable -= 1

    def _raw_buffer(self) -> tuple[str, int]:
        """Join the buffer and report how many trailing words are unstable.

        Adjacent segments share 0.704 s of audio, so the next segment
        re-transcribes the previous one's last words. Plain concatenation would
        print those words twice - "05511 95719 05511 95719". The buffer is
        therefore joined with the same exact 14-word merger the authoritative
        path uses, which removes the repeat. No new logic, and the preview
        agrees with the final transcript about where the seam is.
        """

        texts = [item.text for item in self._segments if item.text.strip()]
        if not texts:
            return "", 0
        try:
            joined = merge_segment_transcripts(texts).text
        except EmptyTranscriptError:
            return "", 0
        return joined, self._segments[-1].unstable_words

    # -- risk --------------------------------------------------------------

    @staticmethod
    def _word_start_offset(raw: str, word_index: int) -> int:
        """Character offset where word number ``word_index`` begins."""

        if word_index <= 0:
            return 0
        index = 0
        seen = 0
        length = len(raw)
        while index < length and seen < word_index:
            while index < length and raw[index].isspace():
                index += 1
            while index < length and not raw[index].isspace():
                index += 1
            seen += 1
        # Land on the word itself rather than the space before it, so the
        # offset can be compared directly against entity spans.
        while index < length and raw[index].isspace():
            index += 1
        return index

    def _overlap_start(self, raw: str, unstable_words: int) -> int:
        """Offset of the words the next segment will transcribe again."""

        words = _words(raw)
        if not words or unstable_words <= 0:
            return len(raw)
        keep = max(0, len(words) - unstable_words)
        return self._word_start_offset(raw, keep)

    @staticmethod
    def _entity_touching_end(raw: str, entities: list[dict[str, Any]]) -> int | None:
        """Part 4: an entity that runs to the last word may still be growing."""

        settled = len(raw.rstrip())
        latest: int | None = None
        for entity in entities:
            try:
                start = int(entity["start"])
                end = int(entity["end"])
            except (KeyError, TypeError, ValueError):
                # A malformed span is ambiguous, so fail toward blurring.
                return 0
            if end >= settled:
                latest = start if latest is None else min(latest, start)
        return latest

    def _unfulfilled_cue_start(
        self, raw: str, entities: list[dict[str, Any]]
    ) -> int | None:
        """Part 5: a cue has arrived but its value has not.

        "my phone number is" with no digits yet gives the masker nothing to
        find, so nothing would be blurred and the digits would print the moment
        they arrive. Blurring from the cue closes that window.

        Only cues within the lookback are considered. A cue far back with no
        value means the speaker moved on, and blurring from there would grey out
        the whole buffer for no benefit.
        """

        words = _words(raw)
        if not words:
            return None
        recent_from = self._word_start_offset(
            raw, max(0, len(words) - self.cue_lookback_words)
        )
        starts = []
        for entity in entities:
            try:
                starts.append(int(entity["start"]))
            except (KeyError, TypeError, ValueError):
                return 0
        earliest: int | None = None
        for pattern in _PREVIEW_CUES:
            for match in pattern.finditer(raw):
                if match.start() < recent_from:
                    continue
                if any(start >= match.end() for start in starts):
                    # The value arrived and was labelled; nothing pending.
                    continue
                earliest = (
                    match.start() if earliest is None else min(earliest, match.start())
                )
        return earliest

    def _risky_start(
        self, raw: str, unstable_words: int, entities: list[dict[str, Any]]
    ) -> int:
        """Where the unstable region begins. Earliest candidate wins."""

        candidates = [self._overlap_start(raw, unstable_words)]
        for candidate in (
            self._entity_touching_end(raw, entities),
            self._unfulfilled_cue_start(raw, entities),
        ):
            if candidate is not None:
                candidates.append(candidate)
        return max(0, min(candidates))

    # -- render ------------------------------------------------------------

    def _labels_for(self, entities: list[dict[str, Any]]) -> dict[int, str]:
        """Stable label per entity start offset, via the salted registry."""

        labels: dict[int, str] = {}
        for entity in entities:
            try:
                start = int(entity["start"])
            except (KeyError, TypeError, ValueError):
                continue
            entity_type = str(entity.get("entity_type", "PII"))
            surface = str(entity.get("text", ""))
            labels[start] = self._registry.label_for(entity_type, surface)
        return labels

    def _render_line(
        self,
        raw: str,
        entities: list[dict[str, Any]],
        risky_start: int,
        labels: dict[int, str],
    ) -> tuple[str, int, bool]:
        """Rebuild the masked line, replacing everything from ``risky_start``.

        The masked string is reconstructed from the raw text and the entity
        spans rather than taken verbatim, because placeholder substitution
        shifts offsets. Rebuilding keeps raw character positions meaningful, so
        the risky cut lands where it was calculated.
        """

        ordered = sorted(
            (item for item in entities if "start" in item and "end" in item),
            key=lambda item: (int(item["start"]), int(item["end"])),
        )
        parts: list[str] = []
        cursor = 0
        for entity in ordered:
            start = int(entity["start"])
            end = int(entity["end"])
            if start < cursor or start >= len(raw):
                continue
            if start >= risky_start:
                break
            if end > risky_start:
                # The entity straddles the cut, so it is still growing. Blur it
                # whole rather than printing the part that has arrived.
                risky_start = start
                break
            parts.append(raw[cursor:start])
            parts.append(labels.get(start, REDACTED_LABEL))
            cursor = end
        head = "".join(parts) + raw[cursor:risky_start]
        blurred = len(_words(raw[risky_start:]))
        line = head.rstrip()
        redacted = blurred > self.blur_cap_words
        if blurred:
            # Past the cap the run stops being "still resolving" and is simply
            # redacted, so a long spoken number cannot stall the display.
            token = REDACTED_LABEL if redacted else BLUR_TOKEN
            line = f"{line} {token}".strip()
        return line, blurred, redacted

    def _blur_everything(self, reason: str) -> PreviewRender:
        self.degraded_cycle_count += 1
        return PreviewRender(
            line=BLUR_TOKEN,
            blurred_words=-1,
            entity_count=0,
            buffered_seconds=self.buffered_seconds,
            degraded=True,
            metadata={"reason": reason},
        )

    def add_segment(
        self,
        text: str,
        *,
        unique_seconds: float,
        force: bool = False,
        generation: int | None = None,
    ) -> PreviewRender | None:
        """Add one decoded segment and repaint.

        Returns the render that was emitted, or None when the cycle was
        skipped. Never raises: a failure here must not disturb the caller.

        A blank segment is *not* evidence of an endpoint - Canary also returns
        empty text for quiet speech, noise and failed decodes - so it never
        releases the unstable tail and never finalizes anything. It simply adds
        no content, and with no content there is nothing new to draw.
        """

        try:
            with self._lock:
                if self._closed:
                    return None
                if generation is not None and generation != self._generation:
                    # Queued before the utterance closed; the buffer it belonged
                    # to is gone. Drawing it now would print after the final line.
                    self.stale_event_count += 1
                    return None
                active_generation = self._generation
                cleaned = (text or "").strip()
                if not cleaned:
                    # Nothing changed, so any repaint would be byte-identical to
                    # the last one. Returning here is what stops the repeated
                    # [PREVIEW] line during silence, and it also keeps TinyBERT
                    # off an unchanged buffer.
                    self.blank_segment_count += 1
                    return None
                self._segments.append(
                    _BufferedSegment(
                        text=cleaned,
                        unique_seconds=max(0.0, float(unique_seconds)),
                        unstable_words=self._unstable_words_for(
                            cleaned, unique_seconds
                        ),
                    )
                )
                now = self.clock()
                if (
                    not force
                    and self._last_cycle is not None
                    and now - self._last_cycle < self.minimum_cycle_seconds
                ):
                    self.skipped_cycle_count += 1
                    return None
                self._last_cycle = now
                raw, unstable = self._raw_buffer()
        except Exception:
            render = self._blur_everything("buffer_error")
            self._emit(render)
            return render

        render = self._build(raw, unstable)

        with self._lock:
            if self._closed or active_generation != self._generation:
                # The utterance closed while this repaint was being computed.
                self.stale_event_count += 1
                return None
            signature = (
                active_generation,
                render.line,
                render.blurred_words,
                render.degraded,
                render.redacted,
            )
            if signature == self._last_signature:
                self.duplicate_suppressed_count += 1
                return None

        if not self._emit(render):
            # The display refused this render. Leaving the signature unrecorded
            # means an identical retry is still allowed to draw later.
            return None
        with self._lock:
            self._last_signature = signature
        return render

    def _build(self, raw: str, unstable_words: int) -> PreviewRender:
        if not raw:
            return PreviewRender("", 0, 0, self.buffered_seconds)
        try:
            result = self.masker.mask_text(raw)
        except Exception:
            # Fail toward blurring: an unmasked line is a breach, a blurred one
            # is a cosmetic problem.
            return self._blur_everything("mask_raised")
        if not getattr(result, "safe_to_release", False):
            # A blocked result has an empty entity list, which is
            # indistinguishable from clean text. Treat it as fully unsafe.
            return self._blur_everything("mask_blocked")
        entities = list(getattr(result, "entities", None) or [])
        try:
            labels = self._labels_for(entities)
            risky_start = self._risky_start(raw, unstable_words, entities)
            line, blurred, redacted = self._render_line(
                raw, entities, risky_start, labels
            )
            # Cosmetic only, and last: word/character counts above are already
            # final, and eviction below fingerprints `raw`, not `line`, so
            # normalising here cannot shift a blur boundary or desync the
            # stability check.
            line = normalize_spoken_digits(line)
            with self._lock:
                self._evict(raw, entities, labels, risky_start)
        except Exception:
            return self._blur_everything("render_error")
        self.render_count += 1
        if redacted:
            self.redacted_render_count += 1
        return PreviewRender(
            line=line,
            blurred_words=blurred,
            entity_count=len(entities),
            buffered_seconds=self.buffered_seconds,
            redacted=redacted,
        )

    def _emit(self, render: PreviewRender) -> bool:
        """Draw one render. Returns False if the display refused it."""

        try:
            self.on_render(render)
        except Exception:
            # The display refusing to draw must not propagate either.
            return False
        return True

    def audit(self) -> dict[str, Any]:
        """Counters only. Never exposes buffered text or surface forms."""

        return {
            "render_count": self.render_count,
            "skipped_cycle_count": self.skipped_cycle_count,
            "degraded_cycle_count": self.degraded_cycle_count,
            "evicted_segment_count": self.evicted_segment_count,
            "redacted_render_count": self.redacted_render_count,
            "duplicate_suppressed_count": self.duplicate_suppressed_count,
            "stale_event_count": self.stale_event_count,
            "blank_segment_count": self.blank_segment_count,
            "finalize_count": self.finalize_count,
            "generation": self._generation,
            "buffered_segment_count": len(self._segments),
            "buffered_seconds": self.buffered_seconds,
            "registry_size": self._registry.size,
            "window_seconds": self.window_seconds,
            "blur_cap_words": self.blur_cap_words,
            "cue_lookback_words": self.cue_lookback_words,
            "closed": self._closed,
        }

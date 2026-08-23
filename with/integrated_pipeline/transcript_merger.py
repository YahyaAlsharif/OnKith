"""Exact notebook-derived 14-word Canary segment transcript merger."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

MAXIMUM_OVERLAP_WORDS = 14
_INNER_APOSTROPHE = re.compile(r"(?<=[a-z0-9])'(?=[a-z0-9])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


class EmptyTranscriptError(RuntimeError):
    """Raised when every independently decoded segment is blank."""


@dataclass(frozen=True)
class MergeDecision:
    segment_index: int
    overlap_words: int
    appended_words: int
    blank: bool = False


@dataclass(frozen=True)
class TranscriptMergeResult:
    text: str
    deduplicated_word_count: int
    blank_segment_indices: tuple[int, ...]
    decisions: tuple[MergeDecision, ...]


def normalize_overlap_text(value: str) -> str:
    """Apply the winning notebook normalizer without preserving punctuation."""

    text = unicodedata.normalize("NFKC", value or "").lower()
    text = text.replace("’", "'").replace("`", "'")
    text = _INNER_APOSTROPHE.sub("", text)
    text = _NON_ALNUM.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def merge_segment_transcripts(
    segment_texts: Sequence[str],
    *,
    maximum_overlap_words: int = MAXIMUM_OVERLAP_WORDS,
) -> TranscriptMergeResult:
    """Merge raw words, dropping the longest normalized suffix/prefix match."""

    if int(maximum_overlap_words) != MAXIMUM_OVERLAP_WORDS:
        raise ValueError("The production merger maximum must be exactly 14 words")
    merged_words: list[str] = []
    blank_indices: list[int] = []
    decisions: list[MergeDecision] = []
    removed_total = 0

    for segment_index, raw_text in enumerate(segment_texts):
        stripped = (raw_text or "").strip()
        if not stripped:
            blank_indices.append(segment_index)
            decisions.append(MergeDecision(segment_index, 0, 0, blank=True))
            continue
        incoming_words = stripped.split()
        overlap = 0
        maximum = min(
            MAXIMUM_OVERLAP_WORDS,
            len(merged_words),
            len(incoming_words),
        )
        for count in range(maximum, 0, -1):
            previous = [normalize_overlap_text(word) for word in merged_words[-count:]]
            incoming = [normalize_overlap_text(word) for word in incoming_words[:count]]
            if previous == incoming:
                overlap = count
                break
        merged_words.extend(incoming_words[overlap:])
        removed_total += overlap
        decisions.append(
            MergeDecision(
                segment_index=segment_index,
                overlap_words=overlap,
                appended_words=len(incoming_words) - overlap,
            )
        )

    if not merged_words:
        raise EmptyTranscriptError("All Canary segment transcripts were blank")
    return TranscriptMergeResult(
        text=" ".join(merged_words),
        deduplicated_word_count=removed_total,
        blank_segment_indices=tuple(blank_indices),
        decisions=tuple(decisions),
    )

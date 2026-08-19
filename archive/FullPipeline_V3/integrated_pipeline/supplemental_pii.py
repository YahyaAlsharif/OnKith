"""Supplemental structured-PII recognizers for spoken-digit sequences.

`structured_pii.StructuredPIIRecognizer` is frozen (it is a protected file), and
two real gaps were confirmed against the live privacy model (measured on
Model V1, TinyBERT-4; both gaps are regex-layer gaps and are independent of
which token classifier runs):

    phone number is a zero five five a six seven three four four two one
        -> released verbatim; the stray ASR filler "a" makes
           `_SEQUENCE_PATTERN.match()` fail at the very first token, so the cue
           finds no sequence at all and 11 digits reach storage in plaintext.

    bank id is four five three three two one one one six five
        -> released verbatim; "bank id" matches neither `_ACCOUNT_CUE`
           (which wants "account"/"bank account") nor `_IDENTIFIER_CUE`
           (which wants "id number"). The token classifier returned no entities either.

This module closes both gaps without editing the frozen recognizer. It runs as
a wrapper *around* the masker and re-merges its spans through the authoritative
`merge_pii_spans`, so the correction lands on the canonical masked text and
therefore on SQLite, Onkith and every display path at once.

Three narrowly-scoped additions, in order of confidence:

1. Extra cue spellings only ("bank id", "account id", "customer id",
   "contact number", ...). Same contextual shape as the frozen recognizer.
2. A tiny ASR-filler allowlist tolerated *inside* a numeric sequence, and only
   after a strong cue. Never a global "skip unknown words" rule.
3. A cue-less floor: a very long run of spoken digits is protected even with no
   cue at all, because nobody says nine consecutive digits conversationally.

Deliberately conservative: short counts ("I have two questions"), room numbers
("room two zero five") and quantities ("in five minutes") must not be masked.
Every threshold here is a fail-closed bound, not a guess at meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .span_merger import SpanCandidate, merge_pii_spans
from .structured_pii import StructuredPIISpan, _DIGIT_WORDS

# Read-only reuse of the frozen recognizer's digit vocabulary so the two stay
# in lockstep. Nothing here mutates it.
_DIGIT_VOCABULARY = _DIGIT_WORDS

# ASR fillers Canary emits mid-sequence. Kept deliberately tiny: every entry is
# a token that carries no numeric meaning and cannot be part of a real value.
_FILLER_WORDS = frozenset({"a", "ah", "uh", "um", "er", "eh", "hmm"})

# At most this many fillers may appear inside one cued numeric sequence.
MAX_FILLERS_IN_SEQUENCE = 2

# A run this long is protected even with no cue at all. Nine is above any
# ordinary spoken count, room number or quantity, and below a full phone
# number, so it catches dictated identifiers without touching normal speech.
UNCUED_RUN_MINIMUM_DIGITS = 9

# Separator characters allowed between two tokens of one sequence.
_SEPARATOR_RE = re.compile(r"[\s().\-]*")
_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")


@dataclass(frozen=True)
class _Cue:
    pattern: re.Pattern[str]
    entity_type: str
    source: str
    minimum_digits: int
    maximum_digits: int


# Only the spellings the frozen recognizer misses. Overlap with it is harmless
# (merge_pii_spans dedups by overlap) but pointless, so it is avoided.
_PHONE_CUE = re.compile(
    r"\b(?:contact\s+number|telephone(?:\s+number)?|cell(?:\s+phone)?(?:\s+number)?|"
    r"whatsapp(?:\s+number)?|call\s+me\s+on|reach\s+me\s+(?:at|on))\b"
    r"(?:\s*(?:is|are|equals|:|-)\s*)?",
    flags=re.IGNORECASE,
)
_ACCOUNT_CUE = re.compile(
    r"\b(?:account\s+(?:id|no)|iban|routing\s+number|sort\s+code)\b"
    r"(?:\s*(?:is|are|equals|:|-)\s*)?",
    flags=re.IGNORECASE,
)
_IDENTIFIER_CUE = re.compile(
    r"\b(?:bank\s+id(?:entifier)?|customer\s+id|client\s+id|member(?:ship)?\s+id|"
    r"employee\s+id|staff\s+id|student\s+id|national\s+id|patient\s+id|"
    r"policy\s+number|id\s+no)\b"
    r"(?:\s*(?:is|are|equals|:|-)\s*)?",
    flags=re.IGNORECASE,
)

_CUES: tuple[_Cue, ...] = (
    _Cue(_PHONE_CUE, "TELEPHONENUM", "supplemental_phone", 7, 15),
    _Cue(_ACCOUNT_CUE, "ACCOUNTNUM", "supplemental_account", 4, 24),
    _Cue(_IDENTIFIER_CUE, "IDCARDNUM", "supplemental_identifier", 4, 24),
)

# Cues the frozen recognizer already spells, re-listed here only so the
# filler-tolerant scanner gets a second pass over them. Without this, a stray
# "a" still defeats the frozen exact-match sequence pattern.
_FROZEN_CUE_RESCAN: tuple[_Cue, ...] = (
    _Cue(
        re.compile(
            r"\b(?:phone(?:\s+number)?|mobile(?:\s+number)?|"
            r"call\s+me\s+at|contact\s+me\s+at)\b"
            r"(?:\s*(?:is|equals|:|-)\s*)?",
            flags=re.IGNORECASE,
        ),
        "TELEPHONENUM",
        "supplemental_phone_filler",
        7,
        15,
    ),
    _Cue(
        re.compile(
            r"\b(?:bank\s+account(?:\s+number)?|account\s+number)\b"
            r"(?:\s*(?:is|equals|:|-)\s*)?",
            flags=re.IGNORECASE,
        ),
        "ACCOUNTNUM",
        "supplemental_account_filler",
        4,
        24,
    ),
    _Cue(
        re.compile(
            r"\b(?:id\s+number|identification\s+number|membership\s+number|"
            r"reference\s+number)\b"
            r"(?:\s*(?:is|equals|:|-)\s*)?",
            flags=re.IGNORECASE,
        ),
        "IDCARDNUM",
        "supplemental_identifier_filler",
        4,
        24,
    ),
)


def _scan_sequence(
    text: str, start: int, *, max_fillers: int
) -> tuple[int, int, int] | None:
    """Walk forward from `start` collecting one filler-tolerant digit run.

    Returns ``(span_start, span_end, digit_count)`` where the span is trimmed to
    the first and last genuine digit token, so leading and trailing fillers are
    never swallowed into the placeholder. Returns ``None`` when no digit is
    found before the run breaks.
    """

    digits = 0
    fillers = 0
    first: int | None = None
    last: int | None = None
    cursor = start
    for match in _TOKEN_RE.finditer(text, start):
        gap = text[cursor : match.start()]
        # Anything other than ordinary number punctuation ends the run.
        if gap and _SEPARATOR_RE.fullmatch(gap) is None:
            break
        token = match.group(0).lower()
        if token.isdigit():
            digits += len(token)
        elif token in _DIGIT_VOCABULARY:
            digits += 1
        elif token in _FILLER_WORDS:
            if fillers >= max_fillers:
                break
            fillers += 1
            cursor = match.end()
            continue
        else:
            break
        if first is None:
            first = match.start()
        last = match.end()
        cursor = match.end()
    if first is None or last is None:
        return None
    return first, last, digits


def _uncued_runs(text: str, minimum_digits: int) -> list[tuple[int, int, int]]:
    """Every maximal run of spoken/written digits, ignoring cues entirely."""

    runs: list[tuple[int, int, int]] = []
    first: int | None = None
    last: int | None = None
    digits = 0
    cursor = 0

    def flush() -> None:
        nonlocal first, last, digits
        if first is not None and last is not None and digits >= minimum_digits:
            runs.append((first, last, digits))
        first = None
        last = None
        digits = 0

    for match in _TOKEN_RE.finditer(text):
        gap = text[cursor : match.start()]
        contiguous = not gap or _SEPARATOR_RE.fullmatch(gap) is not None
        token = match.group(0).lower()
        is_digit = token.isdigit() or token in _DIGIT_VOCABULARY
        if not is_digit or not contiguous:
            flush()
        if is_digit:
            if first is None:
                first = match.start()
            last = match.end()
            digits += len(token) if token.isdigit() else 1
        cursor = match.end()
    flush()
    return runs


class SupplementalPIIRecognizer:
    """Find the structured spans the frozen recognizer provably misses."""

    def __init__(
        self,
        *,
        max_fillers: int = MAX_FILLERS_IN_SEQUENCE,
        uncued_minimum_digits: int = UNCUED_RUN_MINIMUM_DIGITS,
    ):
        self.max_fillers = max_fillers
        self.uncued_minimum_digits = uncued_minimum_digits

    def find_spans(
        self, text: str, covered: Sequence[tuple[int, int]] = ()
    ) -> list[StructuredPIISpan]:
        """Supplemental spans for `text`.

        `covered` lists character ranges an upstream recogniser already claimed.
        The cue-less long-run fallback defers to them: it exists only to stop a
        number escaping entirely, never to relabel one that is already
        correctly identified.
        """

        if not isinstance(text, str):
            raise TypeError(f"text must be str, received {type(text).__name__}.")
        spans: list[StructuredPIISpan] = []
        for cue in (*_CUES, *_FROZEN_CUE_RESCAN):
            for hit in cue.pattern.finditer(text):
                found = _scan_sequence(text, hit.end(), max_fillers=self.max_fillers)
                if found is None:
                    continue
                start, end, digits = found
                if not cue.minimum_digits <= digits <= cue.maximum_digits:
                    continue
                spans.append(
                    StructuredPIISpan(
                        entity_type=cue.entity_type,
                        start_char=start,
                        end_char=end,
                        text=text[start:end],
                        source=cue.source,
                        confidence=1.0,
                    )
                )
        # The generic fallback defers to anything already identified, including
        # the cued spans found just above. Without this it would win the
        # alphabetical tie-break in merge_pii_spans and relabel a correctly
        # detected [TELEPHONENUM_1] as a vaguer [NUMERICID_1].
        claimed = [(span.start_char, span.end_char) for span in spans]
        claimed.extend((int(start), int(end)) for start, end in covered)
        for start, end, _digits in _uncued_runs(text, self.uncued_minimum_digits):
            if any(start < other_end and other_start < end for other_start, other_end in claimed):
                continue
            spans.append(
                StructuredPIISpan(
                    # Deliberately not TELEPHONENUM/IDCARDNUM: with no cue the
                    # kind of number is genuinely unknown, and claiming one
                    # would put a false statement in the stored transcript.
                    entity_type="NUMERICID",
                    start_char=start,
                    end_char=end,
                    text=text[start:end],
                    source="supplemental_long_run",
                    confidence=1.0,
                )
            )
        unique: dict[tuple[str, int, int], StructuredPIISpan] = {}
        for span in spans:
            unique[(span.entity_type, span.start_char, span.end_char)] = span
        return sorted(
            unique.values(),
            key=lambda span: (span.start_char, span.end_char, span.entity_type),
        )


class SupplementalPIIMasker:
    """Re-merge an inner masker's spans with the supplemental recognizer's.

    Wraps the masker rather than replacing it, and rebuilds the masked text
    through the authoritative `merge_pii_spans`, so placeholder numbering,
    overlap resolution and structured-over-model priority stay exactly as the
    frozen merger defines them.

    Exceptions are intentionally allowed to propagate: `FailClosedMasker` sits
    outside this wrapper and converts them into a blocked, non-releasable
    result. Failing to release is the correct direction for a privacy layer.
    """

    def __init__(
        self,
        masker,
        recognizer: SupplementalPIIRecognizer | None = None,
        mask_policy=None,
    ):
        self.masker = masker
        self.recognizer = recognizer or SupplementalPIIRecognizer()
        # This wrapper rebuilds masked_text through merge_pii_spans, so it has
        # to apply the same policy the inner masker used. Otherwise a label the
        # operator disabled would be re-masked here. Inheriting from the inner
        # masker keeps every existing single-argument call site working.
        self.mask_policy = (
            mask_policy if mask_policy is not None else getattr(masker, "mask_policy", None)
        )

    def mask_text(self, text: str):
        result = self.masker.mask_text(text)
        if not isinstance(text, str) or not text.strip():
            return result
        if not result.safe_to_release:
            return result
        already = [
            (int(entity["start"]), int(entity["end"])) for entity in result.entities
        ]
        extra = self.recognizer.find_spans(text, already)
        if not extra:
            return result

        structured: list[SpanCandidate] = []
        model: list[SpanCandidate] = []
        for entity in result.entities:
            candidate = SpanCandidate(
                entity_type=str(entity["entity_type"]),
                start_char=int(entity["start"]),
                end_char=int(entity["end"]),
                text=str(entity["text"]),
                source=str(entity.get("source", "")),
                confidence=float(entity.get("confidence", 1.0)),
                metadata=dict(entity.get("metadata", {})),
            )
            if candidate.source.startswith("structured"):
                structured.append(candidate)
            else:
                model.append(candidate)
        structured.extend(span.as_candidate() for span in extra)

        merged = merge_pii_spans(text, model, structured, mask_policy=self.mask_policy)
        added = sum(
            1
            for entity in merged.entities
            if str(entity.get("source", "")).startswith("structured:supplemental")
        )
        result.masked_text = merged.masked_text
        result.entities = merged.entities
        details = dict(result.raw_model_output or {})
        details["supplemental_spans_added"] = added
        details["supplemental_latency_seconds"] = merged.latency_seconds
        details["mask_policy"] = merged.mask_policy
        details["policy_exempt_span_count"] = merged.policy_exempt_span_count
        result.raw_model_output = details
        return result

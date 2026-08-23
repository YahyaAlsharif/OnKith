"""Whole-word consolidation and deterministic PII span overlap resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

try:
    from .mask_policy import resolve as resolve_mask_policy
except ImportError:  # Direct execution without the package context.
    from mask_policy import resolve as resolve_mask_policy


@dataclass(frozen=True)
class SpanCandidate:
    entity_type: str
    start_char: int
    end_char: int
    text: str
    source: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanDecision:
    span: SpanCandidate
    chosen: bool
    reason: str
    final_placeholder: str | None = None


@dataclass
class SpanMergeResult:
    masked_text: str
    entities: list[dict[str, Any]]
    chosen_spans: list[SpanCandidate]
    decisions: list[SpanDecision]
    latency_seconds: float
    #: Description of the masking policy that produced ``masked_text``.
    mask_policy: dict[str, Any] = field(default_factory=dict)
    #: Spans detected and reported but left unmasked by explicit policy.
    policy_exempt_span_count: int = 0


@dataclass
class WordConsolidationResult:
    labels: list[str]
    spans: list[SpanCandidate]
    warnings: list[str]
    partial_word_boundary_count: int


#: Punctuation that sits *inside* one entity - the ``@`` and ``.`` of an email,
#: the ``://`` and ``/`` of a URL, the dots of an IPv4 address, the dashes of a
#: phone number or a date. A run of these characters joins the words on either
#: side of it rather than separating them.
_TIGHT_CONNECTORS = frozenset("@./:-+_#&=?%~'\\")

#: Longest connector that can still sit inside a single entity ("://").
_MAX_CONNECTOR_LENGTH = 3


def _trim_whitespace_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink an offset onto its first and last non-whitespace character.

    SentencePiece tokenizers (DeBERTa-v2/v3 among them) report the whitespace
    that precedes a token as part of that token's offset. WordPiece does not.
    Trimming first means the rest of this module sees the same thing from both.
    """

    start = max(0, min(int(start), len(text)))
    end = max(start, min(int(end), len(text)))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _holds_alphanumeric(text: str, start: int, end: int) -> bool:
    return any(text[position].isalnum() for position in range(start, end))


def _alphanumeric_word_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand an aligned token only across the containing alphanumeric word.

    Two tokenizer realities are handled before any expansion happens.

    A SentencePiece offset that starts on the space before its word must be
    trimmed, because expanding outwards from that space walks straight into the
    previous word: an ``AGE`` token offset covering ``" 9"`` would otherwise
    grow left across ``"by"`` and yield ``"by 9"``.

    A token that holds no alphanumeric character - the ``@`` in an email, the
    ``.`` in an IPv4 address - belongs to no word at all. Expanding it reaches
    into the words on *both* sides and invents a group that spans the
    punctuation, which is where duplicate overlapping spans come from. Such a
    token is returned exactly as it sits.
    """

    start, end = _trim_whitespace_bounds(text, start, end)
    if start == end or not _holds_alphanumeric(text, start, end):
        return start, end
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return start, end


def _connector_kind(gap: str) -> str | None:
    """Classify the text between two consecutive same-type words.

    Returns ``None`` when the gap is too long or too wordy to sit inside a
    single entity, otherwise one of:

    ``"contiguous"``  nothing separates them.
    ``"whitespace"``  whitespace only - the historical merge rule.
    ``"tight"``       punctuation with no whitespace, as in ``a@b.com``.
    ``"loose"``       punctuation including whitespace, as in ``"13th, 2023"``.
    """

    if not gap:
        return "contiguous"
    if gap.isspace():
        return "whitespace"
    if len(gap) > _MAX_CONNECTOR_LENGTH or any(character.isalnum() for character in gap):
        return None
    if any(character.isspace() for character in gap):
        return "loose"
    if all(character in _TIGHT_CONNECTORS for character in gap):
        return "tight"
    return None


def derive_word_ids(text: str, offsets: Sequence[tuple[int, int]]) -> list[int | None]:
    """Derive stable word groups from exact offsets when tokenizer IDs are absent."""

    key_to_id: dict[tuple[int, int], int] = {}
    result: list[int | None] = []
    for start, end in offsets:
        if start == end:
            result.append(None)
            continue
        bounds = _alphanumeric_word_bounds(text, start, end)
        if bounds[0] == bounds[1]:
            # Nothing but whitespace: the token belongs to no word and must not
            # seed a group of its own. Punctuation keeps its own non-expandable
            # group, which `_alphanumeric_word_bounds` now returns untouched.
            result.append(None)
            continue
        if bounds not in key_to_id:
            key_to_id[bounds] = len(key_to_id)
        result.append(key_to_id[bounds])
    return result


def consolidate_predictions_to_words(
    text: str,
    offsets: Sequence[tuple[int, int]],
    labels: Sequence[str],
    confidences: Sequence[float],
    *,
    word_ids: Sequence[int | None] | None = None,
) -> WordConsolidationResult:
    """Promote subtoken evidence to complete words using a deterministic rule.

    For every original word, each non-O entity type receives a
    character-length-weighted mean of its chosen-label confidences. The highest
    score wins; ties are resolved alphabetically by entity type. A sensitive
    word is expanded only to its complete alphanumeric boundaries.

    Adjacent same-type words become one entity span when nothing but
    whitespace, or a short run of intra-entity punctuation, separates them.
    Punctuation-only gaps that also contain whitespace additionally require the
    tagger to have marked the second word as a continuation. See
    ``_connector_kind``.
    """

    if not (len(offsets) == len(labels) == len(confidences)):
        raise ValueError("offsets, labels, and confidences must have equal lengths.")
    if word_ids is None or len(word_ids) != len(offsets):
        word_ids = derive_word_ids(text, offsets)

    groups: dict[Any, list[int]] = {}
    for index, (offset, word_id) in enumerate(zip(offsets, word_ids)):
        if offset[0] == offset[1]:
            continue
        trimmed_start, trimmed_end = _trim_whitespace_bounds(text, *offset)
        if trimmed_start == trimmed_end:
            # A whitespace-only offset carries no characters to mask.
            continue
        group_key: Any = ("tokenizer", word_id) if word_id is not None else (
            "derived",
            *_alphanumeric_word_bounds(text, *offset),
        )
        groups.setdefault(group_key, []).append(index)

    word_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for indices in groups.values():
        starts = [int(offsets[index][0]) for index in indices]
        ends = [int(offsets[index][1]) for index in indices]
        group_start, group_end = min(starts), max(ends)
        word_start, word_end = _alphanumeric_word_bounds(text, group_start, group_end)
        if word_start == word_end:
            # Fall back to the trimmed group, never to the raw offset: the raw
            # one can begin on the whitespace a SentencePiece token absorbed.
            word_start, word_end = _trim_whitespace_bounds(text, group_start, group_end)
        evidence: dict[str, list[tuple[float, int]]] = {}
        for index in indices:
            label = str(labels[index])
            if label == "O" or "-" not in label:
                continue
            entity_type = label.split("-", maxsplit=1)[1]
            token_length = max(1, int(offsets[index][1]) - int(offsets[index][0]))
            evidence.setdefault(entity_type, []).append(
                (float(confidences[index]), token_length)
            )
        if evidence:
            scored = []
            for entity_type, values in evidence.items():
                total_weight = sum(weight for _, weight in values)
                score = sum(confidence * weight for confidence, weight in values)
                score /= total_weight
                scored.append((score, entity_type))
            scored.sort(key=lambda item: (-item[0], item[1]))
            winning_score, winning_type = scored[0]
            if len(scored) > 1:
                warnings.append(
                    "Conflicting subword entity types for "
                    f"{text[word_start:word_end]!r}: "
                    + ", ".join(f"{kind}={score:.6f}" for score, kind in scored)
                    + f"; chose {winning_type}."
                )
        else:
            winning_type = None
            winning_score = 0.0
        # Does this word open a new entity or continue the one before it? The
        # tagger already answers that with the B-/I- prefix on its first token,
        # and that answer is what keeps two adjacent same-type entities apart
        # when only punctuation separates them.
        leading_index = min(indices, key=lambda index: tuple(offsets[index]))
        leading_label = str(labels[leading_index])
        word_rows.append(
            {
                "indices": indices,
                "start": word_start,
                "end": word_end,
                "entity_type": winning_type,
                "confidence": float(winning_score),
                "continues": (
                    winning_type is not None
                    and leading_label == f"I-{winning_type}"
                ),
            }
        )

    word_rows.sort(key=lambda row: (row["start"], row["end"]))
    consolidated_labels = ["O"] * len(offsets)
    spans: list[SpanCandidate] = []
    active: dict[str, Any] | None = None
    previous_sensitive_word: dict[str, Any] | None = None
    for row in word_rows:
        entity_type = row["entity_type"]
        if entity_type is None:
            if _connector_kind(text[row["start"] : row["end"]]) in ("tight", "loose"):
                # Punctuation the tagger left as O still sits inside whatever
                # entity surrounds it. Stay transparent so the words on either
                # side can still find each other.
                continue
            active = None
            previous_sensitive_word = None
            continue
        connector = (
            _connector_kind(text[previous_sensitive_word["end"] : row["start"]])
            if previous_sensitive_word is not None
            else None
        )
        compatible_previous = (
            previous_sensitive_word is not None
            and previous_sensitive_word["entity_type"] == entity_type
            and connector is not None
            # Whitespace and tight punctuation join unconditionally, which keeps
            # the historical whole-word behaviour and repairs emails, URLs, IPv4
            # addresses and phone numbers. A looser gap such as the ", " in
            # "May 13th, 2023" also has to be confirmed by an I- continuation,
            # so that a list like "London, Paris" stays two entities.
            and (connector != "loose" or row["continues"])
        )
        for position, token_index in enumerate(row["indices"]):
            prefix = "I" if compatible_previous or position > 0 else "B"
            consolidated_labels[token_index] = f"{prefix}-{entity_type}"
        if compatible_previous and active is not None:
            active["end"] = row["end"]
            active["confidence_values"].append(row["confidence"])
        else:
            active = {
                "start": row["start"],
                "end": row["end"],
                "entity_type": entity_type,
                "confidence_values": [row["confidence"]],
            }
            spans.append(
                SpanCandidate(
                    entity_type=entity_type,
                    start_char=row["start"],
                    end_char=row["end"],
                    text=text[row["start"] : row["end"]],
                    source="model_whole_word",
                    confidence=row["confidence"],
                    metadata={"word_level": True},
                )
            )
        if compatible_previous and spans:
            values = active["confidence_values"]
            spans[-1] = SpanCandidate(
                entity_type=entity_type,
                start_char=active["start"],
                end_char=active["end"],
                text=text[active["start"] : active["end"]],
                source="model_whole_word",
                confidence=sum(values) / len(values),
                metadata={"word_level": True, "word_count": len(values)},
            )
        previous_sensitive_word = row

    return WordConsolidationResult(
        labels=consolidated_labels,
        spans=spans,
        warnings=warnings,
        partial_word_boundary_count=count_partial_word_boundaries(text, spans),
    )


def count_partial_word_boundaries(
    text: str, spans: Iterable[SpanCandidate | dict[str, Any]]
) -> int:
    count = 0
    for span in spans:
        start = (
            span.start_char if isinstance(span, SpanCandidate) else int(span["start"])
        )
        end = span.end_char if isinstance(span, SpanCandidate) else int(span["end"])
        if start > 0 and start < len(text) and text[start - 1].isalnum() and text[start].isalnum():
            count += 1
        if end > 0 and end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
            count += 1
    return count


def _overlap(first: SpanCandidate, second: SpanCandidate) -> bool:
    return first.start_char < second.end_char and second.start_char < first.end_char


def merge_pii_spans(
    text: str,
    model_spans: Sequence[SpanCandidate],
    structured_spans: Sequence[SpanCandidate],
    mask_policy: Any = None,
) -> SpanMergeResult:
    """Prefer exact structured spans, then deterministically keep non-overlaps.

    ``mask_policy`` is applied at the very last step: after overlap resolution
    has chosen the authoritative spans, but before any of them is replaced in
    ``masked_text``. A span whose entity type is disabled stays in ``entities``
    with ``masked=False`` and no placeholder, so detection is fully reported
    while the original text is preserved. Omitting the argument masks
    everything, exactly as before.
    """

    import time

    started = time.perf_counter()
    candidates = [*structured_spans, *model_spans]
    for span in candidates:
        if (
            span.start_char < 0
            or span.end_char <= span.start_char
            or span.end_char > len(text)
            or text[span.start_char : span.end_char] != span.text
        ):
            raise ValueError(f"Invalid source-aligned span: {span!r}")
    ordered = sorted(
        candidates,
        key=lambda span: (
            0 if span.source.startswith("structured") else 1,
            -(span.end_char - span.start_char),
            -float(span.confidence),
            span.start_char,
            span.end_char,
            span.entity_type,
            span.source,
        ),
    )
    chosen: list[SpanCandidate] = []
    decisions: list[SpanDecision] = []
    for candidate in ordered:
        conflict = next((existing for existing in chosen if _overlap(candidate, existing)), None)
        if conflict is None:
            chosen.append(candidate)
            decisions.append(SpanDecision(candidate, True, "selected_non_overlapping"))
        else:
            decisions.append(
                SpanDecision(
                    candidate,
                    False,
                    "discarded_overlap_with_"
                    f"{conflict.source}:{conflict.start_char}-{conflict.end_char}",
                )
            )
    policy = resolve_mask_policy(mask_policy)
    chosen.sort(key=lambda span: (span.start_char, span.end_char, span.entity_type))
    parts: list[str] = []
    entities: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    cursor = 0
    policy_exempt = 0
    decision_by_span = {id(decision.span): decision for decision in decisions}
    for span in chosen:
        if span.start_char < cursor:
            raise RuntimeError("Overlap resolution produced overlapping chosen spans.")
        parts.append(text[cursor : span.start_char])
        masked = policy.should_mask(span.entity_type)
        if masked:
            counts[span.entity_type] = counts.get(span.entity_type, 0) + 1
            placeholder = f"[{span.entity_type}_{counts[span.entity_type]}]"
            parts.append(placeholder)
        else:
            # Detected, reported, and deliberately left in the released text.
            placeholder = None
            parts.append(span.text)
            policy_exempt += 1
        entity = {
            "start": span.start_char,
            "end": span.end_char,
            "entity_type": span.entity_type,
            "text": span.text,
            "placeholder": placeholder,
            "source": span.source,
            "confidence": span.confidence,
            "metadata": span.metadata,
            "masked": masked,
        }
        entities.append(entity)
        decision = decision_by_span[id(span)]
        decision.final_placeholder = placeholder
        if not masked:
            decision.reason = f"{decision.reason};mask_policy_disabled"
        cursor = span.end_char
    parts.append(text[cursor:])
    return SpanMergeResult(
        masked_text="".join(parts),
        entities=entities,
        chosen_spans=chosen,
        decisions=decisions,
        latency_seconds=time.perf_counter() - started,
        mask_policy=policy.as_dict(),
        policy_exempt_span_count=policy_exempt,
    )


def decision_to_dict(decision: SpanDecision) -> dict[str, Any]:
    return {
        "span": asdict(decision.span),
        "chosen": decision.chosen,
        "reason": decision.reason,
        "final_placeholder": decision.final_placeholder,
    }

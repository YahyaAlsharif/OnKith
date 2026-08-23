"""Put the real values back after the model has answered.

The masker replaces each private span with a placeholder such as
``[GIVENNAME_1]`` and records what it replaced. The LLM only ever sees
placeholders; this module puts the originals back on the way out, so the
spoken reply says the person's actual name while nothing private was ever
sent anywhere.

The map is local and stays local. It is never sent to the model and never
written to a log.

It does reach a surface, which is a deliberate narrowing of an earlier
guarantee. A surface here is served from localhost to a browser on this
device, in the room with the person who spoke: reading "[GIVENNAME_1]" back
to somebody about their own name is theatre rather than privacy. What the
model is given is unchanged, and so is the transcript database, which keeps
the placeholders because it is the only artifact that outlives the process
and therefore the only one that can be carried off on a card.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# Matches the masker's own placeholder shape, e.g. [TELEPHONENUM_1].
PLACEHOLDER_PATTERN = re.compile(r"\[([A-Z][A-Z0-9_]*)_(\d+)\]")

MAX_TRACKED_PLACEHOLDERS = 200


def map_from_entities(entities: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Build placeholder -> original value from one masked utterance."""

    mapping: dict[str, str] = {}
    for entity in entities or ():
        if not isinstance(entity, dict):
            continue
        placeholder = entity.get("placeholder")
        original = entity.get("text")
        if isinstance(placeholder, str) and isinstance(original, str):
            if placeholder and original:
                mapping[placeholder] = original
    return mapping


def rehydrate(text: str, mapping: dict[str, str]) -> str:
    """Replace every known placeholder in ``text`` with its original value.

    A placeholder with no entry is left exactly as it is: inventing a value
    would be worse than saying the token aloud, and it makes the gap
    audible rather than silent.
    """

    if not text or not mapping:
        return text or ""

    def substitute(match: re.Match[str]) -> str:
        return mapping.get(match.group(0), match.group(0))

    return PLACEHOLDER_PATTERN.sub(substitute, text)


def rehydrate_with_spans(
    text: str, mapping: dict[str, str]
) -> tuple[str, list[dict[str, Any]]]:
    """Put the values back, and say where each one landed.

    The screen marks these, which is the whole demonstration: the room can
    see the exact words that were held back from the model still sitting in
    the sentence, in the one colour reserved for the cover lifting. Offsets
    are into the returned text, not the placeholder text, because the
    returned text is what gets drawn.
    """

    if not text:
        return "", []

    rebuilt: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    length = 0
    for match in PLACEHOLDER_PATTERN.finditer(text):
        before = text[cursor : match.start()]
        rebuilt.append(before)
        length += len(before)
        cursor = match.end()

        placeholder = match.group(0)
        value = mapping.get(placeholder)
        # An unknown placeholder is left exactly as it is and is not marked.
        # Nothing was revealed, so there is nothing to point at.
        shown = placeholder if value is None else value
        rebuilt.append(shown)
        if value is not None:
            spans.append(
                {"start": length, "end": length + len(shown), "kind": match.group(1)}
            )
        length += len(shown)

    rebuilt.append(text[cursor:])
    return "".join(rebuilt), spans


def unresolved_placeholders(text: str, mapping: dict[str, str]) -> list[str]:
    """Report placeholders the reply used that the map cannot fill."""

    return [
        found
        for found in PLACEHOLDER_PATTERN.findall(text or "")
        if f"[{found[0]}_{found[1]}]" not in mapping
    ]


class RehydrationMap:
    """A bounded, rolling placeholder map for one utterance at a time.

    History is kept across turns because a reply can refer to a
    placeholder introduced several turns earlier.

    Not what the assistant runs on. Because the masker numbers placeholders
    per utterance, ``add`` lets a later turn's ``[GIVENNAME_1]`` overwrite an
    earlier turn's, and a reply about the earlier one is then restored with
    the later one's value. ``ConversationPlaceholders`` below renumbers into a
    conversation-wide space so that cannot happen, and is what
    ``AssistantWorker`` uses. This class is kept for single-utterance callers
    and for the benchmark's rehydration probes, which supply their own map.
    """

    def __init__(self, limit: int = MAX_TRACKED_PLACEHOLDERS) -> None:
        self._limit = limit
        self._values: dict[str, str] = {}

    def add(self, entities: Iterable[dict[str, Any]]) -> None:
        self._values.update(map_from_entities(entities))
        excess = len(self._values) - self._limit
        if excess > 0:
            for key in list(self._values)[:excess]:
                del self._values[key]

    def apply(self, text: str) -> str:
        return rehydrate(text, self._values)

    def apply_with_spans(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        return rehydrate_with_spans(text, self._values)

    def missing(self, text: str) -> list[str]:
        return unresolved_placeholders(text, self._values)

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)


class ConversationPlaceholders:
    """Conversation-unique placeholders, allocated locally, never colliding.

    The masker numbers placeholders per utterance. "Remind me to call Fahad"
    and, a minute later, "remind me to call Ali" therefore both arrive as
    ``[GIVENNAME_1]``. A single rolling map across the conversation has Ali
    overwrite Fahad, while the model's own history still holds the first turn
    verbatim - so a reply that refers back to it is restored with the wrong
    person's name. Nothing leaked, but the device told somebody it would call
    the wrong person, which is its own kind of failure.

    This allocates a conversation-scoped number instead. Fahad keeps
    ``[GIVENNAME_1]``; Ali becomes ``[GIVENNAME_2]``; both keep those numbers
    for as long as the conversation lasts. The map only ever grows, so a
    placeholder cannot change meaning underneath a reply that already used it,
    and the wrong value cannot be restored because the right one was never
    displaced.

    A value that comes back in a later turn is given the number it had the
    first time. That is deliberate: it is also the only way the model can tell
    that two turns are about the same person, which single-utterance numbering
    threw away.

    The allocation, the map and the rewriting are all local. What changes on
    the wire is only which number a placeholder carries; no original value is
    sent, and the mapping is never part of a request.
    """

    def __init__(self, limit: int = MAX_TRACKED_PLACEHOLDERS) -> None:
        self._limit = limit
        #: conversation placeholder -> original value
        self._values: dict[str, str] = {}
        #: (kind, original value) -> conversation placeholder
        self._issued: dict[tuple[str, str], str] = {}
        #: highest number handed out per kind. Never reset and never reused,
        #: even after eviction below, because reusing a number is precisely
        #: the collision this class exists to prevent.
        self._counters: dict[str, int] = {}

    def adopt(
        self, masked_text: str, entities: Iterable[dict[str, Any]]
    ) -> tuple[str, dict[str, str]]:
        """Renumber one utterance's placeholders into the conversation's own.

        Returns the masked text to send and the placeholder -> value pairs this
        utterance contributed. The text is rewritten in a single pass so that
        renumbering ``[GIVENNAME_1]`` to ``[GIVENNAME_2]`` cannot then be picked
        up again by the rule for ``[GIVENNAME_2]``.
        """

        local = map_from_entities(entities)
        if not local:
            return masked_text or "", {}

        translation: dict[str, str] = {}
        contributed: dict[str, str] = {}
        for placeholder, original in local.items():
            match = PLACEHOLDER_PATTERN.fullmatch(placeholder)
            if match is None:
                # Not the masker's shape. Left exactly as it is rather than
                # guessed at: an unrecognised token is not a private value this
                # class knows how to account for.
                continue
            kind = match.group(1)
            issued = self._issued.get((kind, original))
            if issued is None:
                number = self._counters.get(kind, 0) + 1
                self._counters[kind] = number
                issued = f"[{kind}_{number}]"
                self._issued[(kind, original)] = issued
            translation[placeholder] = issued
            contributed[issued] = original

        if not translation:
            return masked_text or "", {}

        self._values.update(contributed)
        self._evict()

        def substitute(match: "re.Match[str]") -> str:
            return translation.get(match.group(0), match.group(0))

        rewritten = PLACEHOLDER_PATTERN.sub(substitute, masked_text or "")
        return rewritten, contributed

    def _evict(self) -> None:
        """Bound the map without ever freeing a number for reuse.

        An evicted placeholder becomes unresolvable, and ``rehydrate`` leaves
        an unresolvable placeholder visible rather than filling it. That is the
        safe direction to fail: the token is read out, which is obvious, rather
        than being quietly filled with somebody else's value.
        """

        excess = len(self._values) - self._limit
        if excess <= 0:
            return
        for key in list(self._values)[:excess]:
            value = self._values.pop(key)
            self._issued.pop((PLACEHOLDER_PATTERN.fullmatch(key).group(1), value), None)

    def apply(self, text: str) -> str:
        return rehydrate(text, self._values)

    def apply_with_spans(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        return rehydrate_with_spans(text, self._values)

    def missing(self, text: str) -> list[str]:
        return unresolved_placeholders(text, self._values)

    def value_of(self, placeholder: str) -> str | None:
        return self._values.get(placeholder)

    def clear(self) -> None:
        self._values.clear()
        self._issued.clear()
        self._counters.clear()

    def __len__(self) -> int:
        return len(self._values)

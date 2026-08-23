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
    """A bounded, rolling placeholder map for one conversation.

    History is kept across turns because a reply can refer to a
    placeholder introduced several turns earlier.
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

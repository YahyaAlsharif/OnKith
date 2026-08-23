"""Repair Canary mis-transcriptions of the wake word before interception.

Canary runs with ``pnc: no`` and has no lexicon bias toward "Onkith", so it
routinely emits "on kids", "on kiss", "on care", "uncle" or "aunt Keith"
instead. `onkith_music._find_wake` covers a handful of near-misses ("on kith",
"on kit", fuzzy "on k___") but not the phonetically-distant ones, so those
utterances never reach the interceptor as commands at all.

This module rewrites a known mis-transcription to the canonical ``Onkith``. It
runs on the raw transcript, which is what `_release_final` already hands to
`intercept_onkith_command`, so it lands before both the interceptor and PII
masking. That ordering matters for "aunt Keith": once the masker has turned it
into "aunt [GIVENNAME_1]" the wake word is gone for good.

Position is the whole safety mechanism
--------------------------------------
"uncle", "aunt" and "Ankit" are ordinary words. Rewriting them anywhere in a
sentence would corrupt normal speech and, in Ankit's case, contradict a
deliberate decision in `onkith_music` (a self-test asserts that "Someone named
Ankit called" is not a command). So a variant is only rewritten when it sits at
the *wake position*: the start of the utterance, optionally preceded by a short
greeting such as "hello" or "hey". That is why:

    "on kids play music"          -> "Onkith play music"
    "hello uncle did you hear me" -> "hello Onkith did you hear me"
    "Someone named Ankit called"  -> unchanged
    "my uncle is visiting"        -> unchanged

Only the *interceptor's* copy is normalized. When no command is found the
original transcript is what continues downstream, so storage and the masked
record keep the speaker's actual words.
"""

from __future__ import annotations

import re

CANONICAL_WAKE_WORD = "Onkith"

# Checked before the single-token table so "aunt Keith" wins over bare "aunt".
_TWO_TOKEN_VARIANTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("on", "kith"),
        ("on", "kit"),
        ("on", "kids"),
        ("on", "kid"),
        ("on", "kiss"),
        ("on", "care"),
        ("on", "keith"),
        ("aunt", "keith"),
    }
)

# Ordinary English words. They are only ever rewritten at the wake position.
_SINGLE_TOKEN_VARIANTS: frozenset[str] = frozenset({"uncle", "aunt", "ankit"})

# Words allowed to precede the wake word without breaking "utterance start".
# Deliberately short: these are vocatives and hesitation sounds, not content.
_LEADING_FILLERS: frozenset[str] = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "ok",
        "okay",
        "yo",
        "hallo",
        "hullo",
        "um",
        "uh",
        "er",
        "so",
        "well",
        "please",
    }
)

# At most this many fillers may sit before the wake word. Beyond that the
# position no longer "strongly matches" and the words are left alone.
_MAXIMUM_LEADING_FILLERS = 2

# Same word shape as onkith_music._WORD_RE, so tokenisation agrees with the
# interceptor that consumes the result.
_WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)


def normalize_wake_word(text: str) -> tuple[str, str | None]:
    """Rewrite a leading wake-word mis-transcription to ``Onkith``.

    Returns ``(normalized_text, matched_variant)``. ``matched_variant`` is
    ``None`` when nothing matched, in which case the text is returned
    unchanged and callers should keep using the original transcript.

    Never raises: a wake-word convenience must not be able to drop an
    utterance.
    """

    try:
        if not isinstance(text, str) or not text.strip():
            return text, None

        matches = list(_WORD_RE.finditer(text))
        if not matches:
            return text, None

        words = [match.group(0).lower() for match in matches]

        # Walk past a short greeting, then require a match right there.
        start = 0
        while (
            start < len(words)
            and start < _MAXIMUM_LEADING_FILLERS
            and words[start] in _LEADING_FILLERS
        ):
            start += 1
        if start >= len(words):
            return text, None

        if start + 1 < len(words):
            pair = (words[start], words[start + 1])
            if pair in _TWO_TOKEN_VARIANTS:
                return (
                    _splice(text, matches[start].start(), matches[start + 1].end()),
                    " ".join(pair),
                )

        if words[start] in _SINGLE_TOKEN_VARIANTS:
            return (
                _splice(text, matches[start].start(), matches[start].end()),
                words[start],
            )

        return text, None
    except Exception:
        return text, None


def _splice(text: str, start: int, end: int) -> str:
    """Replace ``[start:end)`` with the canonical word, keeping punctuation.

    Only the matched word span is touched, so a trailing comma or full stop
    after the variant survives: "on kids, play music" -> "Onkith, play music".
    """

    return f"{text[:start]}{CANONICAL_WAKE_WORD}{text[end:]}"

"""Render-time conversion of spoken number words into numerals.

Display only. Canary runs with ``pnc: no``, so numbers arrive as words
("the code is four seven two nine"); this rewrites them into numerals
("the code is 4729") for whatever is about to be printed. Nothing upstream of
rendering ever sees the result: storage, detection, Onkith and the preview's
blur/stability arithmetic all keep operating on the original text.

Five passes:

1. Tokenize - split into whitespace-preserving atoms so the string can be
   rebuilt exactly.
2. Classify - ONES, TEEN, TENS, OH, MULT, MERIDIEM, OPAQUE, UNSTABLE or OTHER.
3. Resolve context words - ``oh``/``double``/``triple`` are numeric only next
   to a real digit word.
4. Build units - one unit is one spoken value, which may span two atoms
   ("twenty five" -> 25, "double four" -> 44).
5. Group and render - units that are adjacent form a run; the run's *context*
   decides how it prints.

Why a grammar rather than one regex: whether a word is a number depends on its
neighbours ("double click" vs "double four"), and how a number prints depends
on the phrase around it ("thirty seconds" -> 30, "code is thirty zero one two"
-> 30012, "at three thirty" -> 3:30). That does not compose into a regex.

Two deliberate invariants:

* **Numeric literals are inert.** Only spelled-out words are ever converted, so
  ``normalize(normalize(x)) == normalize(x)`` holds by construction and a
  previously-rendered "30 20" can never later collapse into "3020".
* **Never convert next to unstable text.** A run touching the preview's blur
  token or ``[[REDACTED]]`` is left as words, because the rest of the number
  has not arrived yet and converting now would print "055" and then "05511".
  Ordinary entity placeholders are settled, so a number beside
  ``[GIVENNAME_1]`` still converts.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_MULT = {"double": 2, "triple": 3}
_MERIDIEM = {"am", "pm", "a.m.", "p.m.", "oclock", "o'clock"}

# A run preceded by one of these prints as a single joined value: these words
# introduce an identifier, not a quantity.
_JOIN_CUES = {
    "code",
    "pin",
    "number",
    "extension",
    "ext",
    "id",
    "account",
    "reference",
    "serial",
    "policy",
    "zip",
    "postcode",
    "passport",
    "licence",
    "license",
    "badge",
}
# Skipped when looking back for a join cue.
_LOOKBACK_SKIP = {
    "is", "are", "was", "were", "the", "my", "your", "our", "his", "her",
    "their", "a", "an", "it", "its", "of", "for",
}
_LOOKBACK_LIMIT = 4

# Minimum length for a run of *bare* single digits to print as one number.
# One or two spoken digits are usually prose ("one of the doctors").
MINIMUM_BARE_RUN = 3

# Mirrors live_preview.BLUR_TOKEN / REDACTED_LABEL. Held as literals rather
# than imported so this module has no dependency on the preview, which imports
# it. A mismatch degrades to "converted when it could have waited", so the
# preview has a test asserting these stay in step.
_BLUR_CHAR = "●"
_REDACTED_LABEL = "[[REDACTED]]"

_ATOM_PATTERN = re.compile(r"\S+|\s+")

_fallback_logged = False
_fallback_lock = threading.Lock()


def _log_fallback_once() -> None:
    global _fallback_logged
    if _fallback_logged:
        return
    with _fallback_lock:
        if _fallback_logged:
            return
        _fallback_logged = True
        logger.warning(
            "digit_normalizer: normalisation failed; rendering text "
            "unchanged for the rest of this session"
        )


def _is_unstable(atom: str) -> bool:
    """Text that may still change: the blur run and the redaction marker."""

    return _BLUR_CHAR in atom or _REDACTED_LABEL in atom


def _is_opaque(atom: str) -> bool:
    """Settled placeholders such as ``[GIVENNAME_1]`` - never rewritten."""

    return atom.startswith("[")


@dataclass
class _Unit:
    """One spoken value, spanning one or two atoms.

    ``sequence`` marks a unit that is part of a *digit string* rather than a
    quantity. "four" and "double four" are sequence units (they contribute
    digits: 4, 44); "twenty" and "twenty five" are not (they are the values 20
    and 25). The distinction decides whether neighbouring units concatenate or
    stay separate, which is why "double four seven" prints 447 while "thirty
    twenty" prints 30 20.

    ``forces`` marks a unit whose presence is unambiguous enough to convert the
    whole run, regardless of how short it is. A spoken "twenty" or an explicit
    "double" is a number; a lone "one" may just be prose.
    """

    start: int
    end: int
    digits: str
    value: int
    sequence: bool
    forces: bool


def _classify(atoms: list[str]) -> list[str]:
    kinds: list[str] = []
    for atom in atoms:
        if not atom.strip():
            kinds.append("WS")
        elif _is_unstable(atom):
            kinds.append("UNSTABLE")
        elif _is_opaque(atom):
            kinds.append("OPAQUE")
        else:
            word = atom.lower().strip(".,;:!?")
            if word in _ONES:
                kinds.append("ONES")
            elif word in _TEENS:
                kinds.append("TEEN")
            elif word in _TENS:
                kinds.append("TENS")
            elif word == "oh":
                kinds.append("OH")
            elif word in _MULT:
                kinds.append("MULT")
            elif atom.lower() in _MERIDIEM:
                kinds.append("MERIDIEM")
            else:
                kinds.append("OTHER")
    return kinds


def _word(atom: str) -> str:
    return atom.lower().strip(".,;:!?")


def _neighbour(kinds: list[str], index: int, step: int) -> int | None:
    cursor = index + step
    while 0 <= cursor < len(kinds):
        if kinds[cursor] != "WS":
            return cursor
        cursor += step
    return None


def _resolve_context_words(atoms: list[str], kinds: list[str]) -> None:
    """``oh`` is 0 only beside a digit; ``double``/``triple`` only before one."""

    changed = True
    while changed:
        changed = False
        for index, kind in enumerate(kinds):
            if kind != "OH":
                continue
            before = _neighbour(kinds, index, -1)
            after = _neighbour(kinds, index, 1)
            if (before is not None and kinds[before] == "ONES") or (
                after is not None and kinds[after] == "ONES"
            ):
                kinds[index] = "ONES"
                changed = True
    for index, kind in enumerate(kinds):
        if kind != "MULT":
            continue
        target = _neighbour(kinds, index, 1)
        if target is not None and kinds[target] == "ONES":
            kinds[index] = "MULT_READY"


def _digit_of(atom: str, kind: str) -> int:
    word = _word(atom)
    if kind == "ONES":
        return _ONES.get(word, 0)
    if kind == "TEEN":
        return _TEENS[word]
    return _TENS[word]


def _build_units(atoms: list[str], kinds: list[str]) -> list[_Unit]:
    units: list[_Unit] = []
    index = 0
    total = len(atoms)
    while index < total:
        kind = kinds[index]
        following = _neighbour(kinds, index, 1)
        if kind == "MULT_READY" and following is not None:
            factor = _MULT[_word(atoms[index])]
            digit = _digit_of(atoms[following], "ONES")
            # Digits, not a quantity: "double four" is 44 the string, so it
            # concatenates with any digit beside it.
            units.append(
                _Unit(
                    index,
                    following,
                    str(digit) * factor,
                    digit,
                    sequence=True,
                    forces=True,
                )
            )
            index = following + 1
            continue
        if kind == "TENS":
            # "twenty five" is one value; "thirty zero" is not (there is no
            # such number), so a following zero starts a new unit instead.
            if (
                following is not None
                and kinds[following] == "ONES"
                and _digit_of(atoms[following], "ONES") != 0
            ):
                value = _TENS[_word(atoms[index])] + _digit_of(
                    atoms[following], "ONES"
                )
                units.append(
                    _Unit(
                        index, following, str(value), value, sequence=False, forces=True
                    )
                )
                index = following + 1
                continue
            value = _TENS[_word(atoms[index])]
            units.append(
                _Unit(index, index, str(value), value, sequence=False, forces=True)
            )
            index += 1
            continue
        if kind == "TEEN":
            value = _TEENS[_word(atoms[index])]
            units.append(
                _Unit(index, index, str(value), value, sequence=False, forces=True)
            )
            index += 1
            continue
        if kind == "ONES":
            digit = _digit_of(atoms[index], "ONES")
            units.append(
                _Unit(index, index, str(digit), digit, sequence=True, forces=False)
            )
            index += 1
            continue
        index += 1
    return units


def _runs(units: list[_Unit], kinds: list[str]) -> list[list[_Unit]]:
    """Units separated only by whitespace belong to one run."""

    runs: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in units:
        if current and all(
            kinds[gap] == "WS" for gap in range(current[-1].end + 1, unit.start)
        ):
            current.append(unit)
        else:
            if current:
                runs.append(current)
            current = [unit]
    if current:
        runs.append(current)
    return runs


def _touches_unstable(atoms: list[str], kinds: list[str], run: list[_Unit]) -> bool:
    before = _neighbour(kinds, run[0].start, -1)
    after = _neighbour(kinds, run[-1].end, 1)
    if before is not None and kinds[before] == "UNSTABLE":
        return True
    if after is not None and kinds[after] == "UNSTABLE":
        return True
    return False


def _has_join_cue(atoms: list[str], kinds: list[str], run: list[_Unit]) -> bool:
    cursor = _neighbour(kinds, run[0].start, -1)
    seen = 0
    while cursor is not None and seen < _LOOKBACK_LIMIT:
        word = _word(atoms[cursor])
        if word in _JOIN_CUES:
            return True
        if word not in _LOOKBACK_SKIP:
            return False
        seen += 1
        cursor = _neighbour(kinds, cursor, -1)
    return False


def _time_render(
    atoms: list[str], kinds: list[str], run: list[_Unit]
) -> str | None:
    """``three thirty pm`` -> ``3:30 pm``; ``at one thirty`` -> ``1:30``."""

    if len(run) != 2:
        return None
    hour, minute = run
    if not 1 <= hour.value <= 12 or hour.digits != str(hour.value):
        return None
    if minute.sequence or not 1 <= minute.value <= 59:
        return None
    after = _neighbour(kinds, run[-1].end, 1)
    if after is not None and kinds[after] == "MERIDIEM":
        return f"{hour.value}:{minute.value:02d}"
    before = _neighbour(kinds, run[0].start, -1)
    if before is not None and _word(atoms[before]) == "at":
        return f"{hour.value}:{minute.value:02d}"
    return None


def _render_run(atoms: list[str], kinds: list[str], run: list[_Unit]) -> str | None:
    """How this run prints, or None to leave it as spoken words."""

    if _touches_unstable(atoms, kinds, run):
        return None

    timed = _time_render(atoms, kinds, run)
    if timed is not None:
        return timed

    if _has_join_cue(atoms, kinds, run):
        return "".join(unit.digits for unit in run)

    if not any(unit.forces for unit in run):
        # Nothing here is unambiguously a number, so it converts only when the
        # digit string is long enough to be one. "one of the doctors said two
        # things" must survive untouched.
        if len(run) < MINIMUM_BARE_RUN:
            return None
        return "".join(unit.digits for unit in run)

    # Something in this run is definitely a number, so the whole run converts
    # and nothing is left half spelled. Consecutive sequence units concatenate
    # into one digit string; quantities stay separate values, because the
    # grouping is genuinely ambiguous ("thirty twenty" is two, not 3020).
    pieces: list[str] = []
    group: list[str] = []

    def flush() -> None:
        if group:
            pieces.append("".join(group))
            group.clear()

    for unit in run:
        if unit.sequence:
            group.append(unit.digits)
        else:
            flush()
            pieces.append(unit.digits)
    flush()
    return " ".join(pieces)


def _normalize_impl(text: str) -> str:
    atoms = _ATOM_PATTERN.findall(text)
    if not atoms:
        return text
    kinds = _classify(atoms)
    _resolve_context_words(atoms, kinds)
    units = _build_units(atoms, kinds)
    if not units:
        return text

    replacements: dict[int, tuple[int, str]] = {}
    for run in _runs(units, kinds):
        rendered = _render_run(atoms, kinds, run)
        if rendered is not None:
            replacements[run[0].start] = (run[-1].end, rendered)

    pieces: list[str] = []
    index = 0
    total = len(atoms)
    while index < total:
        replacement = replacements.get(index)
        if replacement is not None:
            end, value = replacement
            pieces.append(value)
            index = end + 1
        else:
            pieces.append(atoms[index])
            index += 1
    return "".join(pieces)


def normalize_spoken_digits(text: str) -> str:
    """Convert spoken number words in ``text`` into numerals, for display.

    Display only: never call this on anything that will be stored, detected
    against, or fed back into the pipeline. Never raises - any failure falls
    back to returning ``text`` unchanged, logged once per session.
    """

    try:
        if not isinstance(text, str):
            return text
        return _normalize_impl(text)
    except Exception:
        _log_fallback_once()
        return text

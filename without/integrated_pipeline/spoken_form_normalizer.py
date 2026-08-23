"""Rewrite two spoken address forms into the written forms the model knows.

The problem this solves
-----------------------
The privacy model was trained on written text. Speech does not arrive written.
V3's recognizer ran with ``pnc: no`` and produced "billing at example dot org";
V4's Whisper writes "billing at example.org". Neither is an email address as
the model - or ``structured_pii``'s ``_EMAIL_PATTERN``, which requires an ``@`` -
has ever seen one, and the audio benchmark shows the consequence on both
pipelines: the domain gets masked, the local part is released in the clear.

    stt      Please forward the invoice to billing at example.org
    masked   Please forward the invoice to billing at [URL_1]
                                          ^^^^^^^ released

Two rewrites close that, and only two, because only two are supported by
observed failures on the benchmark corpus:

    <local> at <domain>          ->  <local>@<domain>
    <domain> slash <path>        ->  <domain>/<path>

Everything else the recognizer does differently - numerals for number words,
sentence punctuation, capitalisation - is written English already and is left
alone. A rule that rewrote those would be guessing at the model's training
distribution rather than repairing a form that is unambiguously not in it.

Offsets are preserved, not discarded
------------------------------------
The rewrite is a view for the model, never the text the device releases. The
transcript the user sees, the transcript stored, and the offsets every span
carries all stay on the original string: :func:`normalise_spoken_forms` returns
the rewritten text together with the edits that produced it, and
:func:`project_offsets` maps a span found in the rewritten text back onto the
region of the original that produced it. A span that lands inside a rewritten
region is widened to cover that whole region, so a fix can only ever mask more
of the original, never less.

Keeping false positives out
---------------------------
"at" is one of the commonest words in English, so the shape has to carry the
evidence. The left side must be a single unspaced token, the right side must be
a dotted domain whose final label is a real top-level domain, and the domain must
not continue into an address that was already written properly - without that
last guard, "Email me at first.last@alpine-solutions.travel" would be rewritten
into "Email me@first.last@alpine-solutions.travel", corrupting an address the
model would otherwise have caught. Scanning the audio corpus, ``ood_dev``,
``ood_final`` and the deployment benchmark - 3,072 texts - these rules fire on
four regions, and all four are the spoken email and URL forms they are for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

#: Top-level domains a spoken address in this deployment can plausibly end in.
#: A list rather than "any two or more letters" because that pattern matches
#: ordinary prose: "damage at H.No. 26" in the deployment benchmark reads as a
#: ``.no`` domain to a rule without one.
KNOWN_TLDS = frozenset({
    "biz", "co", "com", "dev", "edu", "gov", "info", "io", "me", "mil", "net",
    "org", "uk", "us", "ai", "app", "cloud", "health", "life", "live", "news",
    "online", "shop", "site", "store", "tech", "tv", "xyz",
})

_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
_LOCAL = r"[A-Za-z0-9](?:[A-Za-z0-9._%+-]*[A-Za-z0-9])?"
_PATH = r"[A-Za-z0-9](?:[A-Za-z0-9._~%+-]*[A-Za-z0-9])?"
#: A dotted name whose last label is checked against ``KNOWN_TLDS`` below.
_DOMAIN = rf"{_LABEL}(?:\.{_LABEL})*\.(?P<tld>[A-Za-z]{{2,24}})"
#: The name must end here. ``@`` or another name character means the address was
#: already written properly and the rule would be biting its front off. A bare
#: full stop is allowed through: it is where the sentence ends, not more domain.
_NAME_ENDS = r"(?![A-Za-z0-9@_%+~-])(?!\.[A-Za-z0-9])"

SPOKEN_EMAIL = re.compile(
    rf"(?<![A-Za-z0-9@._%+-])(?P<local>{_LOCAL})\s+at\s+(?P<domain>{_DOMAIN}){_NAME_ENDS}"
)
SPOKEN_URL_PATH = re.compile(
    rf"(?<![A-Za-z0-9@._%+-])(?P<domain>{_DOMAIN})\s+slash\s+(?P<path>{_PATH})"
    + _NAME_ENDS
)


@dataclass(frozen=True)
class Edit:
    """One rewritten region: where it came from and where it went."""

    original_start: int
    original_end: int
    normalised_start: int
    normalised_end: int
    rule: str


@dataclass(frozen=True)
class NormalisedText:
    """A model-facing view of a transcript, with the way back."""

    original: str
    text: str
    edits: tuple[Edit, ...]

    @property
    def changed(self) -> bool:
        return bool(self.edits)


def _tld_ok(match: re.Match[str]) -> bool:
    return match.group("tld").lower() in KNOWN_TLDS


def _matches(text: str) -> list[tuple[re.Match[str], str, str]]:
    """Every non-overlapping rewrite, left to right, email rules first."""

    found: list[tuple[re.Match[str], str, str]] = []
    for rule, pattern, joiner in (
        ("spoken_email", SPOKEN_EMAIL, "@"),
        ("spoken_url_path", SPOKEN_URL_PATH, "/"),
    ):
        for match in pattern.finditer(text):
            if _tld_ok(match):
                found.append((match, rule, joiner))
    found.sort(key=lambda item: item[0].start())

    kept: list[tuple[re.Match[str], str, str]] = []
    end = -1
    for match, rule, joiner in found:
        if match.start() >= end:
            kept.append((match, rule, joiner))
            end = match.end()
    return kept


def normalise_spoken_forms(text: str) -> NormalisedText:
    """Rewrite spoken email and URL forms; return the view and its edits.

    Deterministic and idempotent: the output contains no ``<token> at <domain>``
    shape for the rules to fire on a second time.
    """

    pieces: list[str] = []
    edits: list[Edit] = []
    cursor = 0
    length = 0
    for match, rule, joiner in _matches(text):
        pieces.append(text[cursor:match.start()])
        length += match.start() - cursor
        left = match.group("local") if rule == "spoken_email" else match.group("domain")
        right = match.group("domain") if rule == "spoken_email" else match.group("path")
        replacement = f"{left}{joiner}{right}"
        pieces.append(replacement)
        edits.append(Edit(
            original_start=match.start(),
            original_end=match.end(),
            normalised_start=length,
            normalised_end=length + len(replacement),
            rule=rule,
        ))
        length += len(replacement)
        cursor = match.end()
    if not edits:
        return NormalisedText(original=text, text=text, edits=())
    pieces.append(text[cursor:])
    return NormalisedText(original=text, text="".join(pieces), edits=tuple(edits))


def project_offsets(view: NormalisedText, start: int, end: int) -> tuple[int, int]:
    """Map a span on the rewritten text back onto the original transcript.

    A span that overlaps a rewritten region is widened to the whole of that
    region on the original side. The rewrite drops the words that made the form
    spoken - the " at ", the " slash " - and a masker that covered the rewritten
    address must cover everything those words joined, or the fix would release
    the local part it was written to protect.
    """

    original_start = _project_point(view, start, upper=False)
    original_end = _project_point(view, end, upper=True)
    for edit in view.edits:
        if edit.normalised_start < end and edit.normalised_end > start:
            original_start = min(original_start, edit.original_start)
            original_end = max(original_end, edit.original_end)
    return original_start, max(original_end, original_start)


def _project_point(view: NormalisedText, position: int, *, upper: bool) -> int:
    offset = 0
    for edit in view.edits:
        if position < edit.normalised_start:
            break
        if position <= edit.normalised_end:
            return edit.original_end if upper else edit.original_start
        offset = edit.original_end - edit.normalised_end
    return position + offset


def describe(view: NormalisedText) -> list[dict[str, object]]:
    """The edits in a form the span trace and the run artifacts can carry."""

    return [
        {
            "rule": edit.rule,
            "original": view.original[edit.original_start:edit.original_end],
            "normalised": view.text[edit.normalised_start:edit.normalised_end],
            "start": edit.original_start,
            "end": edit.original_end,
        }
        for edit in view.edits
    ]


def rules() -> Iterable[str]:
    return ("spoken_email", "spoken_url_path")

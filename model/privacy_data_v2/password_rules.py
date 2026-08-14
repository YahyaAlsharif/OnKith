"""Admission rules for the one new Model V2 entity: ``PASSWORD``.

``PASSWORD`` exists to protect a *disclosed secret value* in conversational,
user-like text. It does not exist to memorise the word "password".

Accepted (the span covers the secret itself)::

    My password is BlueHorse!92
    The Wi-Fi password is kaust2026
    My phone password is 482913
    I changed my password to MoonLight44
    Use Summer#2026 to log in

Rejected, and valuable as hard negatives instead (no secret is present)::

    I forgot my password
    How do I reset my password?
    Never share your password
    The password must contain eight characters
    Password authentication is enabled

External corpora annotate ``password`` generously, including credential columns
in synthetic document dumps that look nothing like OnKith speech input. This
module makes the accept/reject decision explicit, deterministic and auditable so
the label is trained on the right distribution instead of on raw volume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

#: Characters inspected on either side of a candidate span.
CONTEXT_WINDOW = 80

# --------------------------------------------------------------------------
# Cue vocabulary
# --------------------------------------------------------------------------

#: Qualifiers that make a credential concrete and personal, e.g. "wifi password".
_QUALIFIER = (
    r"(?:wi[\s-]?fi|wifi|network|router|guest|vpn|account|admin|device|phone|"
    r"laptop|email|login|log[\s-]?in|portal|app|banking|temporary|new|old|"
    r"current|default)"
)

#: Words used for the secret itself. Plurals are matched so generic advice
#: ("passwords should never be reused") is classified as policy talk rather than
#: falling through to the weaker "value is not a secret" rejection.
_SECRET_NOUN = (
    r"(?:passwords?|passcodes?|pass\s?phrases?|passphrases?|pins?(?:\s+codes?)?"
    r"|access\s+codes?)"
)

#: Possessive/definite framing that precedes a real disclosure.
_OWNER = r"(?:my|our|your|his|her|their|the|this|a|an)"

#: Cues that appear immediately BEFORE the secret value.
_PRECEDING_CUES: tuple[tuple[str, str], ...] = (
    (
        rf"\b{_OWNER}(?:\s+{_QUALIFIER}){{0,2}}\s+{_SECRET_NOUN}\s+"
        rf"(?:is|was|will\s+be|reads|remains)\s*[:\-]?\s*$",
        "possessive_disclosure",
    ),
    (
        rf"\b(?:changed|change|reset|set|updated|update|made|switched)\s+"
        rf"{_OWNER}(?:\s+{_QUALIFIER}){{0,2}}\s+{_SECRET_NOUN}\s+to\s*[:\-]?\s*$",
        "change_to_disclosure",
    ),
    (
        rf"\b{_QUALIFIER}\s+{_SECRET_NOUN}\s*[:\-=]\s*$",
        "qualified_field_disclosure",
    ),
    (
        rf"\b{_SECRET_NOUN}\s+(?:is|was)\s*[:\-]?\s*$",
        "bare_disclosure",
    ),
    (
        rf"\b(?:log\s?in|sign\s?in|connect|authenticate|unlock)\s+(?:with|using)\s*$",
        "login_with_disclosure",
    ),
    (
        rf"\b{_SECRET_NOUN}\s*[:\-=]\s*$",
        "field_disclosure",
    ),
)

#: Cues that appear immediately AFTER the secret value, e.g. "Use X to log in".
_FOLLOWING_CUES: tuple[tuple[str, str], ...] = (
    (
        r"^\s*(?:to|and\s+then)\s+(?:log\s?in|sign\s?in|connect|authenticate|"
        r"unlock|get\s+in|access)",
        "use_value_to_login",
    ),
    (
        rf"^\s*(?:as|for)\s+{_OWNER}?\s*(?:{_QUALIFIER}\s+)?{_SECRET_NOUN}\b",
        "value_named_as_password",
    ),
    (
        rf"^\s*is\s+{_OWNER}\s*(?:{_QUALIFIER}\s+)?{_SECRET_NOUN}\b",
        "value_is_the_password",
    ),
)

#: "Use X to log in" needs a preceding imperative as well as the trailing cue.
_USE_IMPERATIVE = re.compile(r"\b(?:use|enter|type|try|say)\s*[:\-]?\s*$", re.IGNORECASE)

#: Sentences that talk *about* passwords without revealing one.
_POLICY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:forgot|forgotten|lost|remember|recall)\b.{0,30}\b" + _SECRET_NOUN, "forgot_or_recall"),
    (r"\bnever\s+share\b|\bdo\s*n[o']?t\s+share\b|\bsharing\b.{0,20}" + _SECRET_NOUN, "sharing_advice"),
    (r"\b" + _SECRET_NOUN + r"\b.{0,40}\b(?:must|should|has\s+to|needs?\s+to|requires?)\b", "policy_requirement"),
    (r"\b(?:at\s+least|minimum\s+of|maximum\s+of)\b.{0,30}\bcharacters?\b", "length_requirement"),
    (r"\b" + _SECRET_NOUN + r"\s+(?:authentication|policy|manager|strength|expiry|expiration|rotation|field|prompt|hint|reset\s+link)\b", "password_feature"),
    (r"\b(?:how\s+do\s+i|how\s+can\s+i|can\s+you\s+help)\b.{0,40}\b" + _SECRET_NOUN, "help_request"),
    (r"\b" + _SECRET_NOUN + r"\s+(?:is\s+)?(?:enabled|disabled|required|optional|incorrect|wrong|expired|invalid)\b", "state_statement"),
)

#: Document/table dumps that do not resemble OnKith conversational input.
_STRUCTURED_HINTS = re.compile(r"[|\t]|\r?\n\s*\r?\n|<[a-z/][^>]{0,40}>", re.IGNORECASE)

_FIELD_PAIR = re.compile(r"\b[A-Za-z][A-Za-z _-]{2,30}\s*[:=]\s*\S", re.IGNORECASE)

#: Words that are never a secret on their own.
_NON_SECRET_WORDS = frozenset(
    {
        "password", "passwords", "passcode", "pin", "none", "null", "unknown",
        "hidden", "redacted", "secret", "blank", "empty", "n/a", "na", "xxx",
        "characters", "character", "letters", "digits", "yes", "no", "true",
        "false", "required", "optional", "enabled", "disabled",
    }
)

_NUMBER_WORDS = frozenset(
    {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "twenty", "thirty",
    }
)

CATEGORY_DISCLOSURE = "disclosure"
CATEGORY_POLICY = "policy_or_help"
CATEGORY_STRUCTURED = "structured_document"
CATEGORY_NO_CUE = "no_disclosure_cue"
CATEGORY_NOT_SECRET = "value_is_not_a_secret"


@dataclass(frozen=True)
class PasswordDecision:
    """One auditable accept/reject decision for a candidate PASSWORD span."""

    accept: bool
    category: str
    reason: str
    cue: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "accept": self.accept,
            "category": self.category,
            "reason": self.reason,
            "cue": self.cue,
        }


def looks_like_secret(value: str) -> bool:
    """Whether ``value`` plausibly is a credential rather than prose.

    Accepts mixed-class strings (``BlueHorse!92``), alphanumeric passphrases
    (``kaust2026``), and numeric PINs of 4-12 digits (``482913``). Rejects bare
    dictionary words, spelled-out numbers and placeholder tokens.
    """

    value = str(value).strip()
    if not (3 <= len(value) <= 64):
        return False
    if " " in value.strip() and len(value.split()) > 3:
        return False
    lowered = value.casefold()
    if lowered in _NON_SECRET_WORDS or lowered in _NUMBER_WORDS:
        return False
    if any(word in _NUMBER_WORDS for word in lowered.split()):
        return False

    has_lower = any(character.islower() for character in value)
    has_upper = any(character.isupper() for character in value)
    has_digit = any(character.isdigit() for character in value)
    has_symbol = any(not character.isalnum() and not character.isspace() for character in value)

    if value.isdigit():
        return 4 <= len(value) <= 12
    classes = sum((has_lower, has_upper, has_digit, has_symbol))
    if classes >= 2 and len(value) >= 5:
        return True
    # A single-class token must at least be long and not a plain dictionary word.
    return classes == 1 and len(value) >= 10 and lowered not in _NON_SECRET_WORDS


def _looks_structured(text: str) -> bool:
    if _STRUCTURED_HINTS.search(text):
        return True
    return len(_FIELD_PAIR.findall(text)) >= 3


def _match_preceding(before: str) -> tuple[str, str] | None:
    for pattern, cue in _PRECEDING_CUES:
        if re.search(pattern, before, re.IGNORECASE):
            return pattern, cue
    return None


def _match_following(before: str, after: str) -> tuple[str, str] | None:
    for pattern, cue in _FOLLOWING_CUES:
        if re.search(pattern, after, re.IGNORECASE):
            if cue == "use_value_to_login" and not _USE_IMPERATIVE.search(before):
                continue
            return pattern, cue
    return None


def classify_password_span(text: str, start: int, end: int) -> PasswordDecision:
    """Decide whether one annotated PASSWORD span is an OnKith-style disclosure."""

    text = str(text)
    if not 0 <= start < end <= len(text):
        return PasswordDecision(False, CATEGORY_NOT_SECRET, "Span offsets are invalid.")
    value = text[start:end]
    before = text[max(0, start - CONTEXT_WINDOW) : start]
    after = text[end : end + CONTEXT_WINDOW]

    if _looks_structured(text):
        return PasswordDecision(
            False,
            CATEGORY_STRUCTURED,
            "Text is a table/markup/multi-field document dump, not conversational input.",
        )

    for pattern, reason in _POLICY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            # A policy sentence that still discloses a value keeps the span, so
            # "I changed my password to MoonLight44 because it must contain a
            # digit" is not thrown away. Otherwise it is advice, not a secret.
            if _match_preceding(before) is None and _match_following(before, after) is None:
                return PasswordDecision(
                    False, CATEGORY_POLICY, f"Password mentioned as {reason}; no value disclosed."
                )
            break

    if not looks_like_secret(value):
        return PasswordDecision(
            False, CATEGORY_NOT_SECRET, f"Annotated value {value[:24]!r} does not look like a credential."
        )

    matched = _match_preceding(before)
    if matched is not None:
        return PasswordDecision(True, CATEGORY_DISCLOSURE, "Credential disclosed in context.", matched[1])
    matched = _match_following(before, after)
    if matched is not None:
        return PasswordDecision(True, CATEGORY_DISCLOSURE, "Credential disclosed in context.", matched[1])

    return PasswordDecision(
        False,
        CATEGORY_NO_CUE,
        "No credential-disclosure cue surrounds the annotated value.",
    )


def filter_password_spans(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[PasswordDecision]]:
    """Drop non-disclosure PASSWORD spans; keep every other supported span.

    Rejected PASSWORD spans become ordinary ``O`` text. The row is retained
    because its other annotations are still correct.
    """

    text = str(record["source_text"])
    kept: list[dict[str, Any]] = []
    decisions: list[PasswordDecision] = []
    for span in record["privacy_mask"]:
        if str(span["label"]) != "PASSWORD":
            kept.append(dict(span))
            continue
        decision = classify_password_span(text, int(span["start"]), int(span["end"]))
        decisions.append(decision)
        if decision.accept:
            kept.append(dict(span))
    result = dict(record)
    result["privacy_mask"] = kept
    return result, decisions


def audit_password_decisions(decisions: Iterable[PasswordDecision]) -> dict[str, Any]:
    """Summarise decisions for the reproducibility manifest."""

    by_category: dict[str, int] = {}
    by_cue: dict[str, int] = {}
    accepted = total = 0
    for decision in decisions:
        total += 1
        accepted += int(decision.accept)
        by_category[decision.category] = by_category.get(decision.category, 0) + 1
        if decision.cue:
            by_cue[decision.cue] = by_cue.get(decision.cue, 0) + 1
    return {
        "candidate_spans": total,
        "accepted_spans": accepted,
        "rejected_spans": total - accepted,
        "acceptance_rate": (accepted / total) if total else 0.0,
        "by_category": dict(sorted(by_category.items())),
        "by_accepted_cue": dict(sorted(by_cue.items())),
    }

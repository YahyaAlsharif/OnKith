"""Conservative, contextual recognizers for explicit structured PII."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .span_merger import SpanCandidate


@dataclass(frozen=True)
class StructuredPIISpan:
    entity_type: str
    start_char: int
    end_char: int
    text: str
    source: str
    confidence: float

    def as_candidate(self) -> SpanCandidate:
        return SpanCandidate(
            entity_type=self.entity_type,
            start_char=self.start_char,
            end_char=self.end_char,
            text=self.text,
            source=f"structured:{self.source}",
            confidence=self.confidence,
            metadata={"recognizer": self.source},
        )


@dataclass
class StructuredPIIConfig:
    enable_email: bool = True
    enable_phone: bool = True
    enable_account: bool = True
    enable_zipcode: bool = True
    enable_identifier: bool = True
    phone_min_digits: int = 7
    phone_max_digits: int = 15
    account_min_digits: int = 4
    account_max_digits: int = 24
    zipcode_min_digits: int = 4
    zipcode_max_digits: int = 10
    identifier_min_digits: int = 4
    identifier_max_digits: int = 24


_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_DIGIT_UNIT = r"(?:\d+|zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_SEQUENCE_PATTERN = re.compile(
    rf"(?P<sequence>\+?\s*{_DIGIT_UNIT}(?:[\s().-]*{_DIGIT_UNIT})*)",
    flags=re.IGNORECASE,
)
_EMAIL_PATTERN = re.compile(
    r"(?<![A-Z0-9._%+\-])"
    r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+"
    # A terminal full stop is normal sentence punctuation, so it must not
    # invalidate the complete address immediately before it.
    r"(?![A-Z0-9_%+\-])",
    flags=re.IGNORECASE,
)


def _digit_count(value: str) -> int:
    count = sum(character.isdigit() for character in value)
    words = re.findall(r"[A-Za-z]+", value.lower())
    return count + sum(word in _DIGIT_WORDS for word in words)


def _trim_number_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ".,;-"):
        end -= 1
    return start, end


class StructuredPIIRecognizer:
    """Find only explicit structured PII and preserve exact original offsets."""

    _PHONE_CUE = re.compile(
        r"\b(?:phone(?:\s+number)?|mobile(?:\s+number)?|"
        r"call\s+me\s+at|contact\s+me\s+at)\b"
        r"(?:\s*(?:is|equals|:|-)\s*)?",
        flags=re.IGNORECASE,
    )
    _ACCOUNT_CUE = re.compile(
        r"\b(?:bank\s+account(?:\s+number)?|account\s+number|account)\b"
        r"(?:\s*(?:is|equals|:|-)\s*)?",
        flags=re.IGNORECASE,
    )
    _ZIP_CUE = re.compile(
        r"\b(?:zip\s+code|postal\s+code|postcode)\b"
        r"(?:\s*(?:is|equals|:|-)\s*)?",
        flags=re.IGNORECASE,
    )
    _IDENTIFIER_CUE = re.compile(
        r"\b(?:id\s+number|identification\s+number|membership\s+number|"
        r"reference\s+number)\b"
        r"(?:\s*(?:is|equals|:|-)\s*)?",
        flags=re.IGNORECASE,
    )

    def __init__(self, config: StructuredPIIConfig | None = None):
        self.config = config or StructuredPIIConfig()

    def _contextual_numeric_spans(
        self,
        text: str,
        cue_pattern: re.Pattern[str],
        *,
        entity_type: str,
        source: str,
        minimum_digits: int,
        maximum_digits: int,
    ) -> list[StructuredPIISpan]:
        spans: list[StructuredPIISpan] = []
        for cue in cue_pattern.finditer(text):
            match = _SEQUENCE_PATTERN.match(text, cue.end())
            if match is None:
                continue
            start, end = _trim_number_span(text, *match.span("sequence"))
            candidate = text[start:end]
            digit_count = _digit_count(candidate)
            if not minimum_digits <= digit_count <= maximum_digits:
                continue
            spans.append(
                StructuredPIISpan(
                    entity_type=entity_type,
                    start_char=start,
                    end_char=end,
                    text=candidate,
                    source=source,
                    confidence=1.0,
                )
            )
        return spans

    def find_spans(self, text: str) -> list[StructuredPIISpan]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, received {type(text).__name__}.")
        spans: list[StructuredPIISpan] = []
        if self.config.enable_email:
            spans.extend(
                StructuredPIISpan(
                    entity_type="EMAIL",
                    start_char=match.start(),
                    end_char=match.end(),
                    text=match.group(0),
                    source="email",
                    confidence=1.0,
                )
                for match in _EMAIL_PATTERN.finditer(text)
            )
        if self.config.enable_phone:
            spans.extend(
                self._contextual_numeric_spans(
                    text,
                    self._PHONE_CUE,
                    entity_type="TELEPHONENUM",
                    source="contextual_phone",
                    minimum_digits=self.config.phone_min_digits,
                    maximum_digits=self.config.phone_max_digits,
                )
            )
        if self.config.enable_account:
            spans.extend(
                self._contextual_numeric_spans(
                    text,
                    self._ACCOUNT_CUE,
                    entity_type="ACCOUNTNUM",
                    source="contextual_account",
                    minimum_digits=self.config.account_min_digits,
                    maximum_digits=self.config.account_max_digits,
                )
            )
        if self.config.enable_zipcode:
            spans.extend(
                self._contextual_numeric_spans(
                    text,
                    self._ZIP_CUE,
                    entity_type="ZIPCODE",
                    source="contextual_zipcode",
                    minimum_digits=self.config.zipcode_min_digits,
                    maximum_digits=self.config.zipcode_max_digits,
                )
            )
        if self.config.enable_identifier:
            spans.extend(
                self._contextual_numeric_spans(
                    text,
                    self._IDENTIFIER_CUE,
                    entity_type="IDCARDNUM",
                    source="contextual_identifier",
                    minimum_digits=self.config.identifier_min_digits,
                    maximum_digits=self.config.identifier_max_digits,
                )
            )
        unique: dict[tuple[str, int, int], StructuredPIISpan] = {}
        for span in spans:
            unique[(span.entity_type, span.start_char, span.end_char)] = span
        return sorted(
            unique.values(),
            key=lambda span: (span.start_char, span.end_char, span.entity_type),
        )

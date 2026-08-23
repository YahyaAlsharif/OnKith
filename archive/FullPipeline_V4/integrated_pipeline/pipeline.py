"""Fail-closed privacy boundary for terminal release."""

from __future__ import annotations

from typing import Protocol

from .schemas import PIIResult

BLOCKED_MARKER = "[BLOCKED: PII MASKING FAILED]"


class Masker(Protocol):
    def mask_text(self, text: str) -> PIIResult: ...


class FailClosedMasker:
    """Convert every masker exception into a non-releasable safe result."""

    def __init__(self, masker: Masker):
        self.masker = masker

    def mask_text(self, text: str) -> PIIResult:
        try:
            result = self.masker.mask_text(text)
        except Exception as exc:
            return PIIResult(
                original_text="",
                masked_text=BLOCKED_MARKER,
                latency_seconds=0.0,
                entities=[],
                safe_to_release=False,
                privacy_status="blocked_masking_failure",
                error=f"{type(exc).__name__}: masking operation failed",
            )
        if not result.safe_to_release:
            result.original_text = ""
            result.masked_text = BLOCKED_MARKER
            result.privacy_status = "blocked_masking_failure"
        return result

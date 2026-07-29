#!/usr/bin/env python3
"""Hardware-free validation for the OnKith PII masking stage.

Everything here runs without a microphone, without Moonshine, and without
network access, so it works identically on a Windows laptop and on the Pi.

Usage:
    python scripts/test_pii_offline.py
    python scripts/test_pii_offline.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrated_pipeline.config import PipelineConfig
from integrated_pipeline.pii_adapter import CONTEXT_LIMIT, PIIMaskerAdapter
from integrated_pipeline.pipeline import BLOCKED_MARKER, FailClosedMasker
from integrated_pipeline.schemas import PIIResult

EXPECTED_LABELS = 67

_FAILURES: list[str] = []
_PASSES = 0
_VERBOSE = False


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASSES
    if condition:
        _PASSES += 1
        if _VERBOSE:
            print(f"  [pass] {name}")
    else:
        _FAILURES.append(f"{name}: {detail}" if detail else name)
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}")


# ---------------------------------------------------------------------------
# 1. Environment and model integrity
# ---------------------------------------------------------------------------
def test_environment(config: PipelineConfig) -> None:
    section("1. Environment and artifacts")
    check(
        "python >= 3.11",
        sys.version_info[:2] >= (3, 11),
        f"found {sys.version.split()[0]}",
    )
    config.validate(require_artifacts=True)
    check("configuration and artifacts validate", True)

    import onnxruntime as ort

    check(
        "CPUExecutionProvider available",
        "CPUExecutionProvider" in ort.get_available_providers(),
        str(ort.get_available_providers()),
    )
    session = ort.InferenceSession(
        str(config.tinybert_model_path), providers=["CPUExecutionProvider"]
    )
    outputs = {output.name: output.shape for output in session.get_outputs()}
    check("ONNX exposes a 'logits' output", "logits" in outputs, str(outputs))
    label_dim = outputs.get("logits", [None, None, None])[-1]
    check(
        f"ONNX emits {EXPECTED_LABELS} labels",
        label_dim == EXPECTED_LABELS,
        f"got {label_dim}",
    )
    input_names = {inp.name for inp in session.get_inputs()}
    check(
        "ONNX inputs are input_ids + attention_mask",
        {"input_ids", "attention_mask"}.issubset(input_names),
        str(input_names),
    )


# ---------------------------------------------------------------------------
# 2. The PII model loads without dragging in PyTorch
# ---------------------------------------------------------------------------
def test_no_torch(config: PipelineConfig) -> PIIMaskerAdapter:
    section("2. Lightweight load (no PyTorch)")
    check(
        "torch absent before load",
        "torch" not in sys.modules,
        "a previous import already pulled in torch",
    )
    adapter = PIIMaskerAdapter(config)
    started = time.perf_counter()
    adapter.load()
    elapsed = time.perf_counter() - started
    check("adapter reports loaded", adapter.loaded)
    check(
        "torch NOT imported by PII model load",
        "torch" not in sys.modules,
        "torch was imported — the transformers dependency has returned",
    )
    check(
        "transformers NOT imported by PII model load",
        "transformers" not in sys.modules,
        "transformers was imported",
    )
    check("label map has 67 entries", len(adapter._id2label) == EXPECTED_LABELS)
    print(f"  model load: {elapsed:.2f}s")
    return adapter


# ---------------------------------------------------------------------------
# 3. Entity coverage
# ---------------------------------------------------------------------------
def test_entities(masker: FailClosedMasker) -> None:
    section("3. Entity coverage")

    cases: list[tuple[str, str, list[str], list[str]]] = [
        # (label, text, must_not_appear, must_appear)
        (
            "person name",
            "My name is Rayyan Khan.",
            ["Rayyan", "Khan"],
            ["[GIVENNAME_1]", "[SURNAME_1]"],
        ),
        (
            "email",
            "Email me at rayyan@example.com please.",
            ["rayyan@example.com", "@example", "example.com"],
            ["[EMAIL_1]"],
        ),
        (
            "email with subdomain",
            "You can reach me at rayyan.khan@sub.example.co.uk.",
            ["rayyan.khan@sub.example.co.uk", "@sub", "example.co.uk"],
            ["[EMAIL_1]"],
        ),
        (
            "phone number",
            "My phone number is 555 123 4567.",
            ["555 123 4567", "4567"],
            ["[TELEPHONENUM_1]"],
        ),
        (
            "spoken-digit phone",
            "Call me at five five five one two three four five six seven.",
            [],
            ["[TELEPHONENUM_1]"],
        ),
        (
            "street address",
            "I live at 42 Baker Street, London.",
            ["Baker Street", "London"],
            ["[STREET_1]", "[CITY_1]"],
        ),
        (
            "zip code",
            "My zip code is 12345.",
            ["12345"],
            ["[ZIPCODE_1]"],
        ),
        (
            "account number",
            "My bank account number is 987654321.",
            ["987654321"],
            ["[ACCOUNTNUM_1]"],
        ),
        (
            "id number",
            "My id number is 4455667788.",
            ["4455667788"],
            ["[IDCARDNUM_1]"],
        ),
        (
            "combined utterance",
            "My name is Rayyan Khan, my email is rayyan@example.com and my "
            "phone number is 555 123 4567.",
            ["Rayyan", "Khan", "rayyan@example.com", "@example", "555 123 4567"],
            ["[GIVENNAME_1]", "[SURNAME_1]", "[EMAIL_1]", "[TELEPHONENUM_1]"],
        ),
    ]

    for label, text, forbidden, required in cases:
        result = masker.mask_text(text)
        check(f"{label}: released", result.safe_to_release, result.error or "")
        for fragment in forbidden:
            check(
                f"{label}: {fragment!r} removed",
                fragment not in result.masked_text,
                f"leaked in {result.masked_text!r}",
            )
        for placeholder in required:
            check(
                f"{label}: {placeholder} present",
                placeholder in result.masked_text,
                f"got {result.masked_text!r}",
            )


# ---------------------------------------------------------------------------
# 4. Clean text must survive untouched
# ---------------------------------------------------------------------------
def test_clean_text(masker: FailClosedMasker) -> None:
    section("4. Clean text is not over-masked")
    for text in (
        "The weather is quite nice today and we discussed many unrelated topics.",
        "I bought three apples and twelve oranges yesterday.",
        "Please turn on the kitchen light and set a timer.",
    ):
        result = masker.mask_text(text)
        check(
            f"unchanged: {text[:40]!r}",
            result.masked_text == text,
            f"got {result.masked_text!r}",
        )

    for blank in ("", "   \n\t  "):
        result = masker.mask_text(blank)
        check(
            f"blank input {blank!r} is safe and empty",
            result.safe_to_release and result.masked_text == "",
            f"got {result!r}",
        )


# ---------------------------------------------------------------------------
# 5. Long input and window-boundary behaviour
# ---------------------------------------------------------------------------
def test_long_input(adapter: PIIMaskerAdapter, masker: FailClosedMasker) -> None:
    section("5. Long input and sliding windows")

    filler = "The weather is quite nice today and we discussed many unrelated topics at length. "
    long_text = filler * 60 + "My name is Rayyan Khan and my email is rayyan@example.com."

    token_count = len(adapter._tokenizer.encode(long_text).ids)
    check(
        f"test input exceeds the {CONTEXT_LIMIT}-token context",
        token_count > CONTEXT_LIMIT,
        f"only {token_count} tokens",
    )

    result = masker.mask_text(long_text)
    windows = result.raw_model_output.get("window_count", 0)
    check("long input released", result.safe_to_release, result.error or "")
    check("long input used multiple windows", windows > 1, f"window_count={windows}")
    for fragment in ("rayyan@example.com", "@example", "Rayyan", "Khan"):
        check(
            f"long input: {fragment!r} removed from tail",
            fragment not in result.masked_text,
            f"tail was {result.masked_text[-120:]!r}",
        )
    check(
        "long input: placeholders present in tail",
        "[EMAIL_1]" in result.masked_text or "[EMAIL_" in result.masked_text,
        f"tail was {result.masked_text[-120:]!r}",
    )

    # PII deliberately placed mid-document, straddling a window seam.
    block = "The quarterly report covered many operational topics in detail. "
    boundary_text = (
        block * 30 + "Contact Rayyan Khan at rayyan@example.com regarding it. " + block * 30
    )
    result = masker.mask_text(boundary_text)
    check(
        "boundary input used multiple windows",
        result.raw_model_output.get("window_count", 0) > 1,
        str(result.raw_model_output.get("window_count")),
    )
    for fragment in ("rayyan@example.com", "@example", "Rayyan", "Khan"):
        check(
            f"window-boundary PII: {fragment!r} removed",
            fragment not in result.masked_text,
            "leaked at a window seam",
        )

    check(
        "no partial-word boundaries produced",
        result.raw_model_output.get("partial_word_boundary_count", 0) == 0,
        str(result.raw_model_output.get("partial_word_boundary_count")),
    )


# ---------------------------------------------------------------------------
# 6. Fail-closed behaviour
# ---------------------------------------------------------------------------
def test_fail_closed() -> None:
    section("6. Fail-closed privacy boundary")

    class Exploding:
        def mask_text(self, text: str) -> PIIResult:
            raise RuntimeError("simulated inference failure")

    secret = "My name is Rayyan Khan and my email is rayyan@example.com"
    result = FailClosedMasker(Exploding()).mask_text(secret)
    check("masking failure is not releasable", not result.safe_to_release)
    check(
        "masking failure returns the blocked marker",
        result.masked_text == BLOCKED_MARKER,
        f"got {result.masked_text!r}",
    )
    check(
        "raw text never survives a masking failure",
        secret not in result.masked_text and result.original_text == "",
        "raw transcript leaked through the failure path",
    )
    check(
        "failure status is reported",
        result.privacy_status == "blocked_masking_failure",
        result.privacy_status,
    )

    class Unsafe:
        def mask_text(self, text: str) -> PIIResult:
            return PIIResult(
                original_text=text,
                masked_text=text,
                latency_seconds=0.0,
                entities=[],
                safe_to_release=False,
                privacy_status="pretend_ok",
            )

    result = FailClosedMasker(Unsafe()).mask_text(secret)
    check(
        "unsafe result is overwritten with the blocked marker",
        result.masked_text == BLOCKED_MARKER and secret not in result.masked_text,
        f"got {result.masked_text!r}",
    )


# ---------------------------------------------------------------------------
# 7. Latency and memory
# ---------------------------------------------------------------------------
def test_performance(masker: FailClosedMasker) -> None:
    section("7. Latency and memory (informational)")
    short = "My name is Rayyan Khan and my email is rayyan@example.com."
    long_text = (
        "The weather is quite nice today and we discussed many unrelated topics at length. "
        * 60
    ) + short

    for label, text in (("short", short), ("long", long_text)):
        masker.mask_text(text)  # warm
        started = time.perf_counter()
        result = masker.mask_text(text)
        elapsed = (time.perf_counter() - started) * 1000.0
        windows = result.raw_model_output.get("window_count")
        print(f"  {label:6s} chars={len(text):<6d} windows={windows}  {elapsed:7.1f} ms")

    try:
        import psutil

        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"  resident memory: {rss:.0f} MiB")
    except ModuleNotFoundError:
        print("  resident memory: psutil not installed (skipped)")


def main() -> int:
    global _VERBOSE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _VERBOSE = args.verbose

    print("OnKith offline PII validation")
    print("=" * 60)

    config = PipelineConfig()
    test_environment(config)
    adapter = test_no_torch(config)
    masker = FailClosedMasker(adapter)

    test_entities(masker)
    test_clean_text(masker)
    test_long_input(adapter, masker)
    test_fail_closed()
    test_performance(masker)

    print("\n" + "=" * 60)
    if _FAILURES:
        print(f"FAILED — {len(_FAILURES)} check(s) failed, {_PASSES} passed")
        for failure in _FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"ALL CHECKS PASSED — {_PASSES} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

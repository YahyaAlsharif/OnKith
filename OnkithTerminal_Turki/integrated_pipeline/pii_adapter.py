"""Local INT8 token classification plus structured PII recognition.

Two trained classifiers can sit behind this. Model V2
(DeBERTa-v3-xsmall, 63 labels) is the deployed one; Model V1 (TinyBERT-4,
67 labels) is the one it replaced and is still loadable. Which one runs is
a single configuration key, ``pii.engine``, overridable by the
``ONKITH_PII_ENGINE`` environment variable.

Both are worth having. Keeping the old one constructible is what makes the
comparison in ``eval/compare_classifiers.py`` a measurement rather than
two numbers from two afternoons, and it is what makes a bad result on the
Pi a config edit rather than a redeploy.

Each model's decode lives in its own module because the tokenizer family
decides the decode, and getting that wrong is silent: measured through the
wrong word ids, Model V2 typed F1 reads 0.59 instead of 0.96. This module
owns everything around the decode that is the same either way - the
structured regex layer, the categories this device releases on purpose,
the device's own name, and the span trace the terminal reports.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .config import PII_WORD_POLICY, PipelineConfig
from .model_v2_masker import ModelV2OnnxMasker
from .schemas import PIIResult
from .span_merger import (
    SpanCandidate,
    count_partial_word_boundaries,
    decision_to_dict,
    merge_pii_spans,
)
from .structured_pii import StructuredPIIRecognizer
from .tinybert_masker import TinyBertOnnxMasker

BATCH_SIZE = 4
#: Which decoder each engine name builds.
ENGINE_MASKERS = {"deberta": ModelV2OnnxMasker, "tinybert": TinyBertOnnxMasker}


def sha256_of(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PIIMaskerAdapter:
    """Load only the packaged tokenizer, label map, and INT8 ONNX model."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._masker: ModelV2OnnxMasker | TinyBertOnnxMasker | None = None
        self._id2label: dict[int, str] = {}
        self._structured = StructuredPIIRecognizer()
        self.never_masked = frozenset(config.never_masked_entities)
        self.never_masked_words = frozenset(config.never_masked_words)
        self.load_count = 0

    @property
    def engine(self) -> str:
        return self.config.pii_engine

    @property
    def loaded(self) -> bool:
        return self._masker is not None and self._masker.loaded

    @staticmethod
    def reject_lfs_pointer(path: Path) -> None:
        """Fail with a readable message when the ONNX was never fetched.

        A Git LFS pointer is a 130-byte text file, so onnxruntime's own
        error for it is an opaque protobuf parse failure. The size floor
        catches a truncated download for the same reason.
        """

        if not path.is_file():
            raise FileNotFoundError(f"PII INT8 model is missing: {path}")
        with path.open("rb") as stream:
            header = stream.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/"):
            raise RuntimeError(f"{path} is a Git LFS pointer, not an ONNX model")
        if path.stat().st_size < 1_000_000:
            raise RuntimeError(f"{path} is unexpectedly small for the INT8 model")

    def verify_artifact(self, path: Path) -> str:
        """Check the packaged ONNX against the sha256 pinned in the config.

        The deployment copy has to stay byte-identical to the artifact the
        held-out evaluation measured; anything else means the device is not
        running the model that was validated. A config that pins nothing
        skips the check, which is how Model V1 loads: it predates the
        practice.
        """

        expected = self.config.pii_model_sha256
        if not expected:
            return ""
        actual = sha256_of(path)
        if actual != expected:
            raise RuntimeError(
                f"{path} sha256 {actual} does not match the pinned {expected}; "
                "the packaged INT8 model is not the validated artifact"
            )
        return actual

    def load(self) -> None:
        if self.loaded:
            return
        engine = self.engine
        if engine not in ENGINE_MASKERS:
            raise RuntimeError(
                f"Unknown PII engine {engine!r}; known engines are "
                f"{sorted(ENGINE_MASKERS)}"
            )
        self.config.validate(require_artifacts=True)
        model_path = self.config.pii_model_path
        self.reject_lfs_pointer(model_path)
        self.verify_artifact(model_path)

        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the pinned onnxruntime package") from exc
        try:
            import transformers  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the pinned transformers package") from exc
        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CPUExecutionProvider is unavailable")

        expected_labels = self.config.pii_label_count
        payload = json.loads(
            self.config.pii_label_map_path.read_text(encoding="utf-8")
        )
        # Model V1 shipped a bare id-to-label map; Model V2 wraps it in a
        # document that carries the ontology version beside it.
        raw = payload.get("id2label", payload) if isinstance(payload, dict) else payload
        labels = {int(key): str(value) for key, value in raw.items()}
        if len(labels) != expected_labels or set(labels) != set(
            range(expected_labels)
        ):
            raise RuntimeError(
                f"The {engine} label map must contain IDs 0 through "
                f"{expected_labels - 1}"
            )

        masker = ENGINE_MASKERS[engine](
            model_path,
            self.config.pii_tokenizer_path,
            labels,
            context_limit=int(self.config.pii["context_limit"]),
            overflow_stride=int(self.config.pii["overflow_stride"]),
            batch_size=BATCH_SIZE,
        )
        masker.load()
        self._masker = masker
        self._id2label = labels
        self.load_count += 1

    def _is_the_devices_own_name(self, span: SpanCandidate) -> bool:
        """A span that is exactly the device's name and nothing else.

        Whole span only. "Onkith" is released; "Onkith Alshuaibi" is not,
        because the second word makes it a person again, and a span that
        merely contains the name is somebody whose name contains it.

        Structured spans are not offered here on purpose. A phone number is
        a phone number whatever it sits next to, and this exists for one
        narrow thing: the classifier reading the device's name as a
        person's. Both models do it, so both are filtered.
        """

        return span.text.strip().casefold() in self.never_masked_words

    def mask_text(self, text: str) -> PIIResult:
        """Mask one complete buffer; callers enforce fail-closed exceptions."""

        if not isinstance(text, str):
            raise TypeError("PII input must be text")
        if not text.strip():
            return PIIResult(
                original_text=text,
                masked_text="",
                latency_seconds=0.0,
                entities=[],
                safe_to_release=True,
                privacy_status="masked_empty",
            )
        self.load()
        assert self._masker is not None
        started = time.perf_counter()

        model_started = time.perf_counter()
        decoding = self._masker.decode(text)
        model_latency = time.perf_counter() - model_started

        # Dropped here rather than left to a mask policy downstream, so a
        # released category never reaches the entity list at all. A city the
        # device deliberately keeps is not a redaction that happened to be
        # undone, and the transcript should not record it as one.
        model_spans: list[SpanCandidate] = [
            span
            for span in decoding.spans
            if span.entity_type not in self.never_masked
            and not self._is_the_devices_own_name(span)
        ]

        structured_started = time.perf_counter()
        structured_spans = [
            span.as_candidate() for span in self._structured.find_spans(text)
        ]
        structured_latency = time.perf_counter() - structured_started

        merged = merge_pii_spans(text, model_spans, structured_spans)
        return PIIResult(
            original_text=text,
            masked_text=merged.masked_text,
            latency_seconds=time.perf_counter() - started,
            entities=merged.entities,
            safe_to_release=True,
            privacy_status="masked",
            raw_model_output={
                "engine": self.engine,
                "bio_repairs": decoding.bio_repairs,
                "window_count": decoding.window_count,
                "label_count": len(self._id2label),
                "word_policy": PII_WORD_POLICY,
                "structured_enabled": True,
                "pii_model_latency_seconds": model_latency,
                "structured_latency_seconds": structured_latency,
                "span_merger_latency_seconds": merged.latency_seconds,
                "partial_word_boundary_count": count_partial_word_boundaries(
                    text, merged.chosen_spans
                ),
                "span_trace": [
                    decision_to_dict(decision) for decision in merged.decisions
                ],
                "warnings": decoding.warnings,
            },
        )

"""Local DeBERTa-v3-xsmall INT8 token classification plus structured PII.

The neural decode itself lives in :mod:`integrated_pipeline.model_v2_masker`,
which is the single canonical Model V2 decoder shared with the offline
evaluation harness. This module owns everything around it that is specific to
the live deployment: pinned-artifact verification, the structured regex layer,
the configurable mask policy, and the span trace the terminal UI reports.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import PII_WORD_POLICY, PipelineConfig
from .mask_policy import MaskPolicy
from .model_v2_masker import ModelV2OnnxMasker
from .schemas import PIIResult
from .span_merger import (
    SpanCandidate,
    count_partial_word_boundaries,
    decision_to_dict,
    merge_pii_spans,
)
from .spoken_form_normalizer import NormalisedText
from .spoken_form_normalizer import describe as describe_spoken_forms
from .spoken_form_normalizer import normalise_spoken_forms, project_offsets
from .structured_pii import StructuredPIIRecognizer

#: Model V2 ontology: 31 entity types, 63 BIO labels.
EXPECTED_LABEL_COUNT = 63
BATCH_SIZE = 4


def _sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_candidate(view: NormalisedText, text: str,
                       span: SpanCandidate) -> SpanCandidate:
    """Move one span off the normalised view and back onto the transcript."""

    start, end = project_offsets(view, span.start_char, span.end_char)
    return SpanCandidate(
        entity_type=span.entity_type,
        start_char=start,
        end_char=end,
        text=text[start:end],
        source=span.source,
        confidence=span.confidence,
        metadata={**span.metadata, "spoken_form_projected": True},
    )


class PIIMaskerAdapter:
    """Load only the packaged tokenizer, label map, and INT8 ONNX model."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._masker: ModelV2OnnxMasker | None = None
        self._id2label: dict[int, str] = {}
        self._structured = StructuredPIIRecognizer()
        # Resolved once at construction. An absent or malformed policy masks
        # everything, so existing deployments are unaffected.
        self.mask_policy = MaskPolicy.from_pipeline_config(config)
        self.load_count = 0

    @property
    def loaded(self) -> bool:
        return self._masker is not None and self._masker.loaded

    @staticmethod
    def reject_lfs_pointer(path: Path) -> None:
        """Fail with a readable message when the ONNX was never fetched.

        A Git LFS pointer is a 130-byte text file, so onnxruntime's own error
        for it is an opaque protobuf parse failure. The size floor catches a
        truncated download for the same reason.
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

        The deployment copy must stay byte-identical to the canonical artifact
        under ``model/DeBERTa-v3-xsmall/`` that the held-out evaluation measured;
        anything else means the pipeline is not the model that was validated.
        Silero is verified the same way, in ``silero_vad.py``.
        """

        expected = str(self.config.pii["sha256"])
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"{path} sha256 {actual} does not match the pinned {expected}; "
                "the packaged INT8 model is not the validated artifact"
            )
        return actual

    def load(self) -> None:
        if self.loaded:
            return
        self.config.validate(require_privacy_artifacts=True)
        model_path = self.config.pii_model_path
        tokenizer_path = self.config.pii_tokenizer_path
        label_path = self.config.pii_label_map_path
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

        payload = json.loads(label_path.read_text(encoding="utf-8"))
        raw = payload.get("id2label", payload) if isinstance(payload, dict) else payload
        labels = {int(key): str(value) for key, value in raw.items()}
        if len(labels) != EXPECTED_LABEL_COUNT or set(labels) != set(
            range(EXPECTED_LABEL_COUNT)
        ):
            raise RuntimeError(
                f"Model V2 label map must contain IDs 0 through {EXPECTED_LABEL_COUNT - 1}"
            )

        masker = ModelV2OnnxMasker(
            model_path,
            tokenizer_path,
            labels,
            context_limit=int(self.config.pii["context_limit"]),
            overflow_stride=int(self.config.pii["overflow_stride"]),
            batch_size=BATCH_SIZE,
        )
        # The tokenizer is packaged beside the model and must never reach the
        # network; transformers only consults the hub when a local file is
        # missing, and on a Pi with no route that is a long timeout, not an
        # error.
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            masker.load()
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline

        self._masker = masker
        self._id2label = labels
        self.load_count += 1

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

        # V4 only. The recognizer writes an address the way it was spoken -
        # "billing at example.org" - and neither the model nor the structured
        # EMAIL pattern has seen one written that way, so the domain was masked
        # and the local part released. The rewrite is a view for the detectors;
        # every span it produces is projected back before anything is masked, so
        # the released transcript is still the transcript, character for
        # character. See spoken_form_normalizer.py.
        view = normalise_spoken_forms(text)
        detect_text = view.text

        model_started = time.perf_counter()
        decoding = self._masker.decode(detect_text)
        model_latency = time.perf_counter() - model_started
        model_spans: list[SpanCandidate] = decoding.spans

        structured_started = time.perf_counter()
        structured_spans = [
            span.as_candidate() for span in self._structured.find_spans(detect_text)
        ]
        structured_latency = time.perf_counter() - structured_started

        if view.changed:
            model_spans = [
                _project_candidate(view, text, span) for span in model_spans
            ]
            structured_spans = [
                _project_candidate(view, text, span) for span in structured_spans
            ]

        merged = merge_pii_spans(
            text, model_spans, structured_spans, mask_policy=self.mask_policy
        )
        return PIIResult(
            original_text=text,
            masked_text=merged.masked_text,
            latency_seconds=time.perf_counter() - started,
            entities=merged.entities,
            safe_to_release=True,
            privacy_status="masked",
            raw_model_output={
                "bio_repairs": decoding.bio_repairs,
                "window_count": decoding.window_count,
                "label_count": len(self._id2label),
                "word_policy": PII_WORD_POLICY,
                "structured_enabled": True,
                "spoken_form_edits": describe_spoken_forms(view),
                "mask_policy": merged.mask_policy,
                "policy_exempt_span_count": merged.policy_exempt_span_count,
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

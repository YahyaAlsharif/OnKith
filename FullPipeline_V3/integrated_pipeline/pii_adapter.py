"""Local TinyBERT INT8 token classification plus structured PII recognition."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import PII_WORD_POLICY, PipelineConfig
from .schemas import PIIResult
from .span_merger import (
    SpanCandidate,
    consolidate_predictions_to_words,
    count_partial_word_boundaries,
    decision_to_dict,
    derive_word_ids,
    merge_pii_spans,
)
from .structured_pii import StructuredPIIRecognizer

CONTEXT_LIMIT = 512
OVERFLOW_STRIDE = 128
EXPECTED_LABEL_COUNT = 67
BATCH_SIZE = 4


def _repair_bio(labels: list[str]) -> tuple[list[str], int]:
    repaired: list[str] = []
    previous_type: str | None = None
    repairs = 0
    for label in labels:
        fixed = label
        if label.startswith("I-"):
            entity_type = label[2:]
            if previous_type != entity_type:
                fixed = f"B-{entity_type}"
                repairs += 1
        repaired.append(fixed)
        previous_type = None if fixed == "O" else fixed[2:]
    return repaired, repairs


class PIIMaskerAdapter:
    """Load only the packaged tokenizer, label map, and INT8 ONNX model."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._tokenizer: Any | None = None
        self._session: Any | None = None
        self._id2label: dict[int, str] = {}
        self._structured = StructuredPIIRecognizer()
        self.load_count = 0

    @property
    def loaded(self) -> bool:
        return self._tokenizer is not None and self._session is not None

    @staticmethod
    def reject_lfs_pointer(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"TinyBERT INT8 model is missing: {path}")
        with path.open("rb") as stream:
            header = stream.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/"):
            raise RuntimeError(f"{path} is a Git LFS pointer, not an ONNX model")
        if path.stat().st_size < 1_000_000:
            raise RuntimeError(f"{path} is unexpectedly small for the INT8 model")

    def load(self) -> None:
        if self.loaded:
            return
        self.config.validate(require_privacy_artifacts=True)
        model_path = self.config.tinybert_model_path
        tokenizer_path = self.config.tinybert_tokenizer_path
        label_path = self.config.tinybert_label_map_path
        self.reject_lfs_pointer(model_path)

        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the pinned onnxruntime package") from exc
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the pinned transformers package") from exc

        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CPUExecutionProvider is unavailable")
        labels = {
            int(key): str(value)
            for key, value in json.loads(
                label_path.read_text(encoding="utf-8")
            ).items()
        }
        if len(labels) != EXPECTED_LABEL_COUNT or set(labels) != set(
            range(EXPECTED_LABEL_COUNT)
        ):
            raise RuntimeError("TinyBERT label map must contain IDs 0 through 66")

        previous_offline = None
        try:
            import os

            previous_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                use_fast=True,
            )
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("A fast local tokenizer is required for exact offsets")

        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        if "logits" not in {output.name for output in session.get_outputs()}:
            raise RuntimeError("TinyBERT ONNX output named 'logits' is missing")
        self._tokenizer = tokenizer
        self._session = session
        self._id2label = labels
        self.load_count += 1

    def _window_encodings(self, text: str) -> tuple[Any, list[dict[str, Any]]]:
        assert self._tokenizer is not None
        full = self._tokenizer(
            text,
            truncation=False,
            padding=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        if len(full["input_ids"]) <= CONTEXT_LIMIT:
            return full, [
                {
                    "input_ids": full["input_ids"],
                    "attention_mask": full["attention_mask"],
                    "offset_mapping": full["offset_mapping"],
                }
            ]
        overflow = self._tokenizer(
            text,
            truncation=True,
            max_length=CONTEXT_LIMIT,
            stride=OVERFLOW_STRIDE,
            padding=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_token_type_ids=False,
        )
        windows = [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offsets,
            }
            for input_ids, attention_mask, offsets in zip(
                overflow["input_ids"],
                overflow["attention_mask"],
                overflow["offset_mapping"],
            )
        ]
        return full, windows

    @staticmethod
    def _word_ids(
        encoding: Any, text: str, offsets: list[tuple[int, int]]
    ) -> list[int | None]:
        try:
            ids = list(encoding.word_ids())
        except (AttributeError, TypeError, ValueError):
            ids = derive_word_ids(text, offsets)
        if len(ids) != len(offsets):
            ids = derive_word_ids(text, offsets)
        return [
            word_id if start != end else None
            for word_id, (start, end) in zip(ids, offsets)
        ]

    @staticmethod
    def _argmax_with_confidence(logits: Any) -> tuple[Any, Any]:
        import numpy as np

        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / np.sum(exp, axis=-1, keepdims=True)
        predictions = np.argmax(probabilities, axis=-1)
        confidence = np.take_along_axis(
            probabilities, predictions[..., None], axis=-1
        )[..., 0]
        return predictions, confidence

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
        assert self._tokenizer is not None
        assert self._session is not None
        started = time.perf_counter()
        tinybert_started = time.perf_counter()
        full, windows = self._window_encodings(text)
        all_offsets = [tuple(map(int, item)) for item in full["offset_mapping"]]
        all_word_ids = self._word_ids(full, text, all_offsets)
        offsets = [item for item in all_offsets if item[0] != item[1]]
        word_ids = [
            word_id
            for word_id, item in zip(all_word_ids, all_offsets)
            if item[0] != item[1]
        ]
        if not offsets:
            raise RuntimeError("Tokenizer produced no source-aligned text tokens")

        prediction_by_offset: dict[tuple[int, int], tuple[int, float]] = {}
        for batch_start in range(0, len(windows), BATCH_SIZE):
            batch = windows[batch_start : batch_start + BATCH_SIZE]
            padded = self._tokenizer.pad(
                [
                    {
                        "input_ids": window["input_ids"],
                        "attention_mask": window["attention_mask"],
                    }
                    for window in batch
                ],
                padding="longest",
                return_tensors="np",
            )
            inputs = {
                "input_ids": padded["input_ids"],
                "attention_mask": padded["attention_mask"],
            }
            logits = self._session.run(["logits"], inputs)[0]
            expected = (
                inputs["input_ids"].shape[0],
                inputs["input_ids"].shape[1],
                EXPECTED_LABEL_COUNT,
            )
            if tuple(logits.shape) != expected:
                raise RuntimeError(
                    f"Unexpected TinyBERT output {tuple(logits.shape)}; expected {expected}"
                )
            predictions, confidences = self._argmax_with_confidence(logits)
            for row, window in enumerate(batch):
                for offset, label_id, confidence in zip(
                    window["offset_mapping"],
                    predictions[row, : len(window["offset_mapping"])],
                    confidences[row, : len(window["offset_mapping"])],
                ):
                    aligned = tuple(map(int, offset))
                    if aligned[0] == aligned[1]:
                        continue
                    previous = prediction_by_offset.get(aligned)
                    if previous is None or float(confidence) > previous[1]:
                        prediction_by_offset[aligned] = (
                            int(label_id),
                            float(confidence),
                        )

        missing = [offset for offset in offsets if offset not in prediction_by_offset]
        if missing:
            raise RuntimeError(f"Missing TinyBERT prediction for offset {missing[0]}")
        labels, repairs = _repair_bio(
            [self._id2label[prediction_by_offset[offset][0]] for offset in offsets]
        )
        confidences = [prediction_by_offset[offset][1] for offset in offsets]
        consolidated = consolidate_predictions_to_words(
            text,
            offsets,
            labels,
            confidences,
            word_ids=word_ids,
        )
        tinybert_latency = time.perf_counter() - tinybert_started
        tinybert_spans: list[SpanCandidate] = consolidated.spans
        structured_started = time.perf_counter()
        structured_spans = [
            span.as_candidate() for span in self._structured.find_spans(text)
        ]
        structured_latency = time.perf_counter() - structured_started
        merged = merge_pii_spans(text, tinybert_spans, structured_spans)
        return PIIResult(
            original_text=text,
            masked_text=merged.masked_text,
            latency_seconds=time.perf_counter() - started,
            entities=merged.entities,
            safe_to_release=True,
            privacy_status="masked",
            raw_model_output={
                "bio_repairs": repairs,
                "window_count": len(windows),
                "word_policy": PII_WORD_POLICY,
                "structured_enabled": True,
                "tinybert_latency_seconds": tinybert_latency,
                "structured_latency_seconds": structured_latency,
                "span_merger_latency_seconds": merged.latency_seconds,
                "partial_word_boundary_count": count_partial_word_boundaries(
                    text, merged.chosen_spans
                ),
                "span_trace": [
                    decision_to_dict(decision) for decision in merged.decisions
                ],
                "warnings": consolidated.warnings,
            },
        )

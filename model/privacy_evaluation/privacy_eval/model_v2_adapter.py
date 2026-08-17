"""A model-agnostic masker for evaluating candidate checkpoints.

``adapters.PIIMaskerAdapter`` is the production path and is pinned to TinyBERT:
it loads the tokenizer named in ``live_config.json`` and rejects any label map
that is not exactly 67 entries. Model V2 uses a different tokenizer
(SentencePiece rather than WordPiece) and 63 labels, so evaluating it through
that adapter is impossible by design.

Rather than weaken the production contract to run an experiment, this module
reimplements only the parts that must vary - tokenizer, label map, window size -
and imports the rest of the decision logic from production unchanged:

* ``consolidate_predictions_to_words`` (whole-word promotion);
* ``merge_pii_spans`` (overlap resolution and placeholder substitution).

Those two functions are pure and already model-independent, so an evaluation run
here exercises the same decoding behaviour the deployed pipeline would.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_ROOT = REPO_ROOT / "FullPipeline_V3"

DEFAULT_CONTEXT_LIMIT = 512
DEFAULT_OVERFLOW_STRIDE = 128
DEFAULT_BATCH_SIZE = 4


def _production_span_api():
    root = str(PRODUCTION_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from integrated_pipeline.span_merger import (
        consolidate_predictions_to_words,
        count_partial_word_boundaries,
        derive_word_ids,
        merge_pii_spans,
    )

    return SimpleNamespace(
        consolidate_predictions_to_words=consolidate_predictions_to_words,
        count_partial_word_boundaries=count_partial_word_boundaries,
        derive_word_ids=derive_word_ids,
        merge_pii_spans=merge_pii_spans,
    )


def repair_bio(labels: Sequence[str]) -> tuple[list[str], int]:
    """Identical to the production BIO repair, kept local to avoid an import.

    ``pii_adapter._repair_bio`` is private and lives beside the 67-label
    constant; duplicating these few lines is cleaner than importing across that
    boundary and accidentally coupling evaluation to the production label count.
    """

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


class ModelV2OnnxMasker:
    """Run any ONNX token classifier through the production decoding path."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        id2label: dict[int, str] | str | Path,
        *,
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        overflow_stride: int = DEFAULT_OVERFLOW_STRIDE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        structured_recognizer: Any | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.context_limit = int(context_limit)
        self.overflow_stride = int(overflow_stride)
        self.batch_size = int(batch_size)
        self.intra_op_threads = intra_op_threads
        self._structured = structured_recognizer
        self._api = _production_span_api()

        if isinstance(id2label, (str, Path)):
            raw = json.loads(Path(id2label).read_text(encoding="utf-8"))
            payload = raw.get("id2label", raw) if isinstance(raw, dict) else raw
            self.id2label = {int(key): str(value) for key, value in payload.items()}
        else:
            self.id2label = {int(key): str(value) for key, value in id2label.items()}
        if set(self.id2label) != set(range(len(self.id2label))):
            raise ValueError("id2label must cover a contiguous range starting at 0.")
        self.num_labels = len(self.id2label)

        self._tokenizer: Any | None = None
        self._session: Any | None = None

    # -- loading ------------------------------------------------------------

    def load(self) -> None:
        if self._tokenizer is not None and self._session is not None:
            return
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model is missing: {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path), local_files_only=True, use_fast=True
        )
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("A fast tokenizer is required for exact character offsets.")

        options = ort.SessionOptions()
        if self.intra_op_threads is not None:
            options.intra_op_num_threads = int(self.intra_op_threads)
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(self.model_path), options, providers=["CPUExecutionProvider"]
        )
        output_names = {output.name for output in session.get_outputs()}
        if "logits" not in output_names:
            raise RuntimeError(f"Expected a 'logits' output, found {sorted(output_names)}.")
        self._tokenizer, self._session = tokenizer, session

    # -- windowing ----------------------------------------------------------

    def _windows(self, text: str) -> tuple[Any, list[dict[str, Any]]]:
        assert self._tokenizer is not None
        full = self._tokenizer(
            text,
            truncation=False,
            padding=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        if len(full["input_ids"]) <= self.context_limit:
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
            max_length=self.context_limit,
            stride=self.overflow_stride,
            padding=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_token_type_ids=False,
        )
        windows = [
            {"input_ids": ids, "attention_mask": mask, "offset_mapping": offsets}
            for ids, mask, offsets in zip(
                overflow["input_ids"],
                overflow["attention_mask"],
                overflow["offset_mapping"],
            )
        ]
        return full, windows

    @staticmethod
    def _argmax_with_confidence(logits: Any) -> tuple[Any, Any]:
        import numpy as np

        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=-1, keepdims=True)
        predictions = np.argmax(probabilities, axis=-1)
        confidence = np.take_along_axis(probabilities, predictions[..., None], axis=-1)[..., 0]
        return predictions, confidence

    # -- inference ----------------------------------------------------------

    def mask_text(self, text: str) -> SimpleNamespace:
        """Return a production-shaped result for one input."""

        if not isinstance(text, str):
            raise TypeError("PII input must be text")
        if not text.strip():
            return SimpleNamespace(
                original_text=text,
                masked_text="",
                entities=[],
                latency_seconds=0.0,
                safe_to_release=True,
                privacy_status="masked_empty",
                raw_model_output={},
            )
        self.load()
        assert self._tokenizer is not None and self._session is not None
        started = time.perf_counter()

        full, windows = self._windows(text)
        all_offsets = [tuple(map(int, item)) for item in full["offset_mapping"]]
        offsets = [item for item in all_offsets if item[0] != item[1]]
        if not offsets:
            raise RuntimeError("Tokenizer produced no source-aligned text tokens")

        prediction_by_offset: dict[tuple[int, int], tuple[int, float]] = {}
        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
            padded = self._tokenizer.pad(
                [
                    {"input_ids": window["input_ids"], "attention_mask": window["attention_mask"]}
                    for window in batch
                ],
                padding="longest",
                return_tensors="np",
            )
            logits = self._session.run(
                ["logits"],
                {
                    "input_ids": padded["input_ids"],
                    "attention_mask": padded["attention_mask"],
                },
            )[0]
            if logits.shape[-1] != self.num_labels:
                raise RuntimeError(
                    f"Model emits {logits.shape[-1]} labels; the supplied map has {self.num_labels}."
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
                        prediction_by_offset[aligned] = (int(label_id), float(confidence))

        missing = [offset for offset in offsets if offset not in prediction_by_offset]
        if missing:
            raise RuntimeError(f"Missing prediction for offset {missing[0]}")

        labels, repairs = repair_bio(
            [self.id2label[prediction_by_offset[offset][0]] for offset in offsets]
        )
        confidences = [prediction_by_offset[offset][1] for offset in offsets]
        # No ``word_ids`` argument: the decoder derives word groups from the
        # offsets itself. Model V2 is SentencePiece, whose pre-tokenizer splits
        # on whitespace only, so the tokenizer calls ``Walusiak,`` and
        # ``23G@tutanota.com.`` one word each. Because a sensitive word is
        # expanded to its full group bounds, handing those ids over appended the
        # trailing comma or period to nearly every entity - a superset of the
        # gold span, which leaves character recall untouched while destroying
        # exact-boundary typed F1. Deriving is also exactly what the
        # training-side ``predict_spans`` does, so both paths now decode
        # identically. ``pii_adapter`` keeps its own tokenizer word ids:
        # WordPiece pre-tokenizes on punctuation, where they are correct.
        consolidated = self._api.consolidate_predictions_to_words(
            text, offsets, labels, confidences
        )
        structured_spans = []
        if self._structured is not None:
            structured_spans = [span.as_candidate() for span in self._structured.find_spans(text)]
        merged = self._api.merge_pii_spans(text, consolidated.spans, structured_spans)
        return SimpleNamespace(
            original_text=text,
            masked_text=merged.masked_text,
            entities=merged.entities,
            latency_seconds=time.perf_counter() - started,
            safe_to_release=True,
            privacy_status="masked",
            raw_model_output={
                "bio_repairs": repairs,
                "window_count": len(windows),
                "label_count": self.num_labels,
                "structured_enabled": self._structured is not None,
                "warnings": consolidated.warnings,
            },
        )


def build_model_v2_masker(
    model_path: str | Path,
    tokenizer_path: str | Path,
    id2label: dict[int, str] | str | Path,
    *,
    include_structured: bool = False,
    **kwargs: Any,
) -> ModelV2OnnxMasker:
    """Build a Model V2 masker, optionally with the production regex layer.

    ``include_structured=False`` is the model-only view used to measure what the
    neural model actually learned; ``True`` adds the frozen structured
    recognizer so the combined stack can be compared with production numbers.
    """

    structured = None
    if include_structured:
        root = str(PRODUCTION_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from integrated_pipeline.structured_pii import StructuredPIIRecognizer

        structured = StructuredPIIRecognizer()
    return ModelV2OnnxMasker(
        model_path, tokenizer_path, id2label, structured_recognizer=structured, **kwargs
    )

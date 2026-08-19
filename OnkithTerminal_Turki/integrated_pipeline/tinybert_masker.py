"""The Model V1 (TinyBERT-4 INT8) ONNX decoding path.

This is the decode that shipped first and ran the whole of the project up
to Model V2. It is kept loadable, and kept here rather than inline in
``PIIMaskerAdapter``, for two reasons.

The first is measurement. "The new classifier is better" is a claim about
two models, and it can only be checked by running both over the same
corpus in the same process, through the same span merger, on the same
afternoon. That is impossible if only one of them can be constructed.

The second is retreat. Model V2 is five and a half times the size of this
one and has never been run on the Pi. If it is too slow there, or too
large, the way back has to be a configuration key rather than a redeploy
the night before a demo.

The difference from :mod:`integrated_pipeline.model_v2_masker` is one
argument. TinyBERT tokenizes with WordPiece, whose pre-tokenizer splits on
punctuation as well as whitespace, so the tokenizer's own word ids group
``Walusiak,`` as ``Walusiak`` plus ``,`` and are exactly what
``consolidate_predictions_to_words`` wants. DeBERTa-v3 tokenizes with
SentencePiece, which splits on whitespace only, so its word ids would
append the comma to the entity; that decoder derives groups from character
offsets instead. Each model is decoded the way its own tokenizer requires,
which is the only way a comparison between them measures the models.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .span_merger import (
    SpanCandidate,
    consolidate_predictions_to_words,
    derive_word_ids,
)

DEFAULT_CONTEXT_LIMIT = 512
DEFAULT_OVERFLOW_STRIDE = 128
DEFAULT_BATCH_SIZE = 4


def repair_bio(labels: list[str]) -> tuple[list[str], int]:
    """Turn every orphan ``I-`` tag into a ``B-`` tag and count the repairs."""

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


@dataclass
class TinyBertDecoding:
    """Everything the model contributed to one input, before span merging.

    Deliberately the same shape as
    :class:`~integrated_pipeline.model_v2_masker.ModelV2Decoding`, so the
    adapter above can hold either engine without asking which one it has.
    """

    spans: list[SpanCandidate]
    labels: list[str]
    warnings: list[str] = field(default_factory=list)
    bio_repairs: int = 0
    window_count: int = 0
    token_count: int = 0


class TinyBertOnnxMasker:
    """Run the Model V1 ONNX token classifier through the shared decoder."""

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        id2label: dict[int, str] | str | Path,
        *,
        context_limit: int = DEFAULT_CONTEXT_LIMIT,
        overflow_stride: int = DEFAULT_OVERFLOW_STRIDE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        intra_op_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.context_limit = int(context_limit)
        self.overflow_stride = int(overflow_stride)
        self.batch_size = int(batch_size)
        self.intra_op_threads = intra_op_threads

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

    @property
    def loaded(self) -> bool:
        return self._tokenizer is not None and self._session is not None

    def load(self) -> None:
        if self.loaded:
            return
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model is missing: {self.model_path}")

        # The tokenizer is packaged beside the model and must never reach
        # the network; transformers only consults the hub when a local file
        # is missing, and on a Pi with no route that is a long timeout
        # rather than an error.
        previous_offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path), local_files_only=True, use_fast=True
            )
        finally:
            if previous_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(
                "A fast tokenizer is required for exact character offsets."
            )

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
            raise RuntimeError(
                f"Expected a 'logits' output, found {sorted(output_names)}."
            )
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
        exponent = np.exp(shifted)
        probabilities = exponent / np.sum(exponent, axis=-1, keepdims=True)
        predictions = np.argmax(probabilities, axis=-1)
        confidence = np.take_along_axis(
            probabilities, predictions[..., None], axis=-1
        )[..., 0]
        return predictions, confidence

    # -- inference ----------------------------------------------------------

    def decode(self, text: str) -> TinyBertDecoding:
        """Model-only view: tokenize, run every window, consolidate to words."""

        self.load()
        assert self._tokenizer is not None and self._session is not None

        full, windows = self._windows(text)
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
        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
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
            logits = self._session.run(
                ["logits"],
                {
                    "input_ids": padded["input_ids"],
                    "attention_mask": padded["attention_mask"],
                },
            )[0]
            if logits.shape[-1] != self.num_labels:
                raise RuntimeError(
                    f"Model emits {logits.shape[-1]} labels; "
                    f"the supplied map has {self.num_labels}."
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
        # WordPiece word ids are handed over on purpose. See the module
        # docstring: this tokenizer splits on punctuation, so its groups are
        # already the ones the consolidator wants. The Model V2 decoder must
        # not do this and does not.
        consolidated = consolidate_predictions_to_words(
            text, offsets, labels, confidences, word_ids=word_ids
        )
        return TinyBertDecoding(
            spans=consolidated.spans,
            labels=consolidated.labels,
            warnings=consolidated.warnings,
            bio_repairs=repairs,
            window_count=len(windows),
            token_count=len(offsets),
        )

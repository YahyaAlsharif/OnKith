#!/usr/bin/env python3
"""Local TinyBERT INT8 PII masking for OnKith/OmitAI."""
from __future__ import annotations
import json, os, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

@dataclass(frozen=True)
class EntitySpan:
    label: str
    text: str
    start: int
    end: int
    confidence: float

@dataclass(frozen=True)
class MaskingResult:
    original_text: str
    masked_text: str
    entities: List[EntitySpan]
    inference_seconds: float

class PrivacyMasker:
    def __init__(self, model_path, tokenizer_path, id2label_path, *, max_length=256,
                 confidence_threshold=0.0, providers=None):
        self.model_path = Path(model_path).resolve()
        self.tokenizer_path = Path(tokenizer_path).resolve()
        self.id2label_path = Path(id2label_path).resolve()
        self.max_length = int(max_length)
        self.confidence_threshold = float(confidence_threshold)
        for path in (self.model_path, self.tokenizer_path, self.id2label_path):
            if not path.exists():
                raise FileNotFoundError(f"Required PII artefact not found: {path}")
        self.id2label: Dict[int, str] = {
            int(k): v for k, v in json.loads(self.id2label_path.read_text(encoding="utf-8")).items()
        }
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(length=self.max_length)
        available = ort.get_available_providers()
        selected = list(providers) if providers else ["CPUExecutionProvider"]
        selected = [p for p in selected if p in available] or ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=selected)
        self.input_names = {i.name for i in self.session.get_inputs()}

    def mask(self, text: str) -> MaskingResult:
        text = (text or "").strip()
        if not text:
            return MaskingResult("", "", [], 0.0)
        enc = self.tokenizer.encode(text)
        ids = np.asarray([enc.ids], dtype=np.int64)
        attn = np.asarray([enc.attention_mask], dtype=np.int64)
        types = np.asarray([enc.type_ids], dtype=np.int64)
        feed = {}
        for name in self.input_names:
            lower = name.lower()
            if "input_ids" in lower:
                feed[name] = ids
            elif "attention_mask" in lower:
                feed[name] = attn
            elif "token_type_ids" in lower or "segment_ids" in lower:
                feed[name] = types
            else:
                raise RuntimeError(f"Unsupported ONNX input {name!r}; inputs={sorted(self.input_names)}")
        started = time.perf_counter()
        logits = np.asarray(self.session.run(None, feed)[0])
        elapsed = time.perf_counter() - started
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")
        probs = _softmax(logits[0])
        pred_ids = probs.argmax(axis=-1)
        confs = probs.max(axis=-1)
        predictions = []
        for offset, attention, label_id, confidence in zip(enc.offsets, enc.attention_mask, pred_ids.tolist(), confs.tolist()):
            start, end = offset
            if not attention or end <= start:
                continue
            label = self.id2label.get(int(label_id), "O")
            if label != "O" and confidence < self.confidence_threshold:
                label = "O"
            predictions.append((start, end, label, float(confidence)))
        entities = _merge_bio(text, predictions)
        return MaskingResult(text, _apply_masks(text, entities), entities, elapsed)

def _softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def _split_bio(label: str) -> Tuple[str, str]:
    if label == "O" or "-" not in label:
        return "O", ""
    return tuple(label.split("-", 1))

def _merge_bio(text, predictions):
    entities = []
    current_type = current_start = current_end = None
    confidences = []
    def flush():
        nonlocal current_type, current_start, current_end, confidences
        if current_type is not None:
            entities.append(EntitySpan(current_type, text[current_start:current_end], current_start,
                                       current_end, sum(confidences)/len(confidences)))
        current_type = current_start = current_end = None
        confidences = []
    for start, end, label, confidence in predictions:
        prefix, entity_type = _split_bio(label)
        if prefix == "O":
            flush(); continue
        can_continue = current_type == entity_type and current_end is not None and start <= current_end + 1
        if prefix == "B" or not can_continue:
            flush(); current_type, current_start, current_end, confidences = entity_type, start, end, [confidence]
        else:
            current_end = max(current_end, end); confidences.append(confidence)
    flush()
    return entities

def _apply_masks(text, entities):
    if not entities:
        return text
    out, cursor = [], 0
    for entity in sorted(entities, key=lambda e: (e.start, e.end)):
        if entity.start < cursor:
            continue
        out += [text[cursor:entity.start], f"[{entity.label}]"]
        cursor = entity.end
    out.append(text[cursor:])
    return "".join(out)

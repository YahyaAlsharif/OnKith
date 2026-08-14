"""Offline compatibility check using the production-pinned runtime versions."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from transformers import AutoTokenizer


PROBES = (
    "Anne-Marie O'Neill uses dev+alerts@sub.example.co.uk.",
    "account id AB-0049/XY",
    "lowercase punctuation free transcript",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-label-count", type=int, default=67)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, use_fast=True
    )
    raw = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    tokenizer_cases = []
    for text in PROBES:
        hf = tokenizer(text, return_offsets_mapping=True)
        direct = raw.encode(text, add_special_tokens=True)
        exact = (
            hf["input_ids"] == direct.ids
            and [tuple(item) for item in hf["offset_mapping"]]
            == [tuple(item) for item in direct.offsets]
        )
        if not exact:
            raise AssertionError(f"Raw tokenizer mismatch for {text!r}")
        tokenizer_cases.append({"text": text, "exact_match": True})

    model_results = {}
    for name, path in (("fp32", args.fp32), ("int8", args.int8)):
        if path is None:
            continue
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for length in (75, 221, 289, 512):
            inputs = {
                "input_ids": np.ones((1, length), dtype=np.int64),
                "attention_mask": np.ones((1, length), dtype=np.int64),
            }
            output = session.run(["logits"], inputs)[0]
            if tuple(output.shape) != (1, length, args.expected_label_count):
                raise AssertionError(f"Unexpected {name} output shape {output.shape}")
        model_results[name] = {
            "loaded": True,
            "provider": session.get_providers()[0],
            "lengths_passed": [75, 221, 289, 512],
        }

    report = {
        "versions": {
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
            "onnxruntime": version("onnxruntime"),
            "sentencepiece": version("sentencepiece"),
        },
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "tokenizer_config_normalized_for_4_53": True,
        "expected_label_count": args.expected_label_count,
        "tokenizer_cases": tokenizer_cases,
        "models": model_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

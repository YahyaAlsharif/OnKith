"""Clean host benchmark for final Phase 4 artifacts and TinyBERT baseline."""

from __future__ import annotations

import gc
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
from transformers import AutoTokenizer


SEED = 42
LENGTHS = (75, 221, 289, 512)
WARMUPS = 3
REPETITIONS = 10
DEFAULT_THREAD_REPETITIONS = 5

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent                 # model/DeBERTa-v3-xsmall
REPO_ROOT = MODEL_DIR.parents[1]
ARTIFACT_ROOT = MODEL_DIR / "artifacts" / "preflight"
DEBERTA_FP32 = ARTIFACT_ROOT / "onnx_model" / "model_fp32.onnx"
DEBERTA_INT8 = ARTIFACT_ROOT / "onnx_model" / "model_int8.onnx"
DEBERTA_TOKENIZER = ARTIFACT_ROOT / "tokenizer"
TINYBERT_INT8 = REPO_ROOT / "FullPipeline_V3" / "models" / "tinybert_pii" / "model_int8.onnx"
TINYBERT_TOKENIZER = REPO_ROOT / "FullPipeline_V3" / "models" / "tinybert_pii" / "tokenizer"
OUTPUT = ARTIFACT_ROOT / "metrics" / "final_benchmark.json"


def summary(seconds: list[float]) -> dict[str, float | int]:
    milliseconds = [value * 1000.0 for value in seconds]
    return {
        "repetitions": len(milliseconds),
        "mean_ms": float(statistics.fmean(milliseconds)),
        "median_ms": float(statistics.median(milliseconds)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "min_ms": float(min(milliseconds)),
        "max_ms": float(max(milliseconds)),
    }


def session(path: Path, *, one_thread: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if one_thread:
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def inputs(length: int) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.ones((1, length), dtype=np.int64),
        "attention_mask": np.ones((1, length), dtype=np.int64),
    }


def model_benchmark(
    model_paths: dict[str, Path], repetitions: int, *, one_thread: bool
) -> dict[str, Any]:
    result = {name: {} for name in model_paths}
    for name, path in model_paths.items():
        current = session(path, one_thread=one_thread)
        for length in LENGTHS:
            model_inputs = inputs(length)
            for _ in range(WARMUPS):
                current.run(["logits"], model_inputs)
            timings = []
            for _ in range(repetitions):
                started = time.perf_counter()
                output = current.run(["logits"], model_inputs)[0]
                timings.append(time.perf_counter() - started)
                if tuple(output.shape) != (1, length, 67):
                    raise AssertionError(f"Unexpected {name} shape {output.shape}")
            result[name][str(length)] = summary(timings)
        del current
        gc.collect()
    return result


def tokenizer_benchmark() -> dict[str, Any]:
    tokenizers = {
        "deberta": AutoTokenizer.from_pretrained(
            DEBERTA_TOKENIZER, local_files_only=True, use_fast=True
        ),
        "tinybert": AutoTokenizer.from_pretrained(
            TINYBERT_TOKENIZER, local_files_only=True, use_fast=True
        ),
    }
    result = {name: {} for name in tokenizers}
    for name, tokenizer in tokenizers.items():
        for length in LENGTHS:
            text = "privacy aware conversational transcript " * (length * 2)
            for _ in range(WARMUPS):
                tokenizer(
                    text,
                    truncation=True,
                    max_length=length,
                    padding="max_length",
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors="np",
                )
            timings = []
            for _ in range(REPETITIONS):
                started = time.perf_counter()
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=length,
                    padding="max_length",
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    return_tensors="np",
                )
                timings.append(time.perf_counter() - started)
                if tuple(encoded["input_ids"].shape) != (1, length):
                    raise AssertionError("Tokenizer did not produce the requested length")
            result[name][str(length)] = summary(timings)
    return result


def main() -> None:
    process = psutil.Process()
    python_processes = [
        item
        for item in psutil.process_iter(["pid", "name"])
        if (item.info.get("name") or "").lower().startswith("python")
    ]
    one_thread_paths = {
        "deberta_fp32": DEBERTA_FP32,
        "deberta_int8": DEBERTA_INT8,
        "tinybert_int8": TINYBERT_INT8,
    }
    report = {
        "measurement_scope": "development machine; not Raspberry Pi",
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "python_process_count_at_start": len(python_processes),
            "benchmark_process_pid": process.pid,
        },
        "one_thread_controlled": {
            "warmups": WARMUPS,
            "repetitions": REPETITIONS,
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "measurement_order": list(one_thread_paths),
            "session_policy": "each model loaded and measured in isolation",
            "models": model_benchmark(
                one_thread_paths, REPETITIONS, one_thread=True
            ),
        },
        "tokenizers": tokenizer_benchmark(),
    }
    report["deberta_int8_default_threads"] = {
        "warmups": WARMUPS,
        "repetitions": DEFAULT_THREAD_REPETITIONS,
        "intra_op_num_threads": 0,
        "inter_op_num_threads": 0,
        "note": "0 means ONNX Runtime default selection, matching FullPipeline_V3",
        "models": model_benchmark(
            {"deberta_int8": DEBERTA_INT8},
            DEFAULT_THREAD_REPETITIONS,
            one_thread=False,
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

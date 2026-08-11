"""Deployment preflight for microsoft/deberta-v3-xsmall.

This creates a deterministic, randomly initialized 67-label token-classification
head only to test the deployment contract. It does not train or evaluate a PII
model. Large weights and ONNX files are generated under ignored directories.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch
from huggingface_hub import HfApi
from onnx import TensorProto
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


# The modern PyTorch exporter writes Unicode progress markers. Windows' legacy
# console encoding can otherwise raise before the export itself is attempted.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


SEED = 42
MODEL_ID = "microsoft/deberta-v3-xsmall"
NUM_LABELS = 67
CONTEXT_LIMIT = 512
OVERFLOW_STRIDE = 128
OPSET = 17
BENCHMARK_LENGTHS = (75, 221, 289, 512)
BENCHMARK_WARMUPS = 3
BENCHMARK_REPETITIONS = 10

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
ARTIFACT_ROOT = HERE / "artifacts" / "preflight"
CACHE_DIR = ARTIFACT_ROOT / "cache"
TOKENIZER_DIR = ARTIFACT_ROOT / "tokenizer"
CONFIG_DIR = ARTIFACT_ROOT / "config"
METRICS_DIR = ARTIFACT_ROOT / "metrics"
ONNX_DIR = ARTIFACT_ROOT / "onnx_model"
FP32_PATH = ONNX_DIR / "model_fp32.onnx"
INT8_PATH = ONNX_DIR / "model_int8.onnx"
PREPROCESSED_FP32_PATH = ONNX_DIR / "model_fp32_preprocessed.onnx"
RESULTS_PATH = METRICS_DIR / "preflight_results.json"
LABEL_MAP_PATH = (
    REPO_ROOT
    / "model"
    / "TinyBERT-4"
    / "artifacts"
    / "tinybert_pii"
    / "config"
    / "id2label.json"
)

TOKENIZER_PROBES = (
    "jordan.mercer@research.example.ai",
    "Contact Anne-Marie O'Neill at dev+alerts@sub.example.co.uk.",
    "dr. łukasz żółć met xi chen and D'Angelo Ruiz",
    "account id AB-0049/XY and passport P<GBRSMITH<<ALICE",
    "my email is john at example dot com uh please repeat it",
    "lowercase transcript with  double   spaces\nand punctuation-free text",
    "ZX-81 is a product code, 09:30 is a time, and 21.5 C is a temperature.",
)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency_summary(seconds: list[float]) -> dict[str, float | int]:
    milliseconds = [item * 1000.0 for item in seconds]
    return {
        "repetitions": len(milliseconds),
        "mean_ms": float(statistics.fmean(milliseconds)),
        "median_ms": float(statistics.median(milliseconds)),
        "p95_ms": percentile(milliseconds, 95),
        "min_ms": float(min(milliseconds)),
        "max_ms": float(max(milliseconds)),
    }


def load_labels() -> tuple[dict[int, str], dict[str, int]]:
    raw = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    id2label = {int(key): str(value) for key, value in raw.items()}
    if set(id2label) != set(range(NUM_LABELS)):
        raise RuntimeError("Existing ontology must contain label IDs 0 through 66")
    label2id = {label: label_id for label_id, label in id2label.items()}
    if len(label2id) != NUM_LABELS:
        raise RuntimeError("Existing ontology labels must be unique")
    return id2label, label2id


def environment_report(resolved_revision: str) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "model_id": MODEL_ID,
        "resolved_revision": resolved_revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_memory_mib": memory.total / (1024**2),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "sentencepiece",
                "onnx",
                "onnxruntime",
                "onnxscript",
                "numpy",
                "psutil",
                "safetensors",
                "huggingface-hub",
            )
        },
        "measurement_scope": "development machine; not Raspberry Pi",
    }


def validate_offsets(text: str, offsets: list[tuple[int, int]]) -> None:
    for start, end in offsets:
        start = int(start)
        end = int(end)
        if start == end:
            continue
        if start < 0 or end <= start or end > len(text):
            raise AssertionError(f"Invalid source offset {(start, end)} for {text!r}")


def tokenizer_preflight(resolved_revision: str) -> tuple[Any, dict[str, Any]]:
    original = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=resolved_revision,
        cache_dir=CACHE_DIR,
        use_fast=True,
    )
    if not getattr(original, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for exact offsets")
    original.save_pretrained(TOKENIZER_DIR)
    tokenizer_json = TOKENIZER_DIR / "tokenizer.json"
    tokenizer_config_path = TOKENIZER_DIR / "tokenizer_config.json"
    if not tokenizer_json.is_file():
        raise RuntimeError("save_pretrained did not produce tokenizer.json")
    # Transformers 5.x writes runtime-only fields that older production
    # Transformers versions interpret differently. Keep a stable, minimal
    # deployment config; tokenizer.json remains the authoritative backend.
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    for unstable_key in (
        "backend",
        "extra_special_tokens",
        "is_local",
        "local_files_only",
        "vocab_type",
    ):
        tokenizer_config.pop(unstable_key, None)
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        saved = AutoTokenizer.from_pretrained(
            TOKENIZER_DIR,
            local_files_only=True,
            use_fast=True,
        )
    finally:
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline
    raw = Tokenizer.from_file(str(tokenizer_json))

    cases: list[dict[str, Any]] = []
    for text in TOKENIZER_PROBES:
        first = original(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        second = saved(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        direct = raw.encode(text, add_special_tokens=True)
        first_offsets = [tuple(map(int, item)) for item in first["offset_mapping"]]
        second_offsets = [tuple(map(int, item)) for item in second["offset_mapping"]]
        direct_offsets = [tuple(map(int, item)) for item in direct.offsets]
        validate_offsets(text, first_offsets)
        validate_offsets(text, second_offsets)
        validate_offsets(text, direct_offsets)
        hf_equal = (
            first["input_ids"] == second["input_ids"]
            and first["attention_mask"] == second["attention_mask"]
            and first_offsets == second_offsets
            and first["special_tokens_mask"] == second["special_tokens_mask"]
        )
        raw_equal = (
            first["input_ids"] == direct.ids
            and first["attention_mask"] == direct.attention_mask
            and first_offsets == direct_offsets
        )
        if not hf_equal or not raw_equal:
            raise AssertionError(f"Tokenizer round-trip mismatch for {text!r}")
        cases.append(
            {
                "text": text,
                "token_count": len(first["input_ids"]),
                "saved_hf_exact_match": hf_equal,
                "raw_tokenizers_exact_match": raw_equal,
                "offsets_valid": True,
            }
        )

    long_text = "privacy aware conversational transcript " * 800
    overflow = saved(
        long_text,
        truncation=True,
        max_length=CONTEXT_LIMIT,
        stride=OVERFLOW_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    lengths = [len(row) for row in overflow["input_ids"]]
    if len(lengths) < 2 or max(lengths) > CONTEXT_LIMIT:
        raise AssertionError("Tokenizer overflow-window contract failed")
    for offsets in overflow["offset_mapping"]:
        validate_offsets(long_text, [tuple(map(int, item)) for item in offsets])

    files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(TOKENIZER_DIR.iterdir())
        if path.is_file()
    }
    report = {
        "class": type(saved).__name__,
        "is_fast": bool(saved.is_fast),
        "vocab_size": int(saved.vocab_size),
        "model_max_length": int(saved.model_max_length),
        "files": files,
        "probe_cases": cases,
        "overflow_window_count": len(lengths),
        "overflow_window_lengths": lengths,
        "context_limit": CONTEXT_LIMIT,
        "overflow_stride": OVERFLOW_STRIDE,
    }
    return saved, report


class LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def build_model(
    resolved_revision: str,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(SEED)
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=resolved_revision,
        cache_dir=CACHE_DIR,
    )
    config.num_labels = NUM_LABELS
    config.id2label = id2label
    config.label2id = label2id
    config.return_dict = True
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        revision=resolved_revision,
        cache_dir=CACHE_DIR,
        config=config,
        ignore_mismatched_sizes=True,
    )
    # The upstream checkpoint is stored in FP16. Force a genuine FP32 source
    # graph before evaluating the repository's dynamic QInt8 deployment path.
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    head = sum(parameter.numel() for parameter in model.classifier.parameters())
    parameter_dtypes = Counter(str(parameter.dtype) for parameter in model.parameters())
    if set(parameter_dtypes) != {"torch.float32"}:
        raise AssertionError(f"Expected an FP32 model, found {dict(parameter_dtypes)}")
    report = {
        "architecture": type(model).__name__,
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "backbone_parameters": int(total - head),
        "classification_head_parameters": int(head),
        "parameter_bytes": int(
            sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        ),
        "parameter_dtypes": dict(sorted(parameter_dtypes.items())),
        "num_labels": int(model.config.num_labels),
        "classifier_is_random_untrained": True,
    }
    config_payload = config.to_dict()
    config_payload["resolved_revision"] = resolved_revision
    (CONFIG_DIR / "token_classification_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return model, report


def representative_inputs(tokenizer: Any) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        [
            "DeBERTa dynamic ONNX shape check.",
            "A deliberately longer second sentence verifies both dynamic exported axes.",
        ],
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(torch.int64),
        "attention_mask": encoded["attention_mask"].to(torch.int64),
    }


def export_onnx(model: torch.nn.Module, tokenizer: Any) -> dict[str, Any]:
    wrapped = LogitsOnly(model).eval()
    inputs = representative_inputs(tokenizer)
    batch_dimension = torch.export.Dim("batch_size", min=1)
    sequence_dimension = torch.export.Dim(
        "sequence_length", min=1, max=CONTEXT_LIMIT
    )
    errors: list[str] = []
    try:
        program = torch.onnx.export(
            wrapped,
            args=(inputs["input_ids"], inputs["attention_mask"]),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=OPSET,
            dynamo=True,
            external_data=False,
            dynamic_shapes=(
                {0: batch_dimension, 1: sequence_dimension},
                {0: batch_dimension, 1: sequence_dimension},
            ),
        )
        program.save(FP32_PATH, external_data=False)
        exporter = "torch.onnx.export Dynamo"
    except Exception as exc:
        errors.append(f"Dynamo: {type(exc).__name__}: {exc}")
        torch.onnx.export(
            wrapped,
            args=(inputs["input_ids"], inputs["attention_mask"]),
            f=str(FP32_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=OPSET,
            dynamo=False,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
        )
        exporter = "torch.onnx.export legacy dynamic_axes fallback"

    graph = onnx.load(str(FP32_PATH))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(FP32_PATH), providers=["CPUExecutionProvider"])
    if [item.name for item in session.get_inputs()] != ["input_ids", "attention_mask"]:
        raise AssertionError("Unexpected ONNX input contract")
    if [item.name for item in session.get_outputs()] != ["logits"]:
        raise AssertionError("Unexpected ONNX output contract")
    output = session.run(
        ["logits"],
        {key: value.numpy() for key, value in inputs.items()},
    )[0]
    if tuple(output.shape) != (2, inputs["input_ids"].shape[1], NUM_LABELS):
        raise AssertionError(f"Unexpected ONNX output shape {output.shape}")
    if session.get_outputs()[0].type != "tensor(float)":
        raise AssertionError(
            f"Expected genuine FP32 logits, found {session.get_outputs()[0].type}"
        )
    return {
        "success": True,
        "exporter": exporter,
        "fallback_errors": errors,
        "opset": int(graph.opset_import[0].version),
        "size_bytes": FP32_PATH.stat().st_size,
        "sha256": sha256(FP32_PATH),
        "inputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_outputs()
        ],
        "checker_passed": True,
        "cpu_execution_provider_loaded": True,
    }


_TYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.BFLOAT16: 2,
}


def graph_audit(path: Path) -> dict[str, Any]:
    graph = onnx.load(str(path))
    dtype_bytes: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    residual_fp32: list[dict[str, Any]] = []
    embedding_fp32_bytes = 0
    for initializer in graph.graph.initializer:
        elements = int(np.prod(initializer.dims, dtype=np.int64)) if initializer.dims else 1
        byte_count = elements * _TYPE_BYTES.get(initializer.data_type, 0)
        dtype_name = TensorProto.DataType.Name(initializer.data_type)
        dtype_bytes[dtype_name] += byte_count
        dtype_counts[dtype_name] += 1
        if initializer.data_type == TensorProto.FLOAT:
            row = {"name": initializer.name, "size_bytes": int(byte_count)}
            residual_fp32.append(row)
            if "embed" in initializer.name.lower():
                embedding_fp32_bytes += byte_count
    residual_fp32.sort(key=lambda row: row["size_bytes"], reverse=True)
    return {
        "operator_counts": dict(sorted(Counter(node.op_type for node in graph.graph.node).items())),
        "initializer_counts_by_dtype": dict(sorted(dtype_counts.items())),
        "initializer_bytes_by_dtype": dict(sorted(dtype_bytes.items())),
        "residual_fp32_initializer_count": len(residual_fp32),
        "largest_residual_fp32_initializers": residual_fp32[:20],
        "embedding_named_fp32_bytes": int(embedding_fp32_bytes),
    }


def quantize() -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []

    def attempt(source: Path, name: str, extra_options: dict[str, Any] | None = None) -> bool:
        if INT8_PATH.exists():
            INT8_PATH.unlink()
        try:
            quantize_dynamic(
                model_input=str(source),
                model_output=str(INT8_PATH),
                weight_type=QuantType.QInt8,
                per_channel=False,
                reduce_range=False,
                extra_options=extra_options,
            )
            graph = onnx.load(str(INT8_PATH))
            onnx.checker.check_model(graph)
            session = ort.InferenceSession(
                str(INT8_PATH), providers=["CPUExecutionProvider"]
            )
            if session.get_outputs()[0].type != "tensor(float)":
                raise AssertionError("Quantized graph must preserve FP32 logits")
            attempts.append({"name": name, "success": True})
            return True
        except Exception as exc:
            attempts.append(
                {"name": name, "success": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            return False

    success = attempt(FP32_PATH, "standard_dynamic_qint8")
    if not success:
        success = attempt(
            FP32_PATH,
            "dynamic_qint8_enable_subgraph",
            {"EnableSubgraph": True},
        )
    if not success:
        try:
            if PREPROCESSED_FP32_PATH.exists():
                PREPROCESSED_FP32_PATH.unlink()
            quant_pre_process(
                input_model=FP32_PATH,
                output_model_path=PREPROCESSED_FP32_PATH,
                skip_symbolic_shape=True,
                skip_onnx_shape=False,
                skip_optimization=False,
                save_as_external_data=False,
            )
            preprocessed_graph = onnx.load(str(PREPROCESSED_FP32_PATH))
            onnx.checker.check_model(preprocessed_graph)
            attempts.append({"name": "ort_quant_pre_process", "success": True})
            success = attempt(
                PREPROCESSED_FP32_PATH,
                "preprocessed_dynamic_qint8",
            )
        except Exception as exc:
            attempts.append(
                {
                    "name": "ort_quant_pre_process",
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not success:
        if INT8_PATH.exists():
            INT8_PATH.unlink()
        return {
            "success": False,
            "method": "ONNX Runtime dynamic quantization",
            "weight_type": "QInt8",
            "per_channel": False,
            "reduce_range": False,
            "attempts": attempts,
        }
    session = ort.InferenceSession(str(INT8_PATH), providers=["CPUExecutionProvider"])
    return {
        "success": True,
        "method": "ONNX Runtime dynamic quantization",
        "weight_type": "QInt8",
        "per_channel": False,
        "reduce_range": False,
        "attempts": attempts,
        "size_bytes": INT8_PATH.stat().st_size,
        "sha256": sha256(INT8_PATH),
        "checker_passed": True,
        "cpu_execution_provider_loaded": True,
        "output_names": [item.name for item in session.get_outputs()],
    }


def fixed_length_input(tokenizer: Any, length: int) -> tuple[str, dict[str, np.ndarray]]:
    text = "privacy aware conversational transcript " * (length * 2)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=length,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="np",
    )
    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64, copy=False),
        "attention_mask": encoded["attention_mask"].astype(np.int64, copy=False),
    }
    if tuple(inputs["input_ids"].shape) != (1, length):
        raise AssertionError(f"Expected exact benchmark length {length}")
    return text, inputs


def create_session(path: Path) -> tuple[Any, dict[str, Any]]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    process = psutil.Process()
    before = process.memory_info().rss
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    load_seconds = time.perf_counter() - started
    after = process.memory_info().rss
    return session, {
        "load_time_seconds": load_seconds,
        "rss_before_mib": before / (1024**2),
        "rss_after_mib": after / (1024**2),
        "rss_delta_mib": (after - before) / (1024**2),
        "intra_op_num_threads": options.intra_op_num_threads,
        "inter_op_num_threads": options.inter_op_num_threads,
        "execution_mode": "ORT_SEQUENTIAL",
    }


def benchmark(tokenizer: Any, model_paths: dict[str, Path]) -> dict[str, Any]:
    sessions: dict[str, Any] = {}
    report: dict[str, Any] = {
        "measurement_scope": "development machine; not Raspberry Pi",
        "warmups": BENCHMARK_WARMUPS,
        "repetitions": BENCHMARK_REPETITIONS,
        "threading": "one intra-op thread, one inter-op thread, sequential execution",
        "models": {},
    }
    for name, path in model_paths.items():
        sessions[name], load_report = create_session(path)
        report["models"][name] = {"session": load_report, "lengths": {}}

    rng = random.Random(SEED)
    for length in BENCHMARK_LENGTHS:
        text, inputs = fixed_length_input(tokenizer, length)
        for _ in range(BENCHMARK_WARMUPS):
            for session in sessions.values():
                session.run(["logits"], inputs)
        model_latencies = {name: [] for name in sessions}
        tokenizer_latencies: list[float] = []
        overall_latencies = {name: [] for name in sessions}
        for _ in range(BENCHMARK_REPETITIONS):
            order = list(sessions)
            rng.shuffle(order)
            for name in order:
                started = time.perf_counter()
                sessions[name].run(["logits"], inputs)
                model_latencies[name].append(time.perf_counter() - started)
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
            tokenizer_latencies.append(time.perf_counter() - started)
            overall_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64, copy=False),
                "attention_mask": encoded["attention_mask"].astype(np.int64, copy=False),
            }
            for name in order:
                started = time.perf_counter()
                sessions[name].run(["logits"], overall_inputs)
                overall_latencies[name].append(
                    tokenizer_latencies[-1] + (time.perf_counter() - started)
                )
        for name in sessions:
            report["models"][name]["lengths"][str(length)] = {
                "model_inference": latency_summary(model_latencies[name]),
                "tokenizer_plus_model": latency_summary(overall_latencies[name]),
                "output_shape": [1, length, NUM_LABELS],
            }
        report.setdefault("tokenizer", {})[str(length)] = latency_summary(tokenizer_latencies)

    batch_two = np.repeat(fixed_length_input(tokenizer, 221)[1]["input_ids"], 2, axis=0)
    mask_two = np.ones_like(batch_two, dtype=np.int64)
    for name, session in sessions.items():
        output = session.run(
            ["logits"], {"input_ids": batch_two, "attention_mask": mask_two}
        )[0]
        if tuple(output.shape) != (2, 221, NUM_LABELS):
            raise AssertionError(f"Dynamic batch check failed for {name}: {output.shape}")
        report["models"][name]["dynamic_batch_two_shape"] = list(output.shape)
    return report


def run_preflight() -> dict[str, Any]:
    for directory in (CACHE_DIR, TOKENIZER_DIR, CONFIG_DIR, METRICS_DIR, ONNX_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    resolved_revision = HfApi().model_info(MODEL_ID, revision="main").sha
    if not resolved_revision:
        raise RuntimeError("Hugging Face did not return a resolved model revision")
    results: dict[str, Any] = {
        "purpose": "deployment compatibility only; classifier head is random and untrained",
        "seed": SEED,
        "environment": environment_report(resolved_revision),
        "stages": {},
    }
    (CONFIG_DIR / "environment_versions.json").write_text(
        json.dumps(results["environment"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    id2label, label2id = load_labels()
    tokenizer, results["stages"]["tokenizer"] = tokenizer_preflight(resolved_revision)
    model, results["stages"]["model"] = build_model(
        resolved_revision, id2label, label2id
    )
    results["stages"]["onnx_fp32"] = export_onnx(model, tokenizer)
    results["stages"]["onnx_fp32"]["graph_audit"] = graph_audit(FP32_PATH)
    fp32_audit = results["stages"]["onnx_fp32"]["graph_audit"]
    if "FLOAT16" in fp32_audit["initializer_bytes_by_dtype"]:
        raise AssertionError("FP32 graph unexpectedly contains FLOAT16 initializers")
    if fp32_audit["initializer_bytes_by_dtype"].get("FLOAT", 0) <= 0:
        raise AssertionError("FP32 graph contains no FLOAT initializers")
    try:
        results["stages"]["onnx_int8"] = quantize()
        if results["stages"]["onnx_int8"]["success"]:
            results["stages"]["onnx_int8"]["graph_audit"] = graph_audit(INT8_PATH)
    except Exception as exc:
        results["stages"]["onnx_int8"] = {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    results["stages"]["benchmark"] = {
        "status": "run separately after export work and other CPU workloads exit",
        "command": "python benchmark_final.py",
        "report": "metrics/final_benchmark.json",
    }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del model
    gc.collect()
    return results


def concise_summary(results: dict[str, Any]) -> dict[str, Any]:
    stages = results["stages"]
    return {
        "resolved_revision": results["environment"]["resolved_revision"],
        "tokenizer": stages["tokenizer"]["class"],
        "parameters": stages["model"]["total_parameters"],
        "fp32_onnx_mib": stages["onnx_fp32"]["size_bytes"] / (1024**2),
        "int8_success": stages["onnx_int8"]["success"],
        "int8_onnx_mib": (
            stages["onnx_int8"].get("size_bytes", 0) / (1024**2)
        ),
        "results_path": str(RESULTS_PATH),
    }


if __name__ == "__main__":
    try:
        payload = run_preflight()
    except Exception as error:
        print(f"PREFLIGHT FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
    print(json.dumps(concise_summary(payload), indent=2))

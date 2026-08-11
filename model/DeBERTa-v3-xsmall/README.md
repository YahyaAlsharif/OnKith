# DeBERTa-v3-xsmall deployment preflight

This folder contains the Phase 4 deployment compatibility test for
`microsoft/deberta-v3-xsmall`. It does not contain a trained Model V2.

The preflight attaches a deterministic random 67-label token-classification
head using OnKith's existing ontology. Its predictions have no quality meaning;
the head exists only to prove tokenizer, ONNX, quantization and runtime contracts.

## Reproduce

From the repository root, using the existing Python 3.11 development environment:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe model\DeBERTa-v3-xsmall\preflight_deberta_v3_xsmall.py
```

`PYTHONUTF8=1` is required on this Windows host because the modern PyTorch ONNX
exporter prints Unicode progress markers. Without it, console encoding fails
before the exporter can finish and the script falls back to a legacy graph whose
control-flow subgraphs cannot be quantized correctly.

The isolated production-pin check uses versions from
`requirements-production-compat.txt` and runs entirely offline after artifacts
have been created:

```powershell
.\.venv\Scripts\python.exe -m venv model\DeBERTa-v3-xsmall\.venv-production
model\DeBERTa-v3-xsmall\.venv-production\Scripts\python.exe -m pip install `
  -r model\DeBERTa-v3-xsmall\requirements-production-compat.txt
model\DeBERTa-v3-xsmall\.venv-production\Scripts\python.exe `
  model\DeBERTa-v3-xsmall\verify_production_runtime.py `
  --tokenizer model\DeBERTa-v3-xsmall\artifacts\preflight\tokenizer `
  --fp32 model\DeBERTa-v3-xsmall\artifacts\preflight\onnx_model\model_fp32.onnx `
  --int8 model\DeBERTa-v3-xsmall\artifacts\preflight\onnx_model\model_int8.onnx `
  --output model\DeBERTa-v3-xsmall\artifacts\preflight\metrics\production_runtime_compat.json
```

Run `benchmark_final.py` only while the machine is otherwise idle. It compares
DeBERTa FP32, DeBERTa INT8 and production TinyBERT INT8 using identical sequence
lengths and one ONNX Runtime thread, then runs a separately labelled default-
thread DeBERTa INT8 check matching the current live adapter.

## Artifact policy

Downloaded weights, caches, disposable environments and generated ONNX graphs
are ignored locally. The tracked evidence is limited to source, configuration,
compact reports and the deployment tokenizer. Existing BiLSTM, TinyBERT and
`FullPipeline_V3` files are not modified.

## Important compatibility note

Transformers 5.13 serialized `extra_special_tokens` as a list. Production-pinned
Transformers 4.53.3 interprets that field as a mapping and initially failed with
`AttributeError: 'list' object has no attribute 'keys'`. The preflight therefore
normalizes `tokenizer_config.json` to stable fields after saving; the authoritative
tokenizer backend remains `tokenizer.json`. The normalized tokenizer reloads as
`DebertaV2TokenizerFast` and matches direct low-level `tokenizers` IDs and offsets.

All latency measurements are development-host results, not Raspberry Pi results.

## Measured result

The checkpoint was pinned to revision
`4b419818330868dff6a60ad3e6b1c730f8b8c0c6`. The temporary token classifier has
70,707,907 parameters: 70,682,112 in the backbone and 25,795 in the random
67-label head. It occupied 282,831,628 bytes after being explicitly converted
to FP32.

The modern exporter produced a checked dynamic opset-17 graph with inputs
`input_ids` and `attention_mask` and output `logits[batch, sequence, 67]`:

| Artifact | Bytes | MiB |
|---|---:|---:|
| FP32 ONNX | 282,977,091 | 269.868 |
| Dynamic QInt8 ONNX | 82,276,586 | 78.465 |

Standard ONNX Runtime dynamic quantization succeeded with QInt8 weights,
`per_channel=False` and `reduce_range=False`. The quantized graph contains 49
`MatMulInteger` nodes and 49 `DynamicQuantizeLinear` nodes. Initializer storage
is 49,190,401 bytes UINT8, 17,720,497 bytes INT8, 15,188,468 bytes residual
FP32 and 216 bytes INT64. Both graphs pass ONNX checker and load through
`CPUExecutionProvider` under development ONNX Runtime 1.27.0 and production-
pinned ONNX Runtime 1.22.1.

The first Windows run exposed an exporter-specific trap. Without UTF-8 console
output, the modern exporter failed while printing a check-mark and the legacy
fallback generated `If` subgraphs that all three QInt8 attempts failed to keep
topologically valid. With UTF-8 enabled, the modern exporter succeeded and the
first standard dynamic-QInt8 attempt produced the valid 78.465 MiB graph. No
Optimum fallback was needed.

### Final uncontended host benchmark

The final comparison ran after the other Python workloads exited. Each model was
loaded and measured in isolation with one intra-op thread, one inter-op thread,
sequential execution, three warmups and ten repetitions.

| Tokens | DeBERTa FP32 median | DeBERTa INT8 median | TinyBERT INT8 median |
|---:|---:|---:|---:|
| 75 | 129.40 ms | 80.97 ms | 12.69 ms |
| 221 | 410.42 ms | 270.87 ms | 53.30 ms |
| 289 | 567.82 ms | 380.92 ms | 78.91 ms |
| 512 | 1,163.82 ms | 849.19 ms | 193.31 ms |

The current `FullPipeline_V3` adapter leaves ONNX Runtime thread counts at their
defaults. A separately labelled DeBERTa INT8 run using that host-default policy
used three warmups and five repetitions:

| Tokens | Default-thread median | Default-thread p95 |
|---:|---:|---:|
| 75 | 46.30 ms | 58.78 ms |
| 221 | 103.32 ms | 105.55 ms |
| 289 | 121.67 ms | 124.24 ms |
| 512 | 246.60 ms | 250.53 ms |

The host has 14 physical and 20 logical CPU cores, so the default-thread speedup
must not be projected onto the four-core Raspberry Pi 5. The one-thread result is
the controlled relative comparison; the default-thread result only confirms that
the current live-session policy works on this development host.

Saved `tokenizer.json` is 8,339,876 bytes. Seven unusual-name/email/identifier/
whitespace probes matched exactly across the original tokenizer, offline saved
Hugging Face tokenizer and direct low-level `tokenizers` loading. A 512-token,
128-stride long-input probe produced nine valid windows. The production-pinned
reload also passed both graphs at 75, 221, 289 and 512 tokens.

# Model V2 artifact workspace

This tree is populated by `../../model_v2_deberta_v3_xsmall.ipynb`.

| Directory | Contents | Policy |
|---|---|---|
| `data/` | materialized train, test, original validation, and supplementary validation JSONL | ignored; reproducible |
| `checkpoints/` | `latest.pt` resume state and selected `best.pt` | ignored; not a deployment artifact |
| `models/huggingface/` | selected PyTorch model | ignored; export/recovery source |
| `models/fp32/` | reference `model_fp32.onnx` | ignored; too large for normal Git |
| `models/int8/` | release candidate `model_int8.onnx` | ignored until reviewed production transfer |
| `tokenizer/` | normalized DeBERTa tokenizer | tracked after the accepted real run; required at runtime |
| `metrics/` | history, evaluation, parity, quantization gate, runtime check, deployment summary | tracked after the accepted real run |
| `smoke/` | isolated bounded Run All evidence and disposable random/tiny artifacts | ignored; never deploy |

The real-run `deployment_summary.json` records hashes and sizes. After the quality gate passes, a separate production migration copies the accepted INT8 graph plus tokenizer/config into `FullPipeline_V3/models/` and stages those files in the reviewed release. This repository currently has no Git LFS policy; do not silently commit the roughly 270 MiB FP32 graph.

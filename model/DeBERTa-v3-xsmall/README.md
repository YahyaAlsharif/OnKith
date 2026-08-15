# DeBERTa-v3-xsmall Model V2

There are two notebooks, and the split between them is deliberate.

| Notebook | Purpose | Safe to **Run All**? |
|---|---|---|
| [`model_v2_deberta_v3_xsmall.ipynb`](model_v2_deberta_v3_xsmall.ipynb) | **Build the candidate.** Data, training, validation, OOD dev, FP32 export, INT8 quantization, INT8 acceptance. | Yes, always |
| [`model_v2_frozen_ood_final.ipynb`](model_v2_frozen_ood_final.ipynb) | **Final exam.** Score the already-accepted INT8 candidate on the frozen OOD final benchmark, once. | No - it spends the holdout |

The teammate entry point is the first one. It builds, trains, evaluates, and exports the 31-entity / 63-BIO-label Model V2 model. Normal notebook defaults are the real run: up to 10 epochs with early stopping. No full Model V2 training has been run yet.

The second notebook is separated for one reason: the frozen OOD final set can be spent exactly once, and a benchmark that sits inside a development notebook eventually gets run by accident. Nothing in the main notebook can load or score the frozen annotations, so no **Run All**, no re-run of a cell, and no smoke test can burn the final holdout.

Model V2 removes `ALLERGIES`, `USERNAME`, and `WEIGHT`; `HEIGHT` remains absent; `PASSWORD` is the only addition. The deployed TinyBERT contract remains 67 labels until a separate production migration.

## Required first step

Run [`../data/saudi_PII_Builder.ipynb`](../data/saudi_PII_Builder.ipynb) with **Run All** before opening the Model V2 notebook. It writes:

- `model/data/saudi_mena_pii_dataset.csv` - required local corpus, gitignored;
- `model/data/saudi_mena_pii_summary.csv` and `.xlsx` - local reports, gitignored;
- `model/data/saudi_mena_pii_manifest.json` - small tracked hash/provenance record.

The main notebook checks the CSV schema, ontology, PASSWORD disclosure rules, row count, and SHA-256 against the manifest immediately. A missing or stale Saudi artifact fails early with a regeneration command.

From the repository root:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute `
  model\data\saudi_PII_Builder.ipynb --output .saudi_builder_execution.ipynb `
  --output-dir model\data --ExecutePreprocessor.timeout=1800
```

Other supplementary sources are streamed into the main notebook: Gretel and Kiji are required; Nemotron is optional and a failure is recorded honestly in the data manifest.

## Main notebook - build the candidate

Open [`model_v2_deberta_v3_xsmall.ipynb`](model_v2_deberta_v3_xsmall.ipynb) and use **Run All**. It runs in release-evaluation order and stops at an accepted candidate:

```text
train -> best checkpoint -> FP32/PyTorch validation -> OOD dev -> save artifacts
      -> FP32 ONNX export -> parity -> INT8 quantization
      -> INT8 validation and quality gate (test, val_original, val_supplementary,
         combined validation, OOD dev)
      -> accepted INT8 candidate manifest
      -> production-pin verification -> deployment summary -> DONE
```

The last step writes `artifacts/model_v2/metrics/accepted_int8_candidate.json`: the INT8 path, size and SHA-256, the FP32 reference hash, the selected checkpoint and best epoch, the run signature, the ontology version, the 63-label count and label-map hash, the tokenizer identity, the training-config hash, the FP32 and INT8 evaluation summaries including OOD dev, the quality-gate thresholds and per-view results, and the frozen fixture fingerprint.

`accepted: true` appears **only** when a real run clears the gate. If INT8 fails, the manifest records `accepted: false` and the benchmark notebook refuses it. A smoke run writes the same shape under `artifacts/model_v2/smoke/metrics/` with `smoke_test: true` and `accepted: false`, so smoke evidence cannot be mistaken for a release candidate.

Section 6 still fingerprints the frozen-final texts for the contamination firewall - hashes and normalised text only. The main notebook never constructs a frozen case and never reads a frozen annotation.

An isolated development smoke is available without changing production defaults:

```powershell
$env:MODEL_V2_SMOKE_TEST = "1"
$env:MODEL_V2_SMOKE_SOURCE_ROWS = "64"
$env:MODEL_V2_SMOKE_TRAIN_STEPS = "3"
$env:MODEL_V2_SMOKE_EVAL_ROWS = "8"
```

Smoke output goes only under `artifacts/model_v2/smoke/` and must never be treated as a trained release model.

## Final notebook - spend the frozen benchmark

[`model_v2_frozen_ood_final.ipynb`](model_v2_frozen_ood_final.ipynb) is run **only after** the INT8 candidate has been accepted and every number in the main notebook has been reviewed. It does not train, build datasets, augment, select checkpoints, quantize, tune thresholds, or run OOD dev. It does one thing:

```text
verify accepted candidate -> verify benchmark identity
  -> load the accepted INT8 model, tokenizer and label map
  -> load the frozen OOD final fixture -> evaluate once
  -> write artifacts/model_v2/metrics/evaluation_ood_final_int8.json
```

It scores the accepted **INT8 ONNX candidate and nothing else** - no PyTorch, no FP32 ONNX, no TinyBERT, no alternate checkpoint - because the point is to measure the artifact that passed release selection. A TinyBERT comparison must use the existing historical results rather than a second pass over the holdout.

It fails loudly, before the fixture is opened, if the candidate manifest is missing, `accepted` is not true, the candidate came from smoke mode, the INT8 file is missing or its SHA-256 no longer matches, the label count is not 63, the ontology or label-map hash differs, the tokenizer or config identity differs, the run signature disagrees with the training config, normal-evaluation evidence is missing, or the INT8 gate did not pass. The fixture itself is then checked against its committed manifest and against the fingerprint recorded when the candidate was built: path, SHA-256, case count, per-case schema, and the frozen marker.

Reruns are refused. If `evaluation_ood_final_int8.json` already exists for this fixture the notebook stops, whether the candidate hash matches or not - a second candidate scored on the final set turns it into a selection metric. `ALLOW_FROZEN_RERUN = False` is the only override, it stays `False`, and it exists purely to recover a report lost after a run that genuinely happened. Do not use it while developing or testing.

Once this benchmark's results influence any future model change, it is no longer pristine for the next model-generation cycle; that cycle needs a newly built frozen fixture.

The structure and every guard above are covered by `../privacy_evaluation/tests/test_frozen_ood_final.py`, which uses synthetic manifests and synthetic fixtures. No test ever reads the real frozen set.

## Artifact locations and policy

| Artifact | Location | Git policy | Deployment |
|---|---|---|---|
| train/test/validation JSONL | `artifacts/model_v2/data/` | ignored, reproducible | no |
| latest/best checkpoints | `artifacts/model_v2/checkpoints/` | ignored, resumable/reproducible | no |
| saved Hugging Face model | `artifacts/model_v2/models/huggingface/` | ignored | recovery/export source |
| tokenizer | `artifacts/model_v2/tokenizer/` | tracked after the accepted real run | required |
| label map, source map, manifests, training config | `config/` | tracked | required metadata |
| training/evaluation/parity/gate reports | `artifacts/model_v2/metrics/` | tracked after the real run | release evidence |
| accepted candidate manifest | `artifacts/model_v2/metrics/accepted_int8_candidate.json` | tracked after the real run | release gate; input to the final notebook |
| frozen benchmark report | `artifacts/model_v2/metrics/evaluation_ood_final_int8.json` | tracked after the one-shot run | release evidence, written once |
| FP32 ONNX | `artifacts/model_v2/models/fp32/model_fp32.onnx` | ignored; about 270 MiB, manual artifact storage or future Git LFS | parity/reference |
| INT8 ONNX | `artifacts/model_v2/models/int8/model_int8.onnx` | ignored during training | production candidate |

Ignored artifacts are regenerated by rerunning the Saudi builder and then the main notebook. `deployment_summary.json` records binary sizes and hashes so an ignored required binary cannot disappear unnoticed.

After the real INT8 quality gate and a separate reviewed TinyBERT-to-Model-V2 migration, manually copy the accepted INT8 graph, tokenizer, label map, and deployment config into `FullPipeline_V3/models/` and stage them in that release. Do not copy smoke artifacts. The later migration must also change the live label-count contract from 67 to 63 and rerun the FullPipeline regression suite.

The reviewed release transfer is intentionally explicit:

```powershell
$target = "FullPipeline_V3\models\deberta_v3_xsmall_pii"
New-Item -ItemType Directory -Force -Path "$target\tokenizer", "$target\config"
Copy-Item "model\DeBERTa-v3-xsmall\artifacts\model_v2\models\int8\model_int8.onnx" "$target\model_int8.onnx"
Copy-Item "model\DeBERTa-v3-xsmall\artifacts\model_v2\tokenizer\*" "$target\tokenizer\"
Copy-Item "model\DeBERTa-v3-xsmall\config\model_v2_label_map.json" "$target\config\label_map.json"
Copy-Item "model\DeBERTa-v3-xsmall\config\model_v2_training_config.json" "$target\config\training_config.json"
```

Only perform that copy after reviewing hashes in the real-run deployment summary; then point the production config at this exact directory and stage it as part of the migration commit.

## Supporting folders

- `config/` - committed ontology and source mapping; real-run manifests/config appear here.
- `artifacts/` - preflight evidence and Model V2 outputs.
- `preflight/` - backbone compatibility notebook, script, and benchmark; its random 67-label head has no quality meaning.
- `deployment/` - production-pin requirements and runtime verifier.

Reproduce the old backbone preflight with:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe model\DeBERTa-v3-xsmall\preflight\preflight_deberta_v3_xsmall.py
```

Run the host benchmark only on an otherwise idle machine:

```powershell
.\.venv\Scripts\python.exe model\DeBERTa-v3-xsmall\preflight\benchmark_final.py
```

Production-pin requirements and verification live under `deployment/`. Raspberry Pi latency, RAM, and thermal behavior remain mandatory deployment gates; Windows host measurements are not Pi evidence.

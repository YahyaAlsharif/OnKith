# OnKith privacy-model evaluation

This folder freezes the current TinyBERT IID baseline and adds reproducible
out-of-distribution evaluation for Model V2. It does not train a model and does
not change the live pipeline.

## Evaluation layers

The runner reports five deliberately separate stages:

1. `fp32_model` — FP32 ONNX neural predictions with the production windowing,
   BIO repair, and whole-word decoding, but no structured recognizer.
2. `int8_model` — the same model-only path with the deployed INT8 graph.
3. `structured` — the frozen regex recognizer without a neural model.
4. `structured_supplemental` — frozen plus current spoken-number supplements.
5. `production` — the current `FullPipeline_V3` stack: INT8 TinyBERT, frozen
   structured recognizer, supplemental recognizer, and fail-closed wrapper.

This separation prevents regex-caught emails and spoken-number fallbacks from
being reported as neural-model generalization.

## Data partitions

Both partitions contain exactly 360 cases:

| Primary category | Cases |
|---|---:|
| Names | 72 |
| Email | 60 |
| Numeric and structured PII | 84 |
| Contextual and multi-entity | 48 |
| Hard negatives | 72 |
| Long-window inputs | 24 |

Speech/ASR behavior is cross-cutting. Tags identify lowercase, punctuation-free,
filler-word, spoken-email, spoken-digit, partial-utterance, and long-window
cases within the primary categories.

- `ood_dev.jsonl` may guide error analysis and future data design.
- `ood_final.jsonl` is frozen. It must not guide training examples,
  augmentation, thresholds, model selection, or prompt/template changes.
- Entity values, template IDs, domain pools, numeric values, contextual values,
  filler vocabularies, and deterministic seeds are disjoint between partitions.

The manifests record exact SHA-256 hashes and category/tag audits. Fixture
annotations use exact character offsets and are checked on load.
`baselines/tinybert_long_window_audit.json` additionally records the actual
packaged-tokenizer token index and production 512/128 window coverage for all
48 long cases; whitespace positions are not treated as tokenizer positions.

## Run

From this directory with the repository Python 3.11 environment:

```powershell
& ..\..\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
& ..\..\.venv\Scripts\python.exe -B -m privacy_eval.run --partition dev
```

Run the frozen final set only for a declared checkpoint comparison:

```powershell
& ..\..\.venv\Scripts\python.exe -B -m privacy_eval.run --partition final
```

The result files contain overall, category, subcategory, tag, per-entity,
privacy-character, hard-negative, BIO-repair, latency, error, and controlled
minimal-pair results. Per-case predictions remain in memory and are not stored
by default, which keeps the checkpoint compact. Add `--retain-predictions` only
for a deliberate error-analysis run. Development-host latency is not Raspberry
Pi latency.

## Metric interpretation

- Typed entity scores require exact boundaries and the correct entity type.
- Binary entity scores ignore type but still require exact boundaries.
- Private-character recall measures whether gold PII characters were covered.
- Leakage is one minus private-character recall.
- Non-private overmasking measures masked characters outside gold PII.
- Hard-negative false-positive rate is the fraction of no-PII cases receiving
  any predicted span.
- OOD fixtures do not provide a token-level gold standard. Historical token
  metrics remain frozen in `baselines/tinybert_iid_baseline.json`.

`FullPipeline_V3` is imported read-only. The older Moonshine package is not the
combined-production reference because the current runtime uses Canary 180M,
immediate Stage 5, and the supplemental PII wrapper.

## Evaluating a candidate model

`adapters.py` is pinned to TinyBERT by design: it loads the tokenizer named in
`live_config.json` and rejects any label map that is not exactly 67 entries.
Model V2 uses a different tokenizer and 63 labels, so it cannot run through that
path — and weakening the production contract to run an experiment would be the
wrong trade.

`model_v2_adapter.py` is the alternative. It reimplements only what must vary —
tokenizer, label map, window size — and imports the rest of the decision logic
from production unchanged: `consolidate_predictions_to_words` for whole-word
promotion and `merge_pii_spans` for overlap resolution. Both are pure and
already model-independent, so results reflect what the deployed pipeline would
emit.

```python
from privacy_eval.model_v2_adapter import build_model_v2_masker

masker = build_model_v2_masker(
    "artifacts/model_v2/onnx_model/model_int8.onnx",
    "artifacts/model_v2/tokenizer",
    "config/model_v2_label_map.json",
    include_structured=False,   # True adds the frozen regex layer
)
```

`run.py` takes `--model-id` so result files are scoped per model. The default
is `tinybert`, so every existing invocation produces byte-identical filenames.

## Dataset-shaped evaluation

`dataset_metrics.py` scores plain annotated records with the same metric
definitions the fixture path uses, which is what makes the six Model V2
evaluation views comparable. It also provides:

- `restrict_to_labels` — for honest cross-ontology comparison. Model V2 removes
  `WEIGHT` and adds `PASSWORD`, so an aggregate over the union of label spaces
  would be meaningless; both sides are restricted to the shared 30 types first.
- `password_analysis` — reports credential-disclosure recall separately from the
  false-positive rate on password-related no-PII rows. A single F1 would hide
  the distinction that actually matters.
- `subset_by` — per-source or per-category breakdowns.

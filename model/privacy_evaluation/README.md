# OnKith privacy-model evaluation

Everything that decides whether a privacy model is good enough, and everything
that measures what the deployed one actually costs. It trains nothing, exports
nothing, and never modifies the live pipeline.

| What | Where |
|---|---|
| Metric definitions | `privacy_eval/metrics.py`, `privacy_eval/dataset_metrics.py` |
| Frozen TinyBERT IID baseline | `baselines/tinybert_iid_baseline.json` |
| Out-of-distribution fixtures | `fixtures/ood_dev.jsonl`, `fixtures/ood_final.jsonl` |
| Frozen TinyBERT OOD results | `results/` |
| 2,000-row CPU deployment benchmark | `privacy_eval/cpu_benchmark.py`, `fixtures/deployment_benchmark/` |
| 44-clip audio deployment benchmark | `privacy_eval/audio_benchmark.py`, `fixtures/audio_benchmark/` |
| Tests | `tests/` |

The two benchmarks are driven from the repository root by
[`Full_Pipeline_CPU_Benchmark.py`](../../Full_Pipeline_CPU_Benchmark.py) and
[`Full_Pipeline_Audio_Benchmark.py`](../../Full_Pipeline_Audio_Benchmark.py), so
the workstation and the Raspberry Pi run one implementation rather than two that
can drift.

## Evaluation layers

The runner reports five deliberately separate stages:

1. `fp32_model` — FP32 ONNX neural predictions with the production windowing,
   BIO repair, and whole-word decoding, but no structured recognizer.
2. `int8_model` — the same model-only path with the deployed INT8 graph.
3. `structured` — the frozen regex recognizer without a neural model.
4. `structured_supplemental` — frozen plus current spoken-number supplements.
5. `production` — whatever `FullPipeline_V6/config/live_config.json` currently
   points at, plus the frozen structured recognizer, the supplemental
   recognizer, and the fail-closed wrapper. Since the Model V2 migration that is
   the DeBERTa-v3-xsmall INT8 stack.

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

`FullPipeline_V6` is the combined-production reference. The masking-parity test
also imports `archive/FullPipeline_V3` read-only so identical text can be checked
against the historical Canary package without loading either recognizer.

### The TinyBERT arms are frozen records, not re-runnable

`adapters.py` builds its `fp32_model` and `int8_model` stages by overriding only
the model artifact inside the production adapter. That worked while
`live_config.json` pointed at TinyBERT. It no longer does, and the adapter
hash-pins its artifact, so those two stages now stop with

```text
RuntimeError: ...model_int8.onnx sha256 9efee1e3... does not match the pinned value
```

That is the intended failure: the production contract refuses an artifact it does
not recognise rather than quietly scoring the wrong model. Every TinyBERT number
quoted in this repository comes from `results/` and
`baselines/tinybert_iid_baseline.json`, written while that path was live, or from
[`../DeBERTa-v3-xsmall/deberta_vs_tinybert_int8_validation.ipynb`](../DeBERTa-v3-xsmall/deberta_vs_tinybert_int8_validation.ipynb),
which loads both INT8 graphs directly and imports no OnKith module at all. Use
that notebook for any new model-to-model comparison.

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


---

## Deployment benchmarks

Model quality and deployment cost are different questions, measured differently.

### CPU / text benchmark — deployment parity

`privacy_eval/cpu_benchmark.py` scores a committed 2,000-row subset of the
held-out test split through the production masking path and records typed entity
F1, character recall and leakage, per-row latency percentiles, throughput, peak
RSS, and an `ood_dev` cross-check against the release number.

The subset is committed because a 4 GB Pi will never hold the multi-gigabyte
corpora. Selection is deterministic (`privacy_eval/deployment_benchmark.py`,
seed 20260817): every zero-PII row, a floor of twelve rows per entity type where
the data allows, then a uniform fill so common labels keep their natural
frequency. Row ids are content-addressed and the file SHA-256 is manifest-pinned.

The point is **not** a quality claim — the held-out text evaluation already
answers that. The point is that a workstation run and a Raspberry Pi run are
comparable measurements of one artifact: both re-hash the INT8 graph against the
deployment record and refuse to report privacy numbers if it does not match, and
both write `prediction_parity_data.jsonl` so per-row agreement can be computed
across devices afterwards.

Cross-device prediction agreement is expected to be high but not perfect. Two
CPUs running different INT8 kernels legitimately disagree on borderline tokens;
the measured figure is 98.35% with a typed-F1 difference of −0.00039.

### Audio benchmark — the whole system, from sound

`privacy_eval/audio_benchmark.py` drives 44 committed WAV clips through the real
deployment: Silero VAD, Whisper STT, transcript normalisation, INT8 masking,
spoken-digit rendering. If the recognizer is not installed it says exactly what is
missing, runs everything that does not depend on it, and exits with code 2. It
never fabricates a transcript.

**The two subsets are treated differently, and that is the methodology.**

| Subset | Clips | Used for | Not used for |
|---|---:|---|---|
| LibriSpeech dev-clean | 12 | word error rate, latency, real-time factor, memory | any privacy metric |
| Synthetic privacy speech | 32 | privacy metrics, plus recognition and cost | — |

The LibriSpeech clips are marked `privacy_annotated: false` and excluded from
every privacy metric. Read prose is full of person names that carry no gold
annotation, so scoring it would charge the model for correct detections. The
synthetic clips carry the annotations, and every value in them is fictional:
RFC 2606 reserved `example` domains, the NANP 555-01xx range reserved for
fiction, and invented names, addresses and document numbers.

### Comparable arms

Privacy after speech recognition cannot be compared to privacy on clean text
unless the arms are defined identically. Three are reported on the same 32 clips:

| Arm | Input to the privacy model |
|---|---|
| `reference_text` | The written reference sentence, cased and punctuated. |
| `reference_uncased` | The same sentence lowercased and stripped of punctuation — what a perfect recognizer would produce, given `pnc: no`. |
| `audio_stt_pipeline` | The real recognizer output. |

Entity recall and character recall use one definition across all three rows: each
gold entity measured over its own characters, whitespace excluded. Typed F1 needs
exact gold offsets, which recognized text does not have, so it is reported for the
control arms only — never for the audio arm.

The `reference_uncased` arm is what makes the audio result interpretable. Going
from cased reference text to uncased reference text *improves* character recall
(0.7202 → 0.8071), so the audio arm's 0.8043 sits close to the ceiling casing
alone sets, and the residual gap is attributable to recognition rather than to
formatting.

### Failure attribution

Every expected entity that was not protected is attributed to a cause rather than
aggregated into one number:

| Outcome | Meaning |
|---|---|
| Protected | Masked end to end. |
| Privacy-model failure | Survived recognition intact, and the model did not mask it. |
| STT-caused, nothing masked | Recognition destroyed the entity; nothing recognisable was left to mask. |
| STT-altered but masked anyway | Recognition changed the surface form and the model masked it regardless. |

Examples of each are written to `audio_failure_analysis.jsonl`, including the
successes whose surface the recognizer changed.

### Post-masking amplification

The benchmark separately reports a failure mode invisible to any per-entity
metric. `app/digit_normalizer.py` runs **after** masking, by design, and renders
spoken numbers as numerals for display. When the privacy model has already missed
a spoken identifier — counted once, as a privacy-model failure — that renderer
can turn the surviving digit words into a compact numeral of identifier length:

```text
released text:  call me back on five five five zero one four seven
displayed as:   call me back on 5550147
```

No additional information is released; the digit words were already in the
released text. But the copyable form is worse, and a metric that counts entities
would never show it. Any identifier of seven or more digits produced this way is
listed in `audio_postprocessing_amplification.csv` with its clip, its label, and
the text on both sides of the transform. The effect reproduced identically on the
workstation and the Raspberry Pi.

## Known test state

`tests/test_model_v2_evaluation.py::NotebookTests::test_notebook_has_no_stored_outputs`
fails, and is expected to. It asserts that
`../DeBERTa-v3-xsmall/model_v2_deberta_v3_xsmall.ipynb` carries no stored cell
output. The committed notebook is the **executed** release run, so 30 of its 31
code cells carry the output of the training and acceptance run. Clearing them
would satisfy the test and delete the human-readable record of that run, so the
failure is left standing rather than papered over. Every other test here passes.

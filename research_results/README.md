# Research results

Two notebooks live at the repository root. Between them they turn the saved
evaluation evidence into report figures and measure what the deployment
actually costs on real hardware.

| Notebook | Run it on | What it does | Safe to Run All |
|---|---|---|---|
| [`Research_Paper_Figures.ipynb`](../Research_Paper_Figures.ipynb) | any machine with the repository | Reads saved metrics and writes publication-quality figures and tables. Trains nothing, evaluates nothing. | Yes |
| [`Full_Pipeline_CPU_Benchmark.ipynb`](../Full_Pipeline_CPU_Benchmark.ipynb) | **any CPU** - workstation *and* the Raspberry Pi | Runs the real production pipeline and the accepted INT8 privacy model, and records latency, memory and quality. | Yes |

Neither notebook trains a model, re-exports ONNX, or touches the frozen OOD
final benchmark.

## The workflow

```text
workstation                          Raspberry Pi
-----------                          ------------
Full_Pipeline_CPU_Benchmark.ipynb    Full_Pipeline_CPU_Benchmark.ipynb
        │                                     │
        └──────────► research_results/cpu_benchmarks/<run_id>/ ◄────┘
                              │
                              ▼
                  Research_Paper_Figures.ipynb
                              │
                              ▼
                figures/  tables/  summaries/
```

1. Run the benchmark notebook here; commit the folder it writes.
2. Faris pulls, opens the same notebook on the Pi, presses **Run All**, and
   commits the folder it writes there.
3. Pull that back and rerun `Research_Paper_Figures.ipynb`. The device
   comparison and the prediction-parity figure appear on their own - no edits.

## Output layout

```text
research_results/
├── figures/          PNG at 300 dpi and PDF, one pair per plot
├── tables/           the numbers behind each figure, as CSV
├── summaries/        figure_index.json and results_overview.md
└── cpu_benchmarks/
    └── <run_id>/     one folder per benchmark run, per device
```

`<run_id>` is `<UTC timestamp>_<os>-<arch>_<mode>[_<label>]`, for example
`20260817T190227Z_windows-amd64_full_workstation` or
`20260818T081500Z_linux-aarch64_full_pi5`. Only characters that are safe on
every filesystem are used.

Each run folder holds `system_info.json`, `benchmark_config.json`,
`artifact_manifest.json`, `deberta_latency.csv` and its summary,
`pipeline_latency.csv` and its summary, `evaluation_metrics.json`,
`per_entity_metrics.csv`, `prediction_parity_data.jsonl`, `memory_metrics.json`
and a human-readable `benchmark_summary.md`.

## Smoke mode

The benchmark notebook ships with `SMOKE_TEST = False`, which scores the whole
2,000-row subset. Set it to `True` in the configuration cell (or export
`ONKITH_BENCHMARK_SMOKE=1`) for a ~50-row check that finishes in a couple of
minutes and exercises every code path: imports, artifact hashes, model loading,
the production decoder, metrics, latency, memory and file writing.

Plan on roughly 5 minutes for a full run on a recent x86 laptop and an hour or
more on a Raspberry Pi 5.

## What is canonical

| Artifact | Location |
|---|---|
| Deployed INT8 ONNX, tokenizer, label map | `FullPipeline_V3/models/deberta_pii/` |
| Deployment manifest with per-file SHA-256 | `FullPipeline_V3/models/deberta_pii/model_manifest.json` |
| Release gate / accepted candidate | `model/DeBERTa-v3-xsmall/artifacts/model_v2/metrics/accepted_int8_candidate.json` |
| Frozen OOD final report (spent once) | `model/DeBERTa-v3-xsmall/artifacts/model_v2/metrics/evaluation_ood_final_int8.json` |
| Benchmark subset the Pi and the laptop both score | `model/privacy_evaluation/fixtures/deployment_benchmark/` |

The benchmark notebook re-hashes the INT8 graph and refuses to produce privacy
numbers unless it matches all three records. That is what lets two runs from two
machines be compared as measurements of the same model.

## The benchmark subset

2,000 rows selected once from the held-out test split and committed, because a
4 GB Pi will never hold the multi-gigabyte corpora. Selection is
`model/privacy_evaluation/privacy_eval/deployment_benchmark.py`: every zero-PII
row, a floor of twelve rows per entity type where the data allows, then a
uniform fill so common labels keep their natural frequency. Seed 20260817,
content-addressed row ids, manifest-pinned SHA-256.

It is a *deployment* check, not a replacement for the full evaluation. The
40,909-row validation and the frozen OOD sets remain the model-quality evidence.

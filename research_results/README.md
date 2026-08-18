# Research results

Three entry points at the repository root. Between them they turn the saved
evaluation evidence into report figures and measure what the deployment actually
costs on real hardware.

| Entry point | Run it on | What it does | Safe to run |
|---|---|---|---|
| [`Full_Pipeline_CPU_Benchmark.py`](../Full_Pipeline_CPU_Benchmark.py) | **any CPU** - workstation *and* the Raspberry Pi | Scores the committed 2,000-row held-out subset through the accepted INT8 privacy model. Records latency, memory, quality and an out-of-distribution parity check. | Yes |
| [`Full_Pipeline_Audio_Benchmark.py`](../Full_Pipeline_Audio_Benchmark.py) | **any CPU** - workstation *and* the Raspberry Pi | Drives committed WAV clips through the whole deployment: VAD, STT, transcript normalisation, masking, spoken-digit rendering. Records word error rate, real-time factor, memory and end-to-end privacy leakage. | Yes |
| [`Research_Paper_Figures.ipynb`](../Research_Paper_Figures.ipynb) | any machine with the repository | Reads saved metrics and writes publication-quality figures and tables. Trains nothing, evaluates nothing. | Yes |

The two benchmarks are plain Python scripts. The Raspberry Pi has no Jupyter,
and one implementation that runs everywhere is worth more than two that can
drift.

Nothing here trains a model, re-exports ONNX, or touches the frozen OOD final
benchmark.

## The workflow

```text
workstation                             Raspberry Pi
-----------                             ------------
python Full_Pipeline_CPU_Benchmark.py   python Full_Pipeline_CPU_Benchmark.py
python Full_Pipeline_Audio_Benchmark.py python Full_Pipeline_Audio_Benchmark.py
        │                                        │
        └────────► research_results/cpu_benchmarks/<run_id>/ ◄───────┘
                              │
                              ▼
                  Research_Paper_Figures.ipynb
                              │
                              ▼
                figures/  tables/  summaries/
```

1. Run both benchmarks on the workstation; commit the folders they write.
2. On the Raspberry Pi, pull, run the same two commands, and commit what they
   write there.
3. Pull that back and rerun `Research_Paper_Figures.ipynb`. The device
   comparison and the prediction-parity figure appear on their own - no edits.

## The command line

```bash
python Full_Pipeline_CPU_Benchmark.py                 # full 2,000-row benchmark
python Full_Pipeline_CPU_Benchmark.py --smoke-test    # ~50 rows, a couple of minutes
python Full_Pipeline_CPU_Benchmark.py --rows 100      # explicit row count
python Full_Pipeline_CPU_Benchmark.py --device-label pi5

python Full_Pipeline_Audio_Benchmark.py               # every committed clip
python Full_Pipeline_Audio_Benchmark.py --smoke-test  # four representative clips
python Full_Pipeline_Audio_Benchmark.py --clip privacy_001
python Full_Pipeline_Audio_Benchmark.py --list-clips
python Full_Pipeline_Audio_Benchmark.py --verify-only
```

`--help` on either lists every option. `ONKITH_BENCHMARK_SMOKE=1` and
`ONKITH_BENCHMARK_DEVICE=pi5` still work for the CPU benchmark.

Plan on roughly 5 minutes for a full CPU run on a recent x86 laptop and an hour
or more on a Raspberry Pi 5. The audio benchmark is dominated by speech
recognition; budget accordingly and use `--smoke-test` first.

## Output layout

```text
research_results/
├── figures/          PNG at 300 dpi and PDF, one pair per plot
├── tables/           the numbers behind each figure, as CSV
├── summaries/        figure_index.json and results_overview.md
└── cpu_benchmarks/
    └── <run_id>/     one folder per benchmark run, per device
```

`<run_id>` is `<UTC timestamp>_<os>-<arch>[_audio]_<mode>[_<label>]`, for example
`20260817T190227Z_windows-amd64_full_workstation` or
`20260818T081500Z_linux-aarch64_audio_full_pi5`. Only characters that are safe on
every filesystem are used.

Both benchmarks write into the same root and are told apart by
`benchmark_config.json`: audio runs carry `"benchmark": "audio_pipeline"`, and a
run without that key is a text run.

| Text run writes | Audio run writes |
|---|---|
| `system_info.json` | `system_info.json` |
| `benchmark_config.json` | `benchmark_config.json` |
| `artifact_manifest.json` | `artifact_manifest.json` |
| `deberta_latency.csv` + `deberta_latency_summary.json` | `audio_clip_results.csv` + `audio_latency_summary.json` |
| `pipeline_latency.csv` + `pipeline_latency_summary.json` | `audio_stt_metrics.json` |
| `evaluation_metrics.json` | `audio_privacy_metrics.json` |
| `per_entity_metrics.csv` | `audio_memory_metrics.json` |
| `prediction_parity_data.jsonl` | `audio_predictions.jsonl` |
| `ood_dev_parity_data.jsonl` | `audio_failure_analysis.jsonl` |
| `memory_metrics.json` | |
| `benchmark_summary.md` | `audio_benchmark_summary.md` |

## The canonical benchmark runs

Four runs, and only four. Each one is a complete, successful run of a committed
entry point against the hash-verified deployment artifact, and every figure and
table in this directory is derived from them.

| Run id | Device | Benchmark |
|---|---|---|
| `20260817T190227Z_windows-amd64_full_workstation` | development workstation, x86-64 | CPU/text, 2,000 rows |
| `20260818T082425Z_linux-aarch64_full` | Raspberry Pi 5 Model B Rev 1.0, 4 GB | CPU/text, 2,000 rows |
| `20260818T062503Z_windows-amd64_audio_full_workstation` | development workstation, x86-64 | audio pipeline, 44 clips |
| `20260818T081310Z_linux-aarch64_audio_full` | Raspberry Pi 5 Model B Rev 1.0, 4 GB | audio pipeline, 44 clips |

Runs that are *not* kept, and why:

- **Smoke runs.** `--smoke-test` scores about 50 rows or four clips. It proves
  the harness works on a machine; it is not a measurement, and leaving one beside
  the full runs invites it to be read as one. Rerun it locally whenever you want
  it - the output just should not be committed.
- **Runs where a stage was unavailable.** An audio run without the speech
  recognizer installed produces `stt_available: false`, no word error rate, no
  real-time factor and no memory profile. One such Raspberry Pi attempt was made
  before the NeMo stack was installed on the device; it was committed by mistake
  and has been removed. The valid Pi audio run, made minutes later on the same
  board with the full stack, is the one listed above.

The figures notebook already refuses to put an incomplete run into the comparison
figures, but the safer rule is that this directory only ever holds evidence.

## What is canonical

| Artifact | Location |
|---|---|
| Deployed INT8 ONNX, tokenizer, label map | `FullPipeline_V3/models/deberta_pii/` |
| Deployment manifest with per-file SHA-256 | `FullPipeline_V3/models/deberta_pii/model_manifest.json` |
| Release gate / accepted candidate | `model/DeBERTa-v3-xsmall/artifacts/model_v2/metrics/accepted_int8_candidate.json` |
| Frozen OOD final report (spent once) | `model/DeBERTa-v3-xsmall/artifacts/model_v2/metrics/evaluation_ood_final_int8.json` |
| Benchmark subset the Pi and the laptop both score | `model/privacy_evaluation/fixtures/deployment_benchmark/` |
| Audio clips and their manifest | `model/privacy_evaluation/fixtures/audio_benchmark/` |
| Benchmark runners | `model/privacy_evaluation/privacy_eval/{cpu,audio}_benchmark.py` |

Both benchmarks re-hash the INT8 graph and refuse to produce privacy numbers
unless it matches the deployment records. That is what lets two runs from two
machines be compared as measurements of the same model.

## The text benchmark subset

2,000 rows selected once from the held-out test split and committed, because a
4 GB Pi will never hold the multi-gigabyte corpora. Selection is
`model/privacy_evaluation/privacy_eval/deployment_benchmark.py`: every zero-PII
row, a floor of twelve rows per entity type where the data allows, then a
uniform fill so common labels keep their natural frequency. Seed 20260817,
content-addressed row ids, manifest-pinned SHA-256.

## The audio benchmark clips

44 clips, about 280 seconds in total, each 16 kHz mono 16-bit PCM with its
SHA-256 pinned in `audio_benchmark.manifest.json`. Two subsets, reported
separately:

* **synthetic privacy speech benchmark** - 32 clips of Piper TTS reading
  sentences written for this project. Every name, number, address, credential
  and document identifier is fictional; e-mail and web addresses use the RFC
  2606 reserved `example` domains and telephone numbers use the NANP 555-01xx
  fictional range. Piper is MIT and its three voices are trained on LJ Speech
  (public domain), LibriTTS-R (CC BY 4.0) and VCTK (CC BY 4.0).
* **LibriSpeech dev-clean subset** - 12 clips copied verbatim from the
  validation split, CC BY 4.0 (Panayotov et al., ICASSP 2015). Real human read
  speech with official transcripts: a word-error-rate reference that owes
  nothing to a vocoder. It is marked `privacy_annotated: false` and is excluded
  from every privacy metric. Read prose is full of person names - Linnell,
  Mason, Polly - that carry no gold annotation, so scoring it would charge the
  model for correct detections. It measures recognition, latency and memory;
  the synthetic subset carries the privacy annotations.

Regenerate them with
`model/privacy_evaluation/scripts/build_audio_benchmark.py`, which needs
`piper-tts`, `scipy`, `soundfile`, `pyarrow` and `huggingface_hub` in a
throwaway environment. **The Raspberry Pi never runs it**: the WAVs are
committed and the Pi only reads them.

## Scope, honestly stated

The text benchmark is a *deployment* check, not a replacement for the full
evaluation. The 40,909-row validation and the frozen OOD sets remain the
model-quality evidence.

The audio benchmark is a **controlled deployment benchmark and a Raspberry Pi
feasibility test** on a few dozen clips. It is not a speech-recognition
benchmark, its privacy half is synthetic and labelled as such everywhere it is
reported, and it replaces none of the held-out text evaluations.

## Speech recognizer

The audio benchmark needs the production recognizer, which is not in the default
install:

```bash
pip install "nemo-toolkit[asr]==2.5.3" "torch==2.7.1"
python FullPipeline_V3/scripts/acquire_canary.py
```

Without it the script verifies the fixtures, runs the VAD and both
reference-text control arms, records the memory profile, prints exactly what is
missing, and exits with code 2. It never fabricates a transcript.

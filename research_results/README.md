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

`<run_id>` is `<UTC timestamp>_<os>-<arch>[_<engine token>]_<mode>[_<label>]`,
for example `20260817T190227Z_windows-amd64_full_workstation`,
`20260818T081310Z_linux-aarch64_audio_full` or
`20260820T144757Z_windows-amd64_audio-whisper_full`. Only characters that are
safe on every filesystem are used.

The engine token is part of the run id so a Whisper run can never land in or be
mistaken for a Canary folder. `audio` on its own is a V3 Canary audio run;
`audio-whisper` and `whisper` are V4. Runs written before V4 carry no engine
token, and its absence is what identifies them rather than a guess.

Both benchmarks write into the same root and are told apart by
`benchmark_config.json`: audio runs carry `"benchmark": "audio_pipeline"`, and a
run without that key is a text run. `pipeline_version` and `stt_engine` say
which pipeline produced it.

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

Each one is a complete, successful run of a committed entry point against the
hash-verified deployment artifact, and every figure and table in this directory
is derived from them. Two pipeline versions are kept side by side, because the
progression from one recognizer to the other is itself a result.

**Pipeline V3 - NVIDIA Canary 180M Flash.** A frozen historical baseline. These
runs are never regenerated: the Canary build is superseded and its recognizer is
not installed anywhere in the current environment.

| Run id | Device | Benchmark |
|---|---|---|
| `20260817T190227Z_windows-amd64_full_workstation` | development workstation, x86-64 | CPU/text, 2,000 rows |
| `20260818T082425Z_linux-aarch64_full` | Raspberry Pi 5 Model B Rev 1.0, 4 GB | CPU/text, 2,000 rows |
| `20260818T062503Z_windows-amd64_audio_full_workstation` | development workstation, x86-64 | audio pipeline, 44 clips |
| `20260818T081310Z_linux-aarch64_audio_full` | Raspberry Pi 5 Model B Rev 1.0, 4 GB | audio pipeline, 44 clips |

**Pipeline V4 - faster-whisper `base.en`.** The current deployment.

| Run id | Device | Benchmark | Privacy path |
|---|---|---|---|
| `20260819T080341Z_windows-amd64_whisper_full_workstation` | development workstation, x86-64 | CPU/text, 2,000 rows | superseded by the run below |
| `20260820T072331Z_linux-aarch64_whisper_full` | Raspberry Pi 5 Model B Rev 1.0, 4 GB | CPU/text, 2,000 rows | before the spoken-address rewrite |
| `20260819T073431Z_windows-amd64_audio-whisper_full_workstation` | development workstation, x86-64 | audio pipeline, 44 clips | superseded by the run below |
| `20260820T071755Z_linux-aarch64_audio-whisper_full` | Raspberry Pi 5 Model B Rev 1.0, 4 GB | audio pipeline, 44 clips | before the spoken-address rewrite |
| `20260820T145035Z_windows-amd64_whisper_full` | development workstation, x86-64 | CPU/text, 2,000 rows | current |
| `20260820T144757Z_windows-amd64_audio-whisper_full` | development workstation, x86-64 | audio pipeline, 44 clips | current |

### Two things changed on 2026-08-20, and the table above says which runs predate them

**The entity scoring definition.** `locate_entity` in the audio benchmark aligns
each gold entity from the reference transcript into the recognized one. It used
to align on raw words and to measure mask coverage over the whitespace-delimited
region, and both broke on Whisper output specifically: a reference reading
"twenty seven" matched nothing in a transcript reading "27", and `Carter.` was
scored at six masked characters out of seven because the sentence full stop was
counted as part of the surname. Canary emitted neither numerals nor punctuation,
so neither defect could occur in a V3 run. The effect was to **overstate
Whisper's privacy failures by roughly a factor of two**. Both are fixed;
`benchmark_config.json` now carries an `entity_scoring` key naming the
definition, and a folder without that key was written under the old one.

Older folders do not have to be re-run to be compared.
`model/privacy_evaluation/privacy_eval/rescore.py` recomputes the whole
entity-level family for any completed folder on the current definition, from the
`privacy_input_text` / `masked_text` pair every run saved. It loads no
recognizer and no model - the predicted spans are exactly recoverable from that
pair - and it self-checks by replaying each folder's own recorded mask coverage.
The figures notebook applies it to all four audio arms, so the engine comparison
is one definition applied to four runs.

**A spoken-address rewrite in the privacy path (V4 only).** Whisper writes
`billing at example.org`, which is neither the spoken form the transcript
records nor the written form the model and the structured EMAIL recognizer
expect, and the result was the domain masked and the local part released.
`FullPipeline_V4/integrated_pipeline/spoken_form_normalizer.py` rewrites two such
shapes for the detectors only and projects every span back before masking, so
the released transcript is unchanged. This one **cannot** be replayed from a
saved folder: it changes what the masker outputs, which needs the model. Runs
marked "before the spoken-address rewrite" therefore hold the V4 privacy path as
it was. Their runtime, memory and word-error figures are unaffected - the
rewrite is a regex pass over already-recognized text.

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
| Deployed INT8 ONNX, tokenizer, label map | `FullPipeline_V4/models/deberta_pii/` |
| Deployment manifest with per-file SHA-256 | `FullPipeline_V4/models/deberta_pii/model_manifest.json` |
| Release gate / accepted candidate | `model/DeBERTa-v3-xsmall/artifacts/model_v2/metrics/accepted_int8_candidate.json` |
| Frozen OOD final report (spent once) | `model/DeBERTa-v3-xsmall/artifacts/model_v2/metrics/evaluation_ood_final_int8.json` |
| Benchmark subset the Pi and the laptop both score | `model/privacy_evaluation/fixtures/deployment_benchmark/` |
| Audio clips and their manifest | `model/privacy_evaluation/fixtures/audio_benchmark/` |
| Benchmark runners | `model/privacy_evaluation/privacy_eval/{cpu,audio}_benchmark.py` |
| Engine-comparable recognition metrics from saved folders | `model/privacy_evaluation/privacy_eval/stt_comparison.py` |
| Entity-level rescoring of saved folders | `model/privacy_evaluation/privacy_eval/rescore.py` |

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
pip install --requirement FullPipeline_V4/requirements-rpi.txt   # or -windows.txt
```

That brings faster-whisper 1.2.1 on CTranslate2. The `base.en` weights are
fetched once on first use and cached under `~/.cache/huggingface`; nothing is
committed for them, which is why the deployment config pins the package version,
the model name and the decoding settings rather than a file hash.

Without it the script verifies the fixtures, runs the VAD and both
reference-text control arms, records the memory profile, prints exactly what is
missing, and exits with code 2. It never fabricates a transcript.

V3's recognizer, NVIDIA Canary 180M Flash, needed `nemo-toolkit[asr]==2.5.3`,
`torch==2.7.1` and a 700 MB `.nemo` artifact fetched by
`FullPipeline_V3/scripts/acquire_canary.py`. Reproducing the frozen V3 runs
would need that stack reinstalled. Nothing in this directory requires it: the V3
numbers are read from the folders those runs wrote.

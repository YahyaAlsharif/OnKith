# Research results

Four entry points at the repository root. Between them they turn the saved
evaluation evidence into report figures and measure what the deployment actually
costs on real hardware.

All three benchmark scripts measure the pipeline named by
`privacy_eval.benchmark_common.PRODUCTION_ROOT`, which is `FullPipeline_V6`.

## Two results roots, and why

| Folder | Holds | Written by |
|---|---|---|
| [`v6_benchmarks/`](v6_benchmarks/) | Every run of the **current system**: audio, CPU/text and the assistant round trip, on the development workstation and on a Raspberry Pi 5. | the three benchmark scripts today |
| [`cpu_benchmarks/`](cpu_benchmarks/) | The frozen V3 Canary, V4 Whisper and V5 Parakeet runs, workstation and Raspberry Pi. Historical provenance. | nothing any more; read-only |

They are separate because V6 runs the same recognizer V4 did, so a V6 run is
honestly tagged `"stt_engine": "whisper"` and would be indistinguishable from a
V4 folder by engine alone. Keeping the roots apart means a V6 run can never be
written into, or read as, a frozen historical one — and the frozen folders are
never regenerated, so their contents cannot drift.

Run ids carry both the version and the recognizer — `v6_whisper`,
`audio_v6_whisper`, `e2e_v6_whisper` for V6; `parakeet`, `whisper`, or nothing
at all for the frozen Canary runs — so a folder is never ambiguous about what
produced it. Every folder also records `pipeline_version` in its
`benchmark_config.json`, and that key, not the directory name, is what
[`Research_Paper_Figures.ipynb`](../Research_Paper_Figures.ipynb) reads.

| Entry point | Run it on | What it does | Safe to run |
|---|---|---|---|
| [`Full_Pipeline_CPU_Benchmark.py`](../Full_Pipeline_CPU_Benchmark.py) | **any CPU** - workstation *and* the Raspberry Pi | Scores the committed 2,000-row held-out subset through the accepted INT8 privacy model. Records latency, memory, quality and an out-of-distribution parity check. | Yes |
| [`Full_Pipeline_Audio_Benchmark.py`](../Full_Pipeline_Audio_Benchmark.py) | **any CPU** - workstation *and* the Raspberry Pi | Drives committed WAV clips through the whole deployment: VAD, STT, transcript normalisation, masking, spoken-digit rendering. Records word error rate, real-time factor, memory and end-to-end privacy leakage. | Yes |
| [`Full_Pipeline_V6_EndToEnd_Benchmark.py`](../Full_Pipeline_V6_EndToEnd_Benchmark.py) | **any CPU with a network connection** | Drives 6 committed clips through the whole V6 demonstration path: Whisper, masking, the outbound LLM request, local rehydration, local TTS. Audits the bytes handed to `urllib` for private values and writes every payload out verbatim. Six live calls by default because the Groq free tier is token-limited; `--extended` makes it ten. | Yes, and it spends API quota |
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
python Full_Pipeline_V6_EndToEnd_...py  python Full_Pipeline_V6_EndToEnd_...py
        │                                        │
        └────────► research_results/v6_benchmarks/<run_id>/ ◄────────┘
                              │
                add the run id to CANONICAL_RUNS
                in Research_Paper_Figures.ipynb, section 1
                              │
                              ▼
                  Research_Paper_Figures.ipynb   (Run All)
                              │
                              ▼
                figures/  tables/  summaries/
```

A new run folder that is **not** added to `CANONICAL_RUNS` is listed by the
notebook as discovered-but-not-canonical and enters no figure. That is
deliberate: no run is ever picked up because it happens to be the newest.

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

python Full_Pipeline_V6_EndToEnd_Benchmark.py               # 6 cases, live LLM
python Full_Pipeline_V6_EndToEnd_Benchmark.py --extended    # 10 cases, live LLM
python Full_Pipeline_V6_EndToEnd_Benchmark.py --smoke-test  # 3 cases
python Full_Pipeline_V6_EndToEnd_Benchmark.py --llm-stub    # no key needed
python Full_Pipeline_V6_EndToEnd_Benchmark.py --offline     # payload audit only
python Full_Pipeline_V6_EndToEnd_Benchmark.py --no-tts
python Full_Pipeline_V6_EndToEnd_Benchmark.py --list-cases
```

The end-to-end benchmark reads `GROQ_API_KEY` from the environment first, then
from the **repository-root `.env`**, which is git-ignored and must never be
committed. Without a key it still runs everything local, still builds and audits
the outbound payload, and reports the API sections as `not run` with exit code
2. `--llm-stub` answers each built request locally so the whole integration can
be exercised without a key; a stubbed run is labelled as such in every artifact
it writes and is never a measurement of Groq.

Use the stub for repeated development runs. The live default is six requests,
which is small, but the free tier's binding limit is tokens per minute rather
than requests per minute: the conversation is reset between cases and requests
are spaced from the `x-ratelimit-*` response headers for that reason. On a 429
the benchmark honours `Retry-After` once, caps the wait at 60 seconds, records
the case as rate-limited and finishes.

No artifact written here ever contains an API key. Saved request bodies are the
masked payloads only; the key travels as a header and is recorded by name.

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
├── summaries/        figure_index.json (run provenance) and results_overview.md
├── v6_benchmarks/
│   └── <run_id>/     the current system, one folder per run per device
└── cpu_benchmarks/
    └── <run_id>/     frozen V3/V4/V5 runs, read-only
```

`<run_id>` is `<UTC timestamp>_<os>-<arch>[_<engine token>]_<mode>[_<label>]`,
for example `20260817T190227Z_windows-amd64_full_workstation`,
`20260818T081310Z_linux-aarch64_audio_full` or
`20260820T144757Z_windows-amd64_audio-whisper_full`. Only characters that are
safe on every filesystem are used.

The engine token is part of the run id so a Whisper run can never land in or be
mistaken for a Canary folder. `audio` on its own is a V3 Canary audio run;
`audio-whisper` and `whisper` are V4; `audio-parakeet`, `parakeet` and
`e2e-parakeet` are V5; `audio-v6-whisper`, `v6-whisper` and `e2e-v6-whisper` are
V6 and live in `v6_benchmarks/`. Runs written before V4 carry no engine token,
and its absence is what identifies them rather than a guess.

**There are no V5 Raspberry Pi results, and none are missing.** V5 was not taken
through Pi evaluation once workstation testing showed its Parakeet core was not
the preferred final path; its dependency tree has no aarch64 wheels. That is the
only device gap in this directory.

**V6 has Raspberry Pi results of its own** — all three benchmarks, listed below.
Nothing here infers Pi performance from a workstation run, and nothing infers V6
Pi performance from a V4 run any more.

The three benchmarks are told apart by `benchmark_config.json`: audio runs carry
`"benchmark": "audio_pipeline"`, round-trip runs carry `"v5_end_to_end"` or
`"v6_end_to_end"`, and a run without that key is a text run. `pipeline_version`
and `stt_engine` say which pipeline produced it.

| Text run writes | Audio run writes | Round-trip run writes |
|---|---|---|
| `system_info.json` | `system_info.json` | `system_info.json` |
| `benchmark_config.json` | `benchmark_config.json` | `benchmark_config.json` |
| `artifact_manifest.json` | `artifact_manifest.json` | `artifact_manifest.json` |
| `deberta_latency.csv` + `deberta_latency_summary.json` | `audio_clip_results.csv` + `audio_latency_summary.json` | `e2e_summary.json` |
| `pipeline_latency.csv` + `pipeline_latency_summary.json` | `audio_stt_metrics.json` | `e2e_case_results.csv` + `e2e_cases.jsonl` |
| `evaluation_metrics.json` | `audio_privacy_metrics.json` | `e2e_outbound_payloads.jsonl` |
| `per_entity_metrics.csv` | `audio_memory_metrics.json` | `e2e_placeholder_probes.jsonl` |
| `prediction_parity_data.jsonl` | `audio_predictions.jsonl` | `e2e_local_command_check.json` |
| `ood_dev_parity_data.jsonl` | `audio_failure_analysis.jsonl` | `e2e_memory_metrics.json` |
| `memory_metrics.json` | | |
| `benchmark_summary.md` | `audio_benchmark_summary.md` | `e2e_benchmark_summary.md` |

`e2e_outbound_payloads.jsonl` records the URL, the **header names** and the body
of every request the LLM client built. Header *values* are deliberately not
captured, so an API key cannot reach a committed artifact.

## The canonical benchmark runs

One run backs each current public claim. The table below is the authority, and
it is mirrored exactly by `CANONICAL_RUNS` in
[`Research_Paper_Figures.ipynb`](../Research_Paper_Figures.ipynb) section 1 and
by `canonical_runs` in
[`summaries/figure_index.json`](summaries/figure_index.json). Nothing anywhere
selects a run by recency.

### The current system — `FullPipeline_V6`

| Benchmark | Device | Run id |
|---|---|---|
| audio pipeline, 44 clips | workstation, x86-64 | `v6_benchmarks/20260823T142306Z_windows-amd64_audio-v6-whisper_full_workstation` |
| audio pipeline, 44 clips | **Raspberry Pi 5, 4 GB** | `v6_benchmarks/20260823T204339Z_linux-aarch64_audio-v6-whisper_full` |
| CPU/text privacy, 2,000 rows | workstation, x86-64 | `v6_benchmarks/20260823T142451Z_windows-amd64_v6-whisper_full_workstation` |
| CPU/text privacy, 2,000 rows | **Raspberry Pi 5, 4 GB** | `v6_benchmarks/20260823T203840Z_linux-aarch64_v6-whisper_full` |
| assistant round trip, 6 clips | workstation, x86-64 | `v6_benchmarks/20260823T142736Z_windows-amd64_e2e-v6-whisper_full_workstation` |
| assistant round trip, 6 clips | **Raspberry Pi 5, 4 GB** | `v6_benchmarks/20260823T204627Z_linux-aarch64_e2e-v6-whisper_full` |

**Scope of the Raspberry Pi round trip.** That run had no `GROQ_API_KEY`, so it
is an outbound-payload audit: the real client built each request and every byte
was audited, but nothing was sent. Its local measurements — recognition,
masking, the placeholder map, rehydration probes, the local-command route — are
real. Its network latency, placeholder-preservation and text-to-speech columns
are **absent, not zero**. The workstation run is the live one.

**Tested Raspberry Pi environment**, taken from the run manifests rather than
from memory: Raspberry Pi 5 Model B Rev 1.0, 4 GB class (3.95 GiB visible),
64-bit Raspberry Pi OS, Linux `6.18.39+rpt-rpi-2712` aarch64, glibc 2.41,
4 cores, Python 3.11.9, ONNX Runtime 1.29.0, NumPy 2.4.6, faster-whisper on
CTranslate2 1.2.1 (CPU, INT8, beam 5), privacy model
`bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8`.

**Tested workstation environment**: Windows 11 (`Windows-10-10.0.26200-SP0`),
x86-64, 14 physical / 20 logical cores, 15.65 GiB RAM, Python 3.11.9, ONNX
Runtime 1.27.0. The privacy artifact SHA-256 is identical on both.

### History, kept as provenance

These are never regenerated. Where more than one folder exists for the same
device and benchmark, the canonical one is marked; the rest stay on disk as
evidence that the run happened and are excluded from every figure.

**V3 — NVIDIA Canary 180M Flash.** A frozen historical baseline; the recognizer
is not installed anywhere in the current environment.

| Benchmark | Device | Run id | Canonical |
|---|---|---|---|
| CPU/text, 2,000 rows | workstation | `cpu_benchmarks/20260817T190227Z_windows-amd64_full_workstation` | yes |
| CPU/text, 2,000 rows | Raspberry Pi 5 | `cpu_benchmarks/20260818T082425Z_linux-aarch64_full` | yes |
| audio, 44 clips | workstation | `cpu_benchmarks/20260818T062503Z_windows-amd64_audio_full_workstation` | yes |
| audio, 44 clips | Raspberry Pi 5 | `cpu_benchmarks/20260818T081310Z_linux-aarch64_audio_full` | yes |

**V4 — faster-whisper `base.en`.** The system the written academic report
evaluates, and the system V6 inherits its local core from.

| Benchmark | Device | Run id | Canonical | Note |
|---|---|---|---|---|
| CPU/text, 2,000 rows | workstation | `cpu_benchmarks/20260819T080341Z_windows-amd64_whisper_full_workstation` | no | superseded |
| CPU/text, 2,000 rows | workstation | `cpu_benchmarks/20260820T145843Z_windows-amd64_whisper_full_workstation` | **yes** | complete |
| CPU/text, 2,000 rows | Raspberry Pi 5 | `cpu_benchmarks/20260820T072331Z_linux-aarch64_whisper_full` | no | superseded |
| CPU/text, 2,000 rows | Raspberry Pi 5 | `cpu_benchmarks/20260821T201941Z_linux-aarch64_whisper_full` | no | superseded |
| CPU/text, 2,000 rows | Raspberry Pi 5 | `cpu_benchmarks/20260821T205958Z_linux-aarch64_whisper_full` | no | superseded |
| CPU/text, 2,000 rows | Raspberry Pi 5 | `cpu_benchmarks/20260822T122249Z_linux-aarch64_whisper_full` | **yes** | latest; peak RSS not recorded |
| audio, 44 clips | workstation | `cpu_benchmarks/20260819T073431Z_windows-amd64_audio-whisper_full_workstation` | no | superseded |
| audio, 44 clips | workstation | `cpu_benchmarks/20260820T145659Z_windows-amd64_audio-whisper_full_workstation` | **yes** | complete |
| audio, 44 clips | Raspberry Pi 5 | `cpu_benchmarks/20260820T071755Z_linux-aarch64_audio-whisper_full` | no | complete runtime/memory, but **before** the spoken-address rewrite |
| audio, 44 clips | Raspberry Pi 5 | `cpu_benchmarks/20260821T201401Z_linux-aarch64_audio-whisper_full` | no | recognizer unavailable; not comparison evidence |
| audio, 44 clips | Raspberry Pi 5 | `cpu_benchmarks/20260821T201653Z_linux-aarch64_audio-whisper_full` | **yes** | current privacy path; VAD and memory unavailable |

The V4 Pi audio choice needs stating, because two folders are defensible and
they disagree. `20260820T071755Z` has the complete runtime and memory profile
but ran **before** the spoken-address rewrite, so its end-to-end privacy figures
(0.7423 character recall, 4 of 25 clips clean) measure a privacy path that no
longer exists and are not comparable with V6's. `20260821T201653Z` ran the same
privacy path V6 runs and reproduces V6's privacy figures exactly, but its
environment lacked `psutil`, so it records no peak RSS. The later folder is
canonical because the comparison V6 needs is a privacy comparison; its memory
column is left blank rather than filled from the earlier run.

**V5 — NVIDIA Parakeet 110M TDT.** The recognizer experiment whose assistant
layer was kept and whose recognizer was not. Workstation only, by decision: its
dependency tree has no aarch64 wheels.

| Benchmark | Device | Run id | Canonical |
|---|---|---|---|
| audio, 44 clips | workstation | `cpu_benchmarks/20260823T095553Z_windows-amd64_audio-parakeet_full` | yes |
| CPU/text, 2,000 rows | workstation | `cpu_benchmarks/20260823T095708Z_windows-amd64_parakeet_full` | yes |
| assistant round trip, 6 clips | workstation | `cpu_benchmarks/20260823T095954Z_windows-amd64_e2e-parakeet_full` | yes |

### Tracing a headline number back to raw data

Every published number is reachable in four steps, with no guessing at any of
them:

```text
README headline
   -> research_results/tables/<name>.csv          the numbers behind the figure
   -> summaries/figure_index.json                 figure_provenance -> run ids
   -> <root>/<run_id>/                            the committed run folder
   -> *.json / *.csv / *.jsonl                    raw benchmark output
```

Worked example — "mean full-pipeline RTF 0.634 on the Raspberry Pi 5":

1. [`tables/current_system_device_cost.csv`](tables/current_system_device_cost.csv),
   column `mean_pipeline_rtf`, row `Raspberry Pi 5`;
2. that row names run
   `20260823T204339Z_linux-aarch64_audio-v6-whisper_full`;
3. in that folder, `audio_latency_summary.json` →
   `real_time_factor.pipeline.mean_ms`.

The field is named `mean_ms` because the statistics helper names every field
that way; the value is a **ratio**, not milliseconds. The same applies to every
key under `real_time_factor`.

### Provenance expectations

- A number in a README, a figure and a table must come from the same run. If a
  figure and a README disagree, the figure is regenerated or the README is
  wrong — never both left standing.
- "Generated from the latest V6 benchmark" is not provenance. A run id is.
- Committed run folders are **read-only evidence**. They are never edited to
  agree with prose. Where an artifact disagrees with prose, the artifact wins
  and the prose changes.
- A superseded run stays on disk. Deleting it would remove the record that the
  measurement happened.

### Known artifact quirks

Two things in the committed folders are worth knowing before reading them
closely. Neither is corrected in place, because a run folder is a record of what
that run produced:

- The `real_time_factor` block reuses the shared statistics helper, so its
  fields are named `mean_ms`, `p90_ms` and so on. The values are dimensionless
  ratios.
- In the round-trip runs, `e2e_benchmark_summary.md` prints a latency table of
  `n/a` and a "peak during run" figure sampled by a different counter from the
  one `e2e_memory_metrics.json` records. The JSON files are the primary
  artifacts; `e2e_summary.json` carries the full local, network and TTS latency
  distributions, and `e2e_memory_metrics.json` carries
  `process_peak_rss_bytes`. The notebook and this repository's documentation
  read the JSON.

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
`FullPipeline_V6/integrated_pipeline/spoken_form_normalizer.py` rewrites two such
shapes for the detectors only and projects every span back before masking, so
the released transcript is unchanged. This one **cannot** be replayed from a
saved folder: it changes what the masker outputs, which needs the model. Runs
marked "before the spoken-address rewrite" therefore hold the V4 privacy path as
it was. Their runtime, memory and word-error figures are unaffected - the
rewrite is a regex pass over already-recognized text.

Runs that are *not* committed, and why:

- **Smoke runs.** `--smoke-test` scores about 50 rows or four clips. It proves
  the harness works on a machine; it is not a measurement, and leaving one beside
  the full runs invites it to be read as one. Rerun it locally whenever you want
  it — the output just should not be committed.
- **Runs where a stage was unavailable** are committed when they happened, but
  are never canonical. An audio run without the speech recognizer installed
  produces `stt_available: false`, no word error rate and no end-to-end STT
  result; `20260821T201401Z` records exactly that failure. The figures notebook
  excludes such runs from every deployment comparison.

## Regenerating the figures and tables

```bash
python -m jupyter nbconvert --to notebook --execute --inplace \
  Research_Paper_Figures.ipynb
```

Run it from the repository root, with an interpreter that has `matplotlib`,
`numpy` and `nbclient`. It trains nothing, evaluates nothing and downloads
nothing: it reads saved metrics and draws them. It writes `figures/`, `tables/`,
`summaries/figure_index.json` and `summaries/results_overview.md`, overwriting
whatever was there.

What it produces:

| Output | Contents |
|---|---|
| `figures/<name>.png` and `.pdf` | 300 dpi raster and vector, same plot |
| `tables/<name>.csv` | the numbers behind the figure of the same name, plus wider per-run tables |
| `summaries/figure_index.json` | `canonical_runs`, `figure_provenance` (figure name → run ids), the full run inventory, and every input that was missing |
| `summaries/results_overview.md` | the written digest, with the canonical run table at the end |

Adding a new run means: run the benchmark, commit the folder, add its run id to
`CANONICAL_RUNS` in section 1 of the notebook, and press Run All. Skipping the
middle step is not a failure mode — the notebook reports the folder as
discovered-but-not-canonical and draws nothing from it.

## What is canonical

| Artifact | Location |
|---|---|
| Deployed INT8 ONNX, tokenizer, label map | `FullPipeline_V6/models/deberta_pii/` |
| Deployment manifest with per-file SHA-256 | `FullPipeline_V6/models/deberta_pii/model_manifest.json` |
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
pip install --requirement FullPipeline_V6/requirements-rpi.txt   # or -windows.txt
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
`archive/FullPipeline_V3/scripts/acquire_canary.py`. Reproducing the frozen V3 runs
would need that stack reinstalled. Nothing in this directory requires it: the V3
numbers are read from the folders those runs wrote.

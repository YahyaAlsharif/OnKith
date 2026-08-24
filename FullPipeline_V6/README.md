# OnKith production pipeline (FullPipeline_V6)

> **This package is FullPipeline_V6: V4's local core plus V5's assistant layer.**
>
> Everything from the microphone to the masked transcript is
> [`FullPipeline_V4`](../archive/FullPipeline_V4/)'s code, byte for byte -
> Silero VAD, the segmenter and its exact sample counts, **faster-whisper
> `base.en`** on CTranslate2 (CPU, INT8, beam 5, with a decoding hint built from
> the device's configured names), transcript merging, wake-word repair, the
> inactivity buffer, DeBERTa-v3-xsmall INT8 masking, span merging, the
> spoken-form normalizer, spoken-digit rendering, SQLite storage and the
> terminal UI. Nothing was retrained and nothing in the privacy path was
> rewritten.
>
> On top of that, V6 adds the assistant layer built and proven in
> [`FullPipeline_V5`](../archive/FullPipeline_V5/): masked-only requests to
> Groq, a placeholder map that never leaves this device, local rehydration of
> the reply, local text to speech, a localhost presentation surface, wake-word
> routing, and a degraded mode that keeps everything local working when no API
> key is configured. V5's Parakeet recognizer was **not** carried over; see
> [`../archive/README.md`](../archive/README.md) for why.
>
> Two tests hold that claim up rather than leaving it as prose.
> [`tests/test_v6_independence.py`](tests/test_v6_independence.py) compares every
> inherited core file against the archived V4 and fails on any drift, and
> `model/privacy_evaluation/tests/test_masking_parity.py` runs fixed text
> through V3, V4 and V6 in three isolated interpreters and requires identical
> masked output, spans, released text and wake-word repair from all three.
>
> **V6 has been benchmarked directly on both devices.** All six runs are in
> [`../research_results/v6_benchmarks/`](../research_results/v6_benchmarks/),
> kept apart from the frozen V3/V4/V5 folders on purpose: audio pipeline, CPU/text
> privacy and the assistant round trip, on the development workstation and on a
> Raspberry Pi 5. Nothing about V6's deployment behaviour is inferred from an
> earlier pipeline any more. The canonical run ids are listed in
> [`../research_results/README.md`](../research_results/README.md).
>
> The parity tests above are still worth having, and they now say something
> checkable: on the same device and the same fixtures, every V6 quality metric is
> identical to V4's to machine precision, so adding the assistant lane changed
> nothing in the local recognition and masking core. The evidence is
> [`../research_results/tables/current_system_inherits_previous_core.csv`](../research_results/tables/current_system_inherits_previous_core.csv).

## The privacy boundary

```
microphone -> Silero VAD -> faster-whisper base.en -> DeBERTa + structured PII
           -> typed placeholders -> [ local placeholder map, never sent ]
           -> MASKED TEXT ONLY -> Groq -> masked reply
           -> local rehydration -> restored answer -> local TTS + screen
```

The external model is given placeholders and is never told what they stood for.
`[GIVENNAME_1] = Fahad` exists only in this process. Placeholders are numbered
per **conversation** rather than per utterance, so two different people one turn
apart cannot both be `[GIVENNAME_1]` and a reply about the first cannot be
restored with the second one's name -
[`app/rehydrate.py`](app/rehydrate.py) explains the allocator and
[`tests/test_assistant_integration.py`](tests/test_assistant_integration.py)
pins the behaviour.

Configuration comes from the process environment first, then the repository-root
`.env`, then safe defaults. See [`.env.example`](.env.example). The root `.env`
is git-ignored and must never be committed.

**Local speech-to-text and privacy processing for Raspberry Pi 5**

This is the **current** OnKith deployment package and the only maintained one.
Earlier integrated pipelines live under [`../archive/`](../archive/) and should
not be used. The Python entry point is still `app.omit_ai_live` and some code
still says "Omit AI" — that is the project's earlier name, kept so the module
paths in existing instructions keep working.

It is a local English speech-to-text pipeline built for the **Raspberry Pi 5**. It uses **faster-whisper `base.en`** on CTranslate2 for transcription, **Silero VAD** for speech detection, overlap-based segmentation for continuous speech, PII masking, and local SQLite storage.

Speech recognition and privacy processing run locally. Internet access is only required during setup to download the model and for optional YouTube playback through Onkith.

## Running it

Full setup is under [Installation](#installation) (Raspberry Pi) below. The
short version, from a clone of the repository:

```bash
cd FullPipeline_V6
python -m pip install --requirement requirements-rpi.txt      # or requirements-windows.txt

python -m app.omit_ai_live --print-config      # what the pipeline will do
python -m app.omit_ai_live --self-test         # no microphone needed
python -m app.omit_ai_live --text-test "My name is Rayyan Khan, rayyan@example.com"
python -m app.omit_ai_live --list-devices
python -m app.omit_ai_live --device 2          # live, with the assistant UI
```

On Raspberry Pi OS, `./run_live.sh` wraps the same entry point with the
installed virtual environment; every flag below works with either.

| Flag | Effect |
|---|---|
| `--headless` | keeps the assistant and local TTS, opens no browser window |
| `--no-assistant` | disables the Groq/TTS/UI add-on; local core only |
| `--assistant-port N` | localhost port for the assistant UI (default 8765) |
| `--once` | process a single utterance and exit |
| `--show-raw` | also print the unmasked transcript — local debugging only |
| `--json-output` / `--quiet` / `--no-colour` | output shaping |

`python -m app.omit_ai_live --help` is the authoritative list.

**What runs locally, and what does not.** Microphone capture, Silero VAD,
faster-whisper `base.en`, DeBERTa-v3-xsmall INT8 masking, the structured
recognizers, span merging, the conversation-scoped placeholder map, SQLite
storage, rehydration, text to speech and the localhost UI all run on the
device. The only thing that leaves it is a **post-mask** request to the Groq
API, and only when a question clears the privacy gate. Optional YouTube search
for the music route is the one other outbound call.

**Configuration and secrets.** Settings resolve as process environment →
`FullPipeline_V6/.env` → repository-root `.env` → defaults. Copy
[`.env.example`](.env.example) to a `.env` at the repository root and edit it
there; both `.env` paths are git-ignored and neither is packaged into a release
ZIP. `.env.example` contains placeholders only.

**Without a `GROQ_API_KEY`** the pipeline still starts and everything local
still works: recognition, masking, storage, local device commands, the terminal
UI, the localhost UI and local TTS. A question clears the privacy gate, the
screen shows the masked text that *would* have been sent, and no network request
is made. That degraded mode is supported, not a failure. When a key is present,
requests go to `GROQ_MODEL` and fall back to `GROQ_FALLBACK_MODEL` on failure.

**Text to speech** picks the first installed backend in the order kokoro, piper,
sapi (Windows), espeak-ng, say; `install_pi.sh` installs espeak-ng, so speech
works on a Pi with no extra model. Set `ONKITH_TTS` to pin one or to `none` to
run silent. TTS speaks the **locally restored** answer, never the masked one,
and a synthesis failure never breaks the pipeline.

**Model acquisition.** The DeBERTa INT8 privacy model and the Silero VAD are
committed here — they are this project's own release artifacts and a Pi cannot
rebuild them. The Whisper `base.en` weights are fetched once by CTranslate2 on
first use and cached under `~/.cache/huggingface`; after that the device runs
offline. Optional larger assets (a Piper voice, a Kokoro graph) are fetched by
[`scripts/download_large_models.py`](scripts/download_large_models.py).

More on the assistant layer: [`ASSISTANT_ADDON.md`](ASSISTANT_ADDON.md).
What is masked and why: [`MASK_POLICY.md`](MASK_POLICY.md).
Open problems: [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

### Benchmarks

Run from the **repository root**, not from this directory:

```bash
python Full_Pipeline_CPU_Benchmark.py           # 2,000 committed rows
python Full_Pipeline_Audio_Benchmark.py         # 44 committed clips, audio in
python Full_Pipeline_V6_EndToEnd_Benchmark.py   # 6 clips through the round trip
```

All three take `--smoke-test` and `--help`. All three measure this package, and
write to `research_results/v6_benchmarks/<run_id>/`. The round-trip benchmark
takes `--offline` to audit the outbound payload without sending it, and
`--llm-stub` to exercise the whole integration without a key.

---

## Measured on a Raspberry Pi 5 (4 GB)

This package has been benchmarked directly on the device, not inferred from an
earlier pipeline. Every number below comes from a committed run folder, named
beside it.

**Tested environment.** Raspberry Pi 5 Model B Rev 1.0, 4 GB class (3.95 GiB
visible), 64-bit Raspberry Pi OS, Linux `6.18.39+rpt-rpi-2712` aarch64,
glibc 2.41, 4 cores, Python 3.11.9, ONNX Runtime 1.29.0, faster-whisper on
CTranslate2 1.2.1 (CPU, INT8, beam 5), privacy model
`bc8d1d7a…048106a8` — the same INT8 artifact the workstation ran.

### Cost, workstation against the Pi

Audio pipeline benchmark, 44 committed clips, 279.7 s of audio.
Runs `20260823T142306Z_windows-amd64_audio-v6-whisper_full_workstation` and
`20260823T204339Z_linux-aarch64_audio-v6-whisper_full`.

| Measurement | Workstation | **Raspberry Pi 5** |
|---|---:|---:|
| Mean STT real-time factor | 0.371 | **0.617** |
| Mean full-pipeline real-time factor | 0.394 | **0.634** |
| P90 / P95 full-pipeline RTF | 0.729 / 0.764 | **1.109 / 1.139** |
| Mean end-to-end latency per clip | 1.76 s | **3.01 s** |
| P50 / P95 end-to-end latency | 1.54 s / 2.74 s | **2.62 s / 5.39 s** |
| Privacy masking, mean per clip | 22.0 ms | **24.9 ms** |
| STT share of processing time | 92.5 % | **96.8 %** |
| Peak RSS, full audio pipeline | 1,272 MiB of 15.6 GiB | **1,571 MiB of 3.95 GiB** |

Resident memory on the Pi climbs 57 MiB at baseline → 90 MiB with the VAD
loaded → 837 MiB once Whisper is loaded → 1,464 MiB with the privacy model on
top; peak during the run was 1,571 MiB, about 39 % of the board's RAM.

### Quality, workstation against the Pi

The same fixtures through the same artifact. These should not move with the
device, and they barely do.

| Measurement | Workstation | **Raspberry Pi 5** | Difference |
|---|---:|---:|---:|
| Aggregate WER, 44 clips | 0.1795 | **0.1841** | +0.0046 |
| Aggregate WER, number-normalised | 0.0854 | **0.0903** | +0.0050 |
| Private character recall after STT | 0.8314 | **0.8300** | −0.0013 |
| Private character leakage after STT | 0.1686 | **0.1700** | +0.0013 |
| Clips fully sanitized | 16 of 25 | **16 of 25** | 0 |
| Hard-negative false-positive clips | 1 of 7 | **1 of 7** | 0 |

CPU/text privacy benchmark, 2,000 committed rows. Runs
`20260823T142451Z_windows-amd64_v6-whisper_full_workstation` and
`20260823T203840Z_linux-aarch64_v6-whisper_full`.

| Measurement | Workstation | **Raspberry Pi 5** |
|---|---:|---:|
| Typed entity F1 | 0.9452 | **0.9448** |
| Private character recall | 0.9950 | **0.9949** |
| Private character leakage | 0.0050 | **0.0051** |
| Mean masking latency | 46.6 ms | **91.6 ms** |
| P50 / P95 / P99 masking latency | 36.5 / 97.2 / 166.0 ms | **62.4 / 233.0 / 475.1 ms** |
| Throughput | 21.40 rows/s | **10.91 rows/s** |
| Peak RSS, text benchmark | 1,671 MiB | **1,928 MiB of 3.95 GiB** |

Predicted spans agree on **98.35 %** of the 2,000 rows (1,967 identical) and the
released masked text on 98.55 %. Two CPUs running different INT8 kernels
legitimately disagree on borderline tokens; the disagreement is small enough to
leave the quality metrics unchanged to three decimal places.

### What this says

* **It fits.** The complete stack peaks at 1,571 MiB during the audio benchmark
  and 1,928 MiB during the 2,000-row text benchmark, both inside a 4 GB board.
* **It runs faster than real time on average, and not on every clip.** Mean
  full-pipeline RTF is 0.634, but the P90 is 1.109. The clips that exceed real
  time are the **shortest** ones: 7 of 44 clips, all under 2.3 s. Whisper decodes
  a fixed window whatever the clip length, so the per-clip cost has a floor, and
  divided by 1.8 s of audio that floor is an RTF above 1. Every clip of 3 s or
  more stayed under real time, the worst at 0.68×. Turn-based interaction is
  comfortable; continuous streaming under load is not characterised.
* **The Pi is about twice as slow at masking, and it does not matter.** Mean
  masking latency is 91.6 ms against the workstation's 46.6 ms, but masking is
  0.8 % of the audio pipeline's processing time. Speech recognition is 96.8 %.
  Any further end-to-end speed-up has to come from the recognizer.
* **It costs recognition accuracy against Canary.** Number-normalised WER on
  this corpus is 0.0903 against Canary's 0.0403, and 2 of 5 identifier-scale
  numbers are no longer transcribed digit for digit. That is what the deployment
  gain was traded for — Canary needed 2.02× real time on the same board — and it
  is the largest remaining source of end-to-end privacy failure.

Reproduce with `python Full_Pipeline_Audio_Benchmark.py` and
`python Full_Pipeline_CPU_Benchmark.py` from the repository root.

### The assistant round trip on the Pi

Run `20260823T204627Z_linux-aarch64_e2e-v6-whisper_full`, 6 committed clips.
It ran **without a `GROQ_API_KEY`**, so it is an outbound-payload audit: the real
client built each request and the bytes were audited, but nothing was sent.

| Check | Workstation (live) | **Raspberry Pi 5 (payload audit)** |
|---|---:|---:|
| Outbound payloads captured | 6 | **6** |
| Values the masker replaced, tested | 9 | **9** |
| …reappearing in a payload | **0** | **0** |
| Raw transcripts in a payload | **0** | **0** |
| Gold entities the detector missed, in a payload | 2 | **2** |
| Deterministic rehydration probes passed | 4 / 4 | **4 / 4** |
| Malformed-placeholder probes passed | 10 / 10 | **10 / 10** |
| Local device command: requests built | 0 | **0** |
| Mean local pre-request latency | 1.355 s | **2.642 s** |
| Local real-time factor | 0.301 | **0.587** |

The live-only measurements — network round trip, whether a real model returns
intact placeholders, API success rate, spoken output — exist for the workstation
only and are reported as such. On the workstation all 6 live requests succeeded,
both replies that used a placeholder used only intact ones, and both were
resolved locally.

## Features

* **faster-whisper `base.en`** speech-to-text on CTranslate2, CPU, INT8
* Silero VAD speech detection
* Automatic startup noise calibration
* Continuous speech segmentation
* 14-word overlap transcript merger
* Live masked transcript preview
* PII detection and masking
* Spoken-address rewrite in front of the detectors
* Spoken-number normalization
* Local SQLite transcript storage
* Optional Onkith commands
* Designed for Raspberry Pi 5 with 4 GB RAM

---

## STT Benchmark

Measured on the committed 44-clip fixture, on a Raspberry Pi 5 Model B Rev 1.0.
Both engines are scored from their saved transcripts by one implementation
(`model/privacy_evaluation/privacy_eval/stt_comparison.py`). Whisper's column is
run `20260823T204339Z_linux-aarch64_audio-v6-whisper_full`; Canary's is the
frozen `20260818T081310Z_linux-aarch64_audio_full`, which is never regenerated.

| Metric | **This package (Whisper `base.en`)** | V3 Canary 180M Flash |
| --- | ---: | ---: |
| Aggregate WER, as each run recorded it | 0.1841 | 0.0403 |
| Aggregate WER, number-normalised | **0.0903** | **0.0403** |
| Long numbers transcribed digit for digit | 3 / 5 | 5 / 5 |
| Mean STT real-time factor, Pi 5 | **0.617** | 2.022 |
| Mean full-pipeline RTF, Pi 5 | **0.634** | 2.041 |
| Peak RSS, audio benchmark, Pi 5 | **1,571 MiB** | 2,031 MiB |
| Model size | 74M parameters | 180M parameters |

> **Read the number-normalised row, not the raw one.** Canary ran with
> `pnc: no` and wrote spoken numbers as words, which is the form the reference
> transcripts use; Whisper writes numerals, and the standard word error rate
> charges eight substitutions for correctly transcribing "five five five, zero
> one eight eight" as `55550188`. Folding numbers to numerals on both sides is
> what makes the comparison about recognition rather than orthography.
>
> Canary's published LibriSpeech reference is **4.90%** WER at RTF 0.0784. That
> is a vendor benchmark on different hardware and a different corpus; it is not
> comparable to anything in this table and is recorded here only so it is not
> mistaken for a Pi measurement.

---

## Configuration

### Whisper decoder

```text
engine       = faster-whisper 1.2.1
runtime      = ctranslate2
model        = base.en  (Systran/faster-whisper-base.en)
device       = cpu
compute_type = int8
beam_size    = 5
language     = en
```

The decoding hint is built from the device's configured names and command shapes
in `integrated_pipeline/whisper_adapter.py`; set `whisper.initial_prompt` in
`config/live_config.json` to a string to override it wholesale.

### VAD & Segmentation

```text
pre_roll = 0.40s
start_confirmation = 0.20s
endpoint_silence = 2.50s
max_segment = 8.50s
forced_overlap = 0.35s
trailing_context = 0.35s
vad_threshold = 0.50
min_speech = 0.20s
input_chunk = 0.10s
```

---

## Pipeline

```text
16 kHz Microphone
        ↓
Startup Noise Calibration
        ↓
Silero VAD + RMS Gate
        ↓
Speech Segmentation
        ↓
faster-whisper base.en
        ↓
14-Word Overlap Merger
        ↓
Onkith
        ↓
Stage 5
        ↓
Spoken-Address Rewrite (detectors only)
        ↓
PII Masking
        ↓
SQLite Storage
        ↓
Masked Terminal Output
```

Long continuous speech is split into smaller overlapping segments before transcription.

The overlap helps avoid losing words around segment boundaries. After transcription, the **14-word merger** removes repeated text and combines the segments into one final utterance.

---

## STT Backend

| Setting                      | Value                               |
| ---------------------------- | ----------------------------------- |
| Model                        | `Systran/faster-whisper-base.en`    |
| faster-whisper               | `1.2.1`                             |
| Runtime                      | CTranslate2                         |
| Device                       | CPU                                 |
| Precision                    | INT8                                |
| Batch size                   | 1                                   |
| Language                     | English                             |
| Decoder                      | Beam search                         |
| Beam size                    | 5                                   |
| Punctuation / capitalization | On                                  |
| Timestamps                   | Off                                 |
| Model loading                | Once at startup                     |

The recognizer stays loaded in memory while each speech segment is transcribed
independently.

**Punctuation and capitalization are on, and V3's were off.** That is not a
setting this package chose so much as one Whisper does not offer, and it changes
what the privacy model is handed: capitalised names, sentence punctuation, and
numbers written as numerals rather than as digit words. The consequences are
measured rather than assumed — see
[`../research_results/README.md`](../research_results/README.md) for what it did
to the benchmark's own scoring, and `KNOWN_ISSUES.md` for what it does to
masking.

V3 pinned its recognizer by artifact SHA-256 because the `.nemo` file was fetched
once and committed. faster-whisper resolves `base.en` through CTranslate2's own
cache instead, so what `config/live_config.json` pins is the package version, the
model id and the decoding settings — the things that change what comes out.

---

## Privacy Backend

PII detection runs on **OnKith Model V2**: a DeBERTa-v3-xsmall token classifier
fine-tuned on the OnKith privacy corpus and quantized to INT8 ONNX.

| Setting          | Value                                          |
| ---------------- | ---------------------------------------------- |
| Base model       | `microsoft/deberta-v3-xsmall`                  |
| Ontology         | `model_v2.ontology.1` — 31 entity types        |
| Labels           | 63 BIO labels                                  |
| Precision        | INT8 ONNX                                      |
| Runtime          | onnxruntime, `CPUExecutionProvider`            |
| Context window   | 512 tokens, 128-token overflow stride          |
| Tokenizer        | SentencePiece, packaged locally                |
| Model loading    | Once at startup, hash-verified                 |

Artifacts live in `models/deberta_pii/` (ONNX, tokenizer, label map,
`model_manifest.json`). They are byte-identical copies of the canonical
artifacts under `model/DeBERTa-v3-xsmall/artifacts/model_v2/`, and the adapter
refuses to start if the INT8 sha256 does not match the value pinned in
`config/live_config.json`.

### Held-out comparison

Model V2 replaced the previous TinyBERT-4 masker (Model V1) after a direct
40,909-row held-out comparison:

| Metric                    | Model V1 (TinyBERT INT8) | **Model V2 (DeBERTa INT8)** |
| ------------------------- | -----------------------: | --------------------------: |
| Typed entity F1           |                   0.9502 |                  **0.9588** |
| Private character leakage |                   0.0029 |                  **0.0017** |
| Non-private overmasking   |                   0.0044 |                  **0.0029** |

Model V1 remains in `model/TinyBERT-4/` and `tinybert_pii_rpi/` for provenance.
It is no longer packaged with or reachable from this pipeline.

### Decoding

`integrated_pipeline/model_v2_masker.py` holds the only Model V2 decoder in the
project; the offline evaluation harness imports the same module, so the numbers
above were measured through the code this pipeline runs.

SentencePiece pre-tokenizes on whitespace only, which means `Walusiak,` is one
tokenizer word — comma included. The decoder therefore derives word groups from
character offsets rather than tokenizer word ids, so an entity never absorbs the
punctuation that follows it. `tests/test_deberta_pii_integration.py` pins that
behaviour end to end.

### Choosing what gets masked

The model always predicts its complete ontology. Which detections are actually
replaced in the released text is a configuration decision, set per entity type
under `pii.mask_policy` in `config/live_config.json`:

```json
{
  "pii": {
    "mask_policy": { "CITY": false, "COUNTRY": false }
  }
}
```

With that policy, *"What time is it in Riyadh?"* is released unchanged while
`Riyadh` is still detected and reported as a `CITY` with `"masked": false`.
Every other entity type keeps masking normally. An empty policy — the shipped
default — masks everything. See [MASK_POLICY.md](MASK_POLICY.md) for the full
contract and its fail-closed rules.

---

## Live Preview

During long continuous speech, Omit AI can display an in-progress **masked preview** instead of waiting for the entire utterance to finish.

The preview:

* stays in memory only
* is not written to SQLite
* is not sent to Onkith
* does not modify the final transcript

PII masking runs across a rolling text buffer so sensitive information split across two speech segments can still be detected.

---

## Spoken Numbers

Spoken number patterns can be converted into numerals for display after privacy masking.

Examples:

```text
four seven two nine → 4729

twenty five → 25

double four → 44
```

The original STT transcript is still used for storage, PII detection, and Onkith processing.

---

## Raspberry Pi 5 Requirements

* Raspberry Pi 5
* 4 GB RAM
* 64-bit Raspberry Pi OS
* Python 3.11
* 16 kHz mono microphone input
* At least 6 GiB of free storage

---

## Installation

From a clone of the repository:

```bash
cd FullPipeline_V6
```

(A packaged release ZIP built by `scripts/build_release_zip.py` unpacks to the
same layout; `cd` into it instead.)

Make the scripts executable:

```bash
chmod +x install_pi.sh run_live.sh check_pi.sh
```

Install the project:

```bash
./install_pi.sh
```

The Whisper `base.en` weights are fetched once by CTranslate2 on first use and
cached under `~/.cache/huggingface`.

After setup, normal speech recognition can run offline.

---

## Microphone Setup

List available audio devices:

```bash
./run_live.sh --list-devices
```

Run the hardware check:

```bash
./check_pi.sh --with-mic --device 1
```

Replace `1` with the microphone device ID you want to use.

Start Omit AI:

```bash
./run_live.sh --device 1
```

---

## Verification

Run the test suite from this directory, with the interpreter of the environment
you installed into:

```bash
python -m pytest -q
```

`python -m unittest discover -s tests -v` works as well; the suite uses no
pytest-only features.

Run the built-in self-test:

```bash
./run_live.sh --self-test
```

Display the active configuration:

```bash
./run_live.sh --print-config
```

---

## Privacy

Omit AI is designed so normal terminal output displays only the **masked transcript**.

If PII masking fails, the output is blocked instead of displaying the unmasked transcript.

SQLite stores both raw and masked transcripts locally in:

```text
session_transcript.db*
```

These database files may contain sensitive information and should **not be uploaded or shared**.

---

## Onkith

Omit AI includes optional Onkith commands for actions such as YouTube playback.

Add the following environment variables:

```text
YOUTUBE_API_KEY=your_key_here
YOUTUBE_REGION=SA
```

Keep them in a local `.env` beside this file or in the shell environment. `.env`
and `*.key` are gitignored repository-wide; never commit a real key.

Example commands:

```text
Onkith play Adele easy on me
Onkith stop the music
```

Only the YouTube request requires network access.

Speech-to-text and PII processing remain local.

---

## Current Limitations

* English-only speech recognition
* Whisper emits punctuation and capitalization; this changes the surface passed
  to the privacy layer compared with V3's Canary
* Live microphone input must be 16 kHz
* PII detection may produce false positives or false negatives
* Raw transcripts are stored locally in SQLite
* YouTube playback requires internet access and API quota
* **Real-time operation has limited headroom on a Raspberry Pi 5.** Mean
  full-pipeline RTF is 0.634 and the P90 is 1.109. 7 of the 44 benchmark clips
  exceeded real time, and every one of them was shorter than 2.3 s: Whisper
  decodes a fixed window whatever the clip length, so short utterances pay a
  fixed cost. Every clip of 3 s or more stayed under real time, the worst at
  0.68×
* **Post-masking rendering is not re-screened.** V3 demonstrated that surviving
  digit words could be rebuilt into compact identifiers after masking. This
  package measures zero such reconstructions on both devices because Whisper
  emits numerals directly, not because the renderer was made safe; see
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
* Repeated mentions of one person receive different placeholder numbers — see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

---

> **Research Notice:** Omit AI is not a medical device. Clinical use requires independent validation and privacy review.

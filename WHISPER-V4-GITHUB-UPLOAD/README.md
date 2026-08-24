# OnKith production pipeline (FullPipeline_V4)

This build includes an optional assistant layer after the existing privacy
gate. Addressed questions send masked text only to Groq, then replies are
rehydrated and spoken locally. Wake routing, music/light/device actions, the
localhost UI, TTS and rehydration do not alter Whisper, segmentation, masking,
SQLite, or the canonical masked transcript. See `ASSISTANT_ADDON.md`.

> **This package is FullPipeline_V4.**
>
> It is V3's pipeline with one component replaced: the speech recognizer is
> **faster-whisper `base.en`** on CTranslate2 (CPU, INT8, beam 5, with a decoding
> hint built from the device's configured names), in place of NVIDIA Canary 180M
> Flash. VAD, segmentation, transcript merging, wake-word repair, the inactivity
> buffer, DeBERTa-v3-xsmall INT8 masking, span merging, spoken-digit rendering,
> SQLite storage and the terminal UI are V3's code.
>
> V4 adds exactly one thing to the privacy path:
> [`integrated_pipeline/spoken_form_normalizer.py`](integrated_pipeline/spoken_form_normalizer.py),
> which rewrites two spoken address forms into their written form for the
> detectors only and projects every span back before masking. It exists because
> Whisper writes `billing at example.org`, which is neither the spoken form nor
> the written form, and the mailbox name was being released while the domain was
> masked. `model/privacy_evaluation/tests/test_masking_parity_v3_v4.py` runs
> fixed text through both packages and requires identical output on everything
> else.
>
> The Canary implementation and artifacts are preserved at
> [`../archive/FullPipeline_V3/`](../archive/FullPipeline_V3/), together with the
> measurements it produced. Both are benchmarked on both devices; the comparison
> is in [`../research_results/`](../research_results/).

**Local speech-to-text and privacy processing for Raspberry Pi 5**

This is the **current** OnKith deployment package and the only maintained one.
Earlier integrated pipelines live under [`../archive/`](../archive/) and should
not be used. The Python entry point is still `app.omit_ai_live` and some code
still says "Omit AI" — that is the project's earlier name, kept so the module
paths in existing instructions keep working.

It is a local English speech-to-text pipeline built for the **Raspberry Pi 5**. It uses **faster-whisper `base.en`** on CTranslate2 for transcription, **Silero VAD** for speech detection, overlap-based segmentation for continuous speech, PII masking, and local SQLite storage.

Speech recognition and privacy processing run locally. Internet access is only required during setup to download the model and for optional YouTube playback through Onkith.

## Measured on a Raspberry Pi 5 (4 GB)

The whole package has been run end to end on a Raspberry Pi 5 Model B Rev 1.0
with 3.95 GiB visible RAM, 64-bit Raspberry Pi OS, Python 3.11.9.

| Measurement | V4 (Whisper) | V3 (Canary) |
|---|---|---|
| Peak RSS, full audio pipeline | **1,155 MiB** of 3.95 GiB | 2,031 MiB |
| Memory after the recognizer loads | 541 MiB | 1,990 MiB |
| Memory added by the DeBERTa INT8 masker | **28 MiB** | 28 MiB |
| Aggregate WER, 44 committed clips (279.7 s) | 0.1841 raw / **0.0903** number-normalised | 0.0403 |
| Mean STT real-time factor | **0.573** | 2.022 |
| Mean full-pipeline real-time factor | **0.589** | 2.041 |
| P90 / P95 full-pipeline RTF | 0.993 / 1.052 | 2.684 / 2.808 |
| Mean end-to-end latency per clip | **2.8 s** | 13.9 s |
| STT share of processing time | 96.7% | 99.2% |
| Privacy masking, mean per clip | 24.9 ms | 30.4 ms |
| Privacy model on 2,000 committed rows | typed F1 0.9448, 72.6 ms mean | typed F1 0.9448, 83.6 ms mean |

Three conclusions:

* **It fits, with room to spare now.** The complete stack peaks at under a third
  of the board's RAM, against roughly half under V3.
* **It is real-time, narrowly.** Mean full-pipeline RTF is 0.589 — the first
  configuration in this repository to process speech faster than it arrives on
  the target board. P95 is 1.05, so the longest utterances sit at the boundary:
  this is real-time-capable, not real-time with headroom.
* **It costs recognition accuracy.** Number-normalised WER roughly doubles
  against Canary, and 2 of 5 identifier-scale numbers are no longer transcribed
  digit for digit. That is what the deployment gain was traded for, and it is
  the largest remaining source of end-to-end privacy failure.

Faris's current-path Pi rerun also exercised the spoken-address rewrite and
reported 0.8300 private-character recall with 16 of 25 privacy clips fully
sanitized. Its environment lacked `onnx` and `psutil`, so the complete VAD and
memory figures above remain sourced from the earlier V4 run on the same board.

The word error rates are only comparable in the number-normalised form. Canary
ran with `pnc: no` and wrote spoken numbers as words, the way the reference
transcripts are written; Whisper writes numerals, and the standard metric charges
eight substitutions for correctly transcribing "five five five, zero one eight
eight" as `55550188`.

The privacy model is not the bottleneck and never was: 0.9% of V4's processing
time and 2.4% of resident memory, against 0.2% and 1.4% of V3's much larger
totals. Reproduce with `python Full_Pipeline_Audio_Benchmark.py`
and `python Full_Pipeline_CPU_Benchmark.py` from the repository root.

## Features

* **faster-whisper `base.en`** speech-to-text on CTranslate2, CPU, INT8
* Silero VAD speech detection
* Automatic startup noise calibration
* Continuous speech segmentation
* 14-word overlap transcript merger
* Live masked transcript preview
* PII detection and masking
* Spoken-address rewrite in front of the detectors (V4)
* Spoken-number normalization
* Local SQLite transcript storage
* Optional Onkith commands
* Designed for Raspberry Pi 5 with 4 GB RAM

---

## STT Benchmark

Measured on the committed 44-clip fixture, on a Raspberry Pi 5 Model B Rev 1.0.
Both engines are scored from their saved transcripts by one implementation
(`model/privacy_evaluation/privacy_eval/stt_comparison.py`).

| Metric | **V4 Whisper `base.en`** | V3 Canary 180M Flash |
| --- | ---: | ---: |
| Aggregate WER, as each run recorded it | 0.1841 | 0.0403 |
| Aggregate WER, number-normalised | **0.0903** | **0.0403** |
| Long numbers transcribed digit for digit | 3 / 5 | 5 / 5 |
| Mean STT real-time factor, Pi 5 | **0.573** | 2.022 |
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
cd FullPipeline_V4
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
  to the privacy layer compared with V3
* Live microphone input must be 16 kHz
* PII detection may produce false positives or false negatives
* Raw transcripts are stored locally in SQLite
* YouTube playback requires internet access and API quota
* **Real-time operation has limited headroom on a Raspberry Pi 5.** Mean
  full-pipeline RTF is 0.589, but P95 is 1.052, so the longest benchmark clips
  sit just beyond real time
* **Post-masking rendering is not re-screened.** V3 demonstrated that surviving
  digit words could be rebuilt into compact identifiers after masking. V4
  measured zero such reconstructions because Whisper emits numerals directly,
  not because the renderer was made safe; see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
* Repeated mentions of one person receive different placeholder numbers — see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

---

> **Research Notice:** Omit AI is not a medical device. Clinical use requires independent validation and privacy review.

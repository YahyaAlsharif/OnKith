# OnKith production pipeline (FullPipeline_V3)

**Local speech-to-text and privacy processing for Raspberry Pi 5**

This is the **current** OnKith deployment package and the only maintained one.
Earlier integrated pipelines live under [`../archive/`](../archive/) and should
not be used. The Python entry point is still `app.omit_ai_live` and some code
still says "Omit AI" — that is the project's earlier name, kept so the module
paths in existing instructions keep working.

It is a local English speech-to-text pipeline built for the **Raspberry Pi 5**. It uses **NVIDIA Canary 180M Flash** for transcription, **Silero VAD** for speech detection, overlap-based segmentation for continuous speech, PII masking, and local SQLite storage.

Speech recognition and privacy processing run locally. Internet access is only required during setup to download the model and for optional YouTube playback through Onkith.

## Measured on a Raspberry Pi 5 (4 GB)

The whole package has been run end to end on a Raspberry Pi 5 Model B Rev 1.0
with 3.95 GiB visible RAM, 64-bit Raspberry Pi OS, Python 3.11.9.

| Measurement | Result |
|---|---|
| Peak RSS, full audio pipeline | **2,031 MiB** of 3.95 GiB |
| Memory after Canary loads | 1,990 MiB |
| Memory added by the DeBERTa INT8 masker | **28 MiB** |
| Aggregate WER, 44 committed clips (279.7 s) | 0.0403 |
| Mean STT real-time factor | **2.022** |
| Mean full-pipeline real-time factor | **2.041** |
| Mean end-to-end latency per clip | 13.9 s |
| STT share of processing time | 99.2% |
| Privacy masking, mean per clip | 30.4 ms |
| Privacy model on 2,000 committed rows | typed F1 0.9448, 83.6 ms mean, 11.95 rows/s |

Two conclusions, both important:

* **It fits.** The complete stack runs on a 4 GB Pi 5 without running out of
  memory, with roughly half the board's RAM to spare.
* **It is not real-time.** At RTF ≈ 2, the device needs about two seconds of
  computation per second of speech. Use it for push-to-talk, turn-based
  interaction and batch utterance processing — not live streaming transcription.

The privacy model is not the bottleneck: it is 0.2% of processing time and 1.4%
of the resident memory. Reproduce with `python Full_Pipeline_Audio_Benchmark.py`
and `python Full_Pipeline_CPU_Benchmark.py` from the repository root.

## Features

* NVIDIA **Canary 180M Flash** speech-to-text
* Silero VAD speech detection
* Automatic startup noise calibration
* Continuous speech segmentation
* 14-word overlap transcript merger
* Live masked transcript preview
* PII detection and masking
* Spoken-number normalization
* Local SQLite transcript storage
* Optional Onkith commands
* Designed for Raspberry Pi 5 with 4 GB RAM

---

## STT Benchmark

### Canary 180M Flash — LibriSpeech

| Metric        |               Result |
| ------------- | -------------------: |
| WER           |            **4.90%** |
| Errors / REF  | **10,300 / 210,269** |
| Valid samples |  **11,126 / 11,126** |
| Substitutions |                5,115 |
| Deletions     |                  465 |
| Insertions    |                4,720 |
| Model RTF     |           **0.0784** |
| Model size    |  **180M parameters** |

> **Note:** The WER and RTF above come from the published LibriSpeech benchmark
> for this model and are **not** Raspberry Pi 5 runtime measurements. The measured
> Pi 5 numbers are at the top of this file: aggregate WER 0.0403 and mean STT RTF
> 2.022 on the committed 44-clip fixture.

---

## Configuration

### Canary Decoder

```text
decoder = beam
beam_size = 2
temperature = 1.0
max_generation_delta = 50
```

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
Canary 180M Flash
        ↓
14-Word Overlap Merger
        ↓
Onkith
        ↓
Stage 5
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

| Setting                      | Value                      |
| ---------------------------- | -------------------------- |
| Model                        | `nvidia/canary-180m-flash` |
| NeMo                         | `2.5.3`                    |
| PyTorch                      | `2.7.1`                    |
| Device                       | CPU                        |
| Precision                    | FP32                       |
| Batch size                   | 1                          |
| Language                     | English                    |
| Decoder                      | Beam search                |
| Beam size                    | 2                          |
| Punctuation / capitalization | Off                        |
| Timestamps                   | Off                        |
| Model loading                | Once at startup            |

The Canary model stays loaded in memory while each speech segment is transcribed independently.

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
cd FullPipeline_V3
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

The Canary model is downloaded during installation.

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
* Punctuation and capitalization are currently disabled
* Live microphone input must be 16 kHz
* PII detection may produce false positives or false negatives
* Raw transcripts are stored locally in SQLite
* YouTube playback requires internet access and API quota
* **Speech recognition is not real-time on a Raspberry Pi 5** (measured RTF ≈ 2.0). Push-to-talk and turn-based use work; live streaming transcription does not
* **Spoken numeric identifiers can survive masking and then be rendered as digits.** With `pnc: no` the recognizer emits digit words. If the privacy model misses one, `app/digit_normalizer.py` — which runs *after* masking, by design, for display only — can rewrite the surviving words into a numeral such as `5550147`. No new information is released, but the compact form is worse. Measured on 2 of 60 entities in the audio benchmark, identically on the workstation and the Pi
* Repeated mentions of one person receive different placeholder numbers — see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

---

> **Research Notice:** Omit AI is not a medical device. Clinical use requires independent validation and privacy review.

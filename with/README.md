# OnKith production pipeline (FullPipeline_V4)

> **This is the strict Parakeet STT replacement build of FullPipeline_V4.**
>
> The former Whisper adapter has been replaced by NVIDIA
> `nvidia/parakeet-tdt_ctc-110m`, pinned by Hugging Face revision, checkpoint
> size and SHA-256. Inference is CPU FP32 through NeMo's **TDT/RNNT** decoder
> (`greedy_batch`, maximum 10 symbols, temperature 1.0); the CTC-only route is
> rejected at startup. The recognizer is loaded and warmed exactly once.
>
> The adapter still consumes `CanarySegment` and returns `SegmentTranscript`.
> Transcript merging, wake-word repair, inactivity buffering, DeBERTa privacy
> masking, spoken-form handling, display rendering and SQLite storage remain
> on the existing V4 path. Wake/intent routing, Groq, local reply rehydration,
> local TTS and a localhost UI are added strictly after its privacy gate.

**Local speech-to-text and privacy processing for Raspberry Pi 5**

This is the **current** OnKith deployment package and the only maintained one.
Earlier integrated pipelines live under [`../archive/`](../archive/) and should
not be used. The Python entry point is still `app.omit_ai_live` and some code
still says "Omit AI" — that is the project's earlier name, kept so the module
paths in existing instructions keep working.

It is a local English speech-to-text pipeline intended for the **Raspberry Pi
5**. It uses NVIDIA Parakeet 110M TDT through NeMo for transcription, Silero VAD
for speech detection, overlap-based segmentation for continuous speech, PII
masking, and local SQLite storage.

Speech recognition and privacy processing run locally. Internet access is
required during setup, for optional YouTube playback, and only for an addressed
question sent to Groq after masking. Ordinary speech and local commands are not
sent to the LLM. See `ASSISTANT_ADDON.md` and `.env.example`.

## Validation status

The supplied winning Parakeet experiment recorded these results on its own test
hardware; they are **not Raspberry Pi measurements**:

| Dataset | WER | Errors / reference | Validity | RTF |
|---|---:|---:|---:|---:|
| Custom 30 | 15.12% | 65 / 430 | 30 / 30 | 0.0192 |
| LibriSpeech | 4.60% | 9,672 / 210,269 | 11,126 / 11,126 | 0.0190 |

The adapter contract, configuration, checkpoint identity and TDT decoder
selection are covered by local automated tests. Raspberry Pi installation,
peak memory, real-time factor, thermals and microphone behavior for this
Parakeet build remain to be measured on the target board. Older V4
Whisper-on-Pi numbers in repository history do not apply to this build.

## Features

* NVIDIA Parakeet 110M hybrid checkpoint through the TDT/RNNT path, CPU FP32
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
* Wake-word question routing with masked-text-only Groq requests
* Local reply rehydration and local espeak/Piper TTS
* Localhost-only assistant and device UI
* Local PWM light and device-panel commands
* Designed for Raspberry Pi 5 with 4 GB RAM

---

## Configuration

### Parakeet TDT decoder

```text
model              = nvidia/parakeet-tdt_ctc-110m
revision           = 431a349f3051ab85c22b9b7a2741b5fe77065665
checkpoint sha256  = 1e6b5c3bcaa390b479bd8112512ede515d2ef5b61495568f2c266d6d5e680c3a
runtime            = NeMo 2.7.3 / torch 2.7.1
device / precision = CPU / FP32
decoder            = TDT (RNNT), greedy_batch
max symbols        = 10
temperature        = 1.0
timestamps         = off
alignments         = off
```

Startup refuses a wrong checkpoint hash, the CTC decoder branch, a non-TDT
greedy implementation or settings that differ from this configuration.

### VAD & Segmentation

```text
pre_roll = 0.40s
speech_pad = 0.03s
start_confirmation = 0.20s
endpoint_silence = 2.50s
max_segment = 8.50s
forced_overlap = 0.39s
trailing_context = 0.35s
vad_threshold = 0.50
vad_negative_threshold = 0.35
min_speech = 0.25s
min_silence = 0.10s
input_chunk = 0.10s
```

Speech pad is added to submitted pre-roll and trailing context, yielding 6,880
pre-roll samples and 6,080 trailing samples at 16 kHz. The unique forced-split
boundary remains exactly 136,000 samples (8.50 seconds).

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
NVIDIA Parakeet 110M TDT
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

| Setting | Value |
|---|---|
| Model | `nvidia/parakeet-tdt_ctc-110m` |
| NeMo / torch | `2.7.3` / `2.7.1` |
| Device / precision | CPU / FP32 |
| Batch size | 1 |
| Language | English |
| Decoder | TDT/RNNT `greedy_batch` |
| Maximum symbols | 10 |
| Temperature | 1.0 |
| Timestamps / alignments | Off / off |
| Model loading | Once at startup |

The recognizer stays loaded in memory while each speech segment is transcribed
independently.

The exact `.nemo` checkpoint is fetched once during setup and stored in the
Hugging Face cache. Runtime startup verifies its revision, filename, byte size
and SHA-256 before restoring it.

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

The exact Parakeet `.nemo` checkpoint is fetched and verified during setup and
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
* Live microphone input must be 16 kHz
* PII detection may produce false positives or false negatives
* Raw transcripts are stored locally in SQLite
* YouTube playback requires internet access and API quota
* **Raspberry Pi performance is unverified for this Parakeet build.** Run the
  included hardware check and audio benchmark before calling it real-time or
  suitable for a 4 GB deployment
* **Post-masking rendering is not re-screened.** V3 demonstrated that surviving
  digit words could be rebuilt into compact identifiers after masking; see
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
* Repeated mentions of one person receive different placeholder numbers — see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)

---

> **Research Notice:** Omit AI is not a medical device. Clinical use requires independent validation and privacy review.

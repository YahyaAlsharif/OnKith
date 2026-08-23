# OnKith FullPipeline_V5 — Parakeet STT + masked LLM round trip

> **V5 is the demonstration pipeline. V4 is still the production pipeline.**
>
> V5 does two things V4 does not. It replaces the Whisper recognizer with NVIDIA
> `nvidia/parakeet-tdt_ctc-110m`, and it adds a complete assistant layer *after*
> the privacy gate: wake/intent routing, a masked-text-only Groq request, local
> reply rehydration, local TTS and a localhost UI.
>
> On this project's own 44-clip audio benchmark, Parakeet is substantially more
> accurate on words and roughly three times faster than Whisper, but it leaks
> **more** private characters and needs ~577 MiB more RSS. That trade is not yet
> settled in V5's favour, so [`../FullPipeline_V4/`](../FullPipeline_V4/) has not
> been archived. See **Measured status** below for the numbers and
> [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for why.
>
> The Parakeet checkpoint is pinned by Hugging Face revision, byte size and
> SHA-256. Inference is CPU FP32 through NeMo's **TDT/RNNT** decoder
> (`greedy_batch`, maximum 10 symbols, temperature 1.0); the CTC-only route is
> rejected at startup. The recognizer is loaded and warmed exactly once.
>
> The adapter still consumes `CanarySegment` and returns `SegmentTranscript`.
> Transcript merging, wake-word repair, inactivity buffering, DeBERTa privacy
> masking, spoken-form handling, display rendering and SQLite storage are
> byte-identical to V4's.

**Local speech-to-text and privacy processing for Raspberry Pi 5**

V5 was assembled from the `with/` integration folder, which is where the
Parakeet swap and the assistant layer were developed. The Python entry point is
still `app.omit_ai_live` and some code still says "Omit AI" — that is the
project's earlier name, kept so the module paths in existing instructions keep
working.

It is a local English speech-to-text pipeline intended for the **Raspberry Pi
5**. It uses NVIDIA Parakeet 110M TDT through NeMo for transcription, Silero VAD
for speech detection, overlap-based segmentation for continuous speech, PII
masking, and local SQLite storage.

Speech recognition and privacy processing run locally. Internet access is
required during setup, for optional YouTube playback, and only for an addressed
question sent to Groq after masking. Ordinary speech and local commands are not
sent to the LLM. See `ASSISTANT_ADDON.md` and `.env.example`.

## Measured status

Both columns below are the **same 44 committed clips through the same benchmark
code on the same Windows workstation**, so they are a like-for-like comparison.
Re-run them with `python Full_Pipeline_Audio_Benchmark.py`.

| Metric | V4 Whisper | V5 Parakeet | |
|---|---:|---:|---|
| Aggregate WER | 0.1795 | **0.0644** | better |
| WER, number-normalised | 0.0854 | **0.0619** | better |
| STT real-time factor (mean) | 0.360 | **0.114** | better |
| Full-pipeline RTF (mean) | 0.382 | **0.135** | better |
| End-to-end P95 | 2,730.9 ms | **1,471.4 ms** | better |
| Entity recall after STT | 0.5500 | **0.5833** | better |
| Hard-negative false positives | 1 / 7 | 1 / 7 | same |
| Character recall after STT | **0.8314** | 0.7299 | **worse** |
| Character leakage after STT | **0.1686** | 0.2701 | **worse** |
| Fully sanitized clips | **16 / 25** | 12 / 25 | **worse** |
| Long identifiers transcribed exactly | **3 / 5** | 0 / 5 | **worse** |
| Peak process RSS | **1,269.5 MiB** | 1,846.3 MiB | **worse** |

The three regressions have one cause between them, and it is orthography rather
than recognition. Parakeet writes a number read aloud digit by digit as digit
*words* ("five zero one four seven") where Whisper writes a numeral
("555-0147"), and it renders a spoken address as "example. org" where Whisper
writes "example.org". The privacy model and the structured recognizers were
trained and tuned on written English, so both surfaces reach them in a shape
they handle less well. Parakeet also drops repeated tokens in a digit run, which
is what costs it every long identifier. Details and evidence:
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

Raspberry Pi installation, peak memory, real-time factor, thermals and
microphone behaviour for this build remain **unmeasured**. The +577 MiB of RSS
is the number to check first on a 4 GB Pi 5. Older V4 Whisper-on-Pi results do
not apply to this build.

### The round trip

`Full_Pipeline_V5_EndToEnd_Benchmark.py` measures the assistant layer: 24 clips
through Parakeet, masking, the outbound request, rehydration and TTS. Its
privacy check reads the bytes handed to `urllib` inside `app/llm.py` and writes
every payload to the run folder verbatim, so "no private value left the device"
is something a reader can verify rather than take on trust.

From `research_results/cpu_benchmarks/20260823T095954Z_windows-amd64_e2e-parakeet_full/`:

| | |
|---|---:|
| Outbound payloads captured | 24 |
| Values the masker replaced that appear in a payload | **0 / 35** |
| Raw transcripts that appear in a payload | **0 / 24** |
| Gold private entities the model missed, and that therefore left | 7 / 37 |
| Deterministic rehydration probes restored every value | 15 / 15 |
| Malformed / unknown / repeated placeholder probes | 9 / 9 |
| End-to-end successful cases | 19 / 24 |
| Local pre-LLM latency, mean / P50 / P95 | 592 / 495 / 1,068 ms |
| Local TTS latency (Windows SAPI), mean | 4,707 ms |

The first two rows are the guarantee this project makes and they hold exactly:
nothing the masker identified, and no raw transcript, reached the request. The
third row is the guarantee it does not make - a value the privacy model never
detected is a value it never masked - and it is the same 7 entities the audio
benchmark attributes to privacy-model failure. All five end-to-end failures are
that one cause.

**The LLM leg of that run used `--llm-stub`**, a local cooperative responder, so
the round-trip latency and the 15/15 placeholder figure describe the integration
and not Groq. No request has yet been sent to Groq from this repository. Set
`GROQ_API_KEY` and re-run without `--llm-stub` for live numbers.

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
runtime            = NeMo 2.7.3 (pinned) or 2.5.3 (verified) / torch 2.7.1
device / precision = CPU / FP32
decoder            = TDT (RNNT), greedy_batch
max symbols        = 10
temperature        = 1.0
timestamps         = off
alignments         = off
```

Startup refuses a wrong checkpoint hash, the CTC decoder branch, a non-TDT
greedy implementation or settings that differ from this configuration.

`PARAKEET_VERIFIED_PACKAGE_VERSIONS` in `integrated_pipeline/config.py` is a
short allowlist rather than a single NeMo pin, because the version the platform
setup scripts install and the interpreter that produced a given measurement are
not always the same. An unlisted release is still refused, the checkpoint bytes
are still verified, the live model still has to read back as TDT, and
`ParakeetAdapter.audit()` records the version that actually ran — so no result
folder is ambiguous about which one produced it.

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
| NeMo / torch | `2.7.3` pinned, `2.5.3` also verified / `2.7.1` |
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
cd FullPipeline_V5
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
* **Parakeet leaks more private characters than Whisper on this corpus** — 0.2701
  against 0.1686 — because it writes spoken digits as words and spoken addresses
  with a space before the TLD. This is the reason V5 has not replaced V4
* **Parakeet drops repeated tokens in a digit run.** "five five five zero one
  four seven" comes back as "five zero one four seven", so no long identifier in
  the corpus survives intact. A mangled number is not a leaked number, but it is
  also not a usable one
* **Peak RSS is ~577 MiB higher than V4's**, which is the first thing to check on
  a 4 GB Pi 5
* **Raspberry Pi performance is unverified for this build.** Run the included
  hardware check and audio benchmark before calling it real-time or suitable for
  a 4 GB deployment
* **Post-masking rendering is not re-screened.** Under V5 the V3-era shape is
  back: surviving digit words can be rebuilt into a compact identifier after
  masking. One clip in 44. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
* Repeated mentions of one person receive different placeholder numbers — see
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
* **Placeholder numbering is per utterance, and the rehydration map is per
  conversation.** `[GIVENNAME_1]` in one utterance and `[GIVENNAME_1]` in the
  next are unrelated, and the later one overwrites the earlier in the map. A
  reply that reaches back to a placeholder from an earlier turn can therefore be
  rehydrated with the wrong value. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
* The wake word itself is masked (the model reads "Onkith" as a given name).
  `app/post_mask_assistant.py` removes that one placeholder from the outbound
  question so the sentence and the rehydration are correct, but it still
  consumes `[GIVENNAME_1]` in the stored transcript, so the first real name in
  an addressed utterance is numbered `_2`

---

> **Research Notice:** Omit AI is not a medical device. Clinical use requires independent validation and privacy review.

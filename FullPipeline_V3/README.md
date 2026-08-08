# Omit AI

**Local speech-to-text and privacy processing for Raspberry Pi 5**

Omit AI is a local English speech-to-text pipeline built for the **Raspberry Pi 5**. It uses **NVIDIA Canary 180M Flash** for transcription, **Silero VAD** for speech detection, overlap-based segmentation for continuous speech, PII masking, and local SQLite storage.

Speech recognition and privacy processing run locally. Internet access is only required during setup to download the model and for optional YouTube playback through Onkith.

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

> **Note:** The reported RTF is from the LibriSpeech benchmark and is not a Raspberry Pi 5 runtime measurement.

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

Extract the project:

```bash
unzip Omit_AI_RPi_Live_CANARY_180M_PI5_4GB_FINAL.zip
cd Omit_AI_RPi_Live
```

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

Run the test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

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
* Raspberry Pi 5 RTF, memory usage, and command latency still need to be measured on the final hardware setup

---

> **Research Notice:** Omit AI is not a medical device. Clinical use requires independent validation and privacy review.

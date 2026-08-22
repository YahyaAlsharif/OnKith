# Audio pipeline benchmark - 20260821T201401Z_linux-aarch64_audio-whisper_full

A **controlled end-to-end deployment benchmark**: audio in, sanitized text
out, on the real production pipeline. It measures whether the deployment is
practical on this device and whether private information survives imperfect
speech recognition. It is not a speech-recognition benchmark and it does not
replace the held-out text evaluation, the combined validation, `ood_dev` or
the frozen `ood_final`.

- **Clips scored:** 44 of 44 available (279.7 s of audio, full mode)
- **Speech recognizer:** UNAVAILABLE - RuntimeError: faster-whisper backend unavailable: faster_whisper import failed: ModuleNotFoundError: No module named 'faster_whisper'; ctranslate2 import failed: ModuleNotFoundError: No module named 'ctranslate2'
- **Started / finished (UTC):** 2026-08-21T20:14:01 -> 2026-08-21T20:14:06 (0.1 min)

## Device

| Field | Value |
|---|---|
| OS | Linux-6.18.39+rpt-rpi-2712-aarch64-with-glibc2.41 |
| Architecture | aarch64 (64bit) |
| CPU | 0x41 |
| Logical / physical cores | 4 / None |
| Board | Raspberry Pi 5 Model B Rev 1.0 |
| RAM | unknown |
| Privacy model SHA-256 | `bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8` |

## Benchmark audio

**synthetic privacy speech benchmark** - 32 clips, 183.7 s, 60 expected entities. License: see manifest.

**LibriSpeech dev-clean subset** - 12 clips, 96.0 s, 0 expected entities. License: CC BY 4.0.

All private information in the synthetic subset is fictional: reserved `example` domains, the 555-01xx fictional telephone range, and invented names, addresses and document numbers.

## Speech recognition

**Unavailable on this machine.** `RuntimeError: faster-whisper backend unavailable: faster_whisper import failed: ModuleNotFoundError: No module named 'faster_whisper'; ctranslate2 import failed: ModuleNotFoundError: No module named 'ctranslate2'`

```
The audio benchmark needs the production speech recognizer. Install it, then run
this script again - nothing else changes:

    pip install "faster-whisper==1.2.1"

faster-whisper runs on CTranslate2, not torch, and publishes wheels for Windows
and for Raspberry Pi OS (aarch64). The base.en weights are fetched once on first
use and cached; after that the device runs offline.
Until then the control arms below still run, so the fixtures, the privacy model
and the memory profile are all validated - only the recognizer is missing.
```

## Latency

| Stage | Statistics |
|---|---|
| audio_load | mean 0.4 ms, P50 0.3, P95 0.6, P99 0.9 (n=44) |
| vad | not measured |
| stt | not measured |
| normalization | not measured |
| privacy | not measured |
| postprocessing | not measured |
| end_to_end | mean 0.4 ms, P50 0.3, P95 0.6, P99 0.9 (n=44) |

Dominant stage: **audio_load** at 100.0% of measured processing time.

## Privacy: reference text against the audio pipeline

| Arm | Typed F1 | Entity recall | Character recall | Leakage |
|---|---:|---:|---:|---:|
| reference_text (control) | 0.5345 | 0.7333 | 0.7202 | 0.2798 |
| reference_uncased (control) | 0.4915 | 0.8167 | 0.8071 | 0.1929 |

Entity recall and character recall use one definition across all three rows: each gold entity measured over its own characters, whitespace excluded. Typed F1 needs exact gold offsets, which recognized text does not have, so it is reported for the control arms only.

Scored on the 32 privacy-annotated clips. The LibriSpeech subset carries no gold annotation - read prose is full of person names - so it contributes to recognition metrics only.

## Memory

| Stage | RSS |
|---|---|
| peak during run | unknown |
| system RAM | unknown |

## Warnings

- VAD unavailable: RuntimeError: Install onnx==1.17.0 to verify Silero opset
- STT unavailable - RuntimeError: faster-whisper backend unavailable: faster_whisper import failed: ModuleNotFoundError: No module named 'faster_whisper'; ctranslate2 import failed: ModuleNotFoundError: No module named 'ctranslate2'

## Sections completed

- clip_processing: yes
- fixture_verification: yes
- privacy_evaluation: yes

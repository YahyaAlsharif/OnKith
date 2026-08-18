# Audio pipeline benchmark - 20260818T081310Z_linux-aarch64_audio_full

A **controlled end-to-end deployment benchmark**: audio in, sanitized text
out, on the real production pipeline. It measures whether the deployment is
practical on this device and whether private information survives imperfect
speech recognition. It is not a speech-recognition benchmark and it does not
replace the held-out text evaluation, the combined validation, `ood_dev` or
the frozen `ood_final`.

- **Clips scored:** 44 of 44 available (279.7 s of audio, full mode)
- **Speech recognizer:** nvidia/canary-180m-flash
- **Started / finished (UTC):** 2026-08-18T08:13:10 -> 2026-08-18T08:23:50 (10.7 min)

## Device

| Field | Value |
|---|---|
| OS | Linux-6.18.39+rpt-rpi-2712-aarch64-with-glibc2.41 |
| Architecture | aarch64 (64bit) |
| CPU | 0x41 |
| Logical / physical cores | 4 / 4 |
| Board | Raspberry Pi 5 Model B Rev 1.0 |
| RAM | 3.95 GiB |
| Privacy model SHA-256 | `bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8` |

## Benchmark audio

**synthetic privacy speech benchmark** - 32 clips, 183.7 s, 60 expected entities. License: see manifest.

**LibriSpeech dev-clean subset** - 12 clips, 96.0 s, 0 expected entities. License: CC BY 4.0.

All private information in the synthetic subset is fictional: reserved `example` domains, the 555-01xx fictional telephone range, and invented names, addresses and document numbers.

## Speech recognition

| Metric | Value |
|---|---|
| Aggregate WER | 0.0403 |
| Mean WER | 0.0486 |
| Median WER | 0.0000 |
| Reference words | 869 |

- `public`: aggregate WER 0.0405 over 12 clips
- `synthetic_privacy`: aggregate WER 0.0402 over 32 clips

## Latency

| Stage | Statistics |
|---|---|
| audio_load | mean 8.8 ms, P50 7.8, P95 12.2, P99 17.9 (n=44) |
| vad | mean 66.2 ms, P50 52.4, P95 142.7, P99 234.9 (n=44) |
| stt | mean 13803.2 ms, P50 8963.6, P95 42756.7, P99 64605.7 (n=44) |
| normalization | mean 0.2 ms, P50 0.1, P95 0.2, P99 0.4 (n=44) |
| privacy | mean 30.4 ms, P50 26.1, P95 56.9, P99 76.9 (n=44) |
| postprocessing | mean 0.1 ms, P50 0.1, P95 0.2, P99 0.2 (n=44) |
| end_to_end | mean 13909.1 ms, P50 9041.2, P95 42962.4, P99 64933.9 (n=44) |

Dominant stage: **stt** at 99.2% of measured processing time.

### Real-time factor

`RTF < 1` means the device processes audio faster than the audio lasts, which is the condition for live use.

| Measure | Mean | P50 | P90 | P95 |
|---|---:|---:|---:|---:|
| STT | 2.022 | 1.998 | 2.666 | 2.791 |
| Full pipeline | 2.041 | 2.016 | 2.684 | 2.808 |

## Privacy: reference text against the audio pipeline

| Arm | Typed F1 | Entity recall | Character recall | Leakage |
|---|---:|---:|---:|---:|
| reference_text (control) | 0.5345 | 0.7333 | 0.7202 | 0.2798 |
| reference_uncased (control) | 0.4915 | 0.8167 | 0.8071 | 0.1929 |
| audio -> STT -> privacy | not applicable | 0.7000 | 0.8043 | 0.1957 |

Entity recall and character recall use one definition across all three rows: each gold entity measured over its own characters, whitespace excluded. Typed F1 needs exact gold offsets, which recognized text does not have, so it is reported for the control arms only.

Scored on the 32 privacy-annotated clips. The LibriSpeech subset carries no gold annotation - read prose is full of person names - so it contributes to recognition metrics only.

## End-to-end privacy outcome

| Metric | Value |
|---|---|
| Expected entities | 60 |
| Survived speech recognition | 52 (86.7%) |
| Fully masked end to end | 42 (70.0%) |
| Character recall after STT | 0.8043 |
| Character leakage after STT | 0.1957 |
| Fully sanitized clips | 14 (56.0%) |
| Clips with any leakage | 11 (44.0%) |
| Hard-negative false-positive clips | 1 of 7 |

### Failure attribution

| Cause | Entities |
|---|---:|
| Protected | 42 |
| Privacy-model failure | 10 |
| STT-caused, nothing masked | 3 |
| STT-altered but masked anyway | 5 |

Examples of both kinds are in `audio_failure_analysis.jsonl`, which also keeps successfully masked entities whose surface the recognizer changed.

### Spoken numbers rebuilt after masking

The recognizer runs with `pnc: no`, so numbers arrive as digit words. Where the privacy model missed one - already counted as a privacy-model failure above - the display renderer (`app/digit_normalizer.py`, which runs after masking by design) rewrote the surviving words into a numeral of 7+ digits:

| Clip | Entity | Released as | Digits |
|---|---|---|---:|
| privacy_003 | TELEPHONENUM | `5550147` | 7 |
| privacy_015 | CREDITCARDNUMBER | `41126683` | 8 |

2 identifier(s) across 2 clip(s). This is not additional leaked information - the digit words were already in the released text - but it is the compact, copyable form of it. Display only. app/digit_normalizer.py is documented as a render-time transform: storage, Onkith and the preview's own arithmetic keep the original text, so the reconstruction reaches the terminal and the JSON display_text field, not the database.

## Memory

| Stage | RSS |
|---|---|
| baseline | 60.0 MiB |
| vad_loaded | 94.1 MiB |
| stt_loaded | 1990.3 MiB |
| privacy_loaded | 2018.5 MiB |
| all_models_loaded | 2020.5 MiB |
| peak during run | 2030.9 MiB |
| system RAM | 3.95 GiB |

## Warnings

- none

## Sections completed

- clip_processing: yes
- fixture_verification: yes
- privacy_evaluation: yes

# Audio pipeline benchmark - 20260818T062503Z_windows-amd64_audio_full_workstation

A **controlled end-to-end deployment benchmark**: audio in, sanitized text
out, on the real production pipeline. It measures whether the deployment is
practical on this device and whether private information survives imperfect
speech recognition. It is not a speech-recognition benchmark and it does not
replace the held-out text evaluation, the combined validation, `ood_dev` or
the frozen `ood_final`.

- **Clips scored:** 44 of 44 available (279.7 s of audio, full mode)
- **Speech recognizer:** nvidia/canary-180m-flash
- **Started / finished (UTC):** 2026-08-18T06:25:03 -> 2026-08-18T06:29:07 (4.1 min)

## Device

| Field | Value |
|---|---|
| OS | Windows-10-10.0.26200-SP0 |
| Architecture | AMD64 (64bit) |
| CPU | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel |
| Logical / physical cores | 20 / 14 |
| Board | not reported |
| RAM | 15.65 GiB |
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
| audio_load | mean 1.5 ms, P50 1.3, P95 3.1, P99 3.6 (n=44) |
| vad | mean 104.8 ms, P50 77.6, P95 205.1, P99 321.7 (n=44) |
| stt | mean 4535.6 ms, P50 2851.0, P95 14388.5, P99 20232.1 (n=44) |
| normalization | mean 0.2 ms, P50 0.2, P95 0.3, P99 0.4 (n=44) |
| privacy | mean 25.1 ms, P50 23.4, P95 35.6, P99 48.5 (n=44) |
| postprocessing | mean 0.1 ms, P50 0.1, P95 0.3, P99 0.3 (n=44) |
| end_to_end | mean 4667.8 ms, P50 2948.1, P95 14618.0, P99 20606.4 (n=44) |

Dominant stage: **stt** at 97.2% of measured processing time.

### Real-time factor

`RTF < 1` means the device processes audio faster than the audio lasts, which is the condition for live use.

| Measure | Mean | P50 | P90 | P95 |
|---|---:|---:|---:|---:|
| STT | 0.646 | 0.625 | 0.884 | 1.000 |
| Full pipeline | 0.671 | 0.644 | 0.909 | 1.019 |

## Privacy: reference text against the audio pipeline

| Arm | Typed F1 | Entity recall | Character recall | Leakage |
|---|---:|---:|---:|---:|
| reference_text (control) | 0.5299 | 0.7333 | 0.7274 | 0.2726 |
| reference_uncased (control) | 0.4957 | 0.8167 | 0.8071 | 0.1929 |
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
| baseline | 48.8 MiB |
| vad_loaded | 72.7 MiB |
| stt_loaded | 2084.0 MiB |
| privacy_loaded | 2315.9 MiB |
| all_models_loaded | 2317.9 MiB |
| peak during run | 2372.9 MiB |
| system RAM | 15.65 GiB |

## Warnings

- none

## Sections completed

- clip_processing: yes
- fixture_verification: yes
- privacy_evaluation: yes

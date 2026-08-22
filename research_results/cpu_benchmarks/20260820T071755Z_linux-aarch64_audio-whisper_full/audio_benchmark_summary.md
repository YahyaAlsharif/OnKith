# Audio pipeline benchmark - 20260820T071755Z_linux-aarch64_audio-whisper_full

A **controlled end-to-end deployment benchmark**: audio in, sanitized text
out, on the real production pipeline. It measures whether the deployment is
practical on this device and whether private information survives imperfect
speech recognition. It is not a speech-recognition benchmark and it does not
replace the held-out text evaluation, the combined validation, `ood_dev` or
the frozen `ood_final`.

- **Clips scored:** 44 of 44 available (279.7 s of audio, full mode)
- **Speech recognizer:** faster-whisper base.en (ctranslate2, cpu, int8, beam 5)
- **Started / finished (UTC):** 2026-08-20T07:17:55 -> 2026-08-20T07:20:14 (2.3 min)

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
| Aggregate WER | 0.1841 |
| Mean WER | 0.1996 |
| Median WER | 0.1642 |
| Reference words | 869 |

The three rows above use the benchmark's standing definition:
Levenshtein over lowercased, punctuation-stripped words. It is
comparable across devices for one recognizer and **not** across
recognizers, because Canary wrote spoken numbers as words and
Whisper writes numerals. The engine-comparable figure folds
numbers to numerals on both sides:

| Metric | Value |
|---|---|
| Aggregate WER, number-normalised | 0.0903 |
| Mean WER, number-normalised | 0.0857 |
| Long numbers transcribed exactly | 3/5 |
| Wake-word recognition | not measurable: no clip in the corpus addresses the device by name |

- `public`: aggregate WER 0.0856 over 12 clips
- `synthetic_privacy`: aggregate WER 0.2179 over 32 clips

## Latency

| Stage | Statistics |
|---|---|
| audio_load | mean 0.3 ms, P50 0.3, P95 0.6, P99 0.8 (n=44) |
| vad | mean 67.7 ms, P50 50.7, P95 146.4, P99 244.4 (n=44) |
| stt | mean 2723.2 ms, P50 2343.3, P95 4950.1, P99 6096.6 (n=44) |
| normalization | mean 0.1 ms, P50 0.1, P95 0.2, P99 0.2 (n=44) |
| privacy | mean 24.9 ms, P50 21.2, P95 44.0, P99 57.2 (n=44) |
| postprocessing | mean 0.1 ms, P50 0.0, P95 0.2, P99 0.2 (n=44) |
| end_to_end | mean 2816.7 ms, P50 2411.2, P95 5132.0, P99 6400.2 (n=44) |

Dominant stage: **stt** at 96.7% of measured processing time.

### Real-time factor

`RTF < 1` means the device processes audio faster than the audio lasts, which is the condition for live use.

| Measure | Mean | P50 | P90 | P95 |
|---|---:|---:|---:|---:|
| STT | 0.573 | 0.520 | 0.974 | 1.028 |
| Full pipeline | 0.589 | 0.535 | 0.993 | 1.052 |

## Privacy: reference text against the audio pipeline

| Arm | Typed F1 | Entity recall | Character recall | Leakage |
|---|---:|---:|---:|---:|
| reference_text (control) | 0.5345 | 0.7333 | 0.7202 | 0.2798 |
| reference_uncased (control) | 0.4915 | 0.8167 | 0.8071 | 0.1929 |
| audio -> STT -> privacy | not applicable | 0.2500 | 0.7423 | 0.2577 |

Entity recall and character recall use one definition across all three rows: each gold entity measured over its own characters, whitespace excluded. Typed F1 needs exact gold offsets, which recognized text does not have, so it is reported for the control arms only.

Scored on the 32 privacy-annotated clips. The LibriSpeech subset carries no gold annotation - read prose is full of person names - so it contributes to recognition metrics only.

## End-to-end privacy outcome

| Metric | Value |
|---|---|
| Expected entities | 60 |
| Survived speech recognition | 30 (50.0%) |
| Fully masked end to end | 15 (25.0%) |
| Character recall after STT | 0.7423 |
| Character leakage after STT | 0.2577 |
| Fully sanitized clips | 4 (16.0%) |
| Clips with any leakage | 21 (84.0%) |
| Hard-negative false-positive clips | 1 of 7 |

### Failure attribution

| Cause | Entities |
|---|---:|
| Protected | 15 |
| Privacy-model failure | 15 |
| STT-caused, nothing masked | 21 |
| STT-altered but masked anyway | 9 |

Examples of both kinds are in `audio_failure_analysis.jsonl`, which also keeps successfully masked entities whose surface the recognizer changed.

## Memory

| Stage | RSS |
|---|---|
| baseline | 60.6 MiB |
| vad_loaded | 94.2 MiB |
| stt_loaded | 540.6 MiB |
| privacy_loaded | 1042.1 MiB |
| all_models_loaded | 1043.3 MiB |
| peak during run | 1155.1 MiB |
| system RAM | 3.95 GiB |

## Warnings

- none

## Sections completed

- clip_processing: yes
- fixture_verification: yes
- privacy_evaluation: yes

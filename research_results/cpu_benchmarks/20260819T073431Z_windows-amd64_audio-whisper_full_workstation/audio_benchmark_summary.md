# Audio pipeline benchmark - 20260819T073431Z_windows-amd64_audio-whisper_full_workstation

A **controlled end-to-end deployment benchmark**: audio in, sanitized text
out, on the real production pipeline. It measures whether the deployment is
practical on this device and whether private information survives imperfect
speech recognition. It is not a speech-recognition benchmark and it does not
replace the held-out text evaluation, the combined validation, `ood_dev` or
the frozen `ood_final`.

- **Clips scored:** 44 of 44 available (279.7 s of audio, full mode)
- **Speech recognizer:** faster-whisper base.en (ctranslate2, cpu, int8, beam 5)
- **Started / finished (UTC):** 2026-08-19T07:34:31 -> 2026-08-19T07:38:54 (4.4 min)

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
| Aggregate WER | 0.1795 |
| Mean WER | 0.1964 |
| Median WER | 0.1569 |
| Reference words | 869 |

The three rows above use the benchmark's standing definition:
Levenshtein over lowercased, punctuation-stripped words. It is
comparable across devices for one recognizer and **not** across
recognizers, because Canary wrote spoken numbers as words and
Whisper writes numerals. The engine-comparable figure folds
numbers to numerals on both sides:

| Metric | Value |
|---|---|
| Aggregate WER, number-normalised | 0.0854 |
| Mean WER, number-normalised | 0.0821 |
| Long numbers transcribed exactly | 3/5 |
| Wake-word recognition | not measurable: no clip in the corpus addresses the device by name |

- `public`: aggregate WER 0.0811 over 12 clips
- `synthetic_privacy`: aggregate WER 0.2133 over 32 clips

## Latency

| Stage | Statistics |
|---|---|
| audio_load | mean 3.6 ms, P50 3.8, P95 6.8, P99 9.1 (n=44) |
| vad | mean 447.6 ms, P50 331.1, P95 998.9, P99 1458.3 (n=44) |
| stt | mean 3841.6 ms, P50 3592.6, P95 5587.5, P99 6549.6 (n=44) |
| normalization | mean 0.3 ms, P50 0.3, P95 0.5, P99 0.6 (n=44) |
| privacy | mean 164.4 ms, P50 143.2, P95 332.6, P99 399.9 (n=44) |
| postprocessing | mean 0.3 ms, P50 0.1, P95 1.8, P99 2.4 (n=44) |
| end_to_end | mean 4459.6 ms, P50 4018.5, P95 6874.6, P99 8386.7 (n=44) |

Dominant stage: **stt** at 86.2% of measured processing time.

### Real-time factor

`RTF < 1` means the device processes audio faster than the audio lasts, which is the condition for live use.

| Measure | Mean | P50 | P90 | P95 |
|---|---:|---:|---:|---:|
| STT | 0.895 | 0.736 | 1.699 | 1.780 |
| Full pipeline | 1.001 | 0.835 | 1.821 | 1.892 |

## Privacy: reference text against the audio pipeline

| Arm | Typed F1 | Entity recall | Character recall | Leakage |
|---|---:|---:|---:|---:|
| reference_text (control) | 0.5254 | 0.7333 | 0.7226 | 0.2774 |
| reference_uncased (control) | 0.4915 | 0.8167 | 0.8071 | 0.1929 |
| audio -> STT -> privacy | not applicable | 0.2667 | 0.7505 | 0.2495 |

Entity recall and character recall use one definition across all three rows: each gold entity measured over its own characters, whitespace excluded. Typed F1 needs exact gold offsets, which recognized text does not have, so it is reported for the control arms only.

Scored on the 32 privacy-annotated clips. The LibriSpeech subset carries no gold annotation - read prose is full of person names - so it contributes to recognition metrics only.

## End-to-end privacy outcome

| Metric | Value |
|---|---|
| Expected entities | 60 |
| Survived speech recognition | 31 (51.7%) |
| Fully masked end to end | 16 (26.7%) |
| Character recall after STT | 0.7505 |
| Character leakage after STT | 0.2495 |
| Fully sanitized clips | 4 (16.0%) |
| Clips with any leakage | 21 (84.0%) |
| Hard-negative false-positive clips | 1 of 7 |

### Failure attribution

| Cause | Entities |
|---|---:|
| Protected | 16 |
| Privacy-model failure | 15 |
| STT-caused, nothing masked | 19 |
| STT-altered but masked anyway | 10 |

Examples of both kinds are in `audio_failure_analysis.jsonl`, which also keeps successfully masked entities whose surface the recognizer changed.

## Memory

| Stage | RSS |
|---|---|
| baseline | 51.4 MiB |
| vad_loaded | 77.5 MiB |
| stt_loaded | 713.8 MiB |
| privacy_loaded | 1138.6 MiB |
| all_models_loaded | 1146.3 MiB |
| peak during run | 1263.4 MiB |
| system RAM | 15.65 GiB |

## Warnings

- none

## Sections completed

- clip_processing: yes
- fixture_verification: yes
- privacy_evaluation: yes

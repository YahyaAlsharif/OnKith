# Audio pipeline benchmark - 20260820T145659Z_windows-amd64_audio-whisper_full_workstation

A **controlled end-to-end deployment benchmark**: audio in, sanitized text
out, on the real production pipeline. It measures whether the deployment is
practical on this device and whether private information survives imperfect
speech recognition. It is not a speech-recognition benchmark and it does not
replace the held-out text evaluation, the combined validation, `ood_dev` or
the frozen `ood_final`.

- **Clips scored:** 44 of 44 available (279.7 s of audio, full mode)
- **Speech recognizer:** faster-whisper base.en (ctranslate2, cpu, int8, beam 5)
- **Started / finished (UTC):** 2026-08-20T14:56:59 -> 2026-08-20T14:58:40 (1.7 min)

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

### Spoken address forms rewritten before detection

The recognizer writes an address the way it was spoken - `billing at example.org`. Neither the privacy model nor the structured EMAIL recognizer has seen one written that way, and the result was the domain masked and the local part released. `integrated_pipeline/spoken_form_normalizer.py` rewrites two such shapes into their written form for the detectors only; every span is projected back before anything is masked, so the released transcript is still the transcript.

| Clip | Rule | Recognized | Shown to the detectors |
|---|---|---|---|
| privacy_004 | `spoken_email` | `alex.carter at example.com` | `alex.carter@example.com` |
| privacy_011 | `spoken_url_path` | `example.com slash booking` | `example.com/booking` |
| privacy_013 | `spoken_email` | `billing at example.org` | `billing@example.org` |
| privacy_021 | `spoken_email` | `contracts at example.net` | `contracts@example.net` |

4 rewrite(s) across 4 clip(s).

## Latency

| Stage | Statistics |
|---|---|
| audio_load | mean 1.0 ms, P50 0.9, P95 1.4, P99 2.7 (n=44) |
| vad | mean 105.2 ms, P50 85.4, P95 221.0, P99 340.0 (n=44) |
| stt | mean 1587.5 ms, P50 1420.4, P95 2484.4, P99 2876.7 (n=44) |
| normalization | mean 0.2 ms, P50 0.2, P95 0.3, P99 0.4 (n=44) |
| privacy | mean 22.1 ms, P50 19.7, P95 35.8, P99 43.4 (n=44) |
| postprocessing | mean 0.1 ms, P50 0.1, P95 0.3, P99 0.4 (n=44) |
| end_to_end | mean 1716.6 ms, P50 1529.7, P95 2730.9, P99 3265.0 (n=44) |

Dominant stage: **stt** at 92.5% of measured processing time.

### Real-time factor

`RTF < 1` means the device processes audio faster than the audio lasts, which is the condition for live use.

| Measure | Mean | P50 | P90 | P95 |
|---|---:|---:|---:|---:|
| STT | 0.360 | 0.295 | 0.668 | 0.739 |
| Full pipeline | 0.382 | 0.317 | 0.694 | 0.766 |

## Privacy: reference text against the audio pipeline

| Arm | Typed F1 | Entity recall | Character recall | Leakage |
|---|---:|---:|---:|---:|
| reference_text (control) | 0.5254 | 0.7333 | 0.7226 | 0.2774 |
| reference_uncased (control) | 0.4915 | 0.8167 | 0.8071 | 0.1929 |
| audio -> STT -> privacy | not applicable | 0.5500 | 0.8314 | 0.1686 |

Entity recall and character recall use one definition across all three rows: each gold entity measured over its own characters, whitespace excluded. Typed F1 needs exact gold offsets, which recognized text does not have, so it is reported for the control arms only.

Scored on the 32 privacy-annotated clips. The LibriSpeech subset carries no gold annotation - read prose is full of person names - so it contributes to recognition metrics only.

## End-to-end privacy outcome

| Metric | Value |
|---|---|
| Expected entities | 60 |
| Survived speech recognition | 40 (66.7%) |
| Fully masked end to end | 33 (55.0%) |
| Character recall after STT | 0.8314 |
| Character leakage after STT | 0.1686 |
| Fully sanitized clips | 16 (64.0%) |
| Clips with any leakage | 9 (36.0%) |
| Hard-negative false-positive clips | 1 of 7 |

### Failure attribution

| Cause | Entities |
|---|---:|
| Protected | 33 |
| Privacy-model failure | 7 |
| STT-caused, nothing masked | 5 |
| STT-altered but masked anyway | 15 |

Examples of both kinds are in `audio_failure_analysis.jsonl`, which also keeps successfully masked entities whose surface the recognizer changed.

## Memory

| Stage | RSS |
|---|---|
| baseline | 52.5 MiB |
| vad_loaded | 77.9 MiB |
| stt_loaded | 713.6 MiB |
| privacy_loaded | 1138.9 MiB |
| all_models_loaded | 1146.5 MiB |
| peak during run | 1269.5 MiB |
| system RAM | 15.65 GiB |

## Warnings

- none

## Sections completed

- clip_processing: yes
- fixture_verification: yes
- privacy_evaluation: yes

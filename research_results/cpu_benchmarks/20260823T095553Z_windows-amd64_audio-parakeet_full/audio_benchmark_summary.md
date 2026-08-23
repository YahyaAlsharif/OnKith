# Audio pipeline benchmark - 20260823T095553Z_windows-amd64_audio-parakeet_full

A **controlled end-to-end deployment benchmark**: audio in, sanitized text
out, on the real production pipeline. It measures whether the deployment is
practical on this device and whether private information survives imperfect
speech recognition. It is not a speech-recognition benchmark and it does not
replace the held-out text evaluation, the combined validation, `ood_dev` or
the frozen `ood_final`.

- **Clips scored:** 44 of 44 available (279.7 s of audio, full mode)
- **Speech recognizer:** NVIDIA parakeet-tdt_ctc-110m (nemo, cpu, fp32, TDT greedy_batch)
- **Started / finished (UTC):** 2026-08-23T09:55:53 -> 2026-08-23T09:57:05 (1.2 min)

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
| Aggregate WER | 0.0644 |
| Mean WER | 0.0651 |
| Median WER | 0.0401 |
| Reference words | 869 |

The three rows above use the benchmark's standing definition:
Levenshtein over lowercased, punctuation-stripped words. It is
comparable across devices for one recognizer and **not** across
recognizers, because Canary wrote spoken numbers as words while
Whisper and Parakeet write numerals. The engine-comparable figure
folds numbers to numerals on both sides:

| Metric | Value |
|---|---|
| Aggregate WER, number-normalised | 0.0619 |
| Mean WER, number-normalised | 0.0670 |
| Long numbers transcribed exactly | 0/5 |
| Wake-word recognition | not measurable: no clip in the corpus addresses the device by name |

- `public`: aggregate WER 0.0495 over 12 clips
- `synthetic_privacy`: aggregate WER 0.0696 over 32 clips

### Spoken address forms rewritten before detection

The recognizer writes an address the way it was spoken - `billing at example.org`. Neither the privacy model nor the structured EMAIL recognizer has seen one written that way, and the result was the domain masked and the local part released. `integrated_pipeline/spoken_form_normalizer.py` rewrites two such shapes into their written form for the detectors only; every span is projected back before anything is masked, so the released transcript is still the transcript.

No clip in this selection contained a spoken address form, so the stage was a no-op.

## Latency

| Stage | Statistics |
|---|---|
| audio_load | mean 1.2 ms, P50 1.1, P95 2.0, P99 2.9 (n=44) |
| vad | mean 96.5 ms, P50 71.3, P95 200.9, P99 323.1 (n=44) |
| stt | mean 597.4 ms, P50 490.9, P95 1124.0, P99 1532.9 (n=44) |
| normalization | mean 0.2 ms, P50 0.2, P95 0.3, P99 0.3 (n=44) |
| privacy | mean 23.0 ms, P50 21.0, P95 37.7, P99 42.5 (n=44) |
| postprocessing | mean 0.1 ms, P50 0.1, P95 0.3, P99 0.3 (n=44) |
| end_to_end | mean 718.9 ms, P50 591.6, P95 1354.2, P99 1899.2 (n=44) |

Dominant stage: **stt** at 83.1% of measured processing time.

### Real-time factor

`RTF < 1` means the device processes audio faster than the audio lasts, which is the condition for live use.

| Measure | Mean | P50 | P90 | P95 |
|---|---:|---:|---:|---:|
| STT | 0.114 | 0.108 | 0.164 | 0.170 |
| Full pipeline | 0.135 | 0.128 | 0.191 | 0.196 |

## Privacy: reference text against the audio pipeline

| Arm | Typed F1 | Entity recall | Character recall | Leakage |
|---|---:|---:|---:|---:|
| reference_text (control) | 0.5299 | 0.7333 | 0.7274 | 0.2726 |
| reference_uncased (control) | 0.4957 | 0.8167 | 0.8071 | 0.1929 |
| audio -> STT -> privacy | not applicable | 0.5833 | 0.7299 | 0.2701 |

Entity recall and character recall use one definition across all three rows: each gold entity measured over its own characters, whitespace excluded. Typed F1 needs exact gold offsets, which recognized text does not have, so it is reported for the control arms only.

Scored on the 32 privacy-annotated clips. The LibriSpeech subset carries no gold annotation - read prose is full of person names - so it contributes to recognition metrics only.

## End-to-end privacy outcome

| Metric | Value |
|---|---|
| Expected entities | 60 |
| Survived speech recognition | 46 (76.7%) |
| Fully masked end to end | 35 (58.3%) |
| Character recall after STT | 0.7299 |
| Character leakage after STT | 0.2701 |
| Fully sanitized clips | 12 (48.0%) |
| Clips with any leakage | 13 (52.0%) |
| Hard-negative false-positive clips | 1 of 7 |

### Failure attribution

| Cause | Entities |
|---|---:|
| Protected | 35 |
| Privacy-model failure | 11 |
| STT-caused, nothing masked | 4 |
| STT-altered but masked anyway | 10 |

Examples of both kinds are in `audio_failure_analysis.jsonl`, which also keeps successfully masked entities whose surface the recognizer changed.

### Spoken numbers rebuilt after masking

Parakeet punctuates, but it writes a number read aloud digit by digit as digit *words* rather than as a numeral, so the V3-era shape is back under V5 after being absent under Whisper. Where the privacy model missed one - already counted as a privacy-model failure above - the display renderer (`app/digit_normalizer.py`, which runs after masking by design) rewrote the surviving words into a numeral of 7+ digits:

| Clip | Entity | Released as | Digits |
|---|---|---|---:|
| privacy_015 | CREDITCARDNUMBER | `4112683` | 7 |

1 identifier(s) across 1 clip(s). This is not additional leaked information - the digit words were already in the released text - but it is the compact, copyable form of it. Display only. app/digit_normalizer.py is documented as a render-time transform: storage, Onkith and the preview's own arithmetic keep the original text, so the reconstruction reaches the terminal and the JSON display_text field, not the database.

## Memory

| Stage | RSS |
|---|---|
| baseline | 50.4 MiB |
| vad_loaded | 72.2 MiB |
| stt_loaded | 1557.6 MiB |
| privacy_loaded | 1789.8 MiB |
| all_models_loaded | 1791.9 MiB |
| peak during run | 1846.3 MiB |
| system RAM | 15.65 GiB |

## Warnings

- none

## Sections completed

- clip_processing: yes
- fixture_verification: yes
- privacy_evaluation: yes

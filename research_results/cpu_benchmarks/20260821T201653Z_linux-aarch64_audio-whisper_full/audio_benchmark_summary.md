# Audio pipeline benchmark - 20260821T201653Z_linux-aarch64_audio-whisper_full

A **controlled end-to-end deployment benchmark**: audio in, sanitized text
out, on the real production pipeline. It measures whether the deployment is
practical on this device and whether private information survives imperfect
speech recognition. It is not a speech-recognition benchmark and it does not
replace the held-out text evaluation, the combined validation, `ood_dev` or
the frozen `ood_final`.

- **Clips scored:** 44 of 44 available (279.7 s of audio, full mode)
- **Speech recognizer:** faster-whisper base.en (ctranslate2, cpu, int8, beam 5)
- **Started / finished (UTC):** 2026-08-21T20:16:53 -> 2026-08-21T20:19:07 (2.2 min)

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
| audio_load | mean 0.4 ms, P50 0.3, P95 0.6, P99 0.8 (n=44) |
| vad | not measured |
| stt | mean 2769.2 ms, P50 2399.8, P95 5051.0, P99 6159.4 (n=44) |
| normalization | mean 0.1 ms, P50 0.1, P95 0.2, P99 0.2 (n=44) |
| privacy | mean 23.1 ms, P50 18.3, P95 46.2, P99 56.5 (n=44) |
| postprocessing | mean 0.1 ms, P50 0.0, P95 0.2, P99 0.2 (n=44) |
| end_to_end | mean 2793.1 ms, P50 2418.7, P95 5098.5, P99 6217.8 (n=44) |

Dominant stage: **stt** at 99.2% of measured processing time.

### Real-time factor

`RTF < 1` means the device processes audio faster than the audio lasts, which is the condition for live use.

| Measure | Mean | P50 | P90 | P95 |
|---|---:|---:|---:|---:|
| STT | 0.581 | 0.496 | 0.997 | 1.036 |
| Full pipeline | 0.586 | 0.500 | 1.005 | 1.044 |

## Privacy: reference text against the audio pipeline

| Arm | Typed F1 | Entity recall | Character recall | Leakage |
|---|---:|---:|---:|---:|
| reference_text (control) | 0.5345 | 0.7333 | 0.7202 | 0.2798 |
| reference_uncased (control) | 0.4915 | 0.8167 | 0.8071 | 0.1929 |
| audio -> STT -> privacy | not applicable | 0.5500 | 0.8300 | 0.1700 |

Entity recall and character recall use one definition across all three rows: each gold entity measured over its own characters, whitespace excluded. Typed F1 needs exact gold offsets, which recognized text does not have, so it is reported for the control arms only.

Scored on the 32 privacy-annotated clips. The LibriSpeech subset carries no gold annotation - read prose is full of person names - so it contributes to recognition metrics only.

## End-to-end privacy outcome

| Metric | Value |
|---|---|
| Expected entities | 60 |
| Survived speech recognition | 40 (66.7%) |
| Fully masked end to end | 33 (55.0%) |
| Character recall after STT | 0.8300 |
| Character leakage after STT | 0.1700 |
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
| peak during run | unknown |
| system RAM | unknown |

## Warnings

- VAD unavailable: RuntimeError: Install onnx==1.17.0 to verify Silero opset

## Sections completed

- clip_processing: yes
- fixture_verification: yes
- privacy_evaluation: yes

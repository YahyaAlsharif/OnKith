# CPU benchmark - 20260823T142451Z_windows-amd64_v6-whisper_full_workstation

- **Completed successfully:** yes
- **Mode:** full (2,000 of 2,000 benchmark rows)
- **Started / finished (UTC):** 2026-08-23T14:24:51 -> 2026-08-23T14:27:22 (2.5 min)

## Device

| Field | Value |
|---|---|
| OS | Windows-10-10.0.26200-SP0 |
| Architecture | AMD64 (64bit) |
| CPU | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel |
| Logical / physical cores | 20 / 14 |
| Board | not reported |
| RAM | 15.65 GiB |
| Python | 3.11.9 |
| ONNX Runtime | 1.27.0 |

## Model artifact

- Path: `FullPipeline_V6/models/deberta_pii/model_int8.onnx`
- SHA-256: `bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8`
- Size: 78.5 MiB
- Identity verified against the release records: yes
- Benchmark subset SHA-256: `7d2c370d08a5df93cf0c59c36ef7c9cbf2844995472a232bdcf1995c9e39df46`
- Predictions digest: `a03a10760efa2ce899c7752d76df6e5d31da2b7ea34a53ae9481231375b5d230`

## Privacy model quality (production path)

| Metric | Value |
|---|---|
| Typed entity F1 | 0.9452 |
| Typed entity precision | 0.9377 |
| Typed entity recall | 0.9529 |
| Binary entity F1 | 0.9577 |
| Private character recall | 0.9950 |
| Private character leakage | 0.0050 |
| Non-private overmasking | 0.0155 |
| Hard-negative FP rate | 0.2500 (4 rows) |
| Model-attributed typed F1 | 0.9045 |

## Out-of-distribution cross-check (ood_dev, model-only)

- Cases scored: 360
- Typed entity F1: 0.6882
- Release reference: 0.6882 (delta +0.0000)

## Latency

| Stage | Statistics |
|---|---|
| DeBERTa full masking call | mean 46.6 ms, P50 36.5, P95 97.2, P99 166.0 (n=2000) |
| - neural decode | mean 45.9 ms, P50 35.9, P95 95.5, P99 164.6 (n=2000) |
| - structured regex | mean 0.2 ms, P50 0.2, P95 0.5, P99 0.7 (n=2000) |
| - span merge | mean 0.1 ms, P50 0.1, P95 0.2, P99 0.3 (n=2000) |
| pipeline: vad_silero | mean 39.1 ms, P50 39.0, P95 39.3, P99 39.3 (n=5) |
| pipeline: stt_whisper | mean 1520.3 ms, P50 1486.4, P95 1609.5, P99 1625.3 (n=5) |
| pipeline: transcript_merge | mean 0.2 ms, P50 0.2, P95 0.3, P99 0.3 (n=20) |
| pipeline: wake_word_normalisation | mean 0.0 ms, P50 0.0, P95 0.0, P99 0.0 (n=20) |
| pipeline: inactivity_buffer | mean 0.1 ms, P50 0.1, P95 0.2, P99 0.2 (n=20) |
| pipeline: privacy_masking | mean 19.7 ms, P50 19.3, P95 23.2, P99 25.4 (n=20) |
| pipeline: digit_normalisation | mean 0.1 ms, P50 0.1, P95 0.1, P99 0.2 (n=20) |
| pipeline: text path end to end | mean 20.2 ms, P50 19.8, P95 23.7, P99 26.0 (n=20) |
| pipeline: assembled 3.0s utterance | 1579.5 ms (complete) |
| throughput | 21.40 rows/s |

## Memory

- Process RSS: 1380.5 MiB
- Process peak RSS: 1671.0 MiB
- Privacy stack resident (session + tokenizer + imports): 918.9 MiB
- System RAM: 15.65 GiB
- INT8 model on disk: 78.5 MiB

## Warnings

- none

## Sections completed

- deberta_benchmark: yes
- full_pipeline: yes
- ood_dev_crosscheck: yes
- pipeline_text_path: yes

# CPU benchmark - 20260823T095708Z_windows-amd64_parakeet_full

- **Completed successfully:** yes
- **Mode:** full (2,000 of 2,000 benchmark rows)
- **Started / finished (UTC):** 2026-08-23T09:57:08 -> 2026-08-23T09:59:51 (2.7 min)

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
| ONNX Runtime | 1.22.1 |

## Model artifact

- Path: `FullPipeline_V5/models/deberta_pii/model_int8.onnx`
- SHA-256: `bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8`
- Size: 78.5 MiB
- Identity verified against the release records: yes
- Benchmark subset SHA-256: `7d2c370d08a5df93cf0c59c36ef7c9cbf2844995472a232bdcf1995c9e39df46`
- Predictions digest: `b4cfc5ee8d37a94798dd7a8d8c4f11154fc6bf13a7fbf800363d351a47496d69`

## Privacy model quality (production path)

| Metric | Value |
|---|---|
| Typed entity F1 | 0.9450 |
| Typed entity precision | 0.9376 |
| Typed entity recall | 0.9526 |
| Binary entity F1 | 0.9576 |
| Private character recall | 0.9949 |
| Private character leakage | 0.0051 |
| Non-private overmasking | 0.0154 |
| Hard-negative FP rate | 0.2500 (4 rows) |
| Model-attributed typed F1 | 0.9043 |

## Out-of-distribution cross-check (ood_dev, model-only)

- Cases scored: 360
- Typed entity F1: 0.6881
- Release reference: 0.6882 (delta -0.0001)

## Latency

| Stage | Statistics |
|---|---|
| DeBERTa full masking call | mean 49.6 ms, P50 38.6, P95 105.3, P99 188.7 (n=2000) |
| - neural decode | mean 48.8 ms, P50 38.0, P95 103.6, P99 187.6 (n=2000) |
| - structured regex | mean 0.2 ms, P50 0.2, P95 0.5, P99 0.7 (n=2000) |
| - span merge | mean 0.1 ms, P50 0.1, P95 0.2, P99 0.3 (n=2000) |
| pipeline: vad_silero | mean 41.5 ms, P50 41.5, P95 43.5, P99 43.8 (n=5) |
| pipeline: stt_parakeet | mean 284.1 ms, P50 272.3, P95 328.6, P99 339.3 (n=5) |
| pipeline: transcript_merge | mean 0.2 ms, P50 0.2, P95 0.3, P99 0.3 (n=20) |
| pipeline: wake_word_normalisation | mean 0.0 ms, P50 0.0, P95 0.0, P99 0.1 (n=20) |
| pipeline: inactivity_buffer | mean 0.1 ms, P50 0.1, P95 0.2, P99 0.3 (n=20) |
| pipeline: privacy_masking | mean 19.0 ms, P50 19.1, P95 20.7, P99 22.0 (n=20) |
| pipeline: digit_normalisation | mean 0.1 ms, P50 0.1, P95 0.1, P99 0.1 (n=20) |
| pipeline: text path end to end | mean 19.4 ms, P50 19.5, P95 21.1, P99 22.5 (n=20) |
| pipeline: assembled 3.0s utterance | 345.0 ms (complete) |
| throughput | 20.14 rows/s |

## Memory

- Process RSS: 1636.3 MiB
- Process peak RSS: 2081.8 MiB
- Privacy stack resident (session + tokenizer + imports): 599.0 MiB
- System RAM: 15.65 GiB
- INT8 model on disk: 78.5 MiB

## Warnings

- none

## Sections completed

- deberta_benchmark: yes
- full_pipeline: yes
- ood_dev_crosscheck: yes
- pipeline_text_path: yes

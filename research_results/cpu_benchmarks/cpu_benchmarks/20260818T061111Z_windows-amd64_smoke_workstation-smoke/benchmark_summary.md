# CPU benchmark - 20260818T061111Z_windows-amd64_smoke_workstation-smoke

- **Completed successfully:** yes
- **Mode:** smoke (54 of 2,000 benchmark rows)
- **Started / finished (UTC):** 2026-08-18T06:11:11 -> 2026-08-18T06:13:47 (2.6 min)

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

- Path: `FullPipeline_V3/models/deberta_pii/model_int8.onnx`
- SHA-256: `bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8`
- Size: 78.5 MiB
- Identity verified against the release records: yes
- Benchmark subset SHA-256: `7d2c370d08a5df93cf0c59c36ef7c9cbf2844995472a232bdcf1995c9e39df46`
- Predictions digest: `eddcd9d4eaba3708a59ef2a0592e3d2ef56468ea325b6d814d48094948c330ff`

## Privacy model quality (production path)

| Metric | Value |
|---|---|
| Typed entity F1 | 0.9625 |
| Typed entity precision | 0.9536 |
| Typed entity recall | 0.9716 |
| Binary entity F1 | 0.9750 |
| Private character recall | 0.9983 |
| Private character leakage | 0.0017 |
| Non-private overmasking | 0.0075 |
| Hard-negative FP rate | 0.2500 (4 rows) |
| Model-attributed typed F1 | 0.9333 |

## Out-of-distribution cross-check (ood_dev, model-only)

- Cases scored: 40 (subset - smoke mode)
- Typed entity F1: 0.8571
- Release reference: 0.6882 (delta +0.1689)

## Latency

| Stage | Statistics |
|---|---|
| DeBERTa full masking call | mean 44.5 ms, P50 37.8, P95 83.3, P99 101.4 (n=54) |
| - neural decode | mean 44.0 ms, P50 37.4, P95 82.8, P99 100.4 (n=54) |
| - structured regex | mean 0.2 ms, P50 0.2, P95 0.4, P99 0.5 (n=54) |
| - span merge | mean 0.1 ms, P50 0.1, P95 0.2, P99 0.2 (n=54) |
| pipeline: vad_silero | mean 39.4 ms, P50 38.9, P95 40.5, P99 40.7 (n=3) |
| pipeline: stt_canary | mean 7023.7 ms, P50 7001.6, P95 7159.3, P99 7173.3 (n=3) |
| pipeline: transcript_merge | mean 0.2 ms, P50 0.2, P95 0.3, P99 0.3 (n=5) |
| pipeline: wake_word_normalisation | mean 0.0 ms, P50 0.0, P95 0.0, P99 0.0 (n=5) |
| pipeline: inactivity_buffer | mean 0.2 ms, P50 0.1, P95 0.3, P99 0.3 (n=5) |
| pipeline: privacy_masking | mean 19.0 ms, P50 19.2, P95 19.6, P99 19.6 (n=5) |
| pipeline: digit_normalisation | mean 0.1 ms, P50 0.1, P95 0.1, P99 0.1 (n=5) |
| pipeline: text path end to end | mean 19.5 ms, P50 19.7, P95 20.1, P99 20.1 (n=5) |
| pipeline: assembled 3.0s utterance | 7082.5 ms (complete) |
| throughput | 22.43 rows/s |

## Memory

- Process RSS: 1636.1 MiB
- Process peak RSS: 2343.2 MiB
- Privacy stack resident (session + tokenizer + imports): 598.2 MiB
- System RAM: 15.65 GiB
- INT8 model on disk: 78.5 MiB

## Warnings

- none

## Sections completed

- deberta_benchmark: yes
- full_pipeline: yes
- ood_dev_crosscheck: yes
- pipeline_text_path: yes

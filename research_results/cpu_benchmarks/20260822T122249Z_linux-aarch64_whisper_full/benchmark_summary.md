# CPU benchmark - 20260822T122249Z_linux-aarch64_whisper_full

- **Completed successfully:** yes
- **Mode:** full (2,000 of 2,000 benchmark rows)
- **Started / finished (UTC):** 2026-08-22T12:22:49 -> 2026-08-22T12:26:56 (4.1 min)

## Device

| Field | Value |
|---|---|
| OS | Linux-6.18.39+rpt-rpi-2712-aarch64-with-glibc2.41 |
| Architecture | aarch64 (64bit) |
| CPU | 0x41 |
| Logical / physical cores | 4 / None |
| Board | Raspberry Pi 5 Model B Rev 1.0 |
| RAM | unknown |
| Python | 3.11.9 |
| ONNX Runtime | 1.29.0 |

## Model artifact

- Path: `FullPipeline_V4/models/deberta_pii/model_int8.onnx`
- SHA-256: `bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8`
- Size: 78.5 MiB
- Identity verified against the release records: yes
- Benchmark subset SHA-256: `7d2c370d08a5df93cf0c59c36ef7c9cbf2844995472a232bdcf1995c9e39df46`
- Predictions digest: `37ccbbd4d870499e78f274cad1868895dec2f3eec93f320fae0097f988f3a3ac`

## Privacy model quality (production path)

| Metric | Value |
|---|---|
| Typed entity F1 | 0.9448 |
| Typed entity precision | 0.9372 |
| Typed entity recall | 0.9526 |
| Binary entity F1 | 0.9572 |
| Private character recall | 0.9949 |
| Private character leakage | 0.0051 |
| Non-private overmasking | 0.0154 |
| Hard-negative FP rate | 0.2500 (4 rows) |
| Model-attributed typed F1 | 0.9041 |

## Out-of-distribution cross-check (ood_dev, model-only)

- Cases scored: 360
- Typed entity F1: 0.6909
- Release reference: 0.6882 (delta +0.0027)

## Latency

| Stage | Statistics |
|---|---|
| DeBERTa full masking call | mean 78.9 ms, P50 48.9, P95 223.5, P99 463.0 (n=2000) |
| - neural decode | mean 78.3 ms, P50 48.4, P95 222.3, P99 462.0 (n=2000) |
| - structured regex | mean 0.2 ms, P50 0.1, P95 0.4, P99 0.8 (n=2000) |
| - span merge | mean 0.1 ms, P50 0.1, P95 0.1, P99 0.3 (n=2000) |
| pipeline: vad_silero | mean 28.9 ms, P50 28.9, P95 29.1, P99 29.1 (n=5) |
| pipeline: stt_whisper | mean 2587.5 ms, P50 2504.9, P95 2874.6, P99 2935.0 (n=5) |
| pipeline: transcript_merge | mean 0.1 ms, P50 0.1, P95 0.2, P99 0.2 (n=20) |
| pipeline: wake_word_normalisation | mean 0.0 ms, P50 0.0, P95 0.0, P99 0.0 (n=20) |
| pipeline: inactivity_buffer | mean 0.1 ms, P50 0.1, P95 0.1, P99 0.1 (n=20) |
| pipeline: privacy_masking | mean 19.3 ms, P50 19.6, P95 20.4, P99 20.4 (n=20) |
| pipeline: digit_normalisation | mean 0.0 ms, P50 0.0, P95 0.1, P99 0.1 (n=20) |
| pipeline: text path end to end | mean 19.6 ms, P50 19.8, P95 20.7, P99 20.7 (n=20) |
| pipeline: assembled 3.0s utterance | 2635.9 ms (complete) |
| throughput | 12.66 rows/s |

## Memory

- Process RSS: unknown
- Process peak RSS: unknown
- Privacy stack resident (session + tokenizer + imports): unknown
- System RAM: unknown
- INT8 model on disk: 78.5 MiB

## Warnings

- ood_dev typed F1 differs from the release number by +0.0027
- psutil is not installed; memory metrics are limited to artifact sizes

## Sections completed

- deberta_benchmark: yes
- full_pipeline: yes
- ood_dev_crosscheck: yes
- pipeline_text_path: yes

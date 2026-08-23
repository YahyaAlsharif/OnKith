# V6 integrated round-trip benchmark - 20260823T204627Z_linux-aarch64_e2e-v6-whisper_full

Audio in, spoken answer out, through the pipeline the presentation
runs. It is an **integration and latency** benchmark: recognition and
masking accuracy are measured on 44 clips by
`Full_Pipeline_Audio_Benchmark.py`, and the numbers below are not a
substitute for those.

- **Cases:** 6 clips, 27.0 s of audio
- **Pipeline:** FullPipeline_V6 (Whisper Systran/faster-whisper-base.en, DeBERTa-v3-xsmall INT8)
- **LLM:** groq openai/gpt-oss-20b - **not called** (no API key); payload audit only
- **TTS:** espeak-ng
- **Started / finished (UTC):** 2026-08-23T20:46:27 -> 2026-08-23T20:46:57 (0.5 min)

## Device

| Field | Value |
|---|---|
| OS | Linux-6.18.39+rpt-rpi-2712-aarch64-with-glibc2.41 |
| Architecture | aarch64 |
| CPU | 0x41 |
| Logical / physical cores | 4 / 4 |
| RAM | 3.95 GiB |
| Privacy model SHA-256 | `bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8` |

## Privacy round trip

Read off the bytes handed to `urllib` inside `app/llm.py`, not off an
assumption that masking worked. Every payload is in
`e2e_outbound_payloads.jsonl` verbatim, so the claim can be checked
rather than believed.

| Check | Value |
|---|---:|
| Outbound payloads captured | 6 |
| Values the masker replaced, tested | 9 |
| ...appearing in a payload | **0** |
| Gold private entities tested | 11 |
| ...spoken surface appearing in a payload | **2** |
| ...recognized surface appearing in a payload | **2** |
| ...surface altered by the recognizer before masking | 1 |
| Raw transcripts appearing in a payload | **0** |
| Leakage rate | 0.1000 |

Checks against the bytes the LLM client handed to urllib, not against the arguments a function was called with. Every side is number-folded first, so a telephone number spoken as words and written by the recognizer as digits still matches. The two headline counts answer different questions and must not be added together: `masked_but_leaked` is a value the masker replaced that left the device anyway, which is a boundary failure and must be zero; `missed_by_detector_and_in_payload` is a gold entity the privacy model never found, which is a recall miss and is why the audio benchmark exists. A pipeline can have a perfect boundary and an imperfect detector, and this run says which.

## Placeholder preservation

| Metric | Value |
|---|---:|
| Replies received | 0 |
| Replies referring to a placeholder | 0 |
| ...using only intact request placeholders | 0 |

Of the replies that used any bracketed token at all, how many used only tokens that were in the request. A reply containing [Name], [GIVENNAME] or an invented [TIME_1] counts as corrupted, because no local map can fill it. Meaningful only against a live model: a stubbed run scores the responder this file wrote.

## Local rehydration

| Metric | Value |
|---|---:|
| Deterministic probe cases | 4 |
| ...restored every value | 4 |
| Live replies referring to a placeholder | 0 |
| ...fully resolved locally | 0 |
| Malformed/unknown-placeholder probes passed | 10/10 |

## Latency

Local and network are reported apart on purpose. The first is this
device; the second is somebody's internet connection.

| Stage | Mean (ms) | P50 | P95 |
|---|---:|---:|---:|
| Whisper STT | n/a | n/a | n/a |
| Privacy masking | n/a | n/a | n/a |
| Local pre-LLM total | n/a | n/a | n/a |
| Groq round trip | n/a | n/a | n/a |
| End to end, text answer | n/a | n/a | n/a |
| Local TTS | n/a | n/a | n/a |

- API requests attempted: 6, succeeded: 0, failed: 6
- Not run: no API key was configured, so no request left this machine. The payload each request would have carried was still built by the real client and is in e2e_outbound_payloads.jsonl.
- TTS speaks not measured in this run.

## End-to-end outcome

**Successful cases: 6 / 6**

usable transcript, masking released, no raw private value in the outbound payload, deterministic rehydration probe restored every value. The four transport-dependent conditions are not counted in this run because no key was configured.

No case failed.

## Memory

| Stage | RSS |
|---|---|
| baseline | 56.4 MiB |
| stt_loaded | 783.0 MiB |
| privacy_loaded | 1433.2 MiB |
| all_models_loaded | 1434.3 MiB |
| peak during run | 1461.0 MiB |

## Warnings

- GROQ_API_KEY is not set: the outbound payload is still built and audited, but no request is sent and no live latency, placeholder-preservation or API success rate is reported.

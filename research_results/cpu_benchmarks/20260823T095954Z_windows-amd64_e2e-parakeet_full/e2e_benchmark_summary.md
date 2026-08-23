# V5 integrated round-trip benchmark - 20260823T095954Z_windows-amd64_e2e-parakeet_full

Audio in, spoken answer out, through the pipeline the presentation
runs. It is an **integration and latency** benchmark: recognition and
masking accuracy are measured on 44 clips by
`Full_Pipeline_Audio_Benchmark.py`, and the numbers below are not a
substitute for those.

- **Cases:** 24 clips, 119.2 s of audio
- **Pipeline:** FullPipeline_V5 (Parakeet nvidia/parakeet-tdt_ctc-110m, DeBERTa-v3-xsmall INT8)
- **LLM:** groq openai/gpt-oss-20b - **stubbed transport**: the request was built and parsed by the real client and answered locally. Not a Groq measurement.
- **TTS:** sapi
- **Started / finished (UTC):** 2026-08-23T09:59:54 -> 2026-08-23T10:01:09 (1.3 min)

## Device

| Field | Value |
|---|---|
| OS | Windows-10-10.0.26200-SP0 |
| Architecture | AMD64 |
| CPU | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel |
| Logical / physical cores | 20 / 14 |
| RAM | 15.65 GiB |
| Privacy model SHA-256 | `bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8` |

## Privacy round trip

Read off the bytes handed to `urllib` inside `app/llm.py`, not off an
assumption that masking worked. Every payload is in
`e2e_outbound_payloads.jsonl` verbatim, so the claim can be checked
rather than believed.

| Check | Value |
|---|---:|
| Outbound payloads captured | 24 |
| Values the masker replaced, tested | 35 |
| ...appearing in a payload | **0** |
| Gold private entities tested | 37 |
| ...spoken surface appearing in a payload | **2** |
| ...recognized surface appearing in a payload | **5** |
| ...surface altered by the recognizer before masking | 10 |
| Raw transcripts appearing in a payload | **0** |
| Leakage rate | 0.0694 |

Three independent checks against the bytes the LLM client handed to urllib. The first asserts that no value the masker replaced appears in the payload. The second asserts that no gold private surface from the manifest appears in it. The third asserts that the text the recognizer actually produced for each gold entity - located by the same alignment the audio benchmark uses - is absent too, which is what catches an entity the recognizer altered and the privacy model then missed. Every side is number-folded first, so a telephone number spoken as words and written as digits still matches.

## Placeholder preservation

| Metric | Value |
|---|---:|
| Replies received | 24 |
| Replies referring to a placeholder | 15 |
| ...using only intact request placeholders | 15 |

Of the replies that used any bracketed token at all, how many used only tokens that were in the request. A reply containing [Name], [GIVENNAME] or an invented [TIME_1] counts as corrupted, because no local map can fill it. Meaningful only against a live model: a stubbed run scores the responder this file wrote.

## Local rehydration

| Metric | Value |
|---|---:|
| Deterministic probe cases | 15 |
| ...restored every value | 15 |
| Live replies referring to a placeholder | 15 |
| ...fully resolved locally | 15 |
| Malformed/unknown-placeholder probes passed | 9/9 |

## Latency

Local and network are reported apart on purpose. The first is this
device; the second is somebody's internet connection.

| Stage | Mean (ms) | P50 | P95 |
|---|---:|---:|---:|
| Parakeet STT | n/a | n/a | n/a |
| Privacy masking | n/a | n/a | n/a |
| Local pre-LLM total | n/a | n/a | n/a |
| Groq round trip | n/a | n/a | n/a |
| End to end, text answer | n/a | n/a | n/a |
| Local TTS | n/a | n/a | n/a |

- API requests attempted: 24, succeeded: 24, failed: 0
- Stubbed transport: the request was built and parsed by the real client but answered locally, so this timing is the integration's own cost with no network in it, and the placeholder-preservation figure describes a cooperative responder rather than Groq.
- TTS speaks the locally restored answer, never the masked reply.

## End-to-end outcome

**Successful cases: 19 / 24**

usable transcript, masking released, no raw private value in the outbound payload, deterministic rehydration probe restored every value, the request succeeded (stubbed transport), no corrupted placeholder in the reply, every returned placeholder resolved, and a non-empty restored answer.

| Failure cause | Cases |
|---|---:|
| pii_leak | 5 |

## Memory

| Stage | RSS |
|---|---|
| baseline | 49.1 MiB |
| stt_loaded | 1543.7 MiB |
| privacy_loaded | 1779.6 MiB |
| all_models_loaded | 1781.7 MiB |
| peak during run | 1454.8 MiB |

## Warnings

- GROQ_API_KEY is not set: the outbound payload is still built and audited, but no request is sent and no live latency, placeholder-preservation or API success rate is reported.
- --llm-stub: the request is built and answered by a local cooperative responder, not by Groq. Latency, placeholder preservation and API success rate from this run describe the integration, not the model, and must not be reported as a measurement of Groq.

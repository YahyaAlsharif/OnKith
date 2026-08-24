# Whisper V4 assistant integration audit

Audit date: 2026-08-23

## Outcome

This package uses the supplied `FullPipeline_V4.zip` as its base and keeps its
faster-whisper `base.en` recognizer. The assistant is an additional consumer at
the existing `LivePrivacyRuntime.on_output` boundary, after fail-closed masking.

The following protected core files are SHA-256 identical to the supplied V4:

- `app/live_audio.py`
- `app/live_runtime.py`
- `app/stt_runtime.py`
- `app/canary_live_segmenter.py` (the retained V4 segmenter class name)
- `config/live_config.json`
- `integrated_pipeline/config.py`
- `integrated_pipeline/whisper_adapter.py`
- `integrated_pipeline/pii_adapter.py`
- `integrated_pipeline/pipeline.py`
- `integrated_pipeline/schemas.py`
- `integrated_pipeline/silero_vad.py`
- `integrated_pipeline/sqlite_storage.py`
- `integrated_pipeline/model_v2_masker.py`
- `integrated_pipeline/structured_pii.py`
- `integrated_pipeline/supplemental_pii.py`
- `integrated_pipeline/transcript_merger.py`

Therefore the Whisper model identity, CTranslate2 CPU INT8 runtime, beam size,
segmentation constants, masking policy, fail-closed behavior, schemas, capture
path and SQLite semantics were not edited.

## Added after masking

- `app/post_mask_assistant.py`: post-mask wake/intent routing and local actions.
- `app/llm.py`: Groq client receiving masked text only.
- `app/rehydrate.py`: local placeholder-to-value restoration.
- `app/assistant.py`: asynchronous LLM/rehydration/TTS worker.
- `app/speech.py`: local Piper, SAPI, espeak and optional Kokoro backends.
- `app/assistant_surface.py`: read-only, loopback-only, in-memory browser UI.
- `app/light.py`: optional local Raspberry Pi hardware-PWM light control.
- `app/onkith_music.py`: existing music route plus deterministic local intents.
- `app/omit_ai_live.py`: composition only; `WhisperAdapter` remains the STT.

## Privacy assertions

1. Unsafe masking results make no assistant or network call.
2. Ordinary speech without the wake word is held locally.
3. Local music/light/volume/device actions make no LLM call.
4. Groq receives `pii.masked_text` only.
5. The entity map remains local and is used only for local reply rehydration.
6. Rehydrated replies are spoken and shown locally but are not stored in SQLite.
7. The UI binds to `127.0.0.1`, accepts no transcript input and keeps state in
   process memory only.
8. No API key or `.env` file is included.

## Verification completed

- AST parsing: 73 Python files passed.
- Focused Whisper/masking/assistant suite: 115 tests passed, 1 skipped.
- Full test discovery: 398 tests passed, 2 skipped.
- Real local `python -m app.omit_ai_live --self-test`: passed, including Whisper
  `base.en`, Silero and the DeBERTa privacy model.
- Tests prove raw email text is absent from LLM input while the returned
  placeholder is rehydrated only for local TTS.
- Tests prove ordinary, blocked and local-command lanes make no LLM call.
- Tests prove the browser surface serves state on IPv4 loopback.
- Tests prove a wake alias that actually triggers the assistant is displayed as
  `Onkith` in both the terminal and browser UI, even when the masker identifies
  the alias as a name. Ordinary non-wake speech is not rewritten, and the stored
  and LLM copies remain unchanged.

## Target-device checks still required

- A real Groq request requires the user's local API key and network.
- Microphone, speaker, Chromium app mode, YouTube playback and GPIO/PWM require
  testing on the intended Raspberry Pi and connected hardware.

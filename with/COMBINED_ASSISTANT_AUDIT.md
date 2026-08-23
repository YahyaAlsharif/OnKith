# Combined package audit

Audit date: 2026-08-23

## Outcome

This package keeps the supplied strict Parakeet V4 STT/privacy core and adds an
assistant only at the existing `LivePrivacyRuntime.on_output` boundary.

The following files are SHA-256 identical to the input Parakeet package:

- `app/live_audio.py`
- `app/live_runtime.py`
- `app/stt_runtime.py`
- `config/live_config.json`
- `integrated_pipeline/config.py`
- `integrated_pipeline/parakeet_adapter.py`
- `integrated_pipeline/pii_adapter.py`
- `integrated_pipeline/pipeline.py`
- `integrated_pipeline/schemas.py`
- `integrated_pipeline/silero_vad.py`
- `integrated_pipeline/sqlite_storage.py`

Therefore the Parakeet model identity/revision/hash, TDT greedy decoder,
segmentation constants, mask policy, fail-closed behavior, schemas, database
semantics and capture path were not edited.

## Additive changes

- `app/assistant.py`, `app/llm.py`, `app/rehydrate.py`, `app/speech.py`, and
  `app/light.py` were reused from the user-supplied Turki package.
- `app/post_mask_assistant.py` owns post-mask wake/intent routing and actions.
- `app/assistant_surface.py` provides a read-only, localhost-only in-memory UI.
- `app/onkith_music.py` retains its existing music interceptor and exposes a
  separate full intent router for the post-mask layer.
- `app/omit_ai_live.py` composes the add-on and preserves the original terminal
  output before invoking it.
- `.env.example` documents secret-free local configuration. `.env` is excluded
  by `.gitignore` and the release builder.
- `install_pi.sh` adds `espeak-ng` and optional `rpi-hardware-pwm`; neither
  changes the Python STT/masking dependency pins.

## Privacy assertions

1. `PostMaskAssistant.handle_output` receives an existing `LiveOutput` only
   after the fail-closed masker has completed.
2. If `safe_to_release` is false, no assistant or network call is made.
3. Ordinary speech without the wake word is held locally.
4. Deterministic light/device/music commands run locally and do not call Groq.
5. `GroqClient.reply_to` receives `pii.masked_text` only.
6. `pii.entities` is passed only to the local `AssistantWorker` and is never an
   argument to `GroqClient`.
7. Reply rehydration and TTS happen locally. The canonical SQLite transcript
   remains masked and rehydrated conversation state is not persisted.
8. The UI binds to `127.0.0.1`, accepts no transcript POST/input, and stores its
   state only in process memory.
9. No `GROQ_API_KEY` is included. The release builder excludes `.env`.

## Verification

- Python compilation: passed.
- Focused routing/privacy regression set: 44 tests passed.
- Full test discovery: 378 tests passed, 2 skipped.
- New tests prove raw email text is absent from the LLM input while the spoken
  reply is rehydrated locally, and prove ordinary/blocked/local-command lanes
  make no LLM call.
- Archive verification is performed by `scripts/build_release_zip.py`: CRC,
  POSIX member paths, traversal rejection, executable shell bits and secret /
  cache / database exclusion.

Groq's official model catalogue was checked on 2026-08-23 and lists both
`openai/gpt-oss-120b` and `openai/gpt-oss-20b` as production models. The
official Chat Completions endpoint remains
`https://api.groq.com/openai/v1/chat/completions`.

The selected primary is `openai/gpt-oss-20b`; `openai/gpt-oss-120b` is the
fallback. Deprecated `llama-3.1-8b-instant` is not configured.

## Not claimed

- No real Groq call was made because no user API key was provided or embedded.
- Raspberry Pi GPIO/PWM, browser, speaker, microphone and sustained resource
  behavior still require testing on the target Pi.
- Local TTS can be picked up by an open microphone/speaker arrangement because
  capture was deliberately not changed. Use headphones, a directional mic, or
  system acoustic echo cancellation if this occurs.

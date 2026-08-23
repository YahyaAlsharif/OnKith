# Combined package audit

Audit dates: 2026-08-23 (assistant layer), 2026-08-23 (V5 finalization)

## Outcome

This package keeps the supplied strict Parakeet V4 STT/privacy core and adds an
assistant only at the existing `LivePrivacyRuntime.on_output` boundary.

## Finalization: `with/` became `FullPipeline_V5/`

The `with/` integration folder was read, run and measured before anything was
changed. The privacy path was found byte-identical to `FullPipeline_V4/` -
`pii_adapter.py`, `model_v2_masker.py`, `span_merger.py`, `structured_pii.py`,
`supplemental_pii.py`, `mask_policy.py` and `pipeline.py` all compare equal, and
`spoken_form_normalizer.py` differs only in line endings - so the masker did not
need re-validating, only confirming. Ten masking cases were run through both
packages and produced byte-identical output.

`FullPipeline_V5/` is a copy of `with/`. `PipelineConfig` resolves its root from
`__file__`, so nothing depended on the folder's name and no path or import
needed changing for the move. The V5 self-test passes from the new location with
the pinned checkpoint restored and the TDT decoder proved.

### Changed on the way in

| File | Reason |
|---|---|
| `models/deberta_pii/model_int8.onnx` | Added. `with/` shipped without it and relied on `scripts/download_large_models.py`; V5 carries it in place, as V4 does, and it is byte-identical to the canonical artifact. |
| `integrated_pipeline/config.py` | `PARAKEET_VERIFIED_PACKAGE_VERSIONS` added. NeMo 2.7.3 stays the pin the setup scripts install; 2.5.3 is recorded as verified so the interpreter that produced the measurements is named rather than smuggled past a check. |
| `integrated_pipeline/parakeet_adapter.py` | `_verify_package_versions` reads that allowlist; `audit()` now records the NeMo and torch versions that actually ran. Checkpoint SHA-256 verification and the TDT read-back are unchanged and still hard-fail. |
| `onkith_ui.py` | `LiveUI.boundary()` added: one line per leg of the cloud exchange, placeholders chipped in the same colours as the masked utterance. |
| `app/terminal_ui.py` | `TerminalUI.exchange()` and `TerminalUI.reply()` added. Both are display-only and never raise. `reply()` deliberately emits nothing under `--json-output`: it carries restored private values, and the JSON stream is the artifact a run can be carried away in. |
| `app/assistant.py` | Two optional callbacks, `on_exchange` and `on_restored`. The outbound line is emitted from the same string the request is built from, immediately before the call; the inbound line from the raw reply, before rehydration. Neither hook is ever given the entity map. A missing key now also produces a visible turn (`UNCONFIGURED_REPLY`) instead of silence. |
| `app/post_mask_assistant.py` | `terminal_exchange`/`terminal_reply` wiring, and `strip_masked_wake_word`. |
| `app/onkith_music.py` | `wake_word_span()` added - the same match `without_wake_word` already makes, exposed as offsets. No routing behaviour changed. |
| `app/omit_ai_live.py` | Passes `ui.exchange` and `ui.reply` to the assistant. |
| `.env.example` | Added. `START_HERE.md` and `ASSISTANT_ADDON.md` both referenced it; `with/` did not contain it. |
| `README.md`, `START_HERE.md`, `KNOWN_ISSUES.md` | Measured V4-versus-V5 numbers, the presentation command, and the four Parakeet findings from 2026-08-23. |

### The one behavioural defect found and fixed

The privacy model masks the wake word: `Onkith, remind me to call Fahad` becomes
`[GIVENNAME_1], remind me to call [GIVENNAME_2] tomorrow`. `without_wake_word`
then ran on the *masked* copy, found no wake word to strip, and the question
went out with a placeholder for a vocative. A reply echoing it was rehydrated
into the device calling the person by its own name.

Nothing leaked - the wake word was masked, not released - but the sentence was
wrong. `strip_masked_wake_word` removes exactly the placeholder whose span lies
entirely inside the wake word's span in the raw transcript, and drops it from the
map handed to rehydration. Containment is the guard: where the masker merged the
wake word into a longer span, it does nothing rather than delete real words.

### Not changed

The recognizer, the VAD, the segmenter, the masker, the span merger, the
structured recognizers, the fail-closed wrapper, the schemas, the SQLite writer,
the LLM client, the rehydration module and the TTS engine are all as `with/` had
them. `app/llm.py`, `app/rehydrate.py` and `app/speech.py` were read closely,
tested, and left alone.

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

# Whisper V4 assistant add-on

The assistant is attached only after Whisper V4's existing fail-closed privacy
gate. The protected core remains:

```text
Microphone -> Silero VAD -> faster-whisper base.en -> V4 transcript merge
           -> DeBERTa + structured PII -> fail-closed masking -> SQLite
                                                        |
                                                        v
                                                  wake routing
             ordinary speech -> held locally          |
             local command   -> local action           |
             question        -> MASKED text -> Groq LLM
                                                |
                                          masked reply
                                                |
                                  local PII rehydration
                                                |
                                           local TTS
                                                |
                                             speaker

                         localhost-only read-only UI
```

## Privacy boundary

- Unsafe or fail-closed masking results never reach the assistant or network.
- Ordinary speech without the Onkith wake word remains local.
- Music, light, volume and device-panel commands run locally without Groq.
- `GroqClient` receives only `pii.masked_text`, never the raw transcript or the
  local entity map.
- Rehydration uses the local entity map only after the reply returns.
- SQLite keeps the original V4 masked canonical transcript. Rehydrated replies
  and the UI conversation are memory-only and are not persisted.
- The read-only UI binds to `127.0.0.1`; it accepts no transcript input.

## Configure

This uses Groq, the inference service, not xAI Grok. No API key is included.

```bash
cp .env.example .env
```

Set your own `GROQ_API_KEY` in `.env`. The default model is
`openai/gpt-oss-20b`, with `openai/gpt-oss-120b` as fallback.

## Run

```bash
./run_live.sh --device 1
```

The assistant UI opens at `http://127.0.0.1:8765/`.

- `--headless`: enable routing, Groq and TTS without opening the UI.
- `--assistant-port 9000`: use another loopback UI port.
- `--no-assistant`: run the original Whisper V4 core without the add-on.

Examples:

- `Onkith, what time is it?` -> masked question to Groq, local reply/TTS.
- `Onkith, turn the light on` -> local light action.
- `Onkith, show the devices` -> local UI action.
- `Onkith, play Believer` -> local music route.

On Raspberry Pi, `install_pi.sh` installs `espeak-ng`. Windows can use its
built-in SAPI voice. A bundled Piper voice can be selected through `.env`.

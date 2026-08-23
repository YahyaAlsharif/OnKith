# Additive wake, Groq, UI and TTS layer

The Parakeet STT, Silero segmentation, DeBERTa/structured PII masking,
fail-closed behavior, transcript schemas, and SQLite database are the same
protected V4 core. The assistant is an additional consumer after masking.

```text
Microphone -> unchanged Silero/Parakeet V4 -> unchanged fail-closed masking
                                                   |
                         +-------------------------+------------------------+
                         |                                                  |
                   unchanged SQLite                                  wake routing
                                                                            |
              ordinary speech -> held locally                              |
              local command   -> local music/light/device action           |
              question        -> MASKED text only -> Groq LLM              |
                                                      |                     |
                                                masked reply                |
                                                      |                     |
                                      local entity-map rehydration <--------+
                                                      |
                                           local TTS -> speaker
                                                      |
                                            localhost-only UI
```

## API key

No API key is included. Do not paste a key into source code, chat, a ZIP, or
Git. Create it locally:

```bash
cp .env.example .env
nano .env
```

Set `GROQ_API_KEY` in `.env`. The program reads it at startup and redacts it
from Groq error messages. The service in this build is **Groq**, not xAI Grok.

Without a Groq key, Parakeet, masking, SQLite, local commands, the local UI,
and local TTS still start. Questions make no network request and the screen
states that the key is missing.

## Run

```bash
./run_live.sh --device 1
```

The assistant UI opens at `http://127.0.0.1:8765/`. Use `--headless` to keep
Groq and TTS enabled without opening a browser, `--assistant-port 9000` to use
a different localhost port, or `--no-assistant` to run the original V4 core
without this add-on.

Say examples such as:

- `Onkith, what time is it?` -> masked question to Groq, local reply/TTS.
- `Onkith, turn the light on` -> local PWM action; no LLM request.
- `Onkith, show the devices` -> local UI action; no LLM request.
- `Onkith, play Believer` -> existing local music route.
- Ordinary room speech without `Onkith` -> held locally.

The browser UI binds only to `127.0.0.1`, accepts no transcript input, and
keeps conversation state only in memory. SQLite continues to store the masked
canonical transcript; rehydrated replies are not persisted.

## Local TTS and light

`install_pi.sh` installs `espeak-ng`, so speech works without another model.
Piper can be selected through `.env` when its local model is installed.

The light action uses Raspberry Pi hardware PWM when `rpi-hardware-pwm` and
the configured PWM chip are available. For GPIO12, add this to
`/boot/firmware/config.txt`, reboot, and verify the chip number:

```text
dtoverlay=pwm,pin=12,func=4
```

```bash
ls /sys/class/pwm/
```

If no PWM lamp is configured, light commands report locally and the rest of
the pipeline continues normally.


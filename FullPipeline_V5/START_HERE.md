# START HERE

> **FullPipeline_V5 — NVIDIA Parakeet STT plus the masked LLM round trip.**
>
> The Whisper adapter is replaced by the pinned NVIDIA
> `nvidia/parakeet-tdt_ctc-110m` checkpoint, running through NeMo's TDT/RNNT
> `greedy_batch` decoder on CPU FP32. The CTC-only route is not used. The
> downstream V4 privacy, merging and storage paths are byte-identical. On top of
> that, V5 adds wake/intent routing, masked-only Groq questions, local reply
> rehydration, local TTS and a localhost assistant UI — all strictly after the
> privacy gate.
>
> V5 is the demonstration pipeline. [`../FullPipeline_V4/`](../FullPipeline_V4/)
> is still the deployment claim; see **Measured status** in
> [README.md](README.md) for why.

---

# The presentation

This is the command to run in front of an audience:

```powershell
python -m app.omit_ai_live --device 1 --show-raw
```

Run it from inside `FullPipeline_V5/`, with the interpreter that has the
Parakeet runtime installed, and replace `1` with your microphone's number
(see step 4 below). What the room sees, per addressed utterance:

```text
[RAW]        Onkith, remind me to call Fahad tomorrow.
[MASKED]     [GIVENNAME_1], remind me to call [GIVENNAME_2] tomorrow.
[-> CLOUD]   remind me to call [GIVENNAME_2] tomorrow.
[<- CLOUD]   Okay, reminder to call [GIVENNAME_2] tomorrow.
[ONKITH]     Okay, reminder to call Fahad tomorrow.
```

The two `CLOUD` lines are printed from the same strings the HTTP client is
given, immediately before and after the request. That is the demonstration:
the name is on the screen, it is not on either cloud line, and it is back in
the answer. Add `--headless` to keep the browser UI closed, or `--no-assistant`
to run the V4 core alone.

Say `Onkith` and then your sentence. Ordinary room speech without the wake word
never reaches the network at all.

---

# A. Windows

You need **Miniconda** installed once:
<https://docs.conda.io/en/latest/miniconda.html>

### 1. Open PowerShell in `FullPipeline_V5/`

Open the folder in File Explorer, then click the address bar, type `powershell`
and press Enter.

### 2. Run setup (once; NeMo and the 438 MiB checkpoint are large downloads)

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

It creates a Python 3.11 environment, installs the pinned runtime, downloads the
exact Parakeet checkpoint and verifies its revision, byte size and SHA-256. It
finishes only after proving that the active decoder is TDT, not CTC.

### 3. Configure the language model (optional but wanted for the demo)

```powershell
Copy-Item .env.example .env
notepad .env
```

Put a free Groq key in `GROQ_API_KEY` (<https://console.groq.com/keys>). Never
commit it; `.env` is gitignored and excluded from the release ZIP.

**Without a key everything still runs.** Parakeet, masking, SQLite, local
commands, the terminal UI and local TTS all start; an addressed question clears
the privacy gate, the screen shows the masked text that *would* have been sent
next to `x  cloud  · GROQ_API_KEY is not set`, and no request is made. That is
a supported way to present, and it is worth knowing about before you are
standing in front of a room with no wifi.

### 4. Find your microphone

```powershell
conda run -n omit311 python -m app.omit_ai_live --list-devices
```

Note the `[number]` of your mic. Prefer one that says
`capture_status=native_16000_hz_reported`.

### 5. Talk to it

```powershell
conda run -n omit311 python -m app.omit_ai_live --device 1 --show-raw
```

**How to speak:**
1. Stay **silent for 1 second** while it calibrates.
2. Say `Onkith`, then a sentence with a name, email or phone number in it.
3. **Pause for 3 seconds.** That silence is what triggers transcription.
4. You will see the raw transcript, the masked one, both cloud lines and the
   restored answer.

Press `Ctrl+C` to stop.

### Just want to see masking without a microphone?

```powershell
conda run -n omit311 python -m app.omit_ai_live --text-test "I am Rayyan, email rayyan@example.com"
```

---

# B. Raspberry Pi 5

Target: Raspberry Pi 5 (4 GB), 64-bit Raspberry Pi OS, Python 3.11.

> Peak process RSS for this build measured **1,846 MiB** on a workstation, about
> 577 MiB more than V4. On a 4 GB Pi that is the first thing to check.

### 1. Copy the folder to the Pi

```bash
cd FullPipeline_V5 && chmod +x *.sh
```

### 2. Install

```bash
./install_pi.sh 2>&1 | tee install.log
```

The installer uses ARM64 wheels where available and installs NeMo's ASR stack.
Allow substantial time and at least 6 GiB of free storage. Whether every pinned
wheel resolves on your Raspberry Pi OS image is verified by this step, not
assumed by the package.

### 3. Check it

```bash
./run_live.sh --self-test
```

```bash
./check_pi.sh --json pi_report.json 2>&1 | tee checkpi.log
```

### 4. Find the microphone

```bash
./run_live.sh --list-devices
```

```bash
arecord --dump-hw-params -D hw:1,0 2>&1 | tee hwparams.log
```

The second command is the only way to know whether the mic really samples at
16 kHz or whether ALSA is converting.

### 5. Configure and run

```bash
cp .env.example .env      # then put your own GROQ_API_KEY in it
./run_live.sh --device 1 --show-raw
```

The assistant UI opens at `http://127.0.0.1:8765/`. See
[ASSISTANT_ADDON.md](ASSISTANT_ADDON.md) for the exact boundary and the full
command list.

---

# What has been measured, and what has not

Measured on a Windows workstation on 2026-08-23, on the same 44 committed clips
and through the same benchmark code as the committed V4 runs:

| | V4 Whisper | V5 Parakeet |
|---|---:|---:|
| Aggregate WER | 0.1795 | **0.0644** |
| STT real-time factor | 0.360 | **0.114** |
| Character leakage after STT | **0.1686** | 0.2701 |
| Peak process RSS | **1,269.5 MiB** | 1,846.3 MiB |

Still open for this exact build:

- NeMo and torch installation on the target Raspberry Pi OS image
- checkpoint load, peak RSS and sustained thermals on a 4 GB Pi 5
- STT and full-pipeline RTF on Pi CPU
- native versus ALSA-converted 16 kHz microphone behaviour
- Pi WER on the committed and live-room fixtures
- a live Groq round trip: the integration is verified end to end against a
  stubbed transport, and the outbound payloads are audited, but no request has
  yet been sent to Groq from this repository

Do not reuse the older Whisper V4 Pi measurements for this Parakeet build. Run
`./check_pi.sh`, then the three benchmarks at the repository root, and preserve
their reports.

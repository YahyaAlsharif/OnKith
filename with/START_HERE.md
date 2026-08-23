# START HERE

> **This is the strict Parakeet STT replacement build of FullPipeline_V4.**
>
> The Whisper adapter was replaced with the pinned NVIDIA
> `nvidia/parakeet-tdt_ctc-110m` checkpoint, running through NeMo's TDT/RNNT
> `greedy_batch` decoder on CPU FP32. The CTC-only route is not used. The
> downstream V4 privacy, merging and storage paths are unchanged. This package
> adds wake/intent routing, masked Groq questions, local reply rehydration,
> local TTS and a localhost assistant UI after the privacy gate.


Two ways to run this. Pick one.

---

# A. Test it on Windows (your laptop)

You need **Miniconda** installed once:
<https://docs.conda.io/en/latest/miniconda.html>

### 1. Unzip

Right-click the ZIP → **Extract All**. You get a folder `Omit_AI_RPi_Live`.

### 2. Open PowerShell in that folder

Open the folder in File Explorer, then click the address bar, type `powershell`
and press Enter.

### 3. Run setup (once; NeMo and the 438 MiB checkpoint are large downloads)

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

It creates a Python 3.11 environment, installs the pinned runtime, downloads the
exact Parakeet checkpoint and verifies its revision, byte size and SHA-256. It
finishes only after proving that the active decoder is TDT, not CTC.

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

Replace `1` with your mic number.

**How to speak:**
1. Stay **silent for 1 second** while it calibrates.
2. Say a sentence with a name, email or phone number in it.
3. **Pause for 3 seconds.** That silence is what triggers transcription.
4. You will see the raw transcript, then the masked one.

Press `Ctrl+C` to stop.

### Just want to see masking without a microphone?

```powershell
conda run -n omit311 python -m app.omit_ai_live --text-test "I am Rayyan, email rayyan@example.com"
```

---

# B. Install it on the Raspberry Pi 5

Target: Raspberry Pi 5 (4 GB), 64-bit Raspberry Pi OS, Python 3.11.

### 1. Copy the ZIP to the Pi and unzip

```bash
unzip FullPipeline_V4_Parakeet_110M_TDT_RPi.zip
```

```bash
cd Omit_AI_RPi_Live && chmod +x *.sh
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

### 5. Run it

First create the local settings file. Do not share or commit it:

```bash
cp .env.example .env
```

Put your own `GROQ_API_KEY` in `.env`. No Groq key is included in this ZIP.

```bash
./run_live.sh --device 1 --show-raw
```

The assistant UI opens at `http://127.0.0.1:8765/`. Say `Onkith` followed by
a question. Only its masked copy goes to Groq; the entity map, rehydration and
TTS remain local. See `ASSISTANT_ADDON.md` for the exact boundary and commands.

---

# What has been measured, and what has not

The supplied winning experiment measured Parakeet at **15.12% WER / 0.0192 RTF**
on Custom 30 and **4.60% WER / 0.0190 RTF** on LibriSpeech. Those results were
produced on the experiment hardware and are not Raspberry Pi results.

Still open for this exact build:

- NeMo and torch installation on the target Raspberry Pi OS image
- checkpoint load, peak RSS and sustained thermals on a 4 GB Pi 5
- STT and full-pipeline RTF on Pi CPU
- native versus ALSA-converted 16 kHz microphone behavior
- Pi WER on the committed and live-room fixtures

Do not reuse the older Whisper V4 Pi measurements for this Parakeet build. Run
`./check_pi.sh`, then the audio and CPU benchmarks, and preserve their reports.

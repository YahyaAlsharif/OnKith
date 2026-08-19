# START HERE

> **V4 status note (documentation sync pending).**
>
> This package is **FullPipeline_V4**. It is V3's pipeline with one component
> replaced: the speech recognizer is now **faster-whisper `base.en`** on
> CTranslate2 (CPU, INT8, beam 5, with a decoding hint built from the device's
> configured names), in place of NVIDIA Canary 180M Flash. VAD, segmentation,
> transcript merging, wake-word repair, the inactivity buffer, DeBERTa-v3-xsmall
> INT8 masking, span merging, spoken-digit rendering, SQLite storage and the
> terminal UI are V3's code, unchanged.
>
> The Canary build is preserved verbatim at
> [`../archive/FullPipeline_V3/`](../archive/FullPipeline_V3/), together with the
> measurements it produced.
>
> **The prose below still describes V3 and its Canary numbers.** It is rewritten
> once the Whisper Raspberry Pi run has been done, so that the documentation is
> updated against measurements rather than twice.


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

### 3. Run setup (once, ~10 minutes, downloads ~2.5 GB)

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

It creates a Python 3.11 environment, installs the pinned runtime, downloads
the 700 MB Canary model, and finishes by printing `Self-test passed.`

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
unzip Omit_AI_RPi_Live_CANARY_180M_PI5_4GB_FINAL_FIXED.zip
```

```bash
cd Omit_AI_RPi_Live && chmod +x *.sh
```

### 2. Install (30–60 minutes; it compiles some packages)

```bash
./install_pi.sh 2>&1 | tee install.log
```

Four packages in NeMo's ASR tree have no ARM64 wheel and are compiled on the
device: `ctc_segmentation`, `texterrors`, `kaldi-python-io`, `sox`. This is
why it takes a while. If it runs out of memory, increase swap first:

```bash
sudo dphys-swapfile swapoff && sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile && sudo dphys-swapfile setup && sudo dphys-swapfile swapon
```

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

```bash
./run_live.sh --device 1 --show-raw
```

---

# What is still unproven

Running it is not the same as validating it. These remain open until measured
on the Pi:

- real-time factor, peak RAM, thermals on a 4 GB Pi 5
- native versus converted 16 kHz microphone capture
- transcript parity against the validated notebook (which used NeMo 2.7.3 and
  Silero 6.2.1 - both differ from what is pinned here)

See `PI5_VALIDATION_REPORT.md` for the full pending list and
`BACKEND_VERSION_DECISION.md` for the version evidence.

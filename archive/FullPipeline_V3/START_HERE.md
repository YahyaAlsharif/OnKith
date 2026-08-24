# START HERE

> **Superseded. This is Pipeline V3, the Canary build, kept as a baseline.**
>
> The current deployment is [`../FullPipeline_V4/`](../FullPipeline_V4/), which
> is this pipeline with faster-whisper `base.en` in place of NVIDIA Canary 180M
> Flash. V3 is kept intact in the archive because the measurements in
> [`../../research_results/`](../../research_results/) are its, and the
> V3-against-V4 comparison is one of the project's results. Nothing here is
> maintained, and its numbers are never regenerated.
>
> Running it again would need the NeMo stack reinstalled and the 700 MB `.nemo`
> artifact re-fetched; see `BACKEND_VERSION_DECISION.md`.

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

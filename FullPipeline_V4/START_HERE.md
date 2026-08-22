# START HERE

> **This package is FullPipeline_V4.**
>
> It is V3's pipeline with one component replaced: the speech recognizer is
> **faster-whisper `base.en`** on CTranslate2 (CPU, INT8, beam 5, with a decoding
> hint built from the device's configured names), in place of NVIDIA Canary 180M
> Flash. VAD, segmentation, transcript merging, wake-word repair, the inactivity
> buffer, DeBERTa-v3-xsmall INT8 masking, span merging, spoken-digit rendering,
> SQLite storage and the terminal UI are V3's code, plus one addition described
> in `README.md`.
>
> The Canary implementation and artifacts are preserved at
> [`../archive/FullPipeline_V3/`](../archive/FullPipeline_V3/), together with the
> measurements it produced. Both have now been benchmarked on a Raspberry Pi 5;
> the comparison is in [`../research_results/`](../research_results/).


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

It creates a Python 3.11 environment, installs the pinned runtime, and
finishes by printing `Self-test passed.` The `base.en` weights (about 75 MB) are
fetched on first use and cached under `~/.cache/huggingface`.

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
unzip Omit_AI_RPi_Live_WHISPER_BASE_EN_PI5_4GB.zip
```

```bash
cd Omit_AI_RPi_Live && chmod +x *.sh
```

### 2. Install

```bash
./install_pi.sh 2>&1 | tee install.log
```

faster-whisper and CTranslate2 both ship ARM64 wheels, so this is an install
rather than a build. V3 needed 30–60 minutes here because four packages in
NeMo's ASR tree had to be compiled on the device; none of that applies now.

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

# What has been measured, and what has not

Measured on a Raspberry Pi 5 Model B Rev 1.0 (4 GB), from committed audio
fixtures, by `Full_Pipeline_Audio_Benchmark.py` and
`Full_Pipeline_CPU_Benchmark.py`:

- mean full-pipeline real-time factor **0.589** (P95 1.05)
- peak process RSS **1,155 MiB** of 3.95 GiB
- aggregate word error rate 0.1841 raw, 0.0903 number-normalised
- privacy model, 2,000 committed rows: typed F1 0.9448

Still open:

- **live microphone capture on the Pi.** Everything above is measured from
  committed WAV fixtures. Native versus ALSA-converted 16 kHz capture, and what
  a real room does to it, are not characterised.
- **thermals and sustained load.** The benchmark is a few minutes of work; a
  throttling Pi is not measured.
- **one complete current-path Pi profile.** The current-path Pi audio run
  measured the spoken-address rewrite, STT and privacy, reporting 0.8300 private
  character recall and 16 of 25 privacy clips fully sanitized. That environment
  lacked `onnx` and `psutil`, so VAD and memory still come from the earlier
  complete V4 run on the same board.

`PI5_VALIDATION_REPORT.md` and `BACKEND_VERSION_DECISION.md` in
[`../archive/FullPipeline_V3/`](../archive/FullPipeline_V3/) are the V3 records and describe the
NeMo stack, not this one.

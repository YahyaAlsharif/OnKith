# OnKith — live speech privacy pipeline (deployment package)

This is the standalone deployment package for the OnKith live pipeline. The
target device is a 64-bit Raspberry Pi 5 with 4 GB RAM running Raspberry Pi OS
with Python 3.11 or newer and a USB or ALSA microphone. The same package runs
natively on Windows for development and testing — **WSL is not required**.

The runtime is:

```text
continuous microphone
→ local speech segmentation
→ Moonshine v2 Medium Streaming
→ finalized transcript fragments only
→ fixed 3.0-second transcript inactivity buffer
→ TinyBERT-4 INT8 ONNX plus structured PII recognition
→ whole-word-safe span merge
→ masked terminal text
```

By default the terminal shows only masked text. It does not show partial text or
raw finalized text, and it does not save audio, transcripts, or log files.

## Dependencies

The runtime deliberately depends on `tokenizers` — the low-level Rust tokenizer
— rather than `transformers`. The Hugging Face auto-tokenizer imports the whole
`transformers` stack, which pulls in PyTorch whenever it is installed. Measured
on an x86 laptop, that difference is **999 MiB → 324 MiB resident** and a model
load time of **30.5 s → 1.4 s**. On a 4 GB Pi that memory belongs to the
Moonshine speech model.

Pinned in [requirements-rpi.txt](requirements-rpi.txt):

| Package | Version | Note |
|---|---|---|
| `moonshine-voice` | `0.0.69` | Linux ARM64 and Windows AMD64 wheels both published |
| `onnxruntime` | `1.27.0` | CPython 3.11 Linux ARM64 wheel |
| `tokenizers` | `0.22.2` | tokenization only, no PyTorch |
| `numpy` | `2.0.2` | |
| `sounddevice` | `0.5.5` | bundles PortAudio |

There is no `armv7l` wheel for `moonshine-voice`, so a 32-bit Raspberry Pi OS
image will not work.

---

## Windows

```powershell
cd Omit_AI_RPi_Live
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-rpi.txt
```

Use the PowerShell launcher, which resolves the project root and virtual
environment for you:

```powershell
.\run_live.ps1 --self-test
.\run_live.ps1 --text-test "My email is rayyan@example.com"
.\run_live.ps1 --list-devices
.\run_live.ps1 --device 2 --once --show-raw
.\run_live.ps1 --device 2
```

If script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Never run the `.sh` installers under PowerShell; `run_live.ps1` covers the
Windows path.

---

## Raspberry Pi OS

Confirm the architecture first — `aarch64` is required:

```bash
uname -m
getconf LONG_BIT
python3 -V
```

Install:

```bash
cd ~/Omit_AI_RPi_Live
chmod +x install_pi.sh run_live.sh check_pi.sh
./install_pi.sh --check-only     # static checks, installs nothing
./install_pi.sh
```

The installer uses a project-local `.venv` and never touches the system Python
with `--break-system-packages`. Internet access is needed once, to download
wheels and cache the English Medium Streaming Moonshine model. The packaged
TinyBERT model and tokenizer are read directly from disk. Normal use works
offline after installation.

Add your user to the `audio` group if it is not already a member, then log out
and back in:

```bash
sudo usermod -aG audio "$USER"
```

## Select and test a microphone

```bash
./run_live.sh --list-devices
./check_pi.sh
./check_pi.sh --with-mic --device 2
```

Replace `2` with the input-device number shown on your machine. The level check
prints RMS and peak only; it does not save audio.

## Run

```bash
./run_live.sh --device 2 --once --show-raw
./run_live.sh --device 2
./run_live.sh --device 2 --once
./run_live.sh --device 2 --show-raw --show-partials
./run_live.sh --text-test "My email is rayyan@example.com"
./run_live.sh --print-config
./run_live.sh --self-test
./run_live.sh --device 2 --json-output
./run_live.sh --device 2 --quiet
./run_live.sh --device 2 --no-colour
```

Use Ctrl+C for a clean shutdown. Pending finalized text is sent through the
privacy masker before shutdown. A live utterance in progress is finalized and
marked unreliable.

Expected states are `[CALIBRATION]` while room noise is measured, `[READY]`
while listening, `[SPEECH]` after confirmed voice energy, `[STT]` after a fresh
Moonshine stream finalizes, `[BUFFER]` while complete fragments may merge,
`[MASKED]` on a safe release, and `[BLOCKED]` if privacy processing fails.

`--show-raw` and `--show-partials` are explicit local terminal debugging
options. They do not enable file logging. Do not use them where people can see
the terminal.

## Offline validation

Neither of these needs a microphone, Moonshine, or a network connection:

```bash
python scripts/test_pii_offline.py            # 80 checks
python scripts/test_pii_offline.py --verbose
./check_pi.sh                                 # adds Moonshine + device checks
```

`test_pii_offline.py` covers entity coverage, clean-text preservation,
inputs longer than the 512-token context, PII straddling a window boundary,
fail-closed behaviour, the 67-label ONNX contract, and asserts that loading the
PII model does **not** import PyTorch.

## Monitor CPU and memory

```bash
watch -n 1 'ps -o pid,rss,pcpu,comm -C python3 --sort=-rss | head'
free -m
vcgencmd measure_temp
vcgencmd get_throttled          # 0x0 means no throttling
/usr/bin/time -v ./run_live.sh --device 2 --once 2>&1 | grep -E "Maximum resident|Elapsed"
```

If `Maximum resident set size` approaches 2 GB, check that PyTorch has not been
pulled into the virtual environment: `pip list | grep -i torch` should be empty.

---

## What remains active

`sounddevice.InputStream` opens once and remains open while its callback copies
100 ms mono float32 chunks, calculates RMS/peak, and performs a nonblocking
bounded-queue write. Moonshine, finalization, the text timer, TinyBERT, and
terminal rendering never execute in the audio callback.

One audio worker performs calibration, keeps 0.4 seconds of pre-roll, confirms
speech start, feeds Moonshine, and applies the endpoint rule. The immutable
Moonshine model resource loads once. Each utterance receives a fresh SDK stream,
listener, collector, and callback state. Concurrent Moonshine streams are
rejected because SDK thread safety is not assumed.

If the bounded audio queue overflows, the program prints a visible warning,
increments its counter, and marks the affected transcript unreliable.

Shutdown waits a bounded 10 seconds for the audio worker. If a Moonshine call
wedges, the pipeline reports `[WARNING]` and exits rather than hanging.

## Fixed transcript timer

A finalized, nonempty Moonshine fragment starts a monotonic wall-clock deadline.
A later fragment before 3.0 seconds is appended and resets the deadline. An
arrival at or after the deadline flushes the previous buffer first. The timer
thread also flushes when no later fragment arrives. Token and duration caps are
safety bounds.

TinyBERT receives only the complete flushed buffer, never partial Moonshine
updates. PII inference runs on its own worker, so the microphone remains active.

The synthetic merging and separation unit tests only verify this fixed buffer
logic. They do not validate microphone behavior and do not prove that the
configured timeout is optimal.

## Privacy behavior

TinyBERT uses one packaged INT8 ONNX artifact, `CPUExecutionProvider`, 512-token
windows, 128-token overlap, and 67 BIO labels. Inputs longer than the context
window are re-encoded as overlapping windows and every token is scored; long
text is never silently truncated. Subword evidence is consolidated to whole
words, so an entity such as an email address is masked in full rather than in
fragments. Structured recognition covers email and explicitly cued phone,
account, postal, and identifier values. Exact structured spans take priority
over overlapping model fragments; remaining spans are deterministically numbered
and replaced without changing unrelated text.

Structured recognition is **cue-dependent**: it fires on phrases such as "my
phone number is" or "call me at". A numeric identifier introduced some other
way, for example `"Call Dr Sarah Chen at 555 123 4567"`, falls back to the model
alone and is currently missed. Treat uncued numeric identifiers as a known
recall gap rather than a covered case.

Masking fails closed. If model loading, inference, alignment, or span processing
fails, the terminal prints:

```text
[BLOCKED] PII masking failed; transcript was not released
```

and the masked value becomes `[BLOCKED: PII MASKING FAILED]`. The original
transcript is never substituted as a masked result.

## Troubleshooting

- No devices: run `arecord -l`, reconnect the USB microphone, and confirm the
  user belongs to the `audio` group.
- Wrong device ID: rerun `--list-devices`; IDs may change after reconnecting.
- `Microphone N cannot supply 1-channel float32 audio`: the device rejected the
  requested format. Pick another device ID from `--list-devices`.
- ALSA warnings: verify `arecord -l`, the `audio` group, and the chosen sample
  device before treating a harmless ALSA probe warning as a model failure.
- `PortAudioError`: select the explicit device ID and rerun the RMS check.
- ONNX Runtime install failure: confirm 64-bit Raspberry Pi OS, Python 3.11 or
  newer, current `pip`, and at least 2 GiB free disk, then rerun the installer.
- Low detection: stay silent during the 2.0-second calibration and speak after
  `[READY]`.
- Queue overflow: close heavy programs; the affected output is flagged as
  unreliable rather than silently accepted.
- Missing TinyBERT model: recopy the deployment folder and confirm
  `models/tinybert_pii/model_int8.onnx` is larger than 1 MB.
- Missing model cache while offline: reconnect once and rerun `./install_pi.sh`.
- Insufficient RAM: stop desktop/browser workloads or use a lighter OS image;
  do not add swap-heavy benchmarking to the live service.
- Slow RTF: close other CPU-heavy processes, check Pi cooling/throttling, and
  keep the accepted model/configuration unchanged while diagnosing.

## Optional systemd service

No unit file is shipped. Create one only after the pipeline runs correctly
interactively, and edit the user, paths, and device number to match your system:

```ini
# /etc/systemd/system/onkith-live.service
[Unit]
Description=OnKith live speech privacy pipeline
After=sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Omit_AI_RPi_Live
ExecStart=/home/pi/Omit_AI_RPi_Live/run_live.sh --device 2
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now onkith-live.service
journalctl -u onkith-live.service -f
```

Do not add `--show-raw` or `--show-partials` to a service.

## Deliberately excluded research components

This deployment package excludes notebooks, training artifacts, downloaded
speech corpora, LibriSpeech loaders, Kaggle paths, Hugging Face `datasets`, WER
benchmark code, pandas, matplotlib, scikit-learn, seqeval, PyTorch, FP32 ONNX
files, model caches, and raw audio.

## Hardware verification boundary

Static checks and model-independent tests run anywhere. Actual microphone
levels, sustained throughput, thermal behavior, USB stability, and live latency
must be verified on the target Raspberry Pi 5 and microphone.

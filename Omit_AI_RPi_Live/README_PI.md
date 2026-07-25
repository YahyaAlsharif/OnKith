# Omit AI — Raspberry Pi 5 live privacy pipeline

This is the standalone deployment package for a 64-bit Raspberry Pi 5 with
4 GB RAM, Raspberry Pi OS, Python 3.11, and a USB or ALSA microphone.

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

The default terminal does not show partial text or raw finalized text, save
audio, save transcripts, or create log files.

## Install

Copy and extract the folder on the Pi, then run:

```bash
unzip Omit_AI_RPi_Live_Deployment.zip
cd Omit_AI_RPi_Live
chmod +x install_pi.sh run_live.sh check_pi.sh scripts/*.sh
./install_pi.sh
```

The installer uses a local `.venv`; it never modifies the system Python with
`--break-system-packages`. Internet access is needed once to install wheels and
cache the explicit English Medium Streaming Moonshine model. The packaged
TinyBERT model and tokenizer are verified with offline flags. Normal use works
offline after installation.

The pinned Linux ARM64 assumptions are:

- `moonshine-voice==0.0.69` has a Linux ARM64 wheel compatible with current
  64-bit Raspberry Pi OS.
- `onnxruntime==1.27.0` has a CPython 3.11 Linux ARM64 wheel.
- The model was tested with the exact versions in `requirements-rpi.txt`.

## Select and test a microphone

```bash
./run_live.sh --list-devices
./check_pi.sh
./check_pi.sh --with-mic --device 2
```

Replace `2` with the input-device number shown on your Pi. The level check
prints RMS and peak only; it does not save audio.

## Run

```bash
./run_live.sh --device 2 --once --show-raw
./run_live.sh --device 2
./run_live.sh --device 2 --once
./run_live.sh --device 2 --show-raw
./run_live.sh --device 2 --show-raw --show-partials
./run_live.sh --text-test "My email is rayyan@example.com"
./run_live.sh --print-config
./run_live.sh --self-test
./run_live.sh --device 2 --json-output
./run_live.sh --device 2 --quiet
./run_live.sh --device 2 --no-colour
```

Use Ctrl+C for a clean shutdown. Pending finalized text is immediately sent
through the privacy masker before shutdown. A live utterance in progress is
finalized and marked unreliable.

Expected states are `[CALIBRATION]` while room noise is measured, `[READY]`
while listening, `[SPEECH]` after confirmed voice energy, `[STT]` after a fresh
Moonshine stream finalizes, `[BUFFER]` while complete fragments may merge,
`[MASKED]` on a safe release, and `[BLOCKED]` if privacy processing fails.

`--show-raw` and `--show-partials` are explicit local terminal debugging
options. They do not enable file logging. Do not use them where people can see
the terminal.

## What remains active

`sounddevice.InputStream` opens once and remains open while its callback copies
100 ms mono float32 chunks, calculates RMS/peak, and performs a nonblocking
bounded-queue write. Moonshine, finalization, the text timer, TinyBERT, and
terminal rendering never execute in the audio callback.

One audio worker performs calibration, keeps 0.4 seconds of pre-roll, confirms
speech start, feeds Moonshine, and applies the endpoint rule. The immutable
Moonshine model resource loads once. Each utterance receives a fresh SDK
stream, listener, collector, and callback state. Concurrent Moonshine streams
are rejected because SDK thread safety is not assumed.

If the bounded audio queue overflows, the program prints a visible warning,
increments its counter, and marks the affected transcript unreliable.

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
windows, 128-token overlap, and 67 BIO labels. Subword evidence is consolidated
to whole words. Structured recognition covers email and explicitly cued phone,
account, postal, and identifier values. Exact structured spans take priority
over overlapping model fragments; remaining spans are deterministically
numbered and replaced without changing unrelated text.

Masking fails closed. If model loading, inference, alignment, or span processing
fails, the terminal prints:

```text
[BLOCKED] PII masking failed; transcript was not released
```

The original transcript is never substituted as a masked result.

## Troubleshooting

- No devices: run `arecord -l`, reconnect the USB microphone, and confirm the
  user belongs to the `audio` group.
- Wrong device ID: rerun `--list-devices`; IDs may change after reconnecting.
- ALSA warnings: verify `arecord -l`, the `audio` group, and the chosen sample
  device before treating a harmless ALSA probe warning as a model failure.
- `PortAudioError`: select the explicit device ID and rerun the RMS check.
- ONNX Runtime install failure: confirm 64-bit Raspberry Pi OS, Python 3.11,
  current `pip`, and at least 2 GiB free disk, then rerun the installer.
- Low detection: stay silent during the 2.0-second calibration and speak after
  `[READY]`.
- Queue overflow: close heavy programs; the affected output is flagged as
  unreliable rather than silently accepted.
- Missing TinyBERT model: recopy the deployment ZIP and confirm
  `models/tinybert_pii/model_int8.onnx` is larger than 1 MB.
- Missing model cache while offline: reconnect once and rerun
  `./install_pi.sh`.
- Insufficient RAM: stop desktop/browser workloads or use a lighter OS image;
  do not add swap-heavy benchmarking to the live service.
- Slow RTF: close other CPU-heavy processes, check Pi cooling/throttling, and
  keep the accepted model/configuration unchanged while diagnosing.
- Service audio permission: run interactively first and verify the service user
  belongs to the `audio` group.

## Optional systemd service

The example is not installed automatically. First edit its user, paths, and
microphone device:

```bash
sudo cp systemd/omit-ai-live.service.example /etc/systemd/system/omit-ai-live.service
sudo systemctl daemon-reload
sudo systemctl enable --now omit-ai-live.service
journalctl -u omit-ai-live.service -f
```

Do not add raw or partial debug flags to a service.

## Deliberately excluded research components

This deployment ZIP excludes notebooks, training artifacts, downloaded speech
corpora, LibriSpeech loaders, Kaggle paths, Hugging Face `datasets`, WER
benchmark code, pandas, matplotlib, scikit-learn, seqeval, PyTorch artifacts,
FP32 ONNX files, model caches, and raw audio.

## Hardware verification boundary

Static checks and model-independent tests can run elsewhere. Actual microphone
levels, sustained throughput, thermal behavior, USB stability, and live latency
must be verified on the target Raspberry Pi 5 and microphone.

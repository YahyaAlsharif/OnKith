# Onkith terminal (Turki)

The voice terminal: wake word, recogniser, PII masking, the assistant, the
player surface, and the lamp. Runs on a Raspberry Pi 5.

**Start with [HANDOFF.md](HANDOFF.md).** It carries the recogniser
decision and the evidence for it, everything the device answers to, the
open issues, and a section written for Faris on bringing it up on the Pi.

```bash
./install_pi.sh          # everything, including both recognisers
./run_live.sh --stt whisper
./install_desktop.sh     # icon on the desktop, and start on boot
```

## Two things to know before reading it

**The recogniser is Whisper base.en, and the config still says
moonshine.** Every launcher passes `--stt whisper` explicitly, so the
icon, the autostart entry and the command above all run Whisper, while a
bare `./run_live.sh` runs Moonshine. That is deliberate: the default does
not move until base.en has been timed on the Pi, which has not happened.
HANDOFF.md explains what the measurement decides.

**The evaluation corpus is not in this folder.** `eval/` has the scorer,
the manifest, the harness and the recorded results, but not the audio.
This repository's `.gitignore` says recorded speech is never committed
and ignores `*.wav`, so the 48 clips were left out on purpose rather than
forced in. Half of them are recordings of a real person.

To run `eval/run_eval.py` you need the clips in `eval/corpus/clips/`.
Either ask Turki for the corpus, or record your own against the same
script:

```bash
.venv/bin/python eval/record.py --speaker faris
```

`eval/results/history.jsonl` still has every score that has been taken,
and `eval/report.py` prints them, so the numbers in HANDOFF.md can be
checked without the audio.

## Canary

`eval/canary_engine.py` scores the recogniser from `FullPipeline_V3` on
the same corpus and the same metrics as the other two, using that build's
pinned artifact with its sha256 checked. Nothing in this folder imports
torch; it is run from a separate interpreter. The result and the four
reasons it was not adopted are in HANDOFF.md, along with the one case
where it is clearly the better choice.

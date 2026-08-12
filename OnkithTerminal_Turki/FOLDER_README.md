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

**Everything needs a `.env`, and it goes in this folder.** Copy
`.env.example` to `.env` beside `onkith` and `run_live.sh`, and fill in
`GROQ_API_KEY` and `YOUTUBE_API_KEY`. It is read from disk rather than
from the shell because the device is started from an icon and from boot
as well as from a terminal, and a launcher inherits nothing. Without it
the device still hears, masks and shows the player, and says it cannot
answer or search.

**The evaluation corpus is here**, including the recorded half, as a
deliberate exception to this repository's rule against committing
recorded speech. The exception and the reason are written into
`.gitignore` in this folder. 24 of the 48 clips are one person reading a
fixed script; the numbers in them are placeholders and are nobody's real
details.

So `eval/run_eval.py` runs as it stands:

```bash
.venv/bin/python eval/run_eval.py --engine whisper
```

To add a second speaker, which the corpus wants and does not have:

```bash
.venv/bin/python eval/record.py --speaker faris
```

## Canary

`eval/canary_engine.py` scores the recogniser from `FullPipeline_V3` on
the same corpus and the same metrics as the other two, using that build's
pinned artifact with its sha256 checked. Nothing in this folder imports
torch; it is run from a separate interpreter. The result and the four
reasons it was not adopted are in HANDOFF.md, along with the one case
where it is clearly the better choice.

"""Regenerate the synthetic half of the corpus.

Synthetic clips are one machine's voice reading a fixed script, so they are
reproducible and cheap. They are a floor on difficulty, never a substitute
for the recorded half: no room, no noise, one accent. Anything that fails
here fails worse in the room.

Run from the project root:  .venv/bin/python eval/corpus/build_synthetic.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIPS = HERE / "clips"
VOICE = "Daniel"
RATE = "175"

# (id, transcript, expect_wake, expect_intent, must_contain)
SCRIPT = [
    # Waking it, one per lane.
    ("wake-play-song", "Onkith play bones", True, "PLAY", []),
    ("wake-play-artist", "Onkith play bones by imagine dragons", True, "PLAY", []),
    ("wake-play-vague", "Onkith put on some jazz", True, "PLAY", []),
    ("wake-stop", "Onkith stop the music", True, "CLOSE", []),
    ("wake-stop-bare", "Onkith stop", True, "CLOSE", []),
    ("wake-show", "Onkith show the player", True, "SHOW", []),
    ("wake-hide", "Onkith close the player", True, "HIDE", []),
    ("wake-ask-time", "Onkith what time is it in Riyadh", True, "ASK", []),
    ("wake-ask-open", "Onkith what is the weather", True, "ASK", []),
    # The other names it answers to.
    ("wake-echo", "Echo what time is it", True, "ASK", []),
    ("wake-sada", "Sada stop the music", True, "CLOSE", []),
    # Must NOT wake it. A device that talks when nobody called it is worse
    # than one that misses a call, and this half of the corpus is the half
    # that usually gets left out.
    ("quiet-ordinary-1", "the quick brown fox jumps over the lazy dog", False, None, []),
    ("quiet-ordinary-2", "please turn the lights on in the kitchen", False, None, []),
    ("quiet-ordinary-3", "the meeting has been moved to the conference room upstairs", False, None, []),
    ("quiet-ordinary-4", "I would like to know the weather forecast for tomorrow", False, None, []),
    ("quiet-near-miss-1", "someone named Ankit called me this morning", False, None, []),
    ("quiet-near-miss-2", "I left it on the kitchen counter", False, None, []),
    ("quiet-near-miss-3", "the jar is empty and so is the tin", False, None, []),
    # Digits and names have to survive intact. Moonshine was observed
    # inventing extra digits inside a phone number, which is worse than a
    # mishearing in a device whose claim is that it catches phone numbers.
    ("pii-phone", "my number is 0501234567", False, None, ["0501234567"]),
    ("pii-name-phone", "my name is Turki and my number is 0501234567", False, None, ["0501234567"]),
    ("pii-id", "my id card number is 1098765432", False, None, ["1098765432"]),
    ("pii-email", "you can email me at turki at example dot com", False, None, []),
    # Longer, to catch drift and repetition.
    ("long-1", "I need to book a flight to London next Tuesday morning and a hotel near the station for three nights", False, None, []),
    ("long-2", "Onkith play something calm and quiet that will help me concentrate while I am working this afternoon", True, "PLAY", []),
]


def synthesise() -> list[dict]:
    if sys.platform != "darwin":
        raise SystemExit("Synthetic clips are built with macOS say. Commit the WAVs instead.")
    CLIPS.mkdir(parents=True, exist_ok=True)
    entries = []
    for clip_id, transcript, wakes, intent, contains in SCRIPT:
        aiff = CLIPS / f"{clip_id}.aiff"
        wav = CLIPS / f"{clip_id}.wav"
        subprocess.run(
            ["say", "-v", VOICE, "-r", RATE, "-o", str(aiff), transcript], check=True
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
             str(aiff), str(wav)],
            check=True,
        )
        aiff.unlink()
        entries.append(
            {
                "id": clip_id,
                "file": f"clips/{wav.name}",
                "transcript": transcript,
                "expect_wake": wakes,
                "expect_intent": intent,
                "must_contain": contains,
                "voice": f"synthetic:{VOICE}@{RATE}",
                "source": "synthetic",
            }
        )
    return entries


if __name__ == "__main__":
    manifest_path = HERE / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"version": 1, "clips": []}
    )
    recorded = [c for c in manifest["clips"] if c.get("source") != "synthetic"]
    manifest["clips"] = synthesise() + recorded
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(manifest['clips'])} clips ({len(recorded)} recorded kept)")

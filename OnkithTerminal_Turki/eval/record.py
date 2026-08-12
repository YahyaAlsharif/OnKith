"""Add real recorded clips to the corpus, one prompt at a time.

The synthetic half is one machine reading a script with no room and no
noise. It is a floor on difficulty and it flatters every recogniser.
Nothing here is decided until the same script exists in the voices that
will actually stand in front of the device, in a room that sounds like the
room.

    .venv/bin/python eval/record.py --speaker turki
    .venv/bin/python eval/record.py --speaker faris --only wake

It prints a line, records until you stop talking, plays it back, and asks
whether to keep it. Recorded clips carry the same fields as synthetic ones
and are never overwritten by build_synthetic.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "corpus" / "manifest.json"
SAMPLE_RATE = 16000
SILENCE_SECONDS = 1.5
MAXIMUM_SECONDS = 12.0
BLOCK = 1600


def prompts(only: str | None) -> list[dict]:
    """The same script the synthetic clips read, so the two are comparable."""

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    synthetic = [c for c in manifest["clips"] if c.get("source") == "synthetic"]
    if only:
        synthetic = [c for c in synthetic if c["id"].startswith(only)]
    return synthetic


def capture(device: int | None):
    import numpy as np
    import sounddevice as sd

    frames: list = []
    quiet_for = 0.0
    started = time.monotonic()
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device
    ) as stream:
        while True:
            block, _ = stream.read(BLOCK)
            audio = block.reshape(-1)
            frames.append(audio.copy())
            level = float(np.sqrt(np.mean(audio**2)))
            quiet_for = quiet_for + BLOCK / SAMPLE_RATE if level < 0.01 else 0.0
            elapsed = time.monotonic() - started
            if frames and quiet_for >= SILENCE_SECONDS and elapsed > 1.0:
                break
            if elapsed > MAXIMUM_SECONDS:
                break
    return np.concatenate(frames) if frames else np.zeros(0, dtype="float32")


def write_wav(path: Path, samples) -> None:
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((clipped * 32767).astype(np.int16).tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--only", help="record only ids starting with this")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips_dir = MANIFEST.parent / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    existing = {clip["id"] for clip in manifest["clips"]}

    script = prompts(args.only)
    if not script:
        print("Nothing to record. Run build_synthetic.py first.")
        return 1

    print(f"{len(script)} lines. Speak when prompted; it stops on silence.")
    print("Enter keeps, r re-records, s skips, q quits.\n")

    added = 0
    for source in script:
        clip_id = f"{source['id']}--{args.speaker}"
        if clip_id in existing:
            continue
        while True:
            print(f"  say: {source['transcript']}")
            input("       ready, press enter ")
            samples = capture(args.device)
            seconds = len(samples) / SAMPLE_RATE
            path = clips_dir / f"{clip_id}.wav"
            write_wav(path, samples)
            choice = input(f"       {seconds:.1f}s. keep? [enter/r/s/q] ").strip().lower()
            if choice == "r":
                continue
            if choice == "q":
                path.unlink(missing_ok=True)
                _save(manifest)
                print(f"\nstopped. {added} added.")
                return 0
            if choice == "s":
                path.unlink(missing_ok=True)
                break
            manifest["clips"].append(
                {
                    **source,
                    "id": clip_id,
                    "file": f"clips/{path.name}",
                    "voice": args.speaker,
                    "source": "recorded",
                }
            )
            _save(manifest)
            added += 1
            break
    print(f"\n{added} added. Run: .venv/bin/python eval/run_eval.py")
    return 0


def _save(manifest: dict) -> None:
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download the exact pinned official Canary .nemo and verify it atomically."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODEL_ID = "nvidia/canary-180m-flash"
REVISION = "c770bfa0f70f66ad89dbf9827985f999675943a3"
FILENAME = "canary-180m-flash.nemo"
SHA256 = "7b97de18b718ef01bf3398715ce8d18486c4d83961e230230b26f5712e112675"
URL = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{FILENAME}"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT / "models" / "canary" / FILENAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    destination = args.destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and sha256_file(destination) == SHA256:
        print(f"[OK] Exact Canary artifact verified: {destination}")
        return 0
    if args.verify_only:
        raise SystemExit(f"Missing or invalid pinned Canary artifact: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        print(f"[DOWNLOAD] {URL}")
        request = urllib.request.Request(URL, headers={"User-Agent": "Omit-AI-Pi5-installer/1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        actual = sha256_file(temporary)
        if actual != SHA256:
            raise RuntimeError(f"Canary SHA-256 mismatch: {actual}; required {SHA256}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"[OK] Exact Canary artifact installed: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

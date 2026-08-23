#!/usr/bin/env python3
"""Download and verify OnKith model assets published with the GitHub release."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get(
    "ONKITH_MODEL_BASE_URL",
    "https://github.com/YahyaAlsharif/OnKith/releases/download/models-v1",
).rstrip("/")

ARTIFACTS = (
    (
        "models/deberta_pii/model_int8.onnx",
        "onkith-deberta-v2-model_int8.onnx",
        82_278_782,
        "bc8d1d7a7fb14304e195e47079c3dbb330c627ea4c9753f3af667676048106a8",
        True,
    ),
    (
        "models/piper/en_US-lessac-medium.onnx",
        "onkith-piper-en_US-lessac-medium.onnx",
        63_201_294,
        "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
        False,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid(path: Path, size: int, expected_hash: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256_file(path) == expected_hash


def fetch(relative: str, asset: str, size: int, expected_hash: str, required: bool) -> None:
    target = ROOT / relative
    if not required and not target.with_suffix(target.suffix + ".json").is_file():
        return
    if valid(target, size, expected_hash):
        print(f"[OK] verified {relative}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".downloading")
    request = urllib.request.Request(f"{BASE_URL}/{asset}", headers={"User-Agent": "OnKith-setup"})
    try:
        print(f"[download] {asset}")
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if not valid(temporary, size, expected_hash):
            raise RuntimeError(f"Downloaded model failed verification: {asset}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[OK] downloaded and verified {relative}")


def main() -> None:
    for artifact in ARTIFACTS:
        fetch(*artifact)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Render the Onkith mark to the PNG a desktop launcher needs.

Desktop environments want a raster icon at a known size, and a Pi may or
may not have librsvg behind its icon loader. Rather than depend on that,
or on a converter being installed on whichever machine builds this, the
mark is drawn here. It is three rounded rectangles, which is little
enough geometry to be worth owning outright and means this script has no
dependency beyond numpy, which the project already has.

The source of truth is still assets/player/onkith-favicon.svg. The
rectangles below are read out of it rather than copied, so the icon
cannot drift from the mark by somebody editing one and not the other.

    .venv/bin/python scripts/build_icon.py

Colours are the brand's, and there are only two of them.
"""

from __future__ import annotations

import re
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = ROOT / "assets" / "player" / "onkith-favicon.svg"
TARGET = ROOT / "assets" / "onkith.png"

SIZE = 256
# Drawn at eight times the final size and averaged down. There is no
# antialiasing here otherwise, and a hard-edged icon at 256px looks
# broken next to every other icon on the desktop.
SUPERSAMPLE = 8
VOID = (0x08, 0x07, 0x06)
LIFT = (0xFF, 0x5A, 0x36)
# The mark sits in a 24 unit box. A launcher icon needs air around it or
# it reads as cropped when the desktop draws it against a panel.
PADDING_UNITS = 3.0


def rectangles(svg: str) -> list[tuple[float, float, float, float, float]]:
    """Every rect in the mark, as x, y, width, height, corner radius."""

    found = []
    for tag in re.findall(r"<rect\b[^>]*>", svg):
        values = dict(re.findall(r'(\w+)="([^"]+)"', tag))
        try:
            found.append(
                (
                    float(values["x"]),
                    float(values["y"]),
                    float(values["width"]),
                    float(values["height"]),
                    float(values.get("rx", 0.0)),
                )
            )
        except KeyError as error:
            raise SystemExit(f"A rect in {MARK.name} is missing {error}")
    if not found:
        raise SystemExit(f"No rectangles found in {MARK}")
    return found


def render() -> "object":
    import numpy as np

    shapes = rectangles(MARK.read_text(encoding="utf-8"))
    box = 24.0 + PADDING_UNITS * 2
    resolution = SIZE * SUPERSAMPLE
    scale = resolution / box

    # Pixel centres, in the mark's own units, with the padding taken off
    # so the coordinates in the file can be used as they are written.
    axis = (np.arange(resolution) + 0.5) / scale - PADDING_UNITS
    x = axis[None, :]
    y = axis[:, None]

    covered = np.zeros((resolution, resolution), dtype=bool)
    for left, top, width, height, radius in shapes:
        right, bottom = left + width, top + height
        radius = min(radius, width / 2, height / 2)
        inside = (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
        if radius > 0:
            # Outside the rounded corner is the only part of the box that
            # is not filled, so it is the only case worth computing.
            nearest_x = np.clip(x, left + radius, right - radius)
            nearest_y = np.clip(y, top + radius, bottom - radius)
            distance = np.hypot(x - nearest_x, y - nearest_y)
            inside &= distance <= radius
        covered |= inside

    coverage = (
        covered.reshape(SIZE, SUPERSAMPLE, SIZE, SUPERSAMPLE)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )
    image = np.empty((SIZE, SIZE, 4), dtype=np.uint8)
    for channel, value in enumerate(LIFT):
        image[:, :, channel] = value
    # The mark over the brand's own black rather than over nothing. A
    # transparent icon disappears against a dark panel, which is what
    # this device's own screen is.
    background = np.array(VOID, dtype=np.float32)
    foreground = np.array(LIFT, dtype=np.float32)
    blended = background + (foreground - background) * coverage[:, :, None]
    image[:, :, :3] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    image[:, :, 3] = 255
    return image


def write_png(path: Path, image) -> None:
    """A PNG, without adding a dependency to draw one."""

    import numpy as np

    height, width, _ = image.shape
    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter type 0, no prediction
        raw.extend(np.ascontiguousarray(image[row]).tobytes())

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def main() -> int:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    write_png(TARGET, render())
    print(f"[OK] {TARGET.relative_to(ROOT)} at {SIZE}x{SIZE} from {MARK.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

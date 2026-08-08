#!/usr/bin/env python3
"""Build the Linux release archive with POSIX-only member paths.

The previous release was packaged on Windows and every member name used
backslashes (``Omit_AI_RPi_Live\\app\\...``), which is invalid for a Linux
Raspberry Pi release and makes ``unzip`` warn and exit non-zero. This builder
writes ``/`` separators only, refuses absolute paths and ``../`` traversal,
excludes caches/databases/secrets/model downloads, and preserves the executable
bit on shell scripts.

Usage:
    python scripts/build_release_zip.py --output ../Omit_AI_RPi_Live_..._FIXED.zip
"""

from __future__ import annotations

import argparse
import hashlib
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PREFIX = "Omit_AI_RPi_Live"

# Directories never shipped in a release.
EXCLUDED_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "node_modules",
    ".ipynb_checkpoints",
}

# Suffixes never shipped: bytecode, databases, logs, archives, audio dumps.
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".db",
    ".db-wal",
    ".db-shm",
    ".sqlite",
    ".sqlite3",
    ".log",
    ".zip",
    ".tar",
    ".gz",
    ".tmp",
    ".swp",
    ".nemo",  # the 700 MB model is acquired on the device, never bundled
}

# Exact names never shipped: secrets and local settings.
EXCLUDED_NAMES = {
    ".env",
    ".env.local",
    ".DS_Store",
    "Thumbs.db",
    "secrets.json",
    "credentials.json",
    "session_transcript.db",
}


class InvalidMemberName(ValueError):
    """Raised when an archive member name would be unsafe or non-POSIX."""


def validate_member_name(name: str) -> str:
    """Reject anything that is not a safe relative POSIX member path."""

    if "\\" in name:
        raise InvalidMemberName(f"Archive member uses a backslash: {name!r}")
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        raise InvalidMemberName(f"Archive member is absolute: {name!r}")
    parts = PurePosixPath(name).parts
    if ".." in parts:
        raise InvalidMemberName(f"Archive member traverses upward: {name!r}")
    if not name or name != name.strip():
        raise InvalidMemberName(f"Archive member name is malformed: {name!r}")
    return name


def is_excluded(path: Path) -> bool:
    if any(part in EXCLUDED_DIRECTORIES for part in path.parts):
        return True
    if path.name in EXCLUDED_NAMES:
        return True
    return path.suffix.lower() in EXCLUDED_SUFFIXES


def collect(root: Path) -> list[Path]:
    """Every shippable file, sorted for a deterministic archive order."""

    files = [
        item
        for item in root.rglob("*")
        if item.is_file() and not is_excluded(item.relative_to(root))
    ]
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def member_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return validate_member_name(f"{ARCHIVE_PREFIX}/{relative}")


def build(root: Path, output: Path) -> dict[str, object]:
    files = collect(root)
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = member_name(root, path)
            info = zipfile.ZipInfo(name)
            # Fixed timestamp keeps the archive reproducible.
            info.date_time = (2026, 8, 5, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            # 0o644 normally; 0o755 so shell scripts stay executable on Linux.
            mode = 0o755 if path.suffix == ".sh" else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.create_system = 3  # Unix
            archive.writestr(info, path.read_bytes())
            names.append(name)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "output": str(output),
        "members": len(names),
        "names": names,
        "sha256": digest,
        "size_bytes": output.stat().st_size,
    }


def verify(output: Path) -> list[str]:
    """Re-open the built archive and re-check every member name."""

    problems: list[str] = []
    with zipfile.ZipFile(output) as archive:
        broken = archive.testzip()
        if broken is not None:
            problems.append(f"CRC failure in {broken}")
        for info in archive.infolist():
            try:
                validate_member_name(info.filename)
            except InvalidMemberName as exc:
                problems.append(str(exc))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = build(args.root.resolve(), args.output.resolve())
    problems = verify(args.output.resolve())
    print(f"members : {result['members']}")
    print(f"bytes   : {result['size_bytes']}")
    print(f"sha256  : {result['sha256']}")
    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1
    print("[OK] all member paths are relative POSIX paths; archive CRC verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

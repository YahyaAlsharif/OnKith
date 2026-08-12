"""Read the device's own .env before anything asks for a key.

The keys this device needs live in a .env file beside the code, but nothing
was reading it: the launcher execs Python directly and every module asks
``os.environ``. That worked only for whoever had exported the values into
their own shell first, so a fresh terminal, a desktop launcher and a boot
service all started with no key and the assistant quietly said transcripts
were staying on the device.

Nothing is overwritten. A value already in the environment was put there
deliberately, by a service file or by hand, and it wins over the file.
"""

from __future__ import annotations

import os
from pathlib import Path

ENVIRONMENT_FILENAME = ".env"


def parse_environment(text: str) -> dict[str, str]:
    """Read KEY=VALUE lines, ignoring comments, blanks and exports.

    Quotes around a value are stripped because a shell would have stripped
    them too, which is the behaviour anyone editing this file expects.
    """

    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        if not separator:
            continue
        name = name.strip()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def load_environment(root: Path, environment: dict[str, str] | None = None) -> list[str]:
    """Fill in missing values from ``root/.env`` and report which were set.

    The names are returned rather than the values so a caller can say what it
    found without putting a key in a log.
    """

    target = os.environ if environment is None else environment
    path = Path(root) / ENVIRONMENT_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    filled: list[str] = []
    for name, value in parse_environment(text).items():
        if target.get(name):
            continue
        target[name] = value
        filled.append(name)
    return filled

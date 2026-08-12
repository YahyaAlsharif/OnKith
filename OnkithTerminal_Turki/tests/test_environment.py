"""The device reads its own .env before anything asks for a key.

The keys sat in a file nothing read, so the assistant reported itself
unconfigured to everyone who had not exported them by hand first. A desktop
launcher and a boot service never do.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.environment import load_environment, parse_environment


class ParseEnvironmentTest(unittest.TestCase):
    def test_it_reads_what_a_shell_would_read(self) -> None:
        parsed = parse_environment(
            "\n".join(
                [
                    "# a comment",
                    "",
                    "BARE=plain",
                    'QUOTED="has spaces"',
                    "SINGLE='single'",
                    "export EXPORTED=exported",
                    "  SPACED  =  trimmed  ",
                ]
            )
        )
        self.assertEqual(
            parsed,
            {
                "BARE": "plain",
                "QUOTED": "has spaces",
                "SINGLE": "single",
                "EXPORTED": "exported",
                "SPACED": "trimmed",
            },
        )

    def test_a_line_that_is_not_an_assignment_is_skipped(self) -> None:
        self.assertEqual(parse_environment("NOT_AN_ASSIGNMENT\n=novalue\n"), {})

    def test_a_value_may_contain_the_separator(self) -> None:
        # Base64 and URLs both carry equals signs, and a key is often either.
        parsed = parse_environment("TOKEN=abc==\nURL=https://x/y?a=1&b=2\n")
        self.assertEqual(parsed["TOKEN"], "abc==")
        self.assertEqual(parsed["URL"], "https://x/y?a=1&b=2")


class LoadEnvironmentTest(unittest.TestCase):
    def directory(self, contents: str | None) -> Path:
        holder = tempfile.TemporaryDirectory(prefix="onkith-env-")
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        if contents is not None:
            (root / ".env").write_text(contents, encoding="utf-8")
        return root

    def test_missing_values_are_filled_and_reported_by_name(self) -> None:
        root = self.directory("GROQ_API_KEY=secret\nGROQ_MODEL=some-model\n")
        environment: dict[str, str] = {}
        filled = load_environment(root, environment)
        self.assertEqual(sorted(filled), ["GROQ_API_KEY", "GROQ_MODEL"])
        self.assertEqual(environment["GROQ_API_KEY"], "secret")
        # Names, never values: this list is printed at startup.
        self.assertNotIn("secret", filled)

    def test_a_value_already_set_wins(self) -> None:
        root = self.directory("GROQ_API_KEY=from-file\n")
        environment = {"GROQ_API_KEY": "from-the-service-file"}
        filled = load_environment(root, environment)
        self.assertEqual(environment["GROQ_API_KEY"], "from-the-service-file")
        self.assertEqual(filled, [])

    def test_an_empty_value_counts_as_unset(self) -> None:
        root = self.directory("GROQ_API_KEY=from-file\n")
        environment = {"GROQ_API_KEY": ""}
        load_environment(root, environment)
        self.assertEqual(environment["GROQ_API_KEY"], "from-file")

    def test_no_file_is_not_an_error(self) -> None:
        self.assertEqual(load_environment(self.directory(None), {}), [])

    def test_the_entry_point_loads_it_before_anything_asks(self) -> None:
        source = (ROOT / "app" / "omit_ai_live.py").read_text(encoding="utf-8")
        loaded = source.index("load_environment(ROOT)")
        # Whichever module reads a key first, the file has already been read.
        for asks in ("from app.llm import", "from app.onkith_music import"):
            with self.subTest(asks=asks):
                self.assertLess(loaded, source.index(asks))


if __name__ == "__main__":
    unittest.main()

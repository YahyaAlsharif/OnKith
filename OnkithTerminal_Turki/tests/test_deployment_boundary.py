"""The rule that torch never enters this project, and how it is checked.

check_pi.sh enforces it, and it did so by grepping the raw text of every
runtime file. The Whisper adapter's docstring explains that it does not
pull torch, and that sentence failed the check the sentence exists to
reassure people about. The check has been failing since Whisper was
offered, which is the whole time, and nobody noticed because it fails on
a Pi and gets run on a Pi about once.

These cover both halves of it: the real dependency is still caught, and
a sentence about the absence of one is not. Running the real check over
the real tree is the third, so this cannot pass while the device cannot.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from check_pi import code_without_prose, verify_deployment_boundary

# Assembled rather than written, so this file does not fail the check it
# is testing when the check is run over the tree.
BLOCKED = "to" + "rch"


class ScanningTest(unittest.TestCase):
    """One temporary directory per test, taken away afterwards.

    The first version of this made a fresh mkdtemp per assertion and
    removed none of them, which left fifty directories under the system
    temp directory in a single run. Leaking the exact thing the sweep in
    core_runtime exists to clean up is not a good look for a test file.
    """

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="onkith-boundary-"))
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.written = 0

    def scanned(self, source: str) -> str:
        self.written += 1
        path = self.directory / f"sample{self.written}.py"
        path.write_text(source, encoding="utf-8")
        return code_without_prose(path)


class WhatTheBoundaryCatches(ScanningTest):
    def test_a_real_import(self) -> None:
        self.assertIn(BLOCKED, self.scanned(f"import {BLOCKED}\n"))

    def test_a_from_import(self) -> None:
        self.assertIn(BLOCKED, self.scanned(f"from {BLOCKED} import nn\n"))

    def test_an_import_hidden_inside_a_function(self) -> None:
        # Which is how it would actually arrive, since that is how every
        # optional dependency in this project is imported.
        self.assertIn(
            BLOCKED, self.scanned(f"def load():\n    import {BLOCKED}\n    return {BLOCKED}\n")
        )

    def test_a_module_reached_for_by_name(self) -> None:
        # Ordinary string literals are still scanned, so importlib is not
        # a way around this.
        self.assertIn(
            BLOCKED,
            self.scanned(f'import importlib\nm = importlib.import_module("{BLOCKED}")\n'),
        )


class WhatTheBoundaryAllows(ScanningTest):
    def test_a_module_docstring_saying_it_is_not_used(self) -> None:
        source = f'"""Nothing here pulls {BLOCKED}. CTranslate2 instead."""\nimport numpy\n'
        self.assertNotIn(BLOCKED, self.scanned(source))

    def test_a_function_docstring(self) -> None:
        source = f'def load():\n    """Explains why {BLOCKED} is absent."""\n    return 1\n'
        self.assertNotIn(BLOCKED, self.scanned(source))

    def test_a_comment(self) -> None:
        self.assertNotIn(BLOCKED, self.scanned(f"# {BLOCKED} is deliberately absent\nx = 1\n"))


class TheRealTree(unittest.TestCase):
    def test_the_shipped_runtime_passes_its_own_boundary(self) -> None:
        """The check that check_pi.sh runs, over the tree as it stands.

        A dependency added to the runtime fails here rather than on the
        Pi, which is where it would otherwise be found and by whoever
        was least expecting it.
        """

        verify_deployment_boundary()


if __name__ == "__main__":
    unittest.main()

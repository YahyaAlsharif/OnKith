from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_notebook_is_output_free_and_python_cells_parse(self):
        notebook = json.loads((ROOT / "privacy_evaluation.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 10)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "moonshine_pi_live.py"

spec = importlib.util.spec_from_file_location("moonshine_pi_live", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

config = module.AppConfig.from_json(ROOT / "config" / "final_config.json")
config.validate()

assert module.normalize_standard("Hello, WORLD!") == "hello world"
assert module.normalize_command("Set it to twenty two degrees.") == "set it to 22 degrees"
assert module.normalize_command("Don't wait ten minutes.") == "do not wait 10 minutes"

counts = module.word_error_counts(
    "set it to twenty two degrees",
    "set it to 22 degrees",
    command_normalized=True,
)
assert counts.errors == 0

counts = module.word_error_counts(
    "turn on the kitchen light",
    "turn off the kitchen light",
    command_normalized=False,
)
assert counts.errors == 1
assert counts.reference_words == 5

print("test_core.py: PASS")

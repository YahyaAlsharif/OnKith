"""V6 must be the V4 core, unaltered, and must stand on its own.

Two claims are made about this pipeline and both are cheap to check
mechanically, so neither is left as a promise in a README:

* **It is V4's privacy and recognition core.** Not "equivalent to", not
  "ported from" - the same bytes. That is what lets V4's Raspberry Pi
  measurements remain evidence about the path V6 runs, and what makes the
  parity question a file comparison rather than an argument.
* **It does not reach into its predecessors.** V4 and V5 move into
  ``archive/`` once this version is accepted. Anything that imported them
  would break on the day they moved, quietly and in the middle of a
  demonstration, so nothing may.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Every file V6 inherits from V4 unchanged: the recognizer, the segmenter's
#: configuration, the privacy model and its decoder, and the two committed
#: artifacts. If one of these ever differs, V4's measurements have stopped
#: describing what V6 runs and the claim in the README is no longer true.
V4_CORE = (
    "integrated_pipeline/config.py",
    "integrated_pipeline/mask_policy.py",
    "integrated_pipeline/model_v2_masker.py",
    "integrated_pipeline/pii_adapter.py",
    "integrated_pipeline/pipeline.py",
    "integrated_pipeline/silero_vad.py",
    "integrated_pipeline/span_merger.py",
    "integrated_pipeline/spoken_form_normalizer.py",
    "integrated_pipeline/structured_pii.py",
    "integrated_pipeline/supplemental_pii.py",
    "integrated_pipeline/transcript_merger.py",
    "integrated_pipeline/inactivity_buffer.py",
    "integrated_pipeline/whisper_adapter.py",
    "integrated_pipeline/schemas.py",
    "app/canary_live_segmenter.py",
    "app/stt_runtime.py",
    "app/digit_normalizer.py",
    "app/wake_word_normalizer.py",
    "config/live_config.json",
    "models/deberta_pii/model_int8.onnx",
    "models/deberta_pii/model_manifest.json",
    "models/deberta_pii/config/model_v2_label_map.json",
    "models/silero/silero_vad.onnx",
)

#: Wherever V4 is on disk. It is beside V6 until V6 is accepted and in the
#: archive afterwards, and the parity check has to keep working across that
#: move rather than starting to skip silently the day it happens.
V4_LOCATIONS = (
    REPO_ROOT / "FullPipeline_V4",
    REPO_ROOT / "archive" / "FullPipeline_V4",
)

#: The predecessor package roots, spelled out in full rather than as the bare
#: word "archive" - which is also ordinary English and the name of a local
#: variable in the release builder.
FORBIDDEN_PATHS = ("FullPipeline_V4", "FullPipeline_V5")

#: This module names the predecessors on purpose, so it is excluded from the
#: scans that would otherwise match its own source.
SELF = "tests/test_v6_independence.py"

#: What actually runs on the device. Diagnostics under ``scripts/`` may name
#: other backends because reporting which ones are installed is what they are
#: for; nothing in these two directories may.
RUNTIME_DIRECTORIES = ("app", "integrated_pipeline")


def find_v4() -> Path | None:
    return next((path for path in V4_LOCATIONS if path.is_dir()), None)


#: Compared byte for byte, with no line-ending normalisation. The manifests
#: record a SHA-256 of the file, and ``.gitattributes`` marks these ``binary``
#: precisely so no checkout can rewrite them.
EXACT_BYTES = (".onnx", ".json")


def digest(path: Path) -> str:
    """Content hash, ignoring how the checkout wrote the line endings.

    ``.gitattributes`` sets ``* text=auto``, so git converts line endings on
    checkout and the same tracked file is legitimately CRLF in one working tree
    and LF in another. Hashing raw bytes would make this test report a
    divergence every time a file happened to be re-materialised, which says
    nothing about whether the code is the same - and it is the code that has to
    be the same for V4's measurements to describe what V6 runs.

    Artifacts are exempt: their bytes are their identity, they are marked
    ``binary`` so nothing converts them, and a difference there is real.
    """

    data = path.read_bytes()
    if path.suffix.lower() not in EXACT_BYTES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def python_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*.py")
        if "__pycache__" not in path.parts and ".venv" not in str(path)
    ]


class CoreParityTests(unittest.TestCase):
    """V6's local path is V4's, byte for byte."""

    def setUp(self):
        self.v4 = find_v4()
        if self.v4 is None:
            self.skipTest(
                "FullPipeline_V4 is not on disk in either of its known "
                "locations, so parity cannot be checked from here."
            )

    def test_every_inherited_core_file_is_byte_identical_to_v4(self):
        differing = []
        for relative in V4_CORE:
            theirs = self.v4 / relative
            ours = ROOT / relative
            self.assertTrue(ours.is_file(), f"V6 is missing {relative}")
            self.assertTrue(theirs.is_file(), f"V4 is missing {relative}")
            if digest(theirs) != digest(ours):
                differing.append(relative)
            if theirs.suffix.lower() == ".onnx":
                # Belt and braces on the two committed model artifacts: these
                # are compared as raw bytes whatever else happens above.
                self.assertEqual(
                    theirs.read_bytes(), ours.read_bytes(),
                    f"{relative} differs byte for byte",
                )
        self.assertEqual(
            differing,
            [],
            "V6 has diverged from the V4 core it claims to be:\n  "
            + "\n  ".join(differing),
        )

    def test_the_whisper_contract_is_v4s(self):
        theirs = json.loads((self.v4 / "config/live_config.json").read_text("utf-8"))
        ours = json.loads((ROOT / "config/live_config.json").read_text("utf-8"))
        self.assertEqual(theirs["whisper"], ours["whisper"])
        # Spelled out as well as compared, so a reader of the test knows what
        # the frozen configuration actually is.
        self.assertEqual(
            {
                key: ours["whisper"][key]
                for key in (
                    "engine", "runtime", "model", "model_id", "package_version",
                    "device", "compute_type", "beam_size", "language",
                )
            },
            {
                "engine": "faster-whisper",
                "runtime": "ctranslate2",
                "model": "base.en",
                "model_id": "Systran/faster-whisper-base.en",
                "package_version": "1.2.1",
                "device": "cpu",
                "compute_type": "int8",
                "beam_size": 5,
                "language": "en",
            },
        )

    def test_the_segmentation_contract_is_v4s(self):
        from integrated_pipeline.config import (
            ENDPOINT_SILENCE_SAMPLES,
            FORCED_OVERLAP_SAMPLES,
            MAXIMUM_UNIQUE_SEGMENT_SAMPLES,
            PRE_ROLL_SAMPLES,
            TRAILING_CONTEXT_SAMPLES,
        )

        # V5 retuned all of these for Parakeet. V6 is on Whisper and keeps V4's.
        self.assertEqual(ENDPOINT_SILENCE_SAMPLES, 39_936)
        self.assertEqual(MAXIMUM_UNIQUE_SEGMENT_SAMPLES, 136_192)
        self.assertEqual(PRE_ROLL_SAMPLES, 6_144)
        self.assertEqual(FORCED_OVERLAP_SAMPLES, 5_632)
        self.assertEqual(TRAILING_CONTEXT_SAMPLES, 5_632)


class IndependenceTests(unittest.TestCase):
    """Nothing here may import or reach into a predecessor."""

    def test_no_module_imports_v4_or_v5(self):
        offences = []
        for path in python_files():
            try:
                tree = ast.parse(path.read_text("utf-8"))
            except (SyntaxError, UnicodeDecodeError) as error:  # pragma: no cover
                self.fail(f"{path}: {error}")
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if any(root in name for root in FORBIDDEN_PATHS):
                        offences.append(
                            f"{path.relative_to(ROOT).as_posix()}:{node.lineno} {name}"
                        )
        self.assertEqual(
            offences, [], "V6 imports a predecessor:\n  " + "\n  ".join(offences)
        )

    def test_no_module_builds_a_path_into_v4_or_v5(self):
        # sys.path manipulation and bare path strings do not show up as
        # imports, and either would break on the day the predecessors move.
        # Comments may still mention them: V4's own files explain what they
        # replaced, and that history is worth keeping.
        offences = []
        for path in python_files():
            relative = path.relative_to(ROOT).as_posix()
            if relative == SELF:
                continue
            for lineno, line in enumerate(path.read_text("utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if any(root in line for root in FORBIDDEN_PATHS):
                    offences.append(f"{relative}:{lineno} {stripped[:80]}")
        self.assertEqual(
            offences,
            [],
            "V6 refers to a predecessor path in live code:\n  " + "\n  ".join(offences),
        )

    def test_the_runtime_has_no_parakeet_or_nemo_dependency(self):
        offences = []
        for path in python_files():
            relative = path.relative_to(ROOT).as_posix()
            if relative.split("/")[0] not in RUNTIME_DIRECTORIES:
                continue
            text = path.read_text("utf-8").lower()
            for token in ("import nemo", "from nemo", "parakeet"):
                if token in text:
                    offences.append(f"{relative}: {token}")
        self.assertEqual(offences, [])

    def test_requirements_do_not_install_the_parakeet_stack(self):
        # Only the lines pip would act on. Both files carry a comment recording
        # that V3's torch + NeMo pair is what faster-whisper replaced, and
        # deleting that note to satisfy a grep would lose the reason.
        for name in ("requirements-windows.txt", "requirements-rpi.txt"):
            requested = [
                line.strip()
                for line in (ROOT / name).read_text("utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            joined = " ".join(requested).lower()
            with self.subTest(requirements=name):
                self.assertNotIn("nemo", joined)
                self.assertNotIn("torch", joined)
                self.assertIn("faster-whisper", joined)

    def test_the_assistant_layer_is_present(self):
        for module in (
            "app/assistant.py",
            "app/assistant_surface.py",
            "app/light.py",
            "app/llm.py",
            "app/post_mask_assistant.py",
            "app/rehydrate.py",
            "app/speech.py",
        ):
            with self.subTest(module=module):
                self.assertTrue((ROOT / module).is_file())


class SecretHygieneTests(unittest.TestCase):
    def test_every_secret_in_the_example_env_is_blank(self):
        # Model names and backend selectors are configuration and may carry a
        # value. Anything that is a credential may not.
        secretish = ("KEY", "TOKEN", "SECRET", "PASSWORD")
        for line in (ROOT / ".env.example").read_text("utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if not any(word in key.upper() for word in secretish):
                continue
            with self.subTest(key=key):
                self.assertEqual(
                    value.strip(),
                    "",
                    f"{key} in .env.example must be blank, not a real value",
                )

    def test_the_example_env_names_the_supported_models(self):
        text = (ROOT / ".env.example").read_text("utf-8")
        self.assertIn("GROQ_MODEL=openai/gpt-oss-20b", text)
        self.assertIn("GROQ_FALLBACK_MODEL=openai/gpt-oss-120b", text)

    def test_no_groq_key_is_hardcoded(self):
        # A Groq key is the four characters below followed by a long
        # token. Nothing shipped here
        # may carry one, whatever the variable is called.
        marker = "gsk" + "_"
        offences = []
        for path in python_files() + list(ROOT.glob("*.md")) + [ROOT / ".env.example"]:
            if path.relative_to(ROOT).as_posix() == SELF:
                continue
            if marker in path.read_text("utf-8", errors="replace"):
                offences.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offences, [])


if __name__ == "__main__":
    unittest.main()

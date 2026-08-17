"""Model V2 integration cover for the live pipeline.

Three things are pinned here, all of which broke silently at some point during
the TinyBERT to DeBERTa swap and none of which the existing suite would notice:

1. **The packaged artifact is the validated one.** The deployment copy under
   ``models/deberta_pii/`` must be byte-identical to the canonical INT8 ONNX in
   ``model/DeBERTa-v3-xsmall/artifacts/model_v2/``, because that is the artifact
   the 40,909-row held-out comparison measured (typed F1 0.9588, leakage 0.0017,
   overmasking 0.0029). A rebuilt or re-quantized graph is a different model.

2. **TinyBERT is gone from the pipeline.** Not deprecated, not a fallback -
   absent. Model V1 stays intact in its own directories for provenance.

3. **The decode goes through the one canonical masker.** ``ModelV2OnnxMasker``
   derives word groups from character offsets rather than SentencePiece word
   ids; the punctuation cases below are the exact failure that motivated it.

The end-to-end tests load the real 82 MB INT8 graph and are skipped when the
artifact or onnxruntime/transformers is unavailable, so a checkout without the
model still runs the rest of the suite.
"""

from __future__ import annotations

import ast
import hashlib
import json
import unittest
from pathlib import Path

from integrated_pipeline.config import (
    PII_ARTIFACT_BYTES,
    PII_ARTIFACT_SHA256,
    PII_LABEL_COUNT,
    PII_MODEL_ID,
    PipelineConfig,
)
from integrated_pipeline.pii_adapter import EXPECTED_LABEL_COUNT, PIIMaskerAdapter
from integrated_pipeline.pipeline import FailClosedMasker

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
CANONICAL = REPO_ROOT / "model" / "DeBERTa-v3-xsmall"
CANONICAL_INT8 = CANONICAL / "artifacts" / "model_v2" / "models" / "int8" / "model_int8.onnx"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PackagedArtifactTests(unittest.TestCase):
    """The pinned identity of the deployed model."""

    def setUp(self) -> None:
        self.config = PipelineConfig()

    def test_the_configuration_points_at_the_deberta_package(self) -> None:
        pii = self.config.pii
        self.assertEqual(pii["model_id"], PII_MODEL_ID)
        self.assertEqual(pii["ontology_version"], "model_v2.ontology.1")
        self.assertEqual(pii["artifact"], "models/deberta_pii/model_int8.onnx")
        self.assertEqual(pii["tokenizer"], "models/deberta_pii/tokenizer")
        self.assertEqual(pii["label_count"], PII_LABEL_COUNT)
        self.assertEqual(EXPECTED_LABEL_COUNT, PII_LABEL_COUNT)

    def test_the_packaged_int8_model_is_the_validated_artifact(self) -> None:
        model = self.config.pii_model_path
        self.assertTrue(model.is_file(), f"packaged model missing: {model}")
        self.assertEqual(model.stat().st_size, PII_ARTIFACT_BYTES)
        self.assertEqual(sha256_file(model), PII_ARTIFACT_SHA256)

    def test_the_packaged_copy_matches_the_canonical_artifact(self) -> None:
        if not CANONICAL_INT8.is_file():
            self.skipTest(f"canonical artifact not present: {CANONICAL_INT8}")
        self.assertEqual(
            sha256_file(self.config.pii_model_path), sha256_file(CANONICAL_INT8)
        )

    def test_the_label_map_is_the_63_label_model_v2_ontology(self) -> None:
        payload = json.loads(
            self.config.pii_label_map_path.read_text(encoding="utf-8")
        )
        self.assertEqual(payload["ontology_version"], "model_v2.ontology.1")
        self.assertEqual(payload["num_labels"], PII_LABEL_COUNT)
        id2label = {int(key): value for key, value in payload["id2label"].items()}
        self.assertEqual(set(id2label), set(range(PII_LABEL_COUNT)))
        self.assertEqual(id2label[0], "O")

    def test_the_manifest_agrees_with_the_deployment_config(self) -> None:
        manifest = json.loads(
            (self.config.pii_model_path.parent / "model_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["model_id"], self.config.pii["model_id"])
        self.assertEqual(manifest["label_count"], self.config.pii["label_count"])
        self.assertEqual(
            manifest["files"]["model_int8.onnx"]["sha256"], self.config.pii["sha256"]
        )

    def test_a_tampered_artifact_is_refused(self) -> None:
        """verify_artifact is what stops an unvalidated graph reaching users."""

        adapter = PIIMaskerAdapter(self.config)
        with self.assertRaisesRegex(RuntimeError, "does not match the pinned"):
            adapter.verify_artifact(self.config.silero_model_path)


class TinyBertIsGoneTests(unittest.TestCase):
    """Model V1 must not survive anywhere inside the deployed pipeline."""

    @staticmethod
    def executable_source(path: Path) -> str:
        """The module with every comment and string literal removed.

        A docstring may still name TinyBERT - ``supplemental_pii.py`` records
        which model its two regex gaps were confirmed against, and rewriting
        that to say DeBERTa would make it false. What must not survive is a
        TinyBERT *path, identifier or configuration value*, which is exactly
        what is left once the prose is stripped.
        """

        import io
        import tokenize

        kept = []
        with path.open("rb") as stream:
            for token in tokenize.tokenize(io.BytesIO(stream.read()).readline):
                if token.type not in (tokenize.COMMENT, tokenize.STRING):
                    kept.append(token.string)
        return " ".join(kept)

    def test_no_tinybert_artifacts_are_packaged(self) -> None:
        self.assertFalse((ROOT / "models" / "tinybert_pii").exists())
        stray = [
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "models").rglob("*")
            if "tinybert" in path.name.lower()
        ]
        self.assertEqual(stray, [])

    def test_no_runtime_code_or_config_refers_to_tinybert(self) -> None:
        offenders = []
        for directory in ("integrated_pipeline", "app", "scripts", "config"):
            for path in (ROOT / directory).rglob("*"):
                if path.suffix not in {".py", ".json"} or "__pycache__" in path.parts:
                    continue
                body = (
                    self.executable_source(path)
                    if path.suffix == ".py"
                    else path.read_text(encoding="utf-8")
                )
                if "tinybert" in body.lower():
                    offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_the_installer_provisions_the_deberta_model(self) -> None:
        installer = (ROOT / "install_pi.sh").read_text(encoding="utf-8")
        self.assertIn("models/deberta_pii/model_int8.onnx", installer)
        self.assertNotIn("tinybert", installer.lower())


class DecoderWiringTests(unittest.TestCase):
    """One decoder, and it is the corrected one."""

    ADAPTER = ROOT / "integrated_pipeline" / "pii_adapter.py"
    MASKER = ROOT / "integrated_pipeline" / "model_v2_masker.py"

    def test_the_adapter_delegates_to_the_canonical_masker(self) -> None:
        source = self.ADAPTER.read_text(encoding="utf-8")
        self.assertIn("from .model_v2_masker import ModelV2OnnxMasker", source)

    def test_the_adapter_does_not_decode_spans_itself(self) -> None:
        """A second consolidation call here is how the two paths would drift."""

        tree = ast.parse(self.ADAPTER.read_text(encoding="utf-8"))
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        self.assertNotIn("consolidate_predictions_to_words", called)

    def test_the_masker_never_uses_sentencepiece_word_ids(self) -> None:
        """The regression that cost 0.37 typed F1 before it was found."""

        source = self.MASKER.read_text(encoding="utf-8")
        self.assertNotIn(".word_ids()", source)
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "consolidate_predictions_to_words"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("word_ids", [keyword.arg for keyword in calls[0].keywords])


def build_adapter(policy: dict | None = None) -> PIIMaskerAdapter:
    """A live adapter, optionally with a mask policy patched into its config."""

    config = PipelineConfig()
    if policy is not None:
        config.data["pii"]["mask_policy"] = policy
    adapter = PIIMaskerAdapter(config)
    adapter.load()
    return adapter


class LiveModelTestCase(unittest.TestCase):
    """Base class that loads the real INT8 graph once for the whole class."""

    adapter: PIIMaskerAdapter

    @classmethod
    def setUpClass(cls) -> None:
        config = PipelineConfig()
        if not config.pii_model_path.is_file():
            raise unittest.SkipTest(f"INT8 model not present: {config.pii_model_path}")
        try:
            import onnxruntime  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as error:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"runtime unavailable: {error}")
        cls.adapter = build_adapter()

    def mask(self, text: str) -> str:
        return self.adapter.mask_text(text).masked_text

    def entities(self, text: str) -> list[dict]:
        return self.adapter.mask_text(text).entities


class EndToEndMaskingTests(LiveModelTestCase):
    """The pipeline, loaded and run exactly as the deployment runs it."""

    def test_the_model_and_tokenizer_load(self) -> None:
        self.assertTrue(self.adapter.loaded)
        self.assertEqual(self.adapter.load_count, 1)
        self.assertEqual(len(self.adapter._id2label), PII_LABEL_COUNT)

    def test_loading_is_idempotent(self) -> None:
        self.adapter.load()
        self.assertEqual(self.adapter.load_count, 1)

    def test_a_name_followed_by_a_comma_keeps_the_comma(self) -> None:
        """The exact SentencePiece boundary defect, through the live model."""

        text = "My name is Rayyan Khan, and I live in Dubai."
        result = self.adapter.mask_text(text)
        self.assertNotIn("Rayyan", result.masked_text)
        for entity in result.entities:
            with self.subTest(entity=entity["entity_type"]):
                self.assertNotIn(
                    entity["text"][-1], ",.;:",
                    f"{entity['entity_type']} span {entity['text']!r} swallowed punctuation",
                )

    def test_an_entity_followed_by_a_period_keeps_the_period(self) -> None:
        result = self.adapter.mask_text("Write to rayyan.khan@example.com.")
        self.assertTrue(
            result.masked_text.endswith("."),
            f"trailing period was absorbed: {result.masked_text!r}",
        )
        self.assertNotIn("rayyan.khan@example.com", result.masked_text)

    def test_an_email_is_masked(self) -> None:
        masked = self.mask("Please email me at ana.hassan@example.com today")
        self.assertNotIn("ana.hassan@example.com", masked)
        self.assertIn("[EMAIL_1]", masked)

    def test_a_phone_number_is_masked(self) -> None:
        masked = self.mask("You can call me on 0555 123 4567 any time")
        self.assertNotIn("0555 123 4567", masked)
        self.assertNotIn("4567", masked)

    def test_a_date_is_masked(self) -> None:
        masked = self.mask("I was born on 28/11/1980 in Riyadh")
        self.assertNotIn("28/11/1980", masked)

    def test_multiple_spans_in_one_utterance(self) -> None:
        text = (
            "My name is Rayyan Khan, my email is rayyan@example.com "
            "and my phone number is 0555 123 4567"
        )
        result = self.adapter.mask_text(text)
        self.assertNotIn("Rayyan", result.masked_text)
        self.assertNotIn("rayyan@example.com", result.masked_text)
        self.assertNotIn("0555 123 4567", result.masked_text)
        self.assertGreaterEqual(len(result.entities), 3)

    def test_a_hard_negative_is_released_unchanged(self) -> None:
        text = "Turn the lights off and play some music please"
        result = self.adapter.mask_text(text)
        self.assertEqual(result.masked_text, text)
        self.assertEqual(result.entities, [])

    def test_the_result_carries_model_v2_telemetry(self) -> None:
        details = self.adapter.mask_text("Call Ana at ana@example.com").raw_model_output
        self.assertEqual(details["label_count"], PII_LABEL_COUNT)
        self.assertEqual(details["window_count"], 1)
        self.assertTrue(details["structured_enabled"])
        self.assertIn("pii_model_latency_seconds", details)
        self.assertIn("span_trace", details)

    def test_the_fail_closed_boundary_still_wraps_the_model(self) -> None:
        probe = FailClosedMasker(self.adapter).mask_text(
            "My email is rayyan@example.com and my account number is 12345678."
        )
        self.assertTrue(probe.safe_to_release)
        self.assertNotIn("rayyan@example.com", probe.masked_text)


class EntityDisablingTests(LiveModelTestCase):
    """Filtering belongs to configuration, never to the model."""

    RIYADH = "What time is it in Riyadh?"

    def test_city_is_masked_when_the_entity_is_enabled(self) -> None:
        adapter = build_adapter({})
        result = adapter.mask_text(self.RIYADH)
        self.assertNotIn("Riyadh", result.masked_text)

    def test_disabling_city_releases_the_location_but_still_detects_it(self) -> None:
        adapter = build_adapter({"CITY": False})
        result = adapter.mask_text(self.RIYADH)
        self.assertEqual(result.masked_text, self.RIYADH)
        city = [e for e in result.entities if e["entity_type"] == "CITY"]
        self.assertTrue(city, "CITY must still be detected and reported")
        self.assertFalse(city[0]["masked"])
        self.assertIsNone(city[0]["placeholder"])
        self.assertEqual(result.raw_model_output["policy_exempt_span_count"], len(city))

    def test_disabling_several_entities_at_once(self) -> None:
        adapter = build_adapter({"CITY": False, "COUNTRY": False})
        text = "I flew from Riyadh to Egypt last week"
        result = adapter.mask_text(text)
        self.assertEqual(result.masked_text, text)
        self.assertEqual(
            adapter.mask_policy.disabled_labels, ("CITY", "COUNTRY")
        )

    def test_disabling_city_does_not_suppress_unrelated_pii(self) -> None:
        """The whole risk of a disable feature: one exemption leaking others."""

        adapter = build_adapter({"CITY": False, "COUNTRY": False})
        text = "I am in Riyadh, email me at ana@example.com or call 0555 123 4567"
        result = adapter.mask_text(text)
        self.assertIn("Riyadh", result.masked_text)
        self.assertNotIn("ana@example.com", result.masked_text)
        self.assertNotIn("0555 123 4567", result.masked_text)

    def test_the_model_still_predicts_its_complete_ontology(self) -> None:
        """Disabling is a masking decision, not a narrower label set."""

        adapter = build_adapter({"CITY": False})
        self.assertEqual(len(adapter._id2label), PII_LABEL_COUNT)
        self.assertIn("B-CITY", set(adapter._id2label.values()))


if __name__ == "__main__":
    unittest.main()

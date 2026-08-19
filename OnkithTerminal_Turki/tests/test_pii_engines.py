"""Two trained classifiers behind one seam, and the traps in swapping them.

These run both real models. They are slower than the rest of the suite and
that is the point: the thing worth guarding here is that the packaged
artifacts load and decode, which no fake can tell you.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrated_pipeline.config import PII_ENGINE_VARIABLE, PipelineConfig
from integrated_pipeline.pii_adapter import PIIMaskerAdapter, sha256_of


class EngineEnvironment(unittest.TestCase):
    """Select an engine for the duration of one test."""

    def using(self, engine: str) -> PipelineConfig:
        previous = os.environ.get(PII_ENGINE_VARIABLE)
        os.environ[PII_ENGINE_VARIABLE] = engine

        def restore() -> None:
            if previous is None:
                os.environ.pop(PII_ENGINE_VARIABLE, None)
            else:
                os.environ[PII_ENGINE_VARIABLE] = previous

        self.addCleanup(restore)
        return PipelineConfig()


class WhichModelRuns(EngineEnvironment):
    def test_the_deployed_default_is_model_v2(self) -> None:
        previous = os.environ.pop(PII_ENGINE_VARIABLE, None)
        if previous is not None:
            self.addCleanup(os.environ.__setitem__, PII_ENGINE_VARIABLE, previous)
        config = PipelineConfig()
        self.assertEqual(config.pii_engine, "deberta")
        self.assertEqual(config.pii_label_count, 63)

    def test_the_environment_outranks_the_file(self) -> None:
        # The way back has to work without editing tracked configuration
        # and remembering to edit it back.
        config = self.using("tinybert")
        self.assertEqual(config.pii_engine, "tinybert")
        self.assertEqual(config.pii_label_count, 67)
        self.assertIn("tinybert_pii", str(config.pii_model_path))

    def test_each_engine_points_at_its_own_artifacts(self) -> None:
        deberta = self.using("deberta")
        self.assertTrue(deberta.pii_model_path.is_file())
        self.assertTrue(deberta.pii_label_map_path.is_file())
        self.assertTrue((deberta.pii_tokenizer_path / "tokenizer.json").is_file())

    def test_an_engine_nobody_shipped_is_refused(self) -> None:
        # At construction, not at first use. A typo in the engine name
        # should stop the device booting, not surface on the first
        # utterance somebody speaks to it.
        with self.assertRaises(ValueError) as raised:
            self.using("llama")
        self.assertIn("llama", str(raised.exception))

    def test_a_config_written_before_there_were_two_models_still_loads(self) -> None:
        # The rollback path that matters most: an old live_config.json,
        # with everything flat under "pii" and no engines section at all.
        source = json.loads(
            (ROOT / "config" / "live_config.json").read_text(encoding="utf-8")
        )
        source["pii"] = {
            **{
                key: value
                for key, value in source["pii"].items()
                if key not in ("engine", "engines")
            },
            **source["pii"]["engines"]["tinybert"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live_config.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            previous = os.environ.pop(PII_ENGINE_VARIABLE, None)
            if previous is not None:
                self.addCleanup(
                    os.environ.__setitem__, PII_ENGINE_VARIABLE, previous
                )
            config = PipelineConfig(config_path=path)
            self.assertEqual(config.pii_label_count, 67)
            self.assertTrue(config.pii_model_path.is_file())


class TheScreenNamesWhatIsLoaded(EngineEnvironment):
    """A banner that cannot contradict the model behind it.

    The boot panel held "TinyBERT-4 INT8 + structured PII" as a literal
    and went on saying it after the classifier became DeBERTa. Nothing
    failed: the right model was loaded and masking correctly, and the
    only symptom was 714 MB of resident memory under a label that
    belonged to a 14 MB model. Somebody spotted it in a screenshot.
    """

    def test_each_engine_names_itself(self) -> None:
        self.assertEqual(
            self.using("deberta").pii_display_name, "DeBERTa-v3-xsmall INT8"
        )
        self.assertEqual(
            self.using("tinybert").pii_display_name, "TinyBERT-4 INT8"
        )

    def test_the_name_belongs_to_the_artifact_beside_it(self) -> None:
        # The display name and the model path come out of the same block,
        # so a name can only be wrong if the path is wrong too.
        for engine, fragment in (
            ("deberta", "deberta"),
            ("tinybert", "tinybert"),
        ):
            with self.subTest(engine=engine):
                config = self.using(engine)
                self.assertIn(fragment, str(config.pii_model_path))
                self.assertIn(
                    fragment.replace("deberta", "DeBERTa").replace(
                        "tinybert", "TinyBERT"
                    ),
                    config.pii_display_name,
                )

    def test_an_engine_without_a_display_name_still_says_something(self) -> None:
        config = self.using("deberta")
        config.pii["engines"]["deberta"].pop("display")
        self.assertTrue(config.pii_display_name)
        config.pii["engines"]["deberta"].pop("model")
        self.assertEqual(config.pii_display_name, "deberta")

    def test_the_boot_banner_reads_the_name_rather_than_holding_one(self) -> None:
        source = (ROOT / "app" / "omit_ai_live.py").read_text(encoding="utf-8")
        for literal in ("TinyBERT-4 INT8", "DeBERTa-v3-xsmall INT8"):
            self.assertNotIn(
                literal,
                source,
                f"{literal!r} is written into the runtime; it will go stale",
            )
        self.assertIn("config.pii_display_name", source)


class ArtifactIntegrity(EngineEnvironment):
    def test_the_packaged_model_is_the_one_that_was_evaluated(self) -> None:
        # Model V2's accepted numbers belong to one exact file. A different
        # file scoring differently would look like the model regressing.
        config = self.using("deberta")
        self.assertEqual(
            sha256_of(config.pii_model_path), config.pii_model_sha256
        )

    def test_a_swapped_model_is_refused_rather_than_run(self) -> None:
        config = self.using("deberta")
        config.pii["engines"]["deberta"]["sha256"] = "0" * 64
        adapter = PIIMaskerAdapter(config)
        with self.assertRaises(RuntimeError) as raised:
            adapter.load()
        self.assertIn("not the validated artifact", str(raised.exception))

    def test_model_v1_pins_nothing_and_still_loads(self) -> None:
        config = self.using("tinybert")
        config.pii["engines"]["tinybert"].pop("sha256", None)
        adapter = PIIMaskerAdapter(config)
        self.assertEqual(adapter.verify_artifact(config.pii_model_path), "")

    def test_a_label_map_of_the_wrong_size_is_refused(self) -> None:
        # Loading Model V2's graph against Model V1's 67-label map would
        # mislabel every entity rather than fail.
        config = self.using("deberta")
        config.pii["engines"]["deberta"]["label_count"] = 67
        adapter = PIIMaskerAdapter(config)
        with self.assertRaises(Exception):
            adapter.load()


class BothModelsMask(EngineEnvironment):
    def adapter(self, engine: str) -> PIIMaskerAdapter:
        adapter = PIIMaskerAdapter(self.using(engine))
        adapter.load()
        return adapter

    def test_both_engines_mask_a_name_and_a_phone_number(self) -> None:
        for engine in ("tinybert", "deberta"):
            with self.subTest(engine=engine):
                result = self.adapter(engine).mask_text(
                    "Hi this is Sarah Walusiak, my number is 055 123 4567."
                )
                self.assertNotIn("Sarah", result.masked_text)
                self.assertNotIn("Walusiak", result.masked_text)
                self.assertNotIn("055 123 4567", result.masked_text)
                self.assertEqual(result.raw_model_output["engine"], engine)

    def test_both_engines_release_the_categories_this_device_releases(self) -> None:
        # Riyadh scores near 1.0 as a CITY on both, so no threshold reaches
        # this. Asked the time somewhere, the device has to keep the where.
        for engine in ("tinybert", "deberta"):
            with self.subTest(engine=engine):
                result = self.adapter(engine).mask_text(
                    "what time is it right now in Riyadh"
                )
                self.assertIn("Riyadh", result.masked_text)

    def test_both_engines_release_the_devices_own_name(self) -> None:
        # Both read "Onkith" as a person's given name, correctly as far as
        # they know. Every utterance to this device starts with it.
        for engine in ("tinybert", "deberta"):
            with self.subTest(engine=engine):
                result = self.adapter(engine).mask_text("Onkith turn the light off")
                self.assertIn("Onkith", result.masked_text)

    def test_both_engines_report_their_own_latency(self) -> None:
        for engine in ("tinybert", "deberta"):
            with self.subTest(engine=engine):
                details = self.adapter(engine).mask_text("call me on 055 123 4567").raw_model_output
                self.assertGreater(details["pii_model_latency_seconds"], 0.0)
                self.assertIn("span_merger_latency_seconds", details)

    def test_empty_text_costs_neither_engine_a_model_run(self) -> None:
        result = self.adapter("deberta").mask_text("   ")
        self.assertEqual(result.privacy_status, "masked_empty")


class SentencePieceBoundaries(EngineEnvironment):
    """The trap that made Model V2 read 0.59 typed F1 instead of 0.96.

    DeBERTa-v3's pre-tokenizer splits on whitespace only, so ``Walusiak,``
    is one tokenizer word, comma included. Grouping by tokenizer word ids
    appends that comma to the entity: a superset of the gold span, which
    leaves character recall untouched and destroys exact-boundary F1. The
    decoder must derive its groups from character offsets instead.
    """

    def spans(self, engine: str, text: str) -> list[tuple[str, str]]:
        adapter = PIIMaskerAdapter(self.using(engine))
        adapter.load()
        result = adapter.mask_text(text)
        return [
            (entity["entity_type"], entity["text"]) for entity in result.entities
        ]

    def test_trailing_punctuation_stays_out_of_the_entity(self) -> None:
        for engine in ("tinybert", "deberta"):
            with self.subTest(engine=engine):
                for _, text in self.spans(
                    engine, "Hi this is Sarah Walusiak, nice to meet you."
                ):
                    self.assertFalse(
                        text.endswith((",", ".")),
                        f"{engine} put trailing punctuation inside {text!r}",
                    )
                    self.assertEqual(text, text.strip())

    def test_a_leading_space_stays_out_of_the_entity(self) -> None:
        # SentencePiece reports the space before a token as part of it, so
        # an unt rimmed offset walks left into the previous word.
        for engine in ("tinybert", "deberta"):
            with self.subTest(engine=engine):
                for _, text in self.spans(engine, "she is 9 years old"):
                    self.assertEqual(text, text.strip())
                    self.assertNotIn("  ", text)

    def test_an_email_survives_as_one_span_on_both(self) -> None:
        # Tight connectors: the "@" and "." have to join the words either
        # side rather than split them into three entities.
        for engine in ("tinybert", "deberta"):
            with self.subTest(engine=engine):
                result = PIIMaskerAdapter(self.using(engine))
                result.load()
                masked = result.mask_text(
                    "write to rayyan.hadi@example.com about it"
                ).masked_text
                self.assertNotIn("rayyan", masked)
                self.assertNotIn("example.com", masked)


if __name__ == "__main__":
    unittest.main()

"""Score NVIDIA Canary 180M Flash on the same corpus as the other two.

This is the recogniser in FullPipeline_V3. It cannot be scored the way
Moonshine and Whisper are, for two reasons that are worth stating rather
than working around quietly.

It needs torch and NeMo. This project does not install torch and is not
going to, so nothing here is imported until the engine is actually asked
for, and it is left out of a bare `run_eval.py`. It is scored from a
separate interpreter that has those packages:

    ~/.onkith-canary-eval/.venv/bin/python eval/run_eval.py --engine canary

Its adapter exposes `transcribe_segment(CanarySegment) -> SegmentTranscript`
rather than the `transcribe_recording` seam the runtime calls, so it is not
drop-in and cannot be compared by swapping one line. What is replicated
here is the part being measured: the same artifact, pinned by the sha256
in `models/canary/model_manifest.json`, restored the same way, with the
same beam decoder and the same single-turn prompt. The scoring, corpus and
wake logic are shared with the other engines, so the numbers sit in the
same table.

The prompt is the reason this comparison matters. Canary's slots are
source_lang, target_lang, task, pnc and timestamp, and the upstream
adapter raises on any slot outside that set. There is nowhere to put the
device's name. That is the same gap Moonshine has, and on this corpus it
is the whole difference between 25% and 100% wake recall.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

SAMPLE_RATE = 16000
MODEL_ID = "nvidia/canary-180m-flash"
REVISION = "c770bfa0f70f66ad89dbf9827985f999675943a3"
ARTIFACT = "canary-180m-flash.nemo"
# From models/canary/model_manifest.json in FullPipeline_V3. Checked after
# download, so this scores the artifact that build pins and not whatever
# the model card points at today.
SHA256 = "7b97de18b718ef01bf3398715ce8d18486c4d83961e230230b26f5712e112675"

DECODER = {
    "strategy": "beam",
    "beam_size": 2,
    "return_best_hypothesis": True,
    "temperature": 1.0,
    "max_generation_delta": 50,
}
PROMPT_SLOTS = {
    "source_lang": "en",
    "target_lang": "en",
    "task": "asr",
    "pnc": "no",
    "timestamp": "no",
}

# Canary does have a way to be told a word exists, and it is not the prompt.
# TransformerAEDBeamInfer takes a boosting tree of key phrases, which the
# shipped build leaves at alpha 0, meaning off. Comparing Canary against a
# Whisper that has been given its hint while leaving Canary's own switch
# off would not be a comparison, so it is exposed here:
#
#     ONKITH_CANARY_BOOST=2.0 ... eval/run_eval.py --engine canary
#
# The usable window is narrow. Measured on the wake clips, alpha 1.0 fixes
# some, 1.5 to 2.0 is the best it does, and by 3.0 the decoder stops
# transcribing and emits the boosted phrase on its own: every clip comes
# back as "onk". That cliff is the finding, not a tuning detail.
BOOST_ALPHA_ENV = "ONKITH_CANARY_BOOST"


def boost_alpha() -> float:
    import os

    try:
        return float(os.environ.get(BOOST_ALPHA_ENV, "0") or 0)
    except ValueError:
        return 0.0


@dataclass
class CanaryCounters:
    model_resource_load_count: int = 0
    utterance_session_count: int = 0
    callback_count: int = 0
    queue_overflow_count: int = 0
    stt_processing_seconds: float = 0.0


def model_path() -> str:
    """Fetch the pinned artifact and refuse anything that is not it."""

    import hashlib
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(MODEL_ID, ARTIFACT, revision=REVISION)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(
            f"Canary artifact is not the pinned one.\n"
            f"  expected {SHA256}\n  got      {digest}"
        )
    return path


class CanaryEngine:
    """The seam run_eval.py calls, wrapped around NeMo's multitask model."""

    def __init__(self, config: Any):
        self.config = config
        self.counters = CanaryCounters()
        self._model: Any | None = None
        self._boost_alpha = 0.0

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def resource_identity(self) -> str | None:
        return None if self._model is None else f"{MODEL_ID}@{REVISION[:12]}"

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from nemo.collections.asr.models import EncDecMultiTaskModel

        model = EncDecMultiTaskModel.restore_from(
            restore_path=model_path(),
            map_location=torch.device("cpu"),
            strict=True,
        )
        model.to(torch.device("cpu"))
        model.float()
        model.eval()

        # Same beam settings the shipped adapter forces on, rather than
        # whatever the checkpoint was saved with.
        from omegaconf import OmegaConf, open_dict

        decoding = OmegaConf.create(
            OmegaConf.to_container(model.cfg.decoding, resolve=True)
        )
        alpha = boost_alpha()
        with open_dict(decoding):
            decoding.strategy = DECODER["strategy"]
            decoding.temperature = DECODER["temperature"]
            decoding.beam.beam_size = DECODER["beam_size"]
            decoding.beam.return_best_hypothesis = DECODER["return_best_hypothesis"]
            decoding.beam.max_generation_delta = DECODER["max_generation_delta"]
            if alpha:
                # The checkpoint's saved decoding config has no boosting_tree
                # key at all, so the whole node is written rather than
                # assigned into. Triton is off because this is CPU.
                names = [
                    str(n) for n in (self.config.wake.get("names") or []) if str(n)
                ]
                decoding.beam.boosting_tree = OmegaConf.create(
                    {
                        "model_path": None,
                        "key_phrases_file": None,
                        "key_phrases_list": names,
                        "context_score": 1.0,
                        "depth_scaling": 1.0,
                        "unk_score": 0.0,
                        "final_eos_score": 1.0,
                        "score_per_phrase": 0.0,
                        "source_lang": "en",
                        "use_triton": False,
                        "uniform_weights": False,
                        "use_bpe_dropout": False,
                        "num_of_transcriptions": 5,
                        "bpe_alpha": 0.3,
                    }
                )
                decoding.beam.boosting_tree_alpha = alpha
        model.change_decoding_strategy(decoding)
        self._boost_alpha = alpha

        engine = model.decoding.decoding
        if engine.__class__.__name__ != "TransformerAEDBeamInfer":
            raise RuntimeError(
                f"Canary decoder is {engine.__class__.__name__}, "
                "expected TransformerAEDBeamInfer"
            )

        from nemo.collections.asr.models.aed_multitask_models import (
            MultiTaskTranscriptionConfig,
        )

        self._transcription_config = MultiTaskTranscriptionConfig(
            batch_size=1,
            return_hypotheses=True,
            num_workers=0,
            verbose=False,
            timestamps=False,
            prompt={"role": "user", "slots": dict(PROMPT_SLOTS)},
            enable_chunking=False,
        )
        self._model = model
        self.counters.model_resource_load_count += 1

        # Canary's first call carries graph setup that is not part of any
        # later one. Leaving it in would put several seconds of warmup on
        # whichever clip happened to be scored first and quietly inflate
        # the RTF of the whole run.
        import numpy as np

        self._decode(np.zeros(SAMPLE_RATE, dtype="float32"))

    def _decode(self, audio) -> str:
        import numpy as np
        import torch

        with torch.inference_mode():
            outputs = self._model.transcribe(
                audio=[np.ascontiguousarray(audio, dtype=np.float32)],
                override_config=self._transcription_config,
            )
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError("Canary must return exactly one best hypothesis")
        best = outputs[0]
        if isinstance(best, str):
            return best.strip()
        text = getattr(best, "text", None)
        if not isinstance(text, str):
            raise RuntimeError("Canary hypothesis has no text string")
        return text.strip()

    def transcribe_recording(self, recording, partial_callback=None):
        from integrated_pipeline.schemas import STTResult

        self.load()
        audio = recording.samples
        if int(recording.sample_rate) != SAMPLE_RATE:
            raise ValueError(
                f"Canary accepts only 16 kHz audio, got {recording.sample_rate}"
            )
        started = time.perf_counter()
        text = self._decode(audio)
        elapsed = time.perf_counter() - started
        self.counters.utterance_session_count += 1
        self.counters.stt_processing_seconds += elapsed
        duration = len(audio) / SAMPLE_RATE
        return STTResult(
            final_text=text,
            duration_seconds=duration,
            processing_seconds=elapsed,
            finalization_seconds=0.0,
            rtf=elapsed / duration if duration else 0.0,
            callback_count=0,
            session_id=recording.session_id,
            batch_id=recording.batch_id,
            sequence_number=recording.sequence_number,
            captured_audio_seconds=duration,
        )

    def note_queue_overflow(self) -> None:
        self.counters.queue_overflow_count += 1

    def audit(self) -> dict:
        return {
            **vars(self.counters),
            "loaded": self.loaded,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "sha256": SHA256,
            "decoder": dict(DECODER),
            "prompt_slots": dict(PROMPT_SLOTS),
            "boosting_tree_alpha": self._boost_alpha,
            "device": "cpu",
            "precision": "fp32",
        }

    def close(self) -> None:
        self._model = None

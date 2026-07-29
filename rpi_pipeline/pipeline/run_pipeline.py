#!/usr/bin/env python3
"""Microphone -> Moonshine STT -> TinyBERT PII masking."""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
from privacy_masker import PrivacyMasker
PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def resolve(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=PIPELINE_DIR / "config.json")
    p.add_argument("--device", type=int)
    p.add_argument("--once", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()

def print_masked(masker, text):
    try:
        result = masker.mask(text)
    except Exception as exc:
        print(f"PII MASKING ERROR: {exc}", file=sys.stderr)
        return
    print("\n" + "-" * 72)
    print("ORIGINAL:\n" + (result.original_text or "[EMPTY]"))
    print("\nMASKED:\n" + (result.masked_text or "[EMPTY]"))
    if result.entities:
        print("\nENTITIES:")
        for e in result.entities:
            print(f"- {e.label}: {e.text!r} (confidence={e.confidence:.3f})")
    else:
        print("\nENTITIES: none detected")
    print(f"\nPII inference: {result.inference_seconds * 1000:.1f} ms")
    print("-" * 72)

def main():
    args = parse_args()
    cfg = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    stt = load_module(resolve(cfg["moonshine_script"]), "onkith_moonshine_live")
    stt_cfg = stt.AppConfig.from_json(resolve(cfg["moonshine_config"]))
    stt_cfg.validate(); stt.verify_package_version(stt_cfg)
    if args.list_devices:
        stt.list_devices(); return 0
    print("Loading TinyBERT INT8 PII model...")
    masker = PrivacyMasker(resolve(cfg["tinybert_model"]), resolve(cfg["tinybert_tokenizer"]),
                           resolve(cfg["tinybert_id2label"]),
                           max_length=cfg.get("max_length", 256),
                           confidence_threshold=cfg.get("confidence_threshold", 0.0))
    print("TinyBERT loaded.")
    device_id, sample_rate, device_name = stt.resolve_device(args.device)
    print(f"Microphone: {device_name} ({sample_rate:.0f} Hz)")
    threshold = stt.calibrate_noise(stt_cfg, device_id, sample_rate)
    transcriber, _listener, collector = stt.create_transcriber(stt_cfg)
    try:
        while not stt.STOP_REQUESTED.is_set():
            result = stt.capture_utterance(stt_cfg, transcriber, collector, device_id,
                                           sample_rate, threshold, prompt_before_arm=False)
            if result is not None:
                stt.print_result(result)
                transcript = (result.transcript or "").strip()
                if transcript:
                    print_masked(masker, transcript)
            if args.once:
                break
            if not stt.STOP_REQUESTED.is_set():
                print("\nRe-armed for the next utterance. Press Ctrl+C to stop.")
    finally:
        if hasattr(transcriber, "close"):
            transcriber.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

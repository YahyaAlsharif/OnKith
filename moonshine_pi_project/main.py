#!/usr/bin/env python3
"""
Live Raspberry Pi pipeline:
USB mic -> optional RNNoise -> Moonshine integrated VAD/segmenter -> streaming STT.

Why this uses Transcriber instead of MicTranscriber:
- MicTranscriber captures the microphone internally.
- We need to insert 
RNNoise before Moonshine.
- So we capture the mic ourselves using sounddevice, denoise the frames, then call transcriber.add_audio().
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
from typing import Optional

import numpy as np
import sounddevice as sd

from moonshine_pi_project.config import (
    LANGUAGE,
    MIC_BLOCK_MS,
    MIC_CHANNELS,
    MIC_DEVICE,
    MIC_SAMPLE_RATE,
    MOONSHINE_OPTIONS,
    USE_RNNOISE,
)

try:
    from moonshine_voice import Transcriber, TranscriptEventListener, get_model_for_language
except Exception as exc:  # pragma: no cover
    print("[ERROR] Could not import moonshine_voice.", file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    raise exc


class PrintTranscriptListener(TranscriptEventListener):
    """Print transcript events from Moonshine."""

    def on_line_started(self, event):
        text = getattr(event.line, "text", "")
        if text:
            print(f"[START] {text}", flush=True)
        else:
            print("[START]", flush=True)

    def on_line_text_changed(self, event):
        print(f"[LIVE]  {event.line.text}", flush=True)

    def on_line_completed(self, event):
        print(f"[FINAL] {event.line.text}", flush=True)
        # Later: send event.line.text to your privacy classifier here.
        # Example:
        # decision = privacy_classifier.predict(event.line.text)
        # print(f"[PRIVACY] {decision}")


class OptionalRNNoise:
    """Small wrapper around PyRnNoise.

    PyRnNoise expects int16 audio shaped as [channels, samples].
    Moonshine expects mono float32 PCM in [-1.0, 1.0], so we convert after denoising.
    """

    def __init__(self, sample_rate: int):
        from pyrnnoise import RNNoise

        self.denoiser = RNNoise(sample_rate=sample_rate)

    def process_int16_mono(self, audio_i16: np.ndarray) -> np.ndarray:
        """Return denoised mono float32 PCM in [-1, 1]."""
        if audio_i16.ndim != 1:
            audio_i16 = np.squeeze(audio_i16)

        # PyRnNoise expects [channels, samples]
        chunk = audio_i16.reshape(1, -1).astype(np.int16, copy=False)

        frames = []
        for _speech_prob, denoised_frame in self.denoiser.denoise_chunk(chunk, partial=True):
            frame = np.asarray(denoised_frame, dtype=np.int16)
            if frame.ndim == 1:
                frame = frame.reshape(1, -1)
            frames.append(frame)

        if not frames:
            return np.empty(0, dtype=np.float32)

        out_i16 = np.concatenate(frames, axis=1).reshape(-1)
        return int16_to_float32(out_i16)


def int16_to_float32(audio_i16: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 PCM in [-1.0, 1.0]."""
    return (audio_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Moonshine STT on Raspberry Pi with optional RNNoise.")
    parser.add_argument("--list-devices", action="store_true", help="List audio input/output devices and exit.")
    parser.add_argument("--device", type=int, default=MIC_DEVICE, help="Input device id. Default uses system default mic.")
    parser.add_argument("--no-rnnoise", action="store_true", help="Disable RNNoise for comparison/testing.")
    parser.add_argument("--language", default=LANGUAGE, help="Moonshine STT language code, e.g. en.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    use_rnnoise = USE_RNNOISE and not args.no_rnnoise
    blocksize = int(MIC_SAMPLE_RATE * MIC_BLOCK_MS / 1000)
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=20)
    stop_requested = False

    def handle_signal(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("\n[INFO] Stopping...", flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"[INFO] Loading Moonshine model for language={args.language!r}...")
    model_path, model_arch = get_model_for_language(args.language)
    transcriber = Transcriber(
        model_path=model_path,
        model_arch=model_arch,
        options=MOONSHINE_OPTIONS,
    )
    transcriber.add_listener(PrintTranscriptListener())

    denoiser: Optional[OptionalRNNoise] = None
    if use_rnnoise:
        print("[INFO] RNNoise enabled.")
        denoiser = None
    else:
        print("[INFO] RNNoise disabled. Feeding raw mic audio to Moonshine.")

    def audio_callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"[AUDIO WARNING] {status}", file=sys.stderr)
        try:
            # indata shape: [frames, channels], dtype int16
            mono = indata[:, 0].copy()
            audio_q.put_nowait(mono)
        except queue.Full:
            # Drop a chunk rather than blocking the real-time callback.
            print("[WARNING] Audio queue full; dropping audio chunk.", file=sys.stderr)

    print("[INFO] Starting transcriber.")
    transcriber.start()

    print("[INFO] Opening microphone stream.")
    print("[INFO] Speak into the mic. Press Ctrl+C to stop.")

    try:
        with sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=MIC_CHANNELS,
            dtype="int16",
            blocksize=blocksize,
            device=args.device,
            callback=audio_callback,
        ):
            while not stop_requested:
                try:
                    audio_i16 = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if denoiser is not None:
                    audio_f32 = denoiser.process_int16_mono(audio_i16)
                    if audio_f32.size == 0:
                        continue
                else:
                    audio_f32 = int16_to_float32(audio_i16)

                # Moonshine docs specify mono float32 PCM and sample rate.
                # Moonshine handles resampling and integrated VAD/segmentation internally.
                transcriber.add_audio(audio_f32, MIC_SAMPLE_RATE)

    finally:
        print("[INFO] Stopping transcriber.")
        transcriber.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

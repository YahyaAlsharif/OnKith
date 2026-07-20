"""
Config for Raspberry Pi 5 live Moonshine STT with RNNoise in front.

Pipeline:
USB mic -> RNNoise denoising -> Moonshine integrated VAD/segmenter -> Moonshine Streaming STT
"""

# Moonshine STT language. Use "en" first. Change later if needed.
LANGUAGE = "en"

# First version: RNNoise is enabled. Turn False to compare Moonshine alone.
USE_RNNOISE = True

# RNNoise is designed around 48 kHz audio and 10 ms internal frames.
# Moonshine can accept this sample rate and resamples internally.
MIC_SAMPLE_RATE = 48_000
MIC_CHANNELS = 1

# We read 100 ms from the microphone at a time.
# RNNoise will internally process this as 10 x 10 ms frames.
MIC_BLOCK_MS = 100

# Use None for default mic. Set to an integer device id if needed.
MIC_DEVICE = 0

# Moonshine's integrated VAD/segmenter settings.
# Values must be strings because Moonshine options are key/value strings.
MOONSHINE_OPTIONS = {
    "vad_threshold": "0.6",
    "vad_max_segment_duration": "10",
    # Uncomment for debugging. It saves what Moonshine received after RNNoise.
    # "save_input_wav_path": "./debug_audio",
}

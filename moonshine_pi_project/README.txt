Moonshine Raspberry Pi Project - Version 1

Pipeline:
USB mic -> RNNoise denoising -> Moonshine integrated VAD/segmenter -> Moonshine Streaming STT

Install:
  unzip moonshine_pi_project_rnnoise_v1.zip
  cd moonshine_pi_project
  bash install.sh

Run:
  bash run.sh

List microphones:
  source .venv/bin/activate
  python main.py --list-devices

Run with a specific mic device:
  python main.py --device 2

Compare without RNNoise:
  python main.py --no-rnnoise

Notes:
- This version uses Moonshine's integrated VAD/segmenter.
- Current VAD settings are vad_threshold=0.6 and vad_max_segment_duration=10.
- The zip contains the project code. The install script installs Python/system packages and downloads Moonshine model files on the Raspberry Pi.
- We capture audio ourselves instead of using MicTranscriber because we need to insert RNNoise before Moonshine.
- If RNNoise makes WER worse or adds latency, run with --no-rnnoise and compare.

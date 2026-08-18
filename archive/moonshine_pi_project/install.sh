#!/usr/bin/env bash
set -e

# Raspberry Pi OS / Debian setup for live microphone + Moonshine + RNNoise.
sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  portaudio19-dev \
  libasound2-dev \
  build-essential \
  git

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Download Moonshine STT model files once so first run is not delayed.
python -m moonshine_voice.download --stt --language en

echo ""
echo "Install finished."
echo "Run: bash run.sh"
echo "If your mic is not default, list devices with: source .venv/bin/activate && python main.py --list-devices"

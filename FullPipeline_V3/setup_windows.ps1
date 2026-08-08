# Windows LOCAL TEST setup for Omit AI. Not the Raspberry Pi deliverable.
#
# Creates an isolated Python 3.11 conda environment, installs the pinned
# runtime, downloads the pinned Canary model, and runs the self-test.
#
# Windows differs from the Pi in one way: two packages in nemo_toolkit[asr]
# 2.5.3's tree need a C/C++ compiler (ctc_segmentation, texterrors<1.0.0).
# The Pi installer compiles them with build-essential; on Windows that would
# need MSVC, so this script installs NeMo without the [asr] extra plus the
# dependencies the ASR collection actually imports, and uses texterrors 1.1.8
# which publishes a Windows wheel.
#
# Usage:   powershell -ExecutionPolicy Bypass -File setup_windows.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$ENVNAME = "omit311"

function Find-Conda {
    foreach ($p in @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
    )) { if (Test-Path $p) { return $p } }
    $c = Get-Command conda -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    throw "conda not found. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
}

$conda = Find-Conda
Write-Host "[1/6] Using conda: $conda" -ForegroundColor Cyan

$envPath = & $conda env list | Select-String "^\s*$ENVNAME\s"
if (-not $envPath) {
    Write-Host "[2/6] Creating Python 3.11 environment '$ENVNAME'..." -ForegroundColor Cyan
    & $conda create -n $ENVNAME python=3.11 -y | Out-Null
} else {
    Write-Host "[2/6] Environment '$ENVNAME' already exists." -ForegroundColor Cyan
}

$py = & $conda run -n $ENVNAME python -c "import sys; print(sys.executable)"
$py = $py.Trim()
Write-Host "      python: $py"

Write-Host "[3/6] Installing pinned runtime (torch 2.7.1, NeMo 2.5.3)..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip wheel setuptools --quiet
# torch from PyPI, not the +cpu index: the adapter requires the exact string "2.7.1".
& $py -m pip install --quiet "torch==2.7.1" "numpy==2.0.2" "transformers==4.53.3"
& $py -m pip install --quiet "nemo_toolkit==2.5.3"

Write-Host "[4/6] Installing ASR dependencies (skipping compiler-only packages)..." -ForegroundColor Cyan
& $py -m pip install --quiet -r "$ROOT\requirements-windows.txt"

Write-Host "[5/6] Downloading the pinned Canary model (about 700 MB, one time)..." -ForegroundColor Cyan
& $py "$ROOT\scripts\acquire_canary.py"

Write-Host "[6/6] Running the self-test..." -ForegroundColor Cyan
& $py -m app.omit_ai_live --self-test

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "List microphones:  & '$py' -m app.omit_ai_live --list-devices"
Write-Host "Talk to it:        & '$py' -m app.omit_ai_live --device 1 --show-raw"

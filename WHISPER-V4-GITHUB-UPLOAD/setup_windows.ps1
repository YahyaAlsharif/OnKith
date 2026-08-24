# Windows LOCAL TEST setup for Omit AI. Not the Raspberry Pi deliverable.
#
# Creates an isolated Python 3.11 conda environment, installs the pinned
# runtime, caches the Whisper model, and runs the self-test.
#
# V3 needed a long Windows-specific dependency list here because two packages
# in nemo_toolkit[asr] 2.5.3's tree need a C/C++ compiler. V4's recognizer is
# faster-whisper on CTranslate2, which publishes Windows wheels, so the same
# requirements file works on both platforms.
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

Write-Host "Downloading and verifying required local models..." -ForegroundColor Cyan
& $py "$ROOT\scripts\download_large_models.py"
if ($LASTEXITCODE -ne 0) { throw "Model download failed" }

Write-Host "[3/5] Upgrading pip..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip wheel setuptools --quiet

Write-Host "[4/5] Installing the pinned runtime (faster-whisper, ONNX Runtime)..." -ForegroundColor Cyan
& $py -m pip install --quiet -r "$ROOT\requirements-windows.txt"

Write-Host "[5/5] Caching faster-whisper base.en (about 150 MB, one time) and running the self-test..." -ForegroundColor Cyan
& $py -m app.omit_ai_live --self-test

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "List microphones:  & '$py' -m app.omit_ai_live --list-devices"
Write-Host "Talk to it:        & '$py' -m app.omit_ai_live --device 1 --show-raw"

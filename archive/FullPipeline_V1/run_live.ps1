<#
.SYNOPSIS
    Windows launcher for the OnKith live speech privacy pipeline.

.DESCRIPTION
    Windows equivalent of run_live.sh. Resolves the project root from this
    script's own location, requires the project-local virtual environment, and
    forwards every argument to app\omit_ai_live.py unchanged.

    WSL is not required.

.EXAMPLE
    .\run_live.ps1 --list-devices

.EXAMPLE
    .\run_live.ps1 --text-test "My email is rayyan@example.com"

.EXAMPLE
    .\run_live.ps1 --device 2 --once --show-raw
#>

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'

$RootDir = $PSScriptRoot
$VenvPython = Join-Path $RootDir '.venv\Scripts\python.exe'
$EntryPoint = Join-Path $RootDir 'app\omit_ai_live.py'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Error @"
Virtual environment not found at:
  $VenvPython

Create it and install the dependencies first:
  cd "$RootDir"
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip wheel setuptools
  python -m pip install -r requirements-rpi.txt
"@
    exit 1
}

if (-not (Test-Path -LiteralPath $EntryPoint)) {
    Write-Error "Pipeline entry point not found at: $EntryPoint"
    exit 1
}

Push-Location -LiteralPath $RootDir
try {
    if ($null -eq $Arguments) { $Arguments = @() }
    & $VenvPython $EntryPoint @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

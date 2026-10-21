[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]] $Paths,

    [ValidateSet("1", "2", "3", "4")]
    [string] $Mode,

    [string] $Endpoint = "http://127.0.0.1:6969",

    [switch] $Overwrite
)

$ErrorActionPreference = "Stop"
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $scriptDir "lyricsee.py"

if (-not $Paths -or $Paths.Count -eq 0) {
    Write-Host "Drag one or more audio files onto Lyricsee.bat or Lyricsee.cmd." -ForegroundColor Yellow
    Write-Host "Supported search inputs: FLAC, MP3, M4A, WAV, OGG, OPUS, AAC, WMA, AIFF, ALAC." -ForegroundColor Yellow
    exit 2
}

if (-not $Mode) {
    # Default is always verified-source / no-model mode.
    # Model calls require the caller to explicitly pass -Mode 2.
    $Mode = "1"
    Write-Host "Lyricsee: verified web translation mode (no LLM / no NetEase)" -ForegroundColor Cyan
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python was not found on PATH."
}
if (-not (Test-Path -LiteralPath $backend -PathType Leaf)) {
    throw "Backend not found: $backend"
}

$modeName = switch ($Mode) {
    "1" { "web" }
    "2" { "llm" }
    "3" { "deepl" }
    "4" { "mymemory" }
}
$arguments = @($backend, "--mode", $modeName, "--endpoint", $Endpoint)
if ($Overwrite) {
    $arguments += "--overwrite"
}
$arguments += $Paths

& $python.Source @arguments
exit $LASTEXITCODE

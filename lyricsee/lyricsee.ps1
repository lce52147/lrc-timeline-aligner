[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]] $Paths,

    [string] $Endpoint,

    [string] $OutputRoot,

    [string] $BkDir,

    [switch] $NoTranslate,

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
    Write-Host "Drag one or more audio files or an M3U8 playlist onto Lyricsee.bat or Lyricsee.cmd." -ForegroundColor Yellow
    Write-Host "Supported audio: FLAC, MP3, M4A, WAV, OGG, OPUS, AAC, WMA, AIFF, ALAC." -ForegroundColor Yellow
    exit 2
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python was not found on PATH."
}
if (-not (Test-Path -LiteralPath $backend -PathType Leaf)) {
    throw "Backend not found: $backend"
}

$arguments = @($backend)
if ($PSBoundParameters.ContainsKey("Endpoint")) {
    $arguments += @("--endpoint", $Endpoint)
}
if ($PSBoundParameters.ContainsKey("OutputRoot")) {
    $arguments += @("--output-root", $OutputRoot)
}
if ($PSBoundParameters.ContainsKey("BkDir")) {
    $arguments += @("--bk-dir", $BkDir)
}
if ($NoTranslate) {
    $arguments += "--no-translate"
}
if ($Overwrite) {
    $arguments += "--overwrite"
}
$arguments += $Paths

& $python.Source @arguments
exit $LASTEXITCODE
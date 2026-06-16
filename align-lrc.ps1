[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]] $Paths,

    [ValidateSet("energy", "even")]
    [string] $Mode = "energy",

    [ValidateSet("auto", "lyrics", "whisperx", "whispercpp", "heuristic", "audio")]
    [string] $TimingSource = "auto",

    [string] $WhisperCli,

    [string] $WhisperModel,

    [string] $WhisperXPython,

    [string] $WhisperXDevice = "auto",

    [string] $WhisperLanguage = "ja",

    [string] $Output,

    [switch] $Overwrite
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $scriptDir "scripts\auto_lrc.py"

if (-not (Test-Path -LiteralPath $backend)) {
    throw "Backend not found: $backend"
}

if (-not $Paths -or $Paths.Count -eq 0) {
    Write-Host "Drag one or more .flac files onto Align LRC.bat, or run:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\align-lrc.ps1 <song.flac>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Lyrics may be beside the FLAC as <song>.lyrics.lrc, <song>.lyrics.txt, or <song>.txt." -ForegroundColor Yellow
    Write-Host "If the FLAC lives under a Music folder, <Music>\<song>.lrc is also accepted as a checked source." -ForegroundColor Yellow
    exit 2
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "Python was not found on PATH."
}

$argsList = @($backend, "--mode", $Mode, "--timing-source", $TimingSource)
if ($WhisperCli) {
    $argsList += @("--whisper-cli", $WhisperCli)
}
if ($WhisperModel) {
    $argsList += @("--whisper-model", $WhisperModel)
}
if ($WhisperXPython) {
    $argsList += @("--whisperx-python", $WhisperXPython)
}
if ($WhisperXDevice) {
    $argsList += @("--whisperx-device", $WhisperXDevice)
}
if ($WhisperLanguage) {
    $argsList += @("--whisper-language", $WhisperLanguage)
}
if ($Overwrite) {
    $argsList += "--overwrite"
}
if ($Output) {
    if ($Paths.Count -ne 1) {
        throw "-Output can only be used with exactly one FLAC path."
    }
    $argsList += @("--output", $Output)
}
$argsList += $Paths

Write-Host "LRC tools: generating same-folder .lrc" -ForegroundColor Cyan
Write-Host "Timing source: $TimingSource" -ForegroundColor DarkCyan
if ($TimingSource -in @("heuristic", "audio")) {
    Write-Host "Heuristic mode: $Mode" -ForegroundColor DarkCyan
}
elseif ($TimingSource -eq "whisperx") {
    Write-Host "Backend request: whisperx hybrid" -ForegroundColor DarkCyan
    Write-Host "WhisperX device request: $WhisperXDevice" -ForegroundColor DarkCyan
}

& $pythonCommand.Source @argsList
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    foreach ($rawPath in $Paths) {
        $audioPath = [System.IO.Path]::GetFullPath($rawPath)
        if ($Output -and $Paths.Count -eq 1) {
            $lrcPath = [System.IO.Path]::GetFullPath($Output)
        }
        else {
            $lrcPath = [System.IO.Path]::ChangeExtension($audioPath, ".lrc")
        }
        $reportPath = [System.IO.Path]::ChangeExtension($lrcPath, ".align-report.json")
        if (-not (Test-Path -LiteralPath $reportPath)) {
            Write-Warning "Alignment report was not written: $reportPath"
            continue
        }
        $report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $backend = if ($report.backend) { $report.backend } else { "(missing)" }
        $resolved = if ($report.resolved_timing_source) { $report.resolved_timing_source } else { "(missing)" }
        $strategy = if ($report.strategy) { $report.strategy } else { "(missing)" }
        Write-Host "Report: $reportPath" -ForegroundColor DarkGray
        Write-Host "Resolved timing source: $resolved" -ForegroundColor Green
        Write-Host "Backend: $backend" -ForegroundColor Green
        Write-Host "Strategy: $strategy" -ForegroundColor Green
        if ($report.whisperx_device) {
            Write-Host "WhisperX device: $($report.whisperx_device)" -ForegroundColor Green
        }
        if ($TimingSource -eq "whisperx" -and $backend -ne "whisperx") {
            throw "Explicit -TimingSource whisperx did not produce a whisperx backend. Actual backend: $backend"
        }
    }
}

exit $exitCode

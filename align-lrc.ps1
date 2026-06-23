[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]] $Paths,

    [ValidateSet("energy", "even")]
    [string] $Mode = "energy",

    [ValidateSet("auto", "lyrics", "ctc", "jactc", "whisperx", "whispercpp", "heuristic", "audio")]
    [string] $TimingSource = "auto",

    [string] $WhisperCli,

    [string] $WhisperModel,

    [string] $WhisperXPython,

    [string] $WhisperXDevice = "auto",

    [string] $WhisperLanguage = "auto",

    [string] $Output,

    [string] $Lyrics,

    [string] $ReportDirectory,

    [switch] $VocalOnsetRefine,

    [switch] $NoCheckedLrcHint,

    [string] $AnchorHints,

    [switch] $NoAnchorHints,

    [switch] $FailOnReviewRequired,

    [double] $MinTrustedPercent = -1,

    [switch] $StrictReview,

    [switch] $Overwrite
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $scriptDir "scripts\auto_lrc.py"
$defaultReportDirectory = Join-Path $scriptDir "outputs\reports"
if (-not $ReportDirectory) {
    $ReportDirectory = $defaultReportDirectory
}
$ReportDirectory = [System.IO.Path]::GetFullPath($ReportDirectory)

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

# A lyric file is a convenient drag target too: resolve its sibling FLAC and
# feed the dropped file to the backend as the explicit lyric source.
$normalizedPaths = @()
$droppedLyrics = $null
foreach ($rawPath in $Paths) {
    $inputPath = [System.IO.Path]::GetFullPath($rawPath)
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "Dropped path was not found: $inputPath"
    }

    $extension = [System.IO.Path]::GetExtension($inputPath).ToLowerInvariant()
    if ($extension -eq ".flac") {
        $normalizedPaths += $inputPath
        continue
    }

    if ($extension -notin @(".txt", ".lrc")) {
        throw "Drop a .flac, or a .txt/.lrc lyric file beside its matching FLAC: $inputPath"
    }
    if ($Paths.Count -ne 1) {
        throw "A dropped lyric file must be processed by itself. Drop FLAC files together, or one .txt/.lrc file."
    }
    if ($Lyrics) {
        throw "Do not combine a dropped lyric file with -Lyrics. Drop the FLAC instead."
    }

    $stem = [System.IO.Path]::GetFileNameWithoutExtension($inputPath)
    if ($stem.EndsWith(".lyrics", [System.StringComparison]::OrdinalIgnoreCase)) {
        $stem = $stem.Substring(0, $stem.Length - ".lyrics".Length)
    }
    # Explorer-created duplicate names can carry an extra trailing dot.
    $stem = $stem.TrimEnd(".")
    $flacPath = Join-Path ([System.IO.Path]::GetDirectoryName($inputPath)) ($stem + ".flac")
    if (-not (Test-Path -LiteralPath $flacPath -PathType Leaf)) {
        throw "No matching FLAC was found for dropped lyric file: $inputPath`nExpected: $flacPath"
    }

    $normalizedPaths += [System.IO.Path]::GetFullPath($flacPath)
    $droppedLyrics = $inputPath
}
$Paths = $normalizedPaths
if ($droppedLyrics) {
    $Lyrics = $droppedLyrics
    Write-Host "Dropped lyric source: $Lyrics" -ForegroundColor DarkCyan
    Write-Host "Resolved FLAC: $($Paths[0])" -ForegroundColor DarkCyan
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
if ($Lyrics) {
    if ($Paths.Count -ne 1) {
        throw "-Lyrics can only be used with exactly one FLAC path."
    }
    $argsList += @("--lyrics", $Lyrics)
}
if ($VocalOnsetRefine) {
    $argsList += "--vocal-onset-refine"
}
if ($Overwrite) {
    $argsList += "--overwrite"
}
if ($NoCheckedLrcHint) {
    $argsList += "--no-checked-lrc-hint"
}
if ($AnchorHints) {
    $argsList += @("--anchor-hints", $AnchorHints)
}
if ($NoAnchorHints) {
    $argsList += "--no-anchor-hints"
}
if ($FailOnReviewRequired) {
    $argsList += "--fail-on-review-required"
}
if ($MinTrustedPercent -ge 0) {
    $argsList += @("--min-trusted-percent", $MinTrustedPercent)
}
if ($StrictReview) {
    $argsList += "--strict-review"
}
if ($Output) {
    if ($Paths.Count -ne 1) {
        throw "-Output can only be used with exactly one FLAC path."
    }
    $argsList += @("--output", $Output)
}
$argsList += @("--report-dir", $ReportDirectory)
$argsList += $Paths

Write-Host "LRC tools: generating same-folder .lrc" -ForegroundColor Cyan
Write-Host "Reports: $ReportDirectory" -ForegroundColor DarkGray
if ($VocalOnsetRefine) {
    Write-Host "Vocal onset refinement: experimental Demucs GPU candidate tiebreak enabled" -ForegroundColor DarkCyan
}
Write-Host "Timing source: $TimingSource" -ForegroundColor DarkCyan
if ($TimingSource -eq "auto") {
    Write-Host "Backend request: checked LRC hint, otherwise CTC + WhisperX candidate selection" -ForegroundColor DarkCyan
    Write-Host "Device request: $WhisperXDevice" -ForegroundColor DarkCyan
}
elseif ($TimingSource -in @("heuristic", "audio")) {
    Write-Host "Heuristic mode: $Mode" -ForegroundColor DarkCyan
}
elseif ($TimingSource -eq "whisperx") {
    Write-Host "Backend request: whisperx hybrid" -ForegroundColor DarkCyan
    Write-Host "WhisperX device request: $WhisperXDevice" -ForegroundColor DarkCyan
}
elseif ($TimingSource -eq "ctc") {
    Write-Host "Backend request: MMS/CTC forced alignment" -ForegroundColor DarkCyan
    Write-Host "CTC device request: $WhisperXDevice" -ForegroundColor DarkCyan
}
elseif ($TimingSource -eq "jactc") {
    Write-Host "Backend request: experimental Japanese Wav2Vec2 CTC" -ForegroundColor DarkCyan
    Write-Host "This backend is a comparison candidate, not an auto-selection winner." -ForegroundColor Yellow
}
if ($StrictReview) {
    Write-Host "Strict review: enabled (fails on review-required or less than 100% trusted)" -ForegroundColor Yellow
}
elseif ($FailOnReviewRequired -or $MinTrustedPercent -ge 0) {
    Write-Host "Strict gate: fail-on-review=$FailOnReviewRequired min-trusted=$MinTrustedPercent" -ForegroundColor Yellow
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
        if (-not (Test-Path -LiteralPath $ReportDirectory)) {
            Write-Warning "Alignment report directory was not written: $ReportDirectory"
            continue
        }
        $reportPath = $null
        $report = $null
        $reportCandidates = Get-ChildItem -LiteralPath $ReportDirectory -File -Filter "*.align-report.json" |
            Sort-Object LastWriteTime -Descending
        foreach ($candidate in $reportCandidates) {
            $candidateReport = Get-Content -LiteralPath $candidate.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($candidateReport.audio_path -eq $audioPath) {
                $reportPath = $candidate.FullName
                $report = $candidateReport
                break
            }
        }
        if (-not $reportPath) {
            Write-Warning "Alignment report was not found for: $audioPath"
            continue
        }
        $backend = if ($report.backend) { $report.backend } else { "(missing)" }
        $resolved = if ($report.resolved_timing_source) { $report.resolved_timing_source } else { "(missing)" }
        $strategy = if ($report.strategy) { $report.strategy } else { "(missing)" }
        Write-Host "Report: $reportPath" -ForegroundColor DarkGray
        Write-Host "Resolved timing source: $resolved" -ForegroundColor Green
        Write-Host "Backend: $backend" -ForegroundColor Green
        Write-Host "Strategy: $strategy" -ForegroundColor Green
        if ($null -ne $report.trusted_percent) {
            Write-Host "Trusted timing: $($report.trusted_percent)%" -ForegroundColor Green
        }
        if ($null -ne $report.review_required_count) {
            Write-Host "Review required: $($report.review_required_count)" -ForegroundColor Green
        }
        if ($null -ne $report.low_confidence_count) {
            Write-Host "Low confidence: $($report.low_confidence_count)" -ForegroundColor Green
        }
        if ($null -ne $report.timing_trusted_entries) {
            Write-Host "Timing-trusted entries: $($report.timing_trusted_entries)" -ForegroundColor Green
        }
        if ($report.whisperx_device) {
            Write-Host "WhisperX device: $($report.whisperx_device)" -ForegroundColor Green
        }
        if ($report.ctc_device) {
            Write-Host "CTC device: $($report.ctc_device)" -ForegroundColor Green
        }
        if ($report.candidate_selection) {
            Write-Host "Auto selected: $($report.candidate_selection.selected_backend) (quality $($report.candidate_selection.selected_quality))" -ForegroundColor Green
            Write-Host "Selection reason: $($report.candidate_selection.selected_reason)" -ForegroundColor DarkGray
        }
        if ($TimingSource -eq "whisperx" -and $backend -ne "whisperx") {
            throw "Explicit -TimingSource whisperx did not produce a whisperx backend. Actual backend: $backend"
        }
        if ($TimingSource -eq "ctc" -and $backend -ne "ctc") {
            throw "Explicit -TimingSource ctc did not produce a ctc backend. Actual backend: $backend"
        }
        if ($TimingSource -eq "jactc" -and $backend -ne "jactc") {
            throw "Explicit -TimingSource jactc did not produce a jactc backend. Actual backend: $backend"
        }
    }
}

exit $exitCode

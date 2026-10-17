[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true, Position = 0)]
    [string[]] $Paths,

    [ValidateSet("energy", "even")]
    [string] $Mode = "energy",

    [ValidateSet("auto", "lyrics", "ctc", "jactc", "whisperx", "whispercpp", "heuristic", "audio")]
    [string] $TimingSource = "auto",

    [ValidateSet("off", "gross-rescue", "reviewer-validity")]
    [string] $Arbiter = "reviewer-validity",

    [ValidateSet("default", "ctc-first")]
    [string] $TimingSourcePolicy = "ctc-first",

    [string] $ReviewerEvidence,

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

    [switch] $IncludeHubReviewer,

    [ValidateSet("none", "strict", "balanced", "loose")]
    [string] $ReviewerProfile = "none",

    [switch] $Overwrite
)

$ErrorActionPreference = "Stop"
$previousConsoleOutputEncoding = [Console]::OutputEncoding
$previousOutputEncoding = $OutputEncoding
$previousPythonEncoding = $env:PYTHONIOENCODING
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $scriptDir "scripts\auto_lrc.py"
$r2Pipeline = Join-Path $scriptDir "scripts\r2_pipeline.py"
$localFfmpegBin = Join-Path $scriptDir "tools\ffmpeg\bin"
if ((Test-Path -LiteralPath (Join-Path $localFfmpegBin "ffmpeg.exe")) -and
    (Test-Path -LiteralPath (Join-Path $localFfmpegBin "ffprobe.exe"))) {
    $env:PATH = "$localFfmpegBin;$env:PATH"
}
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
    Write-Host "Use an explicit lyric/template file when you need to preserve known text or timestamps." -ForegroundColor Yellow
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

if ($Lyrics -and $Paths.Count -eq 1) {
    $lyricSourcePath = [System.IO.Path]::GetFullPath($Lyrics)
    $effectiveOutputPath = if ($Output) {
        [System.IO.Path]::GetFullPath($Output)
    } else {
        [System.IO.Path]::ChangeExtension($Paths[0], ".lrc")
    }
    if ([System.String]::Equals(
        $lyricSourcePath,
        $effectiveOutputPath,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to overwrite lyric source: $lyricSourcePath"
    }
}

$localPython = Join-Path $scriptDir ".venv-asr\Scripts\python.exe"
if (Test-Path -LiteralPath $localPython -PathType Leaf) {
    $pythonExe = $localPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python was not found. Run setup.ps1 to create .venv-asr, or install Python 3.11 on PATH."
    }
    $pythonExe = $pythonCommand.Source
}

$argsList = @($backend, "--mode", $Mode, "--timing-source", $TimingSource, "--arbiter", $Arbiter, "--timing-source-policy", $TimingSourcePolicy)
if ($ReviewerEvidence) {
    if ($Paths.Count -ne 1) {
        throw "-ReviewerEvidence can only be used with exactly one FLAC path."
    }
    $argsList += @("--reviewer-evidence", $ReviewerEvidence)
}
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

Write-Host "LRC tools: generating same-folder .lrc" -ForegroundColor Cyan
Write-Host "Reports: $ReportDirectory" -ForegroundColor DarkGray
Write-Host "Arbiter: $Arbiter" -ForegroundColor DarkCyan
Write-Host "Timing policy: $TimingSourcePolicy" -ForegroundColor DarkCyan
if ($VocalOnsetRefine) {
    Write-Host "Vocal onset refinement: experimental Demucs GPU candidate tiebreak enabled" -ForegroundColor DarkCyan
}
Write-Host "Timing source: $TimingSource" -ForegroundColor DarkCyan
if ($TimingSource -eq "auto") {
    Write-Host "Backend request: automatic timing candidate selection" -ForegroundColor DarkCyan
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

$inputRoot = [System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($Paths[0]))
$runStartIso = [DateTimeOffset]::UtcNow.ToString("o")
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$songCount = $Paths.Count
$completedSeconds = @()

function Format-Elapsed([double] $seconds) {
    $safeSeconds = [Math]::Max(0.0, $seconds)
    $span = [TimeSpan]::FromSeconds($safeSeconds)
    return "{0}:{1:00}:{2:00}" -f [int][Math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds
}

if ($Arbiter -eq "reviewer-validity" -and -not $ReviewerEvidence) {
    if (-not (Test-Path -LiteralPath $r2Pipeline -PathType Leaf)) {
        throw "R2 pipeline not found: $r2Pipeline"
    }
    Write-Host "R2 reviewer acquisition: enabled" -ForegroundColor DarkCyan
    Write-Host "Reviewer trust profile: $ReviewerProfile" -ForegroundColor DarkCyan
}

Write-Host "$songCount songs queued" -ForegroundColor Cyan
$exitCode = 0
$songIndex = 0
foreach ($rawPath in $Paths) {
    $songIndex++
    $songName = [System.IO.Path]::GetFileName($rawPath)
    $songStarted = $stopwatch.Elapsed.TotalSeconds
    Write-Host ("[{0}/{1}] {2} elapsed {3} ETA -- START" -f $songIndex, $songCount, $songName, (Format-Elapsed $songStarted)) -ForegroundColor DarkCyan
    $songExitCode = 1
    try {
        if ($Arbiter -eq "reviewer-validity" -and -not $ReviewerEvidence) {
            $pipelineArgs = @(
                $r2Pipeline,
                "--reviewer-device", $WhisperXDevice,
                "--reviewer-profile", $ReviewerProfile
            )
            if ($WhisperXPython) {
                $pipelineArgs += @("--reviewer-python", $WhisperXPython)
            }
            if ($IncludeHubReviewer) {
                $pipelineArgs += "--include-hub-reviewer"
            }
            $pipelineArgs += "--"
            if ($argsList.Count -gt 1) {
                $pipelineArgs += $argsList[1..($argsList.Count - 1)]
            }
            $pipelineArgs += $rawPath
            & $pythonExe @pipelineArgs
            $songExitCode = $LASTEXITCODE
        }
        else {
            $argsList += $rawPath
            & $pythonExe @argsList
            $songExitCode = $LASTEXITCODE
            $argsList = @($argsList[0..($argsList.Count - 2)])
        }
    }
    catch {
        Write-Warning ("Alignment failed for {0}: {1}" -f $songName, $_.Exception.Message)
        $songExitCode = 1
        if ($argsList.Count -gt 0 -and $argsList[-1] -eq $rawPath) {
            $argsList = @($argsList[0..($argsList.Count - 2)])
        }
    }

    $songElapsed = $stopwatch.Elapsed.TotalSeconds - $songStarted
    $completedSeconds += $songElapsed
    if ($songExitCode -ne 0) {
        $exitCode = $songExitCode
    }
    $remaining = $songCount - $songIndex
    if ($remaining -eq 0) {
        $eta = "0:00:00"
    }
    else {
        $eta = Format-Elapsed ((($completedSeconds | Measure-Object -Average).Average) * $remaining)
    }
    Write-Host ("[{0}/{1}] {2} elapsed {3} ETA {4} exit {5} END" -f $songIndex, $songCount, $songName, (Format-Elapsed $songElapsed), $eta, $songExitCode) -ForegroundColor DarkCyan
}

$summaryScript = Join-Path $scriptDir "scripts\lrc_batch_summary.py"
if (Test-Path -LiteralPath $summaryScript -PathType Leaf) {
    $savedErrorActionPreference = $ErrorActionPreference
    $nativePreferenceVariable = Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue
    $savedNativePreference = if ($nativePreferenceVariable) { $nativePreferenceVariable.Value } else { $null }
    try {
        $ErrorActionPreference = "Continue"
        $PSNativeCommandUseErrorActionPreference = $false
        $summaryArgs = @(
            $summaryScript,
            $inputRoot,
            "--since", $runStartIso,
            "--report-dir", $ReportDirectory,
            "--work-dir", $scriptDir
        )
        & $pythonExe @summaryArgs
        $summaryExitCode = $LASTEXITCODE
        if ($summaryExitCode -ne 0) {
            Write-Warning "Batch summary failed; preserving alignment exit code $exitCode."
        }
    }
    catch {
        Write-Warning ("Batch summary failed; preserving alignment exit code {0}: {1}" -f $exitCode, $_.Exception.Message)
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
        if ($nativePreferenceVariable) {
            $PSNativeCommandUseErrorActionPreference = $savedNativePreference
        }
        else {
            Remove-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue
        }
    }
}
else {
    Write-Warning "Batch summary script was not found: $summaryScript"
}

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
        if ($report.review_tsv_path) { Write-Host "Review TSV: $($report.review_tsv_path)" -ForegroundColor Yellow }
        Write-Host "Resolved timing source: $resolved" -ForegroundColor Green
        Write-Host "Backend: $backend" -ForegroundColor Green
        Write-Host "Strategy: $strategy" -ForegroundColor Green
        if ($null -ne $report.timing_trusted_percent) {
            Write-Host "Trusted timing: $($report.timing_trusted_percent)%" -ForegroundColor Green
        }
        if ($null -ne $report.overall_trusted_percent) {
            Write-Host "Full authority trust: $($report.overall_trusted_percent)%" -ForegroundColor Green
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

$reviewBatchPath = [System.IO.Path]::GetFullPath((Join-Path $scriptDir "Review LRC.bat"))
Write-Host "Review LRC.bat: $reviewBatchPath" -ForegroundColor Yellow
[Console]::OutputEncoding = $previousConsoleOutputEncoding
$OutputEncoding = $previousOutputEncoding
if ($null -eq $previousPythonEncoding) {
    Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue
}
else {
    $env:PYTHONIOENCODING = $previousPythonEncoding
}
exit $exitCode

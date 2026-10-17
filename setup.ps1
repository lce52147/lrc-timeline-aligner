[CmdletBinding()]
param(
    [switch] $Force,
    [switch] $Prefetch,
    [switch] $Cuda,
    [switch] $Cpu,
    [switch] $Check
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

if ($Cuda -and $Cpu) {
    throw "Choose only one of -Cuda or -Cpu."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $scriptDir ".venv-asr"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$downloadDir = Join-Path $scriptDir "tools\downloads"
$ffmpegBin = Join-Path $scriptDir "tools\ffmpeg\bin"
$doctorJson = Join-Path $scriptDir "outputs\setup-doctor.json"

function Get-FileSha256([string] $Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Download-VerifiedFile(
    [string] $Uri,
    [string] $Destination,
    [string] $Sha256
) {
    $Sha256 = $Sha256.ToLowerInvariant()
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $actual = Get-FileSha256 $Destination
        if ($actual -eq $Sha256) {
            Write-Host "Download cache hit: $Destination" -ForegroundColor DarkGreen
            return
        }
        Write-Warning "Existing file hash mismatch; downloading a verified replacement: $Destination"
        Remove-Item -LiteralPath $Destination -Force
    }

    $partial = "$Destination.partial"
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    $downloaded = $false
    for ($attempt = 1; $attempt -le 3 -and -not $downloaded; $attempt++) {
        Write-Host "Download ($attempt/3): $Uri" -ForegroundColor Cyan
        if ($curl) {
            & $curl.Source -L --fail --retry 3 --retry-delay 2 --retry-all-errors -C - -o $partial $Uri
            if ($LASTEXITCODE -ne 0) {
                if ($attempt -eq 1 -and (Test-Path -LiteralPath $partial)) {
                    Remove-Item -LiteralPath $partial -Force
                }
                continue
            }
        } else {
            try {
                Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $partial
            } catch {
                if ($attempt -eq 3) { throw }
                continue
            }
        }

        if (-not (Test-Path -LiteralPath $partial -PathType Leaf)) {
            continue
        }
        $actual = Get-FileSha256 $partial
        if ($actual -ne $Sha256) {
            Remove-Item -LiteralPath $partial -Force
            if ($attempt -eq 3) {
                throw "SHA256 mismatch for $Uri. Expected $Sha256, got $actual."
            }
            continue
        }
        Move-Item -LiteralPath $partial -Destination $Destination -Force
        $downloaded = $true
    }
    if (-not $downloaded) {
        throw "Download failed after retries: $Uri"
    }
}

function Resolve-Python311 {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        & $py.Source -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Exe = $py.Source; Args = @("-3.11") }
        }
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            return @{ Exe = $python.Source; Args = @() }
        }
    }

    throw "Python 3.11 was not found. Install 64-bit Python 3.11, then rerun: powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1"
}

function Invoke-VenvPython([string[]] $Arguments) {
    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $venvPython $($Arguments -join ' ')"
    }
}

# Remove Mark-of-the-Web from an extracted release without requiring admin rights.
Get-ChildItem -LiteralPath $scriptDir -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { -not $_.FullName.StartsWith($venvDir, [System.StringComparison]::OrdinalIgnoreCase) } |
    ForEach-Object { Unblock-File -LiteralPath $_.FullName -ErrorAction SilentlyContinue }

if (Test-Path -LiteralPath (Join-Path $ffmpegBin "ffmpeg.exe")) {
    $env:PATH = "$ffmpegBin;$env:PATH"
}

if ($Check) {
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw ".venv-asr is missing. Run setup.ps1 first."
    }
    $expect = if ($Cuda) { "yes" } elseif ($Cpu) { "no" } else { "auto" }
    $doctorArgs = @("scripts\doctor.py", "--expect-cuda", $expect, "--json-out", $doctorJson)
    if ($Prefetch) { $doctorArgs += "--prefetch" }
    Push-Location $scriptDir
    try {
        Invoke-VenvPython $doctorArgs
    } finally {
        Pop-Location
    }
    exit 0
}

if ($Force -and (Test-Path -LiteralPath $venvDir)) {
    Write-Host "Rebuilding .venv-asr" -ForegroundColor Yellow
    Remove-Item -LiteralPath $venvDir -Recurse -Force
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    $launcher = Resolve-Python311
    $venvArgs = @($launcher.Args) + @("-m", "venv", $venvDir)
    Write-Host "Creating .venv-asr with Python 3.11" -ForegroundColor Cyan
    & $launcher.Exe @venvArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw "Failed to create .venv-asr."
    }
} else {
    Write-Host "Reusing .venv-asr" -ForegroundColor DarkGreen
}

$nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
$hasNvidia = $false
if ($nvidia) {
    & $nvidia.Source -L *> $null
    $hasNvidia = $LASTEXITCODE -eq 0
}
if ($Cuda -and -not $hasNvidia) {
    throw "-Cuda was requested, but nvidia-smi did not detect a usable NVIDIA GPU."
}
if ($Cuda) {
    $useCuda = $true
} elseif ($Cpu) {
    $useCuda = $false
} else {
    $useCuda = $hasNvidia
}

if ($useCuda) {
    $torchIndex = "https://download.pytorch.org/whl/cu128"
    $torchVersion = "2.8.0+cu128"
    $expectCuda = "yes"
    Write-Host "Runtime target: CUDA 12.8" -ForegroundColor Green
} else {
    $torchIndex = "https://download.pytorch.org/whl/cpu"
    $torchVersion = "2.8.0+cpu"
    $expectCuda = "no"
    Write-Warning "No NVIDIA GPU runtime selected. CPU alignment is supported but can be very slow."
}

Invoke-VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-VenvPython @(
    "-m", "pip", "install",
    "--index-url", $torchIndex,
    "torch==$torchVersion",
    "torchaudio==$torchVersion"
)

$constraint = Join-Path $env:TEMP "lrc-tools-torch-$PID.txt"
try {
    @(
        "torch==$torchVersion",
        "torchaudio==$torchVersion"
    ) | Set-Content -LiteralPath $constraint -Encoding ASCII

    Invoke-VenvPython @(
        "-m", "pip", "install",
        "--extra-index-url", $torchIndex,
        "--constraint", $constraint,
        "-r", (Join-Path $scriptDir "requirements-asr.lock.txt")
    )
    Invoke-VenvPython @(
        "-m", "pip", "install",
        "--extra-index-url", $torchIndex,
        "--constraint", $constraint,
        "-r", (Join-Path $scriptDir "requirements.txt")
    )
} finally {
    Remove-Item -LiteralPath $constraint -Force -ErrorAction SilentlyContinue
}

Invoke-VenvPython @(
    "-c",
    "import torch,torchaudio; assert torch.__version__ == '$torchVersion', torch.__version__; assert torchaudio.__version__ == '$torchVersion', torchaudio.__version__"
)
if ($useCuda) {
    Invoke-VenvPython @("-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable after install'")
}

# ffmpeg: use a complete PATH pair when present; otherwise install the pinned Gyan essentials build.
$ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
$ffprobe = Get-Command ffprobe.exe -ErrorAction SilentlyContinue
if (-not $ffmpeg -or -not $ffprobe) {
    $ffmpegVersion = "8.1.2"
    $ffmpegArchive = "ffmpeg-$ffmpegVersion-essentials_build.zip"
    $ffmpegArchivePath = Join-Path $downloadDir $ffmpegArchive
    $ffmpegUri = "https://www.gyan.dev/ffmpeg/builds/packages/$ffmpegArchive"
    $ffmpegSha = "db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec"
    Download-VerifiedFile $ffmpegUri $ffmpegArchivePath $ffmpegSha

    $extract = Join-Path $env:TEMP "lrc-ffmpeg-$PID"
    Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -LiteralPath $ffmpegArchivePath -DestinationPath $extract -Force
    $ffmpegExe = Get-ChildItem -LiteralPath $extract -Recurse -File -Filter "ffmpeg.exe" | Select-Object -First 1
    if (-not $ffmpegExe) {
        throw "Pinned ffmpeg archive did not contain ffmpeg.exe."
    }
    $sourceBin = $ffmpegExe.Directory.FullName
    if (-not (Test-Path -LiteralPath (Join-Path $sourceBin "ffprobe.exe"))) {
        throw "Pinned ffmpeg archive did not contain ffprobe.exe beside ffmpeg.exe."
    }
    New-Item -ItemType Directory -Force -Path $ffmpegBin | Out-Null
    Copy-Item -Path (Join-Path $sourceBin "*") -Destination $ffmpegBin -Recurse -Force
    Remove-Item -LiteralPath $extract -Recurse -Force
    $env:PATH = "$ffmpegBin;$env:PATH"
    Write-Host "Installed ffmpeg $ffmpegVersion to $ffmpegBin" -ForegroundColor Green
} else {
    Write-Host "Using ffmpeg from PATH: $($ffmpeg.Source)" -ForegroundColor DarkGreen
    Write-Host "Using ffprobe from PATH: $($ffprobe.Source)" -ForegroundColor DarkGreen
}

# whisper.cpp: pin the official ggml-org nightly release and archive digest.
$whisperTag = "b5454"
if ($useCuda) {
    $whisperAsset = "whisper-bin-win-cuda-12.4.0-x64.zip"
    $whisperAssetSha = "afef0b881c500958921c3f5523b50e59ee2ec9b6f5cbd25b324c51ed308a957a"
} else {
    $whisperAsset = "whisper-bin-x64.zip"
    $whisperAssetSha = "6ba69e3482d7826214f90a6a9c84ca07782aec1e1d0c6a7c30c994fd5d816ccb"
}
$whisperReleaseDir = Join-Path $scriptDir "tools\whisper.cpp\Release"
$whisperCli = Join-Path $whisperReleaseDir "whisper-cli.exe"
if (-not (Test-Path -LiteralPath $whisperCli -PathType Leaf)) {
    $whisperArchivePath = Join-Path $downloadDir "$whisperTag-$whisperAsset"
    $whisperUri = "https://github.com/ggml-org/whisper.cpp/releases/download/$whisperTag/$whisperAsset"
    Download-VerifiedFile $whisperUri $whisperArchivePath $whisperAssetSha
    $extract = Join-Path $env:TEMP "lrc-whisper-$PID"
    Remove-Item -LiteralPath $extract -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -LiteralPath $whisperArchivePath -DestinationPath $extract -Force
    $foundCli = Get-ChildItem -LiteralPath $extract -Recurse -File -Filter "whisper-cli.exe" | Select-Object -First 1
    if (-not $foundCli) {
        throw "Pinned whisper.cpp archive did not contain whisper-cli.exe."
    }
    New-Item -ItemType Directory -Force -Path $whisperReleaseDir | Out-Null
    Copy-Item -Path (Join-Path $foundCli.Directory.FullName "*") -Destination $whisperReleaseDir -Recurse -Force
    Remove-Item -LiteralPath $extract -Recurse -Force
    @{
        tag = $whisperTag
        asset = $whisperAsset
        archive_sha256 = $whisperAssetSha
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $scriptDir "tools\whisper.cpp\install.json") -Encoding UTF8
    Write-Host "Installed whisper.cpp $whisperTag ($whisperAsset)" -ForegroundColor Green
} else {
    Write-Host "Reusing whisper-cli: $whisperCli" -ForegroundColor DarkGreen
}

$whisperModel = Join-Path $scriptDir "models\whisper.cpp\ggml-large-v3.bin"
$whisperModelRevision = "5359861c739e955e79d9a303bcbc70fb988958b1"
$whisperModelUri = "https://huggingface.co/ggerganov/whisper.cpp/resolve/$whisperModelRevision/ggml-large-v3.bin?download=true"
$whisperModelSha = "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"
Download-VerifiedFile $whisperModelUri $whisperModel $whisperModelSha

Push-Location $scriptDir
try {
    if ($Prefetch) {
        Invoke-VenvPython @("scripts\prefetch_models.py")
    }

    $doctorArgs = @(
        "scripts\doctor.py",
        "--expect-cuda", $expectCuda,
        "--json-out", $doctorJson
    )
    if ($Prefetch) {
        $doctorArgs += "--prefetch"
    }
    Invoke-VenvPython $doctorArgs
} finally {
    Pop-Location
}

Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Python: $venvPython"
Write-Host "Doctor report: $doctorJson"

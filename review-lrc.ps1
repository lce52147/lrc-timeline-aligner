# No param block and no [CmdletBinding()]: every argument of Review LRC.bat arrives in $args as it was written, so -d, -p 1, -Verbose or
# -OutVariable reach review_server.py (whose argparse refuses what it does not know) instead of binding to a PowerShell common parameter.
# Review LRC.bat sets LRC_FROM_LAUNCHER=1: the dropped names may have been percent-expanded by cmd.exe.
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir "launcher-names.ps1")

# Review LRC.bat passes the interpreter it chose in PYTHON.
$python = $env:PYTHON
if (-not $python) {
    $python = Join-Path $scriptDir ".venv-asr\Scripts\python.exe"
}

$ReviewArguments = @($args)
if ($env:LRC_FROM_LAUNCHER -eq "1" -and $ReviewArguments) {
    $ReviewArguments = @(Restore-DroppedNames -Received $ReviewArguments -LauncherPath (Join-Path $scriptDir "Review LRC.bat"))
}

& $python (Join-Path $scriptDir "scripts\review_server.py") @ReviewArguments
exit $LASTEXITCODE

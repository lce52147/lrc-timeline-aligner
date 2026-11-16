[CmdletBinding()]
param(
    # Set by Review LRC.bat: the dropped names may have been percent-expanded by cmd.exe.
    [switch] $FromLauncher,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ReviewArguments
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $scriptDir "launcher-names.ps1")

# Review LRC.bat passes the interpreter it chose in PYTHON.
$python = $env:PYTHON
if (-not $python) {
    $python = Join-Path $scriptDir ".venv-asr\Scripts\python.exe"
}

if ($FromLauncher -and $ReviewArguments) {
    $ReviewArguments = @(Restore-DroppedNames -Received $ReviewArguments -LauncherPath (Join-Path $scriptDir "Review LRC.bat"))
}

& $python (Join-Path $scriptDir "scripts\review_server.py") @ReviewArguments
exit $LASTEXITCODE

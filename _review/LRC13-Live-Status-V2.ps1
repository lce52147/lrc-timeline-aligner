param([switch]$SpawnWindow)
if ($SpawnWindow) {
    $exe = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $params = '-NoLogo -NoProfile -ExecutionPolicy Bypass -File "'+$PSCommandPath+'"'
    $proc = Start-Process -FilePath $exe -ArgumentList $params -WorkingDirectory (Split-Path -Parent $PSCommandPath) -WindowStyle Normal -PassThru
    Write-Output ('LRC13_STATUS_V2_PID=' + $proc.Id)
    exit
}
$ErrorActionPreference='Continue'
$root=Split-Path -Parent $MyInvocation.MyCommand.Path
$path=Join-Path $root 'LRC13_WORK_STATUS_20261011.json'
try { $Host.UI.RawUI.WindowTitle='LRC 1.3 - FULL VALIDATION LIVE STATUS' } catch {}
while ($true) {
    Clear-Host
    Write-Host ('LRC 1.3 FULL VALIDATION  '+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor Cyan
    Write-Host ('=' * 88)
    try {
        $st=Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host ('Branch       : '+$st.branch)
        Write-Host ('Phase        : '+$st.phase) -ForegroundColor Cyan
        Write-Host ('Current work : '+$st.detail)
        if ($st.full7_count -ne $null) {
            Write-Host ('FULL 7 A/B   : '+$st.full7_count+'/14 cases; produced='+$st.full7_produced) -ForegroundColor Green
        }
        if ($st.final_a7_count -ne $null) {
            Write-Host ('FINAL A7     : '+$st.final_a7_count+'/39 full songs') -ForegroundColor Green
        }
        if ($st.last) { Write-Host ('Last output  : '+$st.last) }
        Write-Host ('Quick gate   : '+$st.quick_gate) -ForegroundColor Green
        Write-Host ('Release gate : '+$st.release_gate) -ForegroundColor Yellow
        if ($st.release_pass -ne $null) {
            Write-Host ('Release pass : '+$st.release_pass)
        }
        if ($st.final_a7_active -ne $null) {
            Write-Host ('Final active : '+$st.final_a7_active+'  fallback='+$st.final_a7_fallback)
            Write-Host ('New >1s     : '+$st.final_a7_new_gt1)
        }
        Write-Host ('Commit       : '+$st.frozen_code)
        Write-Host ('Push/merge   : disabled / disabled')
    } catch {
        Write-Host ('STATUS READ ERROR: '+$_) -ForegroundColor Red
    }
    Write-Host ''
    Write-Host 'This window refreshes every five seconds from the real run manifest.'
    Start-Sleep -Seconds 5
}

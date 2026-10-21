@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHON=%~dp0.venv-asr\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo Python runtime not found: "%PYTHON%"
  echo Run setup.ps1 first.
  if not "%LRC_NO_PAUSE%"=="1" pause
  exit /b 2
)
echo Keep this window open. Closing it stops the review server.
set "LAST_TARGETS=%~dp0outputs\reports\batch-targets.txt"
if "%~1"=="" if exist "%LAST_TARGETS%" (
  echo Reviewing the songs from the last Align LRC run. Drag a folder or FLAC onto this file to review something else.
  "%PYTHON%" "%~dp0scripts\review_server.py" --report-dir "%~dp0outputs\reports" --targets-file "%LAST_TARGETS%"
  goto :done
)
if "%~1"=="" (
  "%PYTHON%" "%~dp0scripts\review_server.py" --report-dir "%~dp0outputs\reports"
) else (
  "%PYTHON%" "%~dp0scripts\review_server.py" %*
)
:done
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" if not "%LRC_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%

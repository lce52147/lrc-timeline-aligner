@echo off
setlocal
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%lyricsee.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (echo Done.) else (echo Failed with exit code %EXIT_CODE%.)
if not "%LRC_TOOLS_NO_PAUSE%"=="1" pause
exit /b %EXIT_CODE%

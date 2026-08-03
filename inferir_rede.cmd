@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0inferir_rede.ps1" %*
set "SCRIPT_EXIT_CODE=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %SCRIPT_EXIT_CODE%

@echo off
REM Deploy to neonbeam-lens.richwerks.local using scp (Windows wrapper).
REM Usage:
REM   deploy\deploy.cmd
REM   deploy\deploy.cmd restart

setlocal
set "SCRIPT_DIR=%~dp0"
set "RESTART_FLAG="

if /I "%~1"=="restart" set "RESTART_FLAG=-Restart"

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%deploy.ps1" %RESTART_FLAG%
exit /b %ERRORLEVEL%

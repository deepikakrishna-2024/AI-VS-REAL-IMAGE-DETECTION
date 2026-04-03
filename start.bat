@echo off
echo ==========================================
echo AI vs Real Image Detection - Full Start
echo ==========================================
echo.

REM Find node and npm
set "NODE_DIR="

REM Try to find node in common locations
if exist "C:\Program Files\nodejs\node.exe" (
    set "NODE_DIR=C:\Program Files\nodejs"
) else if exist "C:\Program Files (x86)\nodejs\node.exe" (
    set "NODE_DIR=C:\Program Files (x86)\nodejs"
) else (
    REM Try to use where command to find node
    for /f "delims=" %%i in ('where node.exe 2^>nul') do (
        set "NODE_DIR=%%~dpi"
        goto :found
    )
)

:found
if "%NODE_DIR%"=="" (
    echo ERROR: node.exe not found!
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

echo Found Node.js at: %NODE_DIR%
echo.
echo Starting AI vs Real Image Detection...
echo.

REM Change to project directory
cd /d "%~dp0"

REM Add Node.js to PATH for this session
set "PATH=%NODE_DIR%;%PATH%"

REM Export NODE_DIR for autostart script to use
set "NODE_DIR=%NODE_DIR%"

REM Run npm dev
"%NODE_DIR%\npm.cmd" run dev

pause

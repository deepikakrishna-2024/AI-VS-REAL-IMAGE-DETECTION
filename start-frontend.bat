@echo off
chcp 65001 >nul
echo Starting Frontend Only...
echo.

REM Find Node.js installation
set "NODE_DIR=C:\Program Files\nodejs"
if not exist "%NODE_DIR%\node.exe" (
    for /f "delims=" %%i in ('where node 2^>nul') do (
        set "NODE_DIR=%%~dpi"
        set "NODE_DIR=!NODE_DIR:~0,-1!"
    )
)

REM Add Node.js to PATH for this session
set "PATH=%NODE_DIR%;%PATH%"

REM Start the Vite dev server
echo Starting frontend on http://localhost:8080
npm run dev:raw

@echo off
echo ==========================================
echo Pushing to GitHub
echo ==========================================
echo.

REM Try to find git
echo Looking for Git installation...

if exist "C:\Program Files\Git\bin\git.exe" (
    set "GIT_PATH=C:\Program Files\Git\bin\git.exe"
) else if exist "C:\Program Files\Git\cmd\git.exe" (
    set "GIT_PATH=C:\Program Files\Git\cmd\git.exe"
) else if exist "C:\Program Files (x86)\Git\bin\git.exe" (
    set "GIT_PATH=C:\Program Files (x86)\Git\bin\git.exe"
) else (
    echo ERROR: Git not found! Please install Git first.
    pause
    exit /b 1
)

echo Found Git at: %GIT_PATH%
echo.

cd /d "%~dp0"

echo 1. Initializing Git repository...
"%GIT_PATH%" init

echo.
echo 2. Adding all files...
"%GIT_PATH%" add .

echo.
echo 3. Committing files...
"%GIT_PATH%" commit -m "Initial commit - AI vs Real Image Detection"

echo.
echo 4. Setting branch to main...
"%GIT_PATH%" branch -M main

echo.
echo 5. Adding remote origin...
"%GIT_PATH%" remote add origin https://github.com/deepikakrishna-2024/AI-VS-REAL-IMAGE-DETECTION.git

echo.
echo 6. Pushing to GitHub...
"%GIT_PATH%" push -u origin main --force

echo.
echo ==========================================
echo Done!
echo ==========================================
pause

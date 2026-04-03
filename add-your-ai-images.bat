@echo off
echo ==========================================
echo Adding Your AI Images to Training Data
echo ==========================================
echo.

REM Create folders if they don't exist
if not exist "datasets\train\YOUR_AI" mkdir "datasets\train\YOUR_AI"

echo.
echo Place your AI-generated images in: datasets\train\YOUR_AI\
echo.
echo Instructions:
echo 1. Copy your AI selfies to: datasets\train\YOUR_AI\
echo 2. Then run: cd ml_model ^&^& python train.py
echo.
echo Press any key to open the folder...
pause

start "" "datasets\train\YOUR_AI"

@echo off
REM Windows batch script to run training

cd /d %~dp0
cd multimodal_video_learning

echo ========================================
echo Multimodal Video Learning - Training
echo ========================================
echo.
echo Activating conda environment...
call conda activate multimodal-video

if errorlevel 1 (
    echo Failed to activate conda environment
    exit /b 1
)

echo.
echo Generating dataset...
python download_dataset.py

if errorlevel 1 (
    echo Failed to generate dataset
    exit /b 1
)

echo.
echo ========================================
echo Starting training...
echo ========================================
echo.

python -m multimodal_video_learning.train ^
    --config configs/config.yaml ^
    --data-root ./data ^
    --device cpu

pause

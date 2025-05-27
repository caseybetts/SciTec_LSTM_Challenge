@echo off
echo Training LSTM Model with Docker Toolbox
echo ========================================

REM Set Docker Toolbox environment
for /f "delims=" %%i in ('docker-machine env default --shell cmd 2^>nul') do (
    if not "%%i"=="" (
        if not "%%i:~0,1%"=="#" (
            %%i
        )
    )
)

REM Check Docker connection
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker not accessible. Please run start_docker_toolbox.bat first.
    pause
    exit /b 1
)

REM Check if image exists
docker images lstm-timeseries:latest --format "{{.Repository}}" | findstr "lstm-timeseries" >nul
if %errorlevel% neq 0 (
    echo Docker image not found. Please run build_toolbox.bat first.
    pause
    exit /b 1
)

REM Check for training data
if not exist "data\train.csv" (
    echo Warning: Training data not found at data\train.csv
    echo Please make sure your training data is in the data directory.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

echo Starting training container...
echo This may take several minutes depending on your data size and epochs.
echo.

REM Get the current directory path for volume mounting
set CURRENT_DIR=%CD%

REM Run training with volume mounts
docker run --rm ^
    -v "%CURRENT_DIR%\data":/app/data ^
    -v "%CURRENT_DIR%\models":/app/models ^
    -v "%CURRENT_DIR%\outputs":/app/outputs ^
    lstm-timeseries:latest python train.py

if %errorlevel% eq 0 (
    echo.
    echo ✓ Training completed successfully!
    echo Check the models directory for best_model.pt
    echo Check the outputs directory for training logs
) else (
    echo.
    echo ✗ Training failed. Check the error messages above.
)

echo.
pause
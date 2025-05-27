@echo off
echo Running LSTM Inference with Docker Toolbox
echo ==========================================

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

REM Check for test data
if not exist "data\test.csv" (
    echo Error: Test data not found at data\test.csv
    echo Please make sure your test data is in the data directory.
    pause
    exit /b 1
)

REM Check for trained model
if not exist "models\best_model.pt" (
    if not exist "best_model.pt" (
        echo Error: Trained model not found.
        echo Please run train_toolbox.bat first to train the model.
        pause
        exit /b 1
    )
)

echo Starting inference...
echo.

REM Get the current directory path for volume mounting
set CURRENT_DIR=%CD%

REM Run inference with volume mounts
docker run --rm ^
    -v "%CURRENT_DIR%\data":/app/data ^
    -v "%CURRENT_DIR%\models":/app/models ^
    -v "%CURRENT_DIR%\outputs":/app/outputs ^
    lstm-timeseries:latest python inference.py --input /app/data/test.csv --output /app/outputs/predictions.csv

if %errorlevel% eq 0 (
    echo.
    echo ✓ Inference completed successfully!
    echo Check outputs\predictions.csv for results
) else (
    echo.
    echo ✗ Inference failed. Check the error messages above.
)

echo.
pause
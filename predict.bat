@echo off
echo Starting LSTM Model Inference...

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if test data exists
if not exist "data\test.csv" (
    echo Error: Test data not found at data\test.csv
    echo Please make sure your test data is in the data directory.
    pause
    exit /b 1
)

REM Check if trained model exists
if not exist "best_model.pt" (
    if not exist "models\best_model.pt" (
        echo Error: Trained model not found.
        echo Please run train.bat first to train the model.
        pause
        exit /b 1
    )
)

echo Running inference container...
docker-compose --profile inference up lstm-inference

echo.
echo Inference completed. Check outputs\predictions.csv for results.
pause
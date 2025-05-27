@echo off
echo Starting LSTM Model Training...

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if training data exists
if not exist "data\train.csv" (
    echo Warning: Training data not found at data\train.csv
    echo Please make sure your training data is in the data directory.
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

echo Running training container...
docker-compose up lstm-trainer

echo.
echo Training completed. Check the outputs directory for results.
echo Trained model will be saved as best_model.pt
pause
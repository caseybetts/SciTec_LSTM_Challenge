@echo off
echo LSTM Time Series Classification - Docker Setup
echo ===============================================

REM Check if Docker Desktop is installed
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not installed or not in PATH.
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Desktop is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Desktop is not running. Starting Docker Desktop...
    echo Please wait for Docker Desktop to start completely, then run this script again.
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo.
    echo Once Docker Desktop is running (you'll see it in the system tray),
    echo press any key to continue...
    pause >nul
    
    REM Wait and check again
    timeout /t 10 /nobreak >nul
    docker version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Docker Desktop is still not ready. Please ensure it's fully started and try again.
        pause
        exit /b 1
    )
)

echo ✓ Docker is running

REM Create project structure
echo Creating project directories...
if not exist "data" mkdir data
if not exist "outputs" mkdir outputs  
if not exist "models" mkdir models

echo ✓ Project directories created

REM Create a sample data directory structure
echo.
echo Data Directory Setup:
echo - Place your train.csv in the 'data' folder
echo - Place your test.csv in the 'data' folder for inference
echo.

REM Check if training data exists
if exist "data\train.csv" (
    echo ✓ Training data found
) else (
    echo ! Training data not found - you'll need to add train.csv to the data folder
)

echo.
echo Setup complete! Next steps:
echo.
echo 1. Place your training data (train.csv) in the 'data' folder
echo 2. Run: build.bat    (to build the Docker image)
echo 3. Run: train.bat    (to train the model)
echo 4. Run: predict.bat  (to run inference on test data)
echo.
echo For interactive development, use: shell.bat
echo.

pause
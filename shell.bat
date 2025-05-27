@echo off
echo Opening Interactive Docker Shell...

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Starting interactive container...
echo You can run Python scripts, install packages, or debug your code.
echo Type 'exit' to leave the container.
echo.

docker-compose --profile interactive up -d lstm-interactive
docker exec -it lstm-time-series-interactive /bin/bash
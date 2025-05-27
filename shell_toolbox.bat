@echo off
echo Opening Interactive Docker Shell (Toolbox)
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

echo Starting interactive container...
echo You can run Python scripts, install packages, or debug your code.
echo Type 'exit' to leave the container.
echo.

REM Get the current directory path for volume mounting
set CURRENT_DIR=%CD%

REM Run interactive shell with volume mounts
docker run -it --rm ^
    -v "%CURRENT_DIR%\data":/app/data ^
    -v "%CURRENT_DIR%\models":/app/models ^
    -v "%CURRENT_DIR%\outputs":/app/outputs ^
    -v "%CURRENT_DIR%":/app/code ^
    lstm-timeseries:latest /bin/bash
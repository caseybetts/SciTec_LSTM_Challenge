@echo off
echo Building Docker Image with Docker Toolbox
echo ==========================================

REM Set Docker Toolbox environment
echo Setting up Docker environment...
for /f "delims=" %%i in ('docker-machine env default --shell cmd 2^>nul') do (
    if not "%%i"=="" (
        if not "%%i:~0,1%"=="#" (
            %%i
        )
    )
)

REM Check if Docker is accessible
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker not accessible. Starting Docker Machine...
    docker-machine start default
    
    REM Set environment again
    for /f "delims=" %%i in ('docker-machine env default --shell cmd') do (
        if not "%%i"=="" (
            if not "%%i:~0,1%"=="#" (
                %%i
            )
        )
    )
    
    docker version >nul 2>&1
    if %errorlevel% neq 0 (
        echo Failed to connect to Docker. Please run start_docker_toolbox.bat first.
        pause
        exit /b 1
    )
)

echo ✓ Docker connection established

REM Create local directories for volume mounting
echo Creating project directories...
if not exist "data" mkdir data
if not exist "models" mkdir models  
if not exist "outputs" mkdir outputs

echo ✓ Directories created

REM Build the Docker image
echo Building Docker image (this may take several minutes)...
docker build -f Dockerfile.toolbox -t lstm-timeseries:latest .

if %errorlevel% eq 0 (
    echo.
    echo ✓ Docker image built successfully!
    echo.
    echo Available commands:
    echo   train_toolbox.bat    - Train the model
    echo   predict_toolbox.bat  - Run inference  
    echo   shell_toolbox.bat    - Interactive shell
    echo.
    echo Docker Machine IP: 
    docker-machine ip default
    echo.
) else (
    echo.
    echo ✗ Build failed. Check the error messages above.
    echo Make sure all Python files are in the current directory.
)

pause
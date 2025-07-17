@echo off
REM Build script for LLM4CodeSec Benchmark Docker image (Windows)
REM This script provides an easy way to build and test the Docker environment on Windows

setlocal EnableDelayedExpansion

REM Default values
set IMAGE_NAME=llm4codesec-benchmark
set TAG=latest
set BUILD_ARGS=
set TEST_GPU=true
set PUSH_IMAGE=false

REM Colors for output (if supported)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "NC=[0m"

REM Function to print status
:print_status
echo %GREEN%[INFO]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM Show usage
:show_usage
echo Usage: %0 [OPTIONS]
echo.
echo Build and test LLM4CodeSec Benchmark Docker image with CUDA support.
echo.
echo OPTIONS:
echo     /h, /help           Show this help message
echo     /n NAME             Docker image name (default: llm4codesec-benchmark)
echo     /t TAG              Docker image tag (default: latest)
echo     /c VERSION          CUDA version (default: 12.1)
echo     /no-gpu-test        Skip GPU functionality test
echo     /push               Push image to registry after build
echo     /no-cache           Build without using cache
echo     /cpu-only           Build CPU-only version
echo.
echo EXAMPLES:
echo     %0                          # Build with default settings
echo     %0 /t v1.0 /push           # Build, tag as v1.0, and push
echo     %0 /c 11.8 /no-cache       # Build with CUDA 11.8, no cache
echo     %0 /cpu-only               # Build CPU-only version
echo.
goto :eof

REM Parse command line arguments
:parse_args
if "%~1"=="" goto start_build
if /i "%~1"=="/h" goto show_usage
if /i "%~1"=="/help" goto show_usage
if /i "%~1"=="/n" (
    set IMAGE_NAME=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="/t" (
    set TAG=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="/c" (
    set BUILD_ARGS=%BUILD_ARGS% --build-arg CUDA_VERSION=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="/no-gpu-test" (
    set TEST_GPU=false
    shift
    goto parse_args
)
if /i "%~1"=="/push" (
    set PUSH_IMAGE=true
    shift
    goto parse_args
)
if /i "%~1"=="/no-cache" (
    set BUILD_ARGS=%BUILD_ARGS% --no-cache
    shift
    goto parse_args
)
if /i "%~1"=="/cpu-only" (
    set TAG=%TAG%-cpu
    set BUILD_ARGS=%BUILD_ARGS% --target builder
    set TEST_GPU=false
    shift
    goto parse_args
)
call :print_error "Unknown option: %~1"
goto show_usage

:start_build
call %0 %*
goto parse_args

REM Main build process
:main
set FULL_IMAGE_NAME=%IMAGE_NAME%:%TAG%

call :print_status "Starting build process for %FULL_IMAGE_NAME%"

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    call :print_error "Docker is not running. Please start Docker and try again."
    exit /b 1
)

REM Check if we're in the right directory
if not exist "Dockerfile" (
    call :print_error "Dockerfile not found. Please run this script from the project root directory."
    exit /b 1
)

REM Check NVIDIA Docker support if testing GPU
if "%TEST_GPU%"=="true" (
    call :print_status "Checking NVIDIA Docker support..."
    docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi >nul 2>&1
    if !errorlevel! neq 0 (
        call :print_warning "GPU test failed. NVIDIA Docker support might not be available."
        call :print_warning "Continuing with build, but GPU functionality may not work."
        set TEST_GPU=false
    ) else (
        call :print_status "NVIDIA Docker support confirmed."
    )
)

REM Build the Docker image
call :print_status "Building Docker image: %FULL_IMAGE_NAME%"
call :print_status "Build command: docker build %BUILD_ARGS% -t %FULL_IMAGE_NAME% ."

docker build %BUILD_ARGS% -t "%FULL_IMAGE_NAME%" .
if %errorlevel% neq 0 (
    call :print_error "Docker build failed!"
    exit /b 1
)
call :print_status "Docker image built successfully!"

REM Test the image
call :print_status "Testing the Docker image..."

REM Basic functionality test
call :print_status "Testing basic functionality..."
docker run --rm "%FULL_IMAGE_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    call :print_status "Basic functionality test passed."
) else (
    call :print_warning "Basic functionality test failed. Image might still be usable."
)

REM Python imports test
call :print_status "Testing Python imports..."
set PYTHON_TEST_CMD=python -c "import sys; print(f'Python version: {sys.version}'); import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); import transformers; print(f'Transformers version: {transformers.__version__}'); print('All import tests passed!')"

if "%TEST_GPU%"=="true" (
    docker run --rm --gpus all "%FULL_IMAGE_NAME%" bash -c "%PYTHON_TEST_CMD%"
    if !errorlevel! equ 0 (
        call :print_status "Python imports test with GPU passed."
    ) else (
        call :print_error "Python imports test with GPU failed!"
        exit /b 1
    )
) else (
    docker run --rm "%FULL_IMAGE_NAME%" bash -c "%PYTHON_TEST_CMD%"
    if !errorlevel! equ 0 (
        call :print_status "Python imports test passed."
    ) else (
        call :print_error "Python imports test failed!"
        exit /b 1
    )
)

REM Test benchmark entry points
call :print_status "Testing benchmark entry points..."
for %%i in (castle jitvul cvefixes unified benchmark) do (
    call :print_status "Testing %%i entry point..."
    docker run --rm "%FULL_IMAGE_NAME%" %%i --help >nul 2>&1
    if !errorlevel! equ 0 (
        call :print_status "%%i entry point test passed."
    ) else (
        call :print_warning "%%i entry point test failed. This might be expected if the module is not implemented."
    )
)

REM Show image information
call :print_status "Docker image information:"
docker images "%FULL_IMAGE_NAME%"

REM Push image if requested
if "%PUSH_IMAGE%"=="true" (
    call :print_status "Pushing image to registry..."
    docker push "%FULL_IMAGE_NAME%"
    if !errorlevel! equ 0 (
        call :print_status "Image pushed successfully!"
    ) else (
        call :print_error "Failed to push image!"
        exit /b 1
    )
)

REM Show usage examples
call :print_status "Build completed successfully!"
echo.
call :print_status "Usage examples:"
echo   # Run interactively with GPU:
echo   docker run -it --gpus all -v %cd%/results:/app/results %FULL_IMAGE_NAME%
echo.
echo   # Run CASTLE benchmark:
echo   docker run --gpus all -v %cd%/results:/app/results %FULL_IMAGE_NAME% castle --model qwen3-4b --dataset binary_all --prompt basic_security
echo.
echo   # Use with docker-compose:
echo   docker-compose run --rm llm4codesec-benchmark castle --help
echo.
call :print_status "For more examples, see DOCKER_README.md"

goto :eof

call :main %*

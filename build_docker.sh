#!/bin/bash
# Build script for LLM4CodeSec Benchmark Docker image
# This script provides an easy way to build and test the Docker environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="llm4codesec-benchmark"
TAG="latest"
BUILD_ARGS=""
TEST_GPU=true

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build and test LLM4CodeSec Benchmark Docker image with CUDA support.

OPTIONS:
    -h, --help          Show this help message
    -n, --name NAME     Docker image name (default: llm4codesec-benchmark)
    -t, --tag TAG       Docker image tag (default: latest)
    -c, --cuda VERSION  CUDA version (default: 12.1)
    --no-gpu-test       Skip GPU functionality test
    --no-cache          Build without using cache
    --cpu-only          Build CPU-only version

EXAMPLES:
    $0                          # Build with default settings
    $0 --cuda 11.8 --no-cache  # Build with CUDA 11.8, no cache
    $0 --cpu-only               # Build CPU-only version

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -c|--cuda)
            BUILD_ARGS="$BUILD_ARGS --build-arg CUDA_VERSION=$2"
            shift 2
            ;;
        --no-gpu-test)
            TEST_GPU=false
            shift
            ;;
        --no-cache)
            BUILD_ARGS="$BUILD_ARGS --no-cache"
            shift
            ;;
        --cpu-only)
            TAG="$TAG-cpu"
            BUILD_ARGS="$BUILD_ARGS --target builder"
            TEST_GPU=false
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Full image name
FULL_IMAGE_NAME="$IMAGE_NAME:$TAG"

print_status "Starting build process for $FULL_IMAGE_NAME"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "Dockerfile" ]]; then
    print_error "Dockerfile not found. Please run this script from the project root directory."
    exit 1
fi

# Check NVIDIA Docker support if testing GPU
if [[ $TEST_GPU == true ]]; then
    print_status "Checking NVIDIA Docker support..."
    if ! docker run --rm --gpus all nvidia/cuda:12.8.1-devel-ubuntu24.04 > /dev/null 2>&1; then
        print_warning "GPU test failed. NVIDIA Docker support might not be available."
        print_warning "Continuing with build, but GPU functionality may not work."
        TEST_GPU=false
    else
        print_status "NVIDIA Docker support confirmed."
    fi
fi

# Build the Docker image
print_status "Building Docker image: $FULL_IMAGE_NAME"
print_status "Build command: docker build $BUILD_ARGS -t $FULL_IMAGE_NAME ."

if docker build $BUILD_ARGS -t "$FULL_IMAGE_NAME" .; then
    print_status "Docker image built successfully!"
else
    print_error "Docker build failed!"
    exit 1
fi

# Test the image
print_status "Testing the Docker image..."

# Python imports test
print_status "Testing Python imports..."
PYTHON_TEST_CMD="python -c \"
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError as e:
    print(f'PyTorch import failed: {e}')
    sys.exit(1)

try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
except ImportError as e:
    print(f'Transformers import failed: {e}')
    sys.exit(1)

try:
    from pathlib import Path
    src_dir = Path('/app')
    if src_dir.exists():
        print('Source directory found.')
    else:
        print('Warning: Source directory not found.')
except Exception as e:
    print(f'Path check failed: {e}')

print('All import tests passed!')
\""

if [[ $TEST_GPU == true ]]; then
    # Test with GPU
    if docker run --rm --gpus all "$FULL_IMAGE_NAME" bash -c "$PYTHON_TEST_CMD"; then
        print_status "Python imports test with GPU passed."
    else
        print_error "Python imports test with GPU failed!"
        exit 1
    fi
else
    # Test without GPU
    if docker run --rm "$FULL_IMAGE_NAME" bash -c "$PYTHON_TEST_CMD"; then
        print_status "Python imports test passed."
    else
        print_error "Python imports test failed!"
        exit 1
    fi
fi

# Test benchmark entry points
print_status "Testing benchmark entry points..."
ENTRY_POINTS=("castle" "jitvul" "cvefixes" "unified" "benchmark")

for entry_point in "${ENTRY_POINTS[@]}"; do
    print_status "Testing $entry_point entry point..."
    print_status `docker run --rm "$FULL_IMAGE_NAME" entrypoints/run_"$entry_point"_benchmark.py --help`
    if docker run --rm "$FULL_IMAGE_NAME" entrypoints/run_"$entry_point"_benchmark.py --help > /dev/null 2>&1; then
        print_status "$entry_point entry point test passed."
    else
        print_warning "$entry_point entry point test failed. This might be expected if the module is not implemented."
    fi
done

# Show image information
print_status "Docker image information:"
docker images "$FULL_IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"

# Show usage examples
print_status "Build completed successfully!"
echo
print_status "Usage examples:"
echo "  # Run interactively with GPU:"
echo "  docker run -it --gpus all -v \$(pwd)/results:/app/results $FULL_IMAGE_NAME"
echo
echo "  # Run CASTLE benchmark:"
echo "  docker run --gpus all -v \$(pwd)/results:/app/results $FULL_IMAGE_NAME castle --model qwen3-4b --dataset binary_all --prompt basic_security"
echo
echo "  # Use with docker-compose:"
echo "  docker-compose run --rm llm4codesec-benchmark castle --help"
echo
print_status "For more examples, see DOCKER_README.md"

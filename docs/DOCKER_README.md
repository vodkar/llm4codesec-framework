# Docker Setup for LLM4CodeSec Benchmark

This Docker setup provides a CUDA-enabled environment for running LLM vulnerability detection benchmarks with optimal GPU performance.

## Prerequisites

### System Requirements
- **NVIDIA GPU** with CUDA Compute Capability 6.0+ (required for optimal performance)
- **GPU Memory**: Minimum 8GB VRAM, recommended 16GB+ for larger models
- **System RAM**: Minimum 16GB, recommended 32GB+
- **Docker**: Version 20.10+ with GPU support
- **NVIDIA Container Toolkit**: Latest version

### NVIDIA Container Toolkit Installation

#### Ubuntu/Debian
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker
```

#### CentOS/RHEL/Fedora
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

# Install nvidia-container-toolkit
sudo yum install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker
```

### Verify GPU Support
```bash
# Test NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

## Building the Docker Image

### Option 1: Using Docker Compose (Recommended)
```bash
# Build and start the container
docker-compose up --build

# Or build only
docker-compose build
```

### Option 2: Using Docker CLI
```bash
# Build the image
docker build -t llm4codesec-benchmark:latest .

# Build with specific CUDA version
docker build -t llm4codesec-benchmark:cuda12.1 \
  --build-arg CUDA_VERSION=12.1 .
```

## Running Benchmarks

### Using Docker Compose (Recommended)
```bash
# Start the container interactively
docker-compose run --rm llm4codesec-benchmark

# Run specific benchmark
docker-compose run --rm llm4codesec-benchmark castle \
  --model qwen3-4b \
  --dataset binary_all \
  --prompt basic_security \
  --sample-limit 100

# Run with custom configuration
docker-compose run --rm llm4codesec-benchmark unified jitvul \
  --model deepseek-coder-v2-lite-16b \
  --dataset binary_all \
  --prompt detailed_analysis
```

### Using Docker CLI
```bash
# Run interactively with GPU support
docker run -it --gpus all \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  llm4codesec-benchmark:latest

# Run CASTLE benchmark
docker run --gpus all \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  llm4codesec-benchmark:latest castle \
  --model qwen3-4b \
  --dataset binary_all \
  --prompt basic_security

# Run JitVul benchmark
docker run --gpus all \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  llm4codesec-benchmark:latest jitvul \
  --model llama3.2-3B \
  --dataset binary_all \
  --prompt context_aware

# Run CVEFixes benchmark
docker run --gpus all \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  llm4codesec-benchmark:latest cvefixes \
  --model deepseek-coder-v2 \
  --dataset binary_c_file \
  --prompt detailed_analysis
```

## Available Benchmark Types

### 1. CASTLE Benchmark
```bash
docker run --gpus all llm4codesec-benchmark:latest castle --help
```

### 2. JitVul Benchmark
```bash
docker run --gpus all llm4codesec-benchmark:latest jitvul --help
```

### 3. CVEFixes Benchmark
```bash
docker run --gpus all llm4codesec-benchmark:latest cvefixes --help
```

### 4. Unified Benchmark Runner
```bash
docker run --gpus all llm4codesec-benchmark:latest unified castle --help
```

## Performance Optimization

### GPU Memory Management
```bash
# For large models, use quantization
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  llm4codesec-benchmark:latest castle \
  --model llama4-scout-17b-16e \
  --use-quantization

# Limit GPU memory usage
docker run --gpus all \
  --memory=16g \
  --shm-size=8g \
  llm4codesec-benchmark:latest
```

### Multi-GPU Setup
```bash
# Use specific GPUs
docker run --gpus '"device=0,1"' \
  llm4codesec-benchmark:latest

# Use all available GPUs
docker run --gpus all \
  llm4codesec-benchmark:latest
```

### Environment Variables for Optimization
```bash
docker run --gpus all \
  -e CUDA_LAUNCH_BLOCKING=0 \
  -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
  -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  llm4codesec-benchmark:latest
```

## Volume Mounts

### Essential Mounts
```bash
docker run --gpus all \
  -v $(pwd)/results:/app/results \      # Results output
  -v $(pwd)/data:/app/data \           # Input datasets
  -v $(pwd)/configs:/app/configs:ro \  # Configuration files
  -v $(pwd)/logs:/app/logs \           # Log files
  llm4codesec-benchmark:latest
```

### Cache Optimization
```bash
# Mount cache directories for faster subsequent runs
docker run --gpus all \
  -v huggingface_cache:/app/.cache/huggingface \
  -v torch_cache:/app/.cache/torch \
  llm4codesec-benchmark:latest
```

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Check container GPU access
docker run --gpus all llm4codesec-benchmark:latest \
  python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, GPU Count: {torch.cuda.device_count()}')"
```

### Out of Memory Errors
```bash
# Use smaller models
docker run --gpus all llm4codesec-benchmark:latest castle \
  --model qwen3-4b \
  --sample-limit 50

# Enable quantization
docker run --gpus all llm4codesec-benchmark:latest castle \
  --model llama3.2-3B \
  --use-quantization

# Increase shared memory
docker run --gpus all --shm-size=16g llm4codesec-benchmark:latest
```

### Performance Issues
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check container resources
docker stats

# Optimize batch size
docker run --gpus all llm4codesec-benchmark:latest castle \
  --batch-size 1 \
  --max-tokens 256
```

## Development and Debugging

### Interactive Shell
```bash
# Enter container shell
docker run -it --gpus all \
  -v $(pwd):/app \
  llm4codesec-benchmark:latest bash

# Run with debugging
docker run --gpus all \
  -e LOG_LEVEL=DEBUG \
  llm4codesec-benchmark:latest castle --verbose
```

### Building for Development
```bash
# Build development image
docker build -t llm4codesec-benchmark:dev \
  --target builder .

# Mount source code for development
docker run -it --gpus all \
  -v $(pwd):/app \
  llm4codesec-benchmark:dev bash
```

## Example Workflows

### Quick Test Run
```bash
docker-compose run --rm llm4codesec-benchmark castle \
  --model qwen3-4b \
  --dataset binary_all \
  --prompt basic_security \
  --sample-limit 10
```

### Full Benchmark Suite
```bash
# Run comprehensive evaluation
docker-compose run --rm llm4codesec-benchmark unified castle \
  --plan comprehensive_evaluation
```

### Batch Processing
```bash
# Process multiple models
for model in qwen3-4b llama3.2-3B deepseek-coder-v2; do
  docker-compose run --rm llm4codesec-benchmark castle \
    --model $model \
    --dataset binary_all \
    --prompt basic_security \
    --output-dir results/$model
done
```

## Resource Requirements by Model Size

| Model Size | GPU Memory | System RAM | Recommended GPU |
|------------|------------|------------|-----------------|
| 1-3B       | 4-8GB      | 8GB        | RTX 3070, A4000 |
| 7-8B       | 8-16GB     | 16GB       | RTX 4080, A5000 |
| 13-16B     | 16-32GB    | 32GB       | RTX 4090, A6000 |
| 30-70B     | 32GB+      | 64GB+      | A100, H100      |

## Support

For issues specific to the Docker setup:
1. Check GPU driver compatibility with CUDA 12.1
2. Verify NVIDIA Container Toolkit installation
3. Ensure sufficient GPU memory for chosen models
4. Check Docker daemon configuration for GPU support

For benchmark-specific issues, refer to the main project documentation.

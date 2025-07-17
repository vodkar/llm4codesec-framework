# Quick Reference: Docker Commands for LLM4CodeSec Benchmark

## Build Commands

```bash
# Quick build with default settings
./build_docker.sh

# Build with custom tag
./build_docker.sh -t v1.0

# Build CPU-only version
./build_docker.sh --cpu-only

# Build with no cache
./build_docker.sh --no-cache

# Windows
build_docker.bat /t v1.0
```

## Run Commands

### Interactive Shell
```bash
# With GPU support
docker run -it --gpus all -v $(pwd)/results:/app/results llm4codesec-benchmark:latest bash

# CPU only
docker run -it -v $(pwd)/results:/app/results llm4codesec-benchmark:cpu bash
```

### Quick Benchmark Tests
```bash
# Test CASTLE benchmark (small sample)
docker run --gpus all -v $(pwd)/results:/app/results llm4codesec-benchmark:latest \
  castle --model qwen3-4b --dataset binary_all --prompt basic_security --sample-limit 10

# Test JitVul benchmark
docker run --gpus all -v $(pwd)/results:/app/results llm4codesec-benchmark:latest \
  jitvul --model llama3.2-3B --dataset binary_all --prompt context_aware --sample-limit 10

# Test CVEFixes benchmark
docker run --gpus all -v $(pwd)/results:/app/results llm4codesec-benchmark:latest \
  cvefixes --model deepseek-coder-v2 --dataset binary_c_file --prompt detailed_analysis --sample-limit 10
```

### Production Runs
```bash
# Full CASTLE evaluation
docker run --gpus all -v $(pwd)/results:/app/results -v $(pwd)/configs:/app/configs:ro \
  llm4codesec-benchmark:latest castle --plan comprehensive_evaluation

# Custom configuration
docker run --gpus all -v $(pwd)/results:/app/results -v $(pwd)/configs:/app/configs:ro \
  llm4codesec-benchmark:latest unified castle --config custom_config.json
```

## Docker Compose Commands

```bash
# Build and start
docker-compose up --build

# Run specific benchmark
docker-compose run --rm llm4codesec-benchmark castle --model qwen3-4b --dataset binary_all

# Interactive session
docker-compose run --rm llm4codesec-benchmark bash

# CPU-only profile
docker-compose --profile cpu-only run llm4codesec-benchmark-cpu
```

## Monitoring and Debugging

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check container stats
docker stats

# View logs
docker logs <container_id>

# Debug Python environment
docker run --gpus all llm4codesec-benchmark:latest python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

## Common Issues and Solutions

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Test Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker
```

### Out of Memory
```bash
# Use smaller model or quantization
docker run --gpus all llm4codesec-benchmark:latest castle \
  --model qwen3-4b --use-quantization --batch-size 1

# Increase shared memory
docker run --gpus all --shm-size=16g llm4codesec-benchmark:latest
```

### Performance Optimization
```bash
# Limit to specific GPU
docker run --gpus '"device=0"' llm4codesec-benchmark:latest

# Optimize for specific architecture
docker run --gpus all \
  -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
  llm4codesec-benchmark:latest
```

## File Structure in Container

```
/app/
├── src/                    # Source code
├── results/               # Output directory (mounted)
├── datasets_processed/                  # Input datasets (mounted)
├── benchmarks/                  # Benchmarks datasets (mounted)
├── .cache/               # Model cache directory
├── setup_env.py          # Entry point handler
├── run_benchmark.sh      # Main runner script
└── pyproject.toml        # Python dependencies
```

## Environment Variables

```bash
# CUDA settings
CUDA_VISIBLE_DEVICES=0,1
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Performance tuning
CUDA_LAUNCH_BLOCKING=0
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Cache directories
HF_HOME=/app/.cache/huggingface
TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
```

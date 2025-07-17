# Multi-stage Dockerfile for LLM4CodeSec Benchmark with NVIDIA CUDA support
FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu24.04

# Set environment variables for non-interactive builds
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app


# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-packaging \
    python3-poetry \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*


# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR /app

COPY pyproject.toml poetry.lock ./
    
RUN poetry install --only main

# Install flash attention
RUN poetry run pip install ninja && \
    FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=8 poetry run pip install flash-attn==2.8.0.post2 --no-build-isolation

WORKDIR /app

COPY src/ .

# Create a Docker healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

ENTRYPOINT ["poetry", "run"]

CMD []

# Labels for documentation
LABEL maintainer="llm4codesec-benchmark"
LABEL description="LLM4CodeSec Benchmark with NVIDIA CUDA support for vulnerability detection"
LABEL version="0.1.0"
LABEL cuda.version="12.8.1"
LABEL python.version="3.12"

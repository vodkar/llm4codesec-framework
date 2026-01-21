# Multi-stage Dockerfile for LLM4CodeSec Benchmark with NVIDIA CUDA support
FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04

# Set environment variables for non-interactive builds
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app


# Install system dependencies
RUN apt-get update && apt-get install -y  --no-install-recommends \
    build-essential python3-dev git ca-certificates curl \
    python3-pip \
    python3-packaging \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

WORKDIR /app

# Disable development dependencies
ENV UV_NO_DEV=1

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv UV_HTTP_TIMEOUT=600 uv sync --locked

# Install flash attention
RUN uv pip install ninja setuptools && \
    MAX_JOBS=2 uv pip install flash-attn --no-build-isolation

COPY src/ .

# Disable development dependencies
ENV UV_NO_DEV=1

# Create a Docker healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

ENTRYPOINT ["uv", "run"]

CMD []

# Labels for documentation
LABEL maintainer="llm4codesec-benchmark"
LABEL description="LLM4CodeSec Benchmark with NVIDIA CUDA support for vulnerability detection"
LABEL version="0.1.0"
LABEL cuda.version="12.8.0"
LABEL python.version="3.12"

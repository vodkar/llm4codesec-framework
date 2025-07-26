"""
LLM Code Security Benchmark Framework

A comprehensive framework for benchmarking Large Language Models on static code analysis tasks.
Supports binary and multi-class vulnerability detection with configurable models and datasets.
"""

import importlib.util
import logging

import torch


def is_flash_attention_available() -> bool:
    """Check if FlashAttention is available."""
    if importlib.util.find_spec("flash_attn"):
        logging.info("FlashAttention is available")
        return True
    else:
        logging.warning(
            "FlashAttention is not available! You can install it to optimize an inference"
        )
        return False


def is_flash_attention_supported() -> bool:
    """Check if a GPU supports FlashAttention 2."""
    major, minor = torch.cuda.get_device_capability(0)

    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    if is_sm8x or is_sm90:
        logging.info(f"FlashAttention is supported on this GPU (SM {major}.{minor})")
        return True
    else:
        logging.warning(
            f"FlashAttention is not supported on this GPU (SM {major}.{minor}). "
        )
        return False

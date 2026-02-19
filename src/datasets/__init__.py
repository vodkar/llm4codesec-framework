"""
Dataset loaders and utilities for the LLM Code Security Benchmark Framework.

This module provides loaders for various vulnerability detection datasets including
CASTLE and CVEFixes benchmarks.
"""

from .loaders.castle import CastleDatasetLoader
from .loaders.cvefixes import CVEFixesDatasetLoader

__all__ = ["CastleDatasetLoader", "CVEFixesDatasetLoader"]

"""Pinned run configuration for the reference-context oracle evaluation."""

import hashlib
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel


class Pins(BaseModel):
    """Pinned llm_scanner checkout that produced the artifacts under analysis."""

    llm_scanner_path: Path
    llm_scanner_git_sha: str


class FrameworkSection(BaseModel):
    """Framework-side run location, bootstrap and probe settings."""

    plan: str = "reference_context_sweep"
    experiments_config: Path
    datasets_config: Path
    results_dir: Path
    datasets_dir: Path
    bootstrap_resamples: int = 2000
    bootstrap_seed: int
    probe_seed: int
    mismatch_tolerance: float = 0.02


class ReferenceRunConfig(BaseModel):
    """Top-level pinned configuration for one reference-context oracle run."""

    pins: Pins
    llm_scanner: dict[str, object]
    framework: FrameworkSection

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Load and validate a :class:`ReferenceRunConfig` from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            The validated configuration.
        """

        with path.open("r", encoding="utf-8") as handle:
            raw: object = yaml.safe_load(handle)
        return cls.model_validate(raw)

    @classmethod
    def sha256(cls, path: Path) -> str:
        """Return the sha256 hex digest of a file's raw bytes.

        Args:
            path: Path to the file to hash.

        Returns:
            Hex digest string.
        """

        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

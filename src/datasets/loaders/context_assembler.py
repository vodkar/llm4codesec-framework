#!/usr/bin/env python3
"""
ContextAssembler Dataset Loader

Loads the CVEFixes-with-Context-Benchmark produced by the ContextAssembler tool.
Each sample is a Python code snippet with both vulnerable (label=1) and safe (label=0)
samples, enriched with extended CVE context information.

Fields in the source JSON that differ from the standard schema and are normalised here:
- ``metadata.CVEFixes-Number`` → ``metadata.cve_id``
- ``metadata.cwe_number`` (raw int)  → ``"CWE-<N>"`` string stored in ``cwe_types``
- ``cwe_types`` in source is always ``[]`` – populated from ``cwe_number`` instead
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Unpack

from pydantic import PrivateAttr, field_validator

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, Dataset, DatasetMetadata, SampleCollection
from datasets.loaders.base import DatasetLoadParams, IDatasetLoader


class ContextAssemblerDatasetLoader(IDatasetLoader):
    """Loads and processes the ContextAssembler CVEFixes context benchmark."""

    source_path: Path
    __logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__)
    )

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, v: Path) -> Path:
        """Ensure the benchmark source file exists."""
        if not v.exists() or not v.is_file():
            raise ValueError(f"ContextAssembler benchmark file not found: {v}")
        return v

    @staticmethod
    def _normalize_cwe_number(cwe_number: int | str | None) -> str | None:
        """
        Normalize a raw CWE number to ``'CWE-NNN'`` format.

        Args:
            cwe_number: Raw CWE identifier – may be an integer (e.g. ``212``),
                a string (e.g. ``"212"`` or ``"CWE-212"``), or ``None``.

        Returns:
            Normalised ``"CWE-<N>"`` string, or ``None`` when the input is absent.
        """
        if cwe_number is None:
            return None
        raw: str = str(cwe_number).strip()
        if raw.upper().startswith("CWE-"):
            return raw.upper()
        if raw.isdigit():
            return f"CWE-{raw}"
        return None

    def _build_sample(self, raw: dict[str, Any]) -> BenchmarkSample:
        """
        Convert a raw ContextAssembler sample dict to a ``BenchmarkSample``.

        Args:
            raw: Single sample dict as read from the source JSON.

        Returns:
            Normalised ``BenchmarkSample``.
        """
        raw_meta: dict[str, Any] = raw.get("metadata", {})

        cwe_number: int | str | None = raw_meta.get("cwe_number")
        normalized_cwe: str | None = self._normalize_cwe_number(cwe_number)

        try:
            cwe_number_int: int = int(cwe_number) if cwe_number is not None else 0
        except (TypeError, ValueError):
            cwe_number_int = 0

        metadata: dict[str, Any] = {
            "cve_id": raw_meta.get("CVEFixes-Number", ""),
            "description": raw_meta.get("description", ""),
            "cwe_number": cwe_number_int,
            "source": "ContextAssembler",
        }

        return BenchmarkSample(
            id=raw["id"],
            code=raw["code"],
            label=int(raw["label"]),
            metadata=metadata,
            cwe_types=[normalized_cwe] if normalized_cwe is not None else [],
            severity=raw.get("severity"),
        )

    def load_dataset(self, **kwargs: Unpack[DatasetLoadParams]) -> SampleCollection:
        """
        Load the ContextAssembler benchmark from its JSON source file.

        Args:
            **kwargs: Supports ``limit`` to cap the number of returned samples.

        Returns:
            ``SampleCollection`` of normalised ``BenchmarkSample`` instances.
        """
        limit: int | None = kwargs.get("limit")

        with open(self.source_path, "r", encoding="utf-8") as fh:
            payload: dict[str, Any] = json.load(fh)

        raw_samples: list[dict[str, Any]] = payload.get("samples", [])
        samples: list[BenchmarkSample] = []

        for raw in raw_samples:
            try:
                samples.append(self._build_sample(raw))
            except Exception:
                self.__logger.exception(
                    "Skipping malformed ContextAssembler sample id=%s", raw.get("id")
                )

        if limit is not None and limit < len(samples):
            samples = samples[:limit]

        self.__logger.info(
            "Loaded %d samples from ContextAssembler source (%s)",
            len(samples),
            self.source_path,
        )
        return SampleCollection(samples)

    def create_dataset_json(
        self,
        output_path: str,
        **kwargs: Unpack[DatasetLoadParams],
    ) -> None:
        """
        Serialize the ContextAssembler benchmark to a processed JSON file
        compatible with the benchmark framework's ``JsonDatasetLoader``.

        Args:
            output_path: Destination path for the output JSON file.
            **kwargs: Passed through to ``load_dataset`` (supports ``limit``).
        """
        task_type: TaskType = kwargs.get("task_type", TaskType.BINARY_VULNERABILITY)
        samples: SampleCollection = self.load_dataset(**kwargs)

        dataset = Dataset(
            metadata=DatasetMetadata(
                name="ContextAssembler-CVEFixes-Benchmark",
                version="1.0",
                task_type=task_type,
                programming_language="Python",
                change_level="file",
            ),
            samples=samples,
        )

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "w", encoding="utf-8") as fh:
            json.dump(dataset.model_dump(), fh, indent=2, ensure_ascii=False)

        self.__logger.info(
            "Created ContextAssembler dataset JSON with %d samples at %s",
            len(samples),
            output_path,
        )

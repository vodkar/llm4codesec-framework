#!/usr/bin/env python3
"""
CASTLE Dataset Loader and Integration

This module provides functionality to load and process the CASTLE benchmark dataset
for use with the LLM code security benchmark framework.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Unpack

from pydantic import ConfigDict

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, Dataset, DatasetMetadata, SampleCollection
from src.datasets.loaders.base import DatasetLoadParams, IDatasetLoader


@dataclass
class CastleMetadata:
    """Metadata extracted from CASTLE file headers."""

    name: str
    version: str
    vulnerable: bool
    description: str
    cwe: int
    line_marker: Optional[int] = None


class CastleDatasetLoader(IDatasetLoader):
    """Loads and processes CASTLE benchmark dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_dir: Path = Path("benchmarks/CASTLE-Source/dataset")
    logger: logging.Logger = logging.getLogger(__name__)

    def _parse_file_metadata(self, content: str) -> CastleMetadata:
        """
        Parse metadata from CASTLE file header comments.

        Args:
            content: File content as string

        Returns:
            CastleMetadata: Parsed metadata
        """
        # Extract header comment block
        header_match = re.search(r"/\*\s*\n(.*?)\*/", content, re.DOTALL)
        if not header_match:
            raise ValueError("No header comment block found")

        header = header_match.group(1)

        # Parse individual fields
        metadata: dict[str, Any] = {}
        for line in header.split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                metadata[key] = value

        # Extract vulnerable flag
        vulnerable = metadata.get("vulnerable", "false").lower() == "true"

        # Extract CWE number
        cwe = int(metadata.get("cwe", "0"))

        return CastleMetadata(
            name=metadata.get("name", ""),
            version=metadata.get("version", ""),
            vulnerable=vulnerable,
            description=metadata.get("description", ""),
            cwe=cwe,
        )

    def _extract_code_content(self, content: str) -> str:
        """
        Extract the actual code content, removing header comments.

        Args:
            content: Full file content

        Returns:
            str: Clean code content
        """
        # Remove header comment block
        code = re.sub(r"/\*\s*\n.*?\*/\s*\n", "", content, flags=re.DOTALL)
        return code.strip()

    def _find_vulnerable_line(self, content: str) -> Optional[int]:
        """
        Find the line marked as vulnerable with {!LINE} marker.

        Args:
            content: File content

        Returns:
            Optional[int]: Line number (1-based) or None if not found
        """
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if "{!LINE}" in line:
                return i
        return None

    def load_single_file(self, file_path: Path) -> BenchmarkSample:
        """
        Load a single CASTLE file and convert to BenchmarkSample.

        Args:
            file_path: Path to the CASTLE .c file

        Returns:
            BenchmarkSample: Processed sample
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse metadata
        metadata = self._parse_file_metadata(content)

        # Extract clean code
        code = self._extract_code_content(content)

        # Find vulnerable line
        vulnerable_line = self._find_vulnerable_line(content)

        # Create sample ID from filename
        sample_id = file_path.stem

        # Determine label based on task type
        # For binary classification: 1 = vulnerable, 0 = safe
        binary_label = 1 if metadata.vulnerable else 0

        # For CWE-specific classification: use CWE number
        cwe_label = f"CWE-{metadata.cwe}" if metadata.vulnerable else "SAFE"

        # Create comprehensive metadata
        sample_metadata: dict[str, Any] = {
            "original_filename": file_path.name,
            "version": metadata.version,
            "description": metadata.description,
            "cwe_number": metadata.cwe,
            "vulnerable_line": vulnerable_line,
            "source": "CASTLE-Benchmark",
        }

        return BenchmarkSample(
            id=sample_id,
            code=code,
            label=binary_label,  # Default to binary label
            metadata=sample_metadata,
            cwe_types=[cwe_label],
            severity="high" if metadata.vulnerable else "none",
        )

    def load_dataset(self, **kwargs: Unpack[DatasetLoadParams]) -> SampleCollection:
        """
        Load the complete CASTLE dataset.

        Args:
            task_type: Type of task ("binary", "cwe_specific", "multiclass")

        Returns:
            SampleCollection: All processed samples
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"CASTLE source directory not found: {self.source_dir}"
            )

        task_type = kwargs.get("task_type", TaskType.BINARY_VULNERABILITY)

        samples: list[BenchmarkSample] = []

        # Iterate through all CWE directories
        for cwe_dir in sorted(self.source_dir.iterdir()):
            if not cwe_dir.is_dir():
                continue

            self.logger.info(f"Processing CWE directory: {cwe_dir.name}")

            # Process all .c files in the directory
            c_files = list(cwe_dir.glob("*.c"))
            for c_file in sorted(c_files):
                try:
                    sample = self.load_single_file(c_file)

                    # Adjust label based on task type
                    if task_type == TaskType.MULTICLASS_VULNERABILITY:
                        if not sample.cwe_types:
                            raise ValueError(f"No CWE types found for {c_file.name}")
                        # We don't have multiclass labels in CASTLE
                        sample.label = sample.cwe_types[0]
                    elif task_type == TaskType.BINARY_CWE_SPECIFIC:
                        # For CWE-specific tasks, we might filter later
                        pass

                    samples.append(sample)

                except Exception as e:
                    self.logger.error(f"Error processing {c_file}: {e}")
                    continue

        self.logger.info(f"Loaded {len(samples)} samples from CASTLE dataset")
        return SampleCollection(root=samples)

    def create_dataset_json(
        self, output_path: str, task_type: TaskType = TaskType.BINARY_VULNERABILITY
    ) -> None:
        """
        Create a JSON dataset file compatible with the benchmark framework.

        Args:
            output_path: Path to output JSON file
            task_type: Type of task classification
        """
        samples = self.load_dataset(task_type=task_type)

        dataset = Dataset(
            metadata=DatasetMetadata(
                name="CASTLE-Benchmark",
                version="1.2",
                task_type=task_type,
                programming_language="C",
            ),
            samples=samples,
        )

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)

        self.logger.info(f"Dataset saved to {output_path}")
        self.logger.info(f"Total samples: {len(samples)}")
        self.logger.info(f"CWE distribution: {dataset.metadata.cwe_distribution}")

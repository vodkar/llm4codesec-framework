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
from typing import Optional

from benchmark.benchmark_framework import DatasetLoader
from benchmark.models import BenchmarkSample


@dataclass
class CastleMetadata:
    """Metadata extracted from CASTLE file headers."""

    name: str
    version: str
    vulnerable: bool
    description: str
    cwe: int
    line_marker: Optional[int] = None


class CastleDatasetLoader:
    """Loads and processes CASTLE benchmark dataset."""

    def __init__(self, source_dir: str = "benchmarks/CASTLE-Source/dataset"):
        """
        Initialize the CASTLE dataset loader.

        Args:
            source_dir: Path to the CASTLE source dataset directory
        """
        self.source_dir = Path(source_dir)
        self.logger = logging.getLogger(__name__)

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
        metadata = {}
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
        sample_metadata = {
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

    def load_dataset(self, task_type: str = "binary") -> list[BenchmarkSample]:
        """
        Load the complete CASTLE dataset.

        Args:
            task_type: Type of task ("binary", "cwe_specific", "multiclass")

        Returns:
            list[BenchmarkSample]: All processed samples
        """
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"CASTLE source directory not found: {self.source_dir}"
            )

        samples = []

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
                    if task_type == "multiclass":
                        if not sample.cwe_types:
                            raise ValueError(f"No CWE types found for {c_file.name}")
                        # We don't have multiclass labels in CASTLE
                        sample.label = sample.cwe_types[0]
                    elif task_type == "cwe_specific":
                        # For CWE-specific tasks, we might filter later
                        pass

                    samples.append(sample)

                except Exception as e:
                    self.logger.error(f"Error processing {c_file}: {e}")
                    continue

        self.logger.info(f"Loaded {len(samples)} samples from CASTLE dataset")
        return samples

    def create_dataset_json(self, output_path: str, task_type: str = "binary") -> None:
        """
        Create a JSON dataset file compatible with the benchmark framework.

        Args:
            output_path: Path to output JSON file
            task_type: Type of task classification
        """
        samples = self.load_dataset(task_type)

        # Convert to dictionary format
        dataset_dict = {
            "metadata": {
                "name": "CASTLE-Benchmark",
                "version": "1.2",
                "task_type": task_type,
                "total_samples": len(samples),
                "vulnerable_samples": sum(
                    1 for s in samples if s.metadata.get("cwe_number", 0) > 0
                ),
                "cwe_distribution": {},
            },
            "samples": [],
        }

        # Calculate CWE distribution
        cwe_counts = {}
        for sample in samples:
            cwe = sample.metadata.get("cwe_number", 0)
            cwe_key = f"CWE-{cwe}" if cwe > 0 else "SAFE"
            cwe_counts[cwe_key] = cwe_counts.get(cwe_key, 0) + 1

        dataset_dict["metadata"]["cwe_distribution"] = cwe_counts

        # Convert samples to dict format
        for sample in samples:
            sample_dict = {
                "id": sample.id,
                "code": sample.code,
                "label": sample.label,
                "cwe_type": sample.cwe_types,
                "severity": sample.severity,
                "metadata": sample.metadata,
            }
            dataset_dict["samples"].append(sample_dict)

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Dataset saved to {output_path}")
        self.logger.info(f"Total samples: {len(samples)}")
        self.logger.info(f"CWE distribution: {cwe_counts}")


class CastleDatasetLoaderFramework(DatasetLoader):
    """
    CASTLE dataset loader that implements the DatasetLoader protocol
    for use with the benchmark framework.
    """

    def __init__(self, source_dir: str = "benchmarks/CASTLE-Source/dataset"):
        self.castle_loader = CastleDatasetLoader(source_dir)

    def load_dataset(self, path: str) -> list[BenchmarkSample]:
        """
        Load dataset from JSON file created by CastleDatasetLoader.

        Args:
            path: Path to the JSON dataset file

        Returns:
            list[BenchmarkSample]: Loaded samples
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for sample_dict in data["samples"]:
            sample = BenchmarkSample(
                id=sample_dict["id"],
                code=sample_dict["code"],
                label=sample_dict["label"],
                metadata=sample_dict["metadata"],
                cwe_types=sample_dict.get("cwe_type"),
                severity=sample_dict.get("severity"),
            )
            samples.append(sample)

        return samples


def filter_by_cwe(
    samples: list[BenchmarkSample], target_cwe: str
) -> list[BenchmarkSample]:
    """
    Filter samples by specific CWE type.

    Args:
        samples: list of benchmark samples
        target_cwe: Target CWE type (e.g., "CWE-125")

    Returns:
        list[BenchmarkSample]: Filtered samples
    """
    filtered = []
    for sample in samples:
        sample_cwe = f"CWE-{sample.metadata.get('cwe_number', 0)}"
        if sample_cwe == target_cwe or sample.cwe_types == ["SAFE"]:
            # Create a copy with binary label for CWE-specific task
            sample_copy = BenchmarkSample(
                id=sample.id,
                code=sample.code,
                label=1 if sample_cwe == target_cwe else 0,
                metadata=sample.metadata.copy(),
                cwe_types=sample.cwe_types,
                severity=sample.severity,
            )
            filtered.append(sample_copy)

    return filtered


def get_available_cwes(samples: list[BenchmarkSample]) -> list[str]:
    """
    Get list of available CWE types in the dataset.

    Args:
        samples: list of benchmark samples

    Returns:
        list[str]: Available CWE types
    """
    cwes = set()
    for sample in samples:
        cwe_num = sample.metadata.get("cwe_number", 0)
        if cwe_num > 0:
            cwes.add(f"CWE-{cwe_num}")

    return sorted(list(cwes))

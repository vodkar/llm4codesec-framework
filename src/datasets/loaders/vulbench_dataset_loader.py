#!/usr/bin/env python3
"""
VulBench Dataset Loader

This module provides functionality to load and process VulBench datasets
for vulnerability detection benchmarks.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.models import BenchmarkSample


class VulBenchDatasetLoader:
    """Dataset loader for VulBench benchmark format."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_dataset(
        self,
        data_path: str,
        task_type: str = "binary",
        dataset_name: str = "d2a",
        max_samples: Optional[int] = None,
        vulnerability_type: Optional[str] = None,
    ) -> List[BenchmarkSample]:
        """
        Load VulBench dataset from JSON file.

        Args:
            data_path: Path to the dataset JSON file
            task_type: Type of task ("binary", "multiclass", or "binary_vulnerability_specific")
            dataset_name: Name of the dataset (d2a, ctf, magma, etc.)
            max_samples: Maximum number of samples to load
            vulnerability_type: Specific vulnerability type to filter for (e.g., "Buffer-Overflow")

        Returns:
            List of BenchmarkSample objects
        """
        self.logger.info(f"Loading VulBench dataset: {data_path}")
        if vulnerability_type:
            self.logger.info(f"Filtering for vulnerability type: {vulnerability_type}")

        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        samples = self._load_raw_dataset(
            data_file, task_type, dataset_name, max_samples, vulnerability_type
        )

        self.logger.info(f"Loaded {len(samples)} samples from {data_path}")
        return samples

    def _load_raw_dataset(
        self,
        data_path: Path,
        task_type: str,
        dataset_name: str,
        max_samples: Optional[int],
        vulnerability_type: Optional[str] = None,
    ) -> List[BenchmarkSample]:
        """Load raw dataset from JSON file and convert to BenchmarkSample objects."""
        samples = []

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for i, item in enumerate(data):
                if max_samples and i >= max_samples:
                    break

                try:
                    vulbench_samples = self._convert_vulbench_item_to_samples(
                        item, i, task_type, dataset_name, vulnerability_type
                    )
                    samples.extend(vulbench_samples)
                except Exception as e:
                    self.logger.warning(f"Error processing item {i}: {e}")
                    continue

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in dataset file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

        return samples

    def _convert_vulbench_item_to_samples(
        self,
        item: Dict[str, Any],
        line_num: int,
        task_type: str,
        dataset_name: str,
        vulnerability_type: Optional[str] = None,
    ) -> List[BenchmarkSample]:
        """Convert a VulBench item to BenchmarkSample objects."""
        samples = []

        # Extract basic information
        code = item.get("code", "").strip()
        vulnerable = item.get("vulnerable", False)
        vulnerability_types = item.get("vulnerability_types", [])
        identifier = item.get("identifier", f"{dataset_name}_{line_num}")
        func_name = item.get("func_name", "unknown")

        if not code:
            return samples

        # If filtering for specific vulnerability type, check if this item matches
        if vulnerability_type and task_type == "binary_vulnerability_specific":
            # For vulnerability-specific binary classification, we include:
            # 1. Samples that contain the specific vulnerability type (labeled as 1)
            # 2. Samples that are safe (no vulnerabilities, labeled as 0)
            has_target_vulnerability = vulnerability_type in vulnerability_types
            is_safe = not vulnerable

            if not (has_target_vulnerability or is_safe):
                # Skip samples with other vulnerability types
                return samples

        # Common metadata
        base_metadata = {
            "dataset": dataset_name,
            "identifier": identifier,
            "func_name": func_name,
            "source": "vulbench",
            "line_number": line_num,
        }

        if task_type == "binary":
            # Binary classification: vulnerable vs non-vulnerable
            label = 1 if vulnerable else 0

            sample = BenchmarkSample(
                id=f"vulbench_{dataset_name}_{line_num}",
                code=code,
                label=label,
                metadata={**base_metadata, "original_vulnerable": vulnerable},
                cwe_types=vulnerability_types if vulnerability_types else None,
                severity=self._get_vulnerability_severity(vulnerability_types),
            )
            samples.append(sample)

        elif task_type == "binary_vulnerability_specific":
            # Binary classification: specific vulnerability type vs safe
            has_target_vulnerability = vulnerability_type in vulnerability_types
            label = 1 if has_target_vulnerability else 0

            sample = BenchmarkSample(
                id=f"vulbench_{dataset_name}_{line_num}_{vulnerability_type.replace('-', '_')}",
                code=code,
                label=label,
                metadata={
                    **base_metadata,
                    "original_vulnerable": vulnerable,
                    "target_vulnerability_type": vulnerability_type,
                    "has_target_vulnerability": has_target_vulnerability,
                },
                cwe_types=[vulnerability_type] if has_target_vulnerability else None,
                severity=self._get_vulnerability_severity(
                    [vulnerability_type] if has_target_vulnerability else []
                ),
            )
            samples.append(sample)

        elif task_type == "multiclass":
            # Multi-class classification: vulnerability type identification
            if vulnerability_types:
                # If vulnerable, use the vulnerability type
                label = (
                    vulnerability_types[0]
                    if len(vulnerability_types) == 1
                    else "Multiple-Vulnerabilities"
                )
            else:
                # If not vulnerable, label as safe
                label = "SAFE"

            sample = BenchmarkSample(
                id=f"vulbench_{dataset_name}_{line_num}_multiclass",
                code=code,
                label=label,
                metadata={
                    **base_metadata,
                    "original_vulnerable": vulnerable,
                    "all_vulnerability_types": vulnerability_types,
                },
                cwe_types=vulnerability_types if vulnerability_types else None,
                severity=self._get_vulnerability_severity(vulnerability_types),
            )
            samples.append(sample)

        return samples

    def _get_vulnerability_severity(
        self, vulnerability_types: List[str]
    ) -> Optional[str]:
        """Determine severity based on vulnerability types."""
        if not vulnerability_types:
            return None

        # Define severity mapping based on common vulnerability types
        high_severity = [
            "Buffer-Overflow",
            "Integer-Overflow",
            "Use-After-Free",
            "Double-Free",
            "Format-String-Vulnerability",
        ]
        medium_severity = [
            "Null-Pointer-Dereference",
            "Memory-Leak",
            "Race-Condition",
            "Improper-Access-Control",
        ]

        for vul_type in vulnerability_types:
            if vul_type in high_severity:
                return "high"
            elif vul_type in medium_severity:
                return "medium"

        return "low"

    def create_dataset_from_vulbench_data(
        self,
        vulbench_data_dir: str,
        output_path: str,
        dataset_name: str,
        task_type: str = "binary",
    ) -> None:
        """
        Create a processed dataset from VulBench raw data directory.

        Args:
            vulbench_data_dir: Path to VulBench data directory (e.g., benchmarks/VulBench/data/d2a)
            output_path: Output path for processed JSON file
            dataset_name: Name of the dataset (d2a, ctf, magma, etc.)
            task_type: Type of task ("binary" or "multiclass")
        """
        self.logger.info(
            f"Creating {task_type} dataset for {dataset_name} from {vulbench_data_dir}"
        )

        data_dir = Path(vulbench_data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"VulBench data directory not found: {vulbench_data_dir}"
            )

        processed_data = []

        # Process each subdirectory in the VulBench data
        for subdir in data_dir.iterdir():
            if not subdir.is_dir():
                continue

            try:
                # Read metadata
                meta_file = subdir / "meta_data.json"
                if not meta_file.exists():
                    self.logger.warning(f"No metadata found for {subdir.name}")
                    continue

                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # Read source code
                src_file = subdir / "src.c"
                if not src_file.exists():
                    self.logger.warning(f"No source code found for {subdir.name}")
                    continue

                with open(src_file, "r", encoding="utf-8") as f:
                    code = f.read()

                # Create processed item
                item = {
                    "identifier": subdir.name,
                    "code": code,
                    "vulnerable": metadata.get("vulnerable", False),
                    "vulnerability_types": metadata.get("vulnerability_type", []),
                    "func_name": subdir.name,
                    "dataset": dataset_name,
                }

                processed_data.append(item)

            except Exception as e:
                self.logger.warning(f"Error processing {subdir.name}: {e}")
                continue

        # Save processed data
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"Created {task_type} dataset with {len(processed_data)} samples: {output_path}"
        )

    def get_dataset_stats(self, data_file: str) -> Dict[str, Any]:
        """
        Generate statistics for a VulBench dataset.

        Args:
            data_file: Path to the dataset JSON file

        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "total_samples": 0,
            "vulnerable_samples": 0,
            "safe_samples": 0,
            "vulnerability_distribution": defaultdict(int),
            "average_code_length": 0,
            "dataset_name": Path(data_file).stem,
        }

        try:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            code_lengths = []

            for item in data:
                stats["total_samples"] += 1

                # Vulnerability status
                if item.get("vulnerable", False):
                    stats["vulnerable_samples"] += 1
                else:
                    stats["safe_samples"] += 1

                # Vulnerability types
                vul_types = item.get("vulnerability_types", [])
                if vul_types:
                    for vul_type in vul_types:
                        stats["vulnerability_distribution"][vul_type] += 1
                else:
                    stats["vulnerability_distribution"]["No-Vulnerability"] += 1

                # Code length
                code = item.get("code", "")
                code_lengths.append(len(code))

            # Calculate average code length
            if code_lengths:
                stats["average_code_length"] = sum(code_lengths) / len(code_lengths)

        except Exception as e:
            self.logger.error(f"Error generating statistics: {e}")
            return {}

        # Convert defaultdict to regular dict
        stats["vulnerability_distribution"] = dict(stats["vulnerability_distribution"])

        return stats

    def generate_vulnerability_specific_datasets(
        self,
        source_datasets: List[str],
        output_dir: str = "datasets_processed/vulbench",
    ) -> None:
        """
        Generate vulnerability-specific binary classification datasets.

        Args:
            source_datasets: List of paths to multiclass VulBench datasets
            output_dir: Directory to save the generated datasets
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Track all vulnerability types across datasets
        all_vulnerability_types = set()

        # First pass: collect all vulnerability types
        for dataset_path in source_datasets:
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item in data:
                    vulnerability_types = item.get("vulnerability_types", [])
                    all_vulnerability_types.update(vulnerability_types)

            except Exception as e:
                self.logger.warning(f"Error reading {dataset_path}: {e}")
                continue

        self.logger.info(
            f"Found vulnerability types: {sorted(all_vulnerability_types)}"
        )

        # Generate datasets for each vulnerability type
        for vuln_type in sorted(all_vulnerability_types):
            if vuln_type == "No-Vulnerability":  # Skip this as it's handled separately
                continue

            vuln_type_safe = vuln_type.replace("-", "_").replace(" ", "_")

            # Combine all datasets for this vulnerability type
            combined_samples = []

            for dataset_path in source_datasets:
                dataset_name = Path(dataset_path).stem.replace(
                    "vulbench_multiclass_", ""
                )

                try:
                    samples = self.load_dataset(
                        dataset_path,
                        task_type="binary_vulnerability_specific",
                        dataset_name=dataset_name,
                        vulnerability_type=vuln_type,
                    )
                    combined_samples.extend(samples)

                except Exception as e:
                    self.logger.warning(
                        f"Error processing {dataset_path} for {vuln_type}: {e}"
                    )
                    continue

            if combined_samples:
                # Convert to JSON format
                json_data = []
                for sample in combined_samples:
                    json_data.append(
                        {
                            "identifier": sample.id,
                            "code": sample.code,
                            "label": sample.label,
                            "vulnerable": sample.label == 1,
                            "vulnerability_type": vuln_type,
                            "dataset": sample.metadata.get("dataset", "unknown"),
                            "target_vulnerability_type": vuln_type,
                        }
                    )

                # Save the dataset
                output_file = (
                    output_path / f"vulbench_binary_{vuln_type_safe.lower()}.json"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)

                self.logger.info(
                    f"Generated {vuln_type} dataset: {len(json_data)} samples -> {output_file}"
                )

                # Generate statistics
                stats = self._calculate_dataset_statistics(json_data, "binary")
                stats_file = (
                    output_path / f"vulbench_{vuln_type_safe.lower()}_stats.json"
                )
                with open(stats_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)

    def _calculate_dataset_statistics(
        self, data: List[Dict[str, Any]], task_type: str
    ) -> Dict[str, Any]:
        """
        Calculate statistics for a dataset.

        Args:
            data: The dataset as a list of dictionaries
            task_type: The type of task ("binary" or "multiclass")

        Returns:
            A dictionary containing the calculated statistics
        """
        stats = {
            "total_samples": len(data),
            "vulnerable_samples": sum(1 for item in data if item.get("label") == 1),
            "safe_samples": sum(1 for item in data if item.get("label") == 0),
            "vulnerability_distribution": defaultdict(int),
        }

        # Vulnerability type distribution
        for item in data:
            vul_types = item.get("vulnerability_types", [])
            if vul_types:
                for vul_type in vul_types:
                    stats["vulnerability_distribution"][vul_type] += 1
            else:
                stats["vulnerability_distribution"]["No-Vulnerability"] += 1

        # Convert defaultdict to regular dict
        stats["vulnerability_distribution"] = dict(stats["vulnerability_distribution"])

        return stats


class VulBenchDatasetLoaderFramework:
    """Framework-compatible wrapper for VulBench dataset loader."""

    def __init__(self):
        self.loader = VulBenchDatasetLoader()

    def load_dataset(
        self,
        dataset_path: str,
        max_samples: Optional[int] = None,
        **kwargs,
    ) -> List[BenchmarkSample]:
        """
        Load dataset compatible with benchmark framework.

        Args:
            dataset_path: Path to the dataset file
            task_type: Task type (binary_vulnerability, multiclass_vulnerability)
            max_samples: Maximum number of samples to load
            **kwargs: Additional parameters

        Returns:
            List of BenchmarkSample objects
        """
        # Extract dataset name from path
        dataset_name = (
            Path(dataset_path).stem.split("_")[-1]
            if "_" in Path(dataset_path).stem
            else "vulbench"
        )

        # Convert task type
        if "multiclass" in dataset_path:
            vulbench_task_type = "multiclass"
        else:
            vulbench_task_type = "binary"

        return self.loader.load_dataset(
            dataset_path,
            task_type=vulbench_task_type,
            dataset_name=dataset_name,
            max_samples=max_samples,
        )

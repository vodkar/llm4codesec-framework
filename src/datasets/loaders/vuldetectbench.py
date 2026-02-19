#!/usr/bin/env python3
"""
VulDetectBench Dataset Loader

This module provides functionality to load and process VulDetectBench datasets
for vulnerability detection benchmarks. VulDetectBench contains 5 tasks of increasing
difficulty for evaluating LLM vulnerability detection capabilities.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.models import BenchmarkSample


class VulDetectBenchDatasetLoader:
    """Dataset loader for VulDetectBench benchmark format."""

    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)

    @staticmethod
    def _normalize_cwe(raw_cwe: Any) -> str:
        """Normalize raw CWE values to CWE-* format."""
        if raw_cwe is None:
            return ""

        raw_cwe_string: str = str(raw_cwe).strip()
        if not raw_cwe_string:
            return ""

        cwe_match = re.search(r"CWE-\d+|\b\d+\b", raw_cwe_string.upper())
        if cwe_match is None:
            return ""

        matched_value: str = cwe_match.group(0)
        return (
            matched_value
            if matched_value.startswith("CWE-")
            else f"CWE-{matched_value}"
        )

    def load_dataset(
        self,
        data_path: str,
        task_type: str = "task1",
        max_samples: int | None = None,
    ) -> list[BenchmarkSample]:
        """
        Load VulDetectBench dataset from JSONL file.

        Args:
            data_path: Path to the dataset JSONL file
            task_type: Type of task (task1, task2, task3, task4, task5)
            max_samples: Maximum number of samples to load

        Returns:
            list of BenchmarkSample objects
        """
        self.logger.info(f"Loading VulDetectBench dataset: {data_path}")

        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        samples = self._load_raw_dataset(data_file, task_type, max_samples)

        self.logger.info(f"Loaded {len(samples)} samples from {data_path}")
        return samples

    def load_processed_dataset(self, data_file: Path) -> list[BenchmarkSample]:
        self.logger.info(f"Loading processed dataset: {data_file}")

        if not data_file.exists():
            raise FileNotFoundError(f"Processed dataset file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            try:
                data: dict[str, Any] | list[dict[str, Any]] = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("samples"), list):
                    samples_payload: list[Any] = data["samples"]
                    return [
                        BenchmarkSample(**sample)
                        for sample in samples_payload
                        if isinstance(sample, dict)
                    ]
                else:
                    items: list[Any] = data if isinstance(data, list) else []
                    return [
                        BenchmarkSample(**item)
                        for item in items
                        if isinstance(item, dict)
                    ]
            except json.JSONDecodeError:
                self.logger.exception("Error decoding JSON from %s", data_file)
                raise

    def _load_raw_dataset(
        self, data_path: Path, task_type: str, max_samples: int | None
    ) -> list[BenchmarkSample]:
        """Load raw dataset from JSONL file and convert to BenchmarkSample objects."""
        samples: list[BenchmarkSample] = []

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                        vuldetectbench_samples = (
                            self._convert_vuldetectbench_item_to_samples(
                                item, i, task_type
                            )
                        )
                        samples.extend(vuldetectbench_samples)
                    except json.JSONDecodeError:
                        self.logger.exception("Error parsing line %s", i + 1)
                        continue
                    except Exception:
                        self.logger.exception("Error processing item %s", i)
                        continue

        except Exception:
            self.logger.exception("Error loading dataset")
            raise

        return samples

    def _convert_vuldetectbench_item_to_samples(
        self, item: dict[str, Any], line_num: int, task_type: str
    ) -> list[BenchmarkSample]:
        """Convert a VulDetectBench item to BenchmarkSample objects."""
        samples: list[BenchmarkSample] = []

        # Extract basic information
        code: str = str(item.get("code", "")).strip()
        answer: str = str(item.get("answer", ""))
        cwe: str = self._normalize_cwe(item.get("cwe", ""))
        idx: str = str(item.get("idx", str(line_num)))

        if not code:
            return samples

        # Handle task-specific data extraction
        if task_type == "task1":
            # Task 1: Binary vulnerability detection (YES/NO)
            label: int | str = 1 if answer.upper() == "YES" else 0
            task_description: str = "binary_vulnerability"
        elif task_type == "task2":
            # Task 2: Multi-choice vulnerability type inference
            label = answer  # Keep as string for multi-class
            task_description = "multiclass_vulnerability"
        elif task_type in ["task3", "task4", "task5"]:
            # Task 3-5: Code analysis tasks (keep answer as string)
            label = answer
            if task_type == "task3":
                task_description = "key_objects_identification"
            elif task_type == "task4":
                task_description = "root_cause_location"
            else:  # task5
                task_description = "trigger_point_location"
        else:
            self.logger.warning(f"Unknown task type: {task_type}")
            return samples

        # Common metadata
        base_metadata: dict[str, Any] = {
            "task_type": task_type,
            "task_description": task_description,
            "original_idx": idx,
            "cwe": cwe,
            "source": "vuldetectbench",
            "line_number": line_num,
        }

        # Add task-specific metadata
        if task_type == "task2" and "selection" in item:
            base_metadata["selection_choices"] = item["selection"]

        sample = BenchmarkSample(
            id=f"vuldetectbench_{task_type}_{idx}",
            code=code,
            label=label,
            metadata=base_metadata,
            cwe_types=[cwe] if cwe else [],
            severity=self._get_vulnerability_severity(cwe) if cwe else None,
        )
        samples.append(sample)

        return samples

    def _get_vulnerability_severity(self, cwe: str) -> str | None:
        """Determine severity based on CWE type."""
        if not cwe:
            return None

        # Extract CWE number
        cwe_num = cwe.split("-")[-1] if "-" in cwe else cwe

        # Define severity mapping based on common CWE patterns
        high_severity_cwes = {
            "78",
            "79",
            "89",
            "94",
            "119",
            "120",
            "121",
            "122",
            "123",
            "124",
            "787",
            "788",
            "416",
            "415",
            "400",
        }
        medium_severity_cwes = {
            "125",
            "190",
            "476",
            "401",
            "403",
            "404",
            "502",
            "611",
            "776",
        }

        if cwe_num in high_severity_cwes:
            return "high"
        elif cwe_num in medium_severity_cwes:
            return "medium"
        else:
            return "low"

    def create_dataset_from_vuldetectbench_data(
        self, vuldetectbench_data_dir: str, output_path: str, task_type: str = "task1"
    ) -> None:
        """
        Create a processed dataset from VulDetectBench raw data directory.

        Args:
            vuldetectbench_data_dir: Path to VulDetectBench data directory
            output_path: Output path for processed JSON file
            task_type: Type of task (task1, task2, task3, task4, task5)
        """
        self.logger.info(
            f"Creating dataset for {task_type} from {vuldetectbench_data_dir}"
        )

        data_dir = Path(vuldetectbench_data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"VulDetectBench data directory not found: {vuldetectbench_data_dir}"
            )

        # Find the appropriate task file
        task_file = data_dir / f"{task_type}_code.jsonl"
        if not task_file.exists():
            raise FileNotFoundError(f"Task file not found: {task_file}")

        processed_data: list[BenchmarkSample] = []

        with open(task_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item: dict[str, Any] = json.loads(line)
                    cwe_value: str = self._normalize_cwe(item.get("cwe", ""))
                    answer_value: str = str(item.get("answer", ""))
                    processed_item = BenchmarkSample(
                        id=f"vuldetectbench_{task_type}_{item.get('idx', line_num)}",
                        code=str(item.get("code", "")).strip(),
                        label=1
                        if task_type == "task1" and answer_value.upper() == "YES"
                        else (0 if task_type == "task1" else answer_value),
                        metadata={
                            "task_type": task_type,
                            "task_description": self._get_task_description(task_type),
                            "original_idx": item.get("idx", str(line_num)),
                            "cwe": cwe_value,
                            "source": "vuldetectbench",
                            "line_number": line_num,
                        },
                        cwe_types=[cwe_value] if cwe_value else [],
                        severity=self._get_vulnerability_severity(cwe_value),
                    )

                    # Add task-specific fields
                    if task_type == "task2" and "selection" in item:
                        processed_item.metadata["selection_choices"] = item["selection"]

                    processed_data.append(processed_item)

                except Exception:
                    self.logger.exception("Error processing line %s", line_num)
                    continue

        # Save processed data
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create metadata
        dataset_info: dict[str, Any] = {
            "metadata": {
                "task_type": task_type,
                "total_samples": len(processed_data),
                "source": "vuldetectbench",
                "description": self._get_task_description(task_type),
            },
            "samples": [row.model_dump() for row in processed_data],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"Created {task_type} dataset with {len(processed_data)} samples: {output_path}"
        )

    def _get_task_description(self, task_type: str) -> str:
        """Get description for each task type."""
        descriptions = {
            "task1": "Binary vulnerability detection (YES/NO)",
            "task2": "Multi-choice vulnerability type inference",
            "task3": "Key objects and functions identification",
            "task4": "Root cause location identification",
            "task5": "Trigger point location identification",
        }
        return descriptions.get(task_type, f"VulDetectBench {task_type}")

    def get_dataset_stats(self, data_file: str) -> dict[str, Any]:
        """
        Generate statistics for a VulDetectBench dataset.

        Args:
            data_file: Path to the dataset file

        Returns:
            dictionary containing dataset statistics
        """
        vulnerability_distribution: dict[str, int] = defaultdict(int)
        cwe_distribution: dict[str, int] = defaultdict(int)
        stats: dict[str, Any] = {
            "total_samples": 0,
            "task_type": "",
            "average_code_length": 0.0,
            "dataset_name": Path(data_file).stem,
        }

        try:
            data_path = Path(data_file)
            items: list[dict[str, Any]]

            # Handle both JSON and JSONL formats
            if data_path.suffix == ".json":
                with open(data_file, "r", encoding="utf-8") as f:
                    data: dict[str, Any] | list[dict[str, Any]] = json.load(f)
                    if isinstance(data, dict) and "samples" in data:
                        items = [
                            sample
                            for sample in data["samples"]
                            if isinstance(sample, dict)
                        ]
                        stats["task_type"] = data.get("metadata", {}).get(
                            "task_type", "unknown"
                        )
                    else:
                        items = data if isinstance(data, list) else []
            else:  # JSONL format
                items = []
                with open(data_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            items.append(json.loads(line))

            code_lengths: list[int] = []

            for item in items:
                stats["total_samples"] += 1

                # CWE distribution
                cwe: str = str(item.get("cwe", "Unknown"))
                if cwe:
                    cwe_distribution[cwe] += 1
                else:
                    cwe_distribution["No-CWE"] += 1

                # Answer distribution (for Task 1)
                answer: str = str(item.get("answer", ""))
                if answer:
                    vulnerability_distribution[answer] += 1

                # Code length
                code: str = str(item.get("code", ""))
                code_lengths.append(len(code))

            # Calculate average code length
            if code_lengths:
                stats["average_code_length"] = sum(code_lengths) / len(code_lengths)

        except Exception:
            self.logger.exception("Error generating statistics")
            return {}

        # Convert defaultdicts to regular dicts
        stats["vulnerability_distribution"] = dict(vulnerability_distribution)
        stats["cwe_distribution"] = dict(cwe_distribution)

        return stats


class VulDetectBenchDatasetLoaderFramework:
    """Framework-compatible wrapper for VulDetectBench dataset loader."""

    def __init__(self) -> None:
        self.loader: VulDetectBenchDatasetLoader = VulDetectBenchDatasetLoader()
        self.logger: logging.Logger = logging.getLogger(__name__)

    def load_dataset(
        self,
        dataset_path: str,
        task_type: str = "task1_vulnerability",
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> list[BenchmarkSample]:
        """
        Load dataset compatible with benchmark framework.

        Args:
            dataset_path: Path to the dataset file
            task_type: Task type (task1_vulnerability, task2_multiclass, etc.)
            max_samples: Maximum number of samples to load
            **kwargs: Additional parameters

        Returns:
            list of BenchmarkSample objects
        """
        # Convert framework task type to VulDetectBench task type
        if "task1" in task_type:
            vuldetectbench_task_type: str = "task1"
        elif "task2" in task_type:
            vuldetectbench_task_type = "task2"
        elif "task3" in task_type:
            vuldetectbench_task_type = "task3"
        elif "task4" in task_type:
            vuldetectbench_task_type = "task4"
        elif "task5" in task_type:
            vuldetectbench_task_type = "task5"
        else:
            vuldetectbench_task_type = "task1"

        return self.loader.load_dataset(
            dataset_path, task_type=vuldetectbench_task_type, max_samples=max_samples
        )

    def load_processed_dataset(
        self,
        dataset_path: Path,
    ) -> list[BenchmarkSample]:
        """
        Load processed dataset for benchmark framework.

        Args:
            dataset_path: Path to the processed dataset file
            task_type: Task type (task1, task2, etc.)
            max_samples: Maximum number of samples to load

        Returns:
            list of BenchmarkSample objects
        """
        return self.loader.load_processed_dataset(dataset_path)

    def get_dataset_info(self, data_file: str) -> dict[str, Any]:
        """
        Get information about the dataset.

        Args:
            data_file: Path to the dataset file

        Returns:
            dictionary with dataset information
        """
        return self.loader.get_dataset_stats(data_file)

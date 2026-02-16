#!/usr/bin/env python3
"""
JitVul Dataset Loader

This module provides dataset loading functionality for the JitVul benchmark,
which contains real-world vulnerability data with function pairs (vulnerable vs. non-vulnerable)
and associated call graph information.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Unpack

from pydantic import BaseModel, PrivateAttr

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, CWEType, SampleCollection
from datasets.loaders.base import DatasetLoadParams, IDatasetLoader


class JitVulDatasetLoader(IDatasetLoader):
    """Framework-compatible dataset loader for JitVul."""

    source_path: Path
    __logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__)
    )

    def load_dataset(
        self,
        is_use_call_graph: bool = True,
        **kwargs: Unpack[DatasetLoadParams],
    ) -> SampleCollection:
        """
        Load raw JSONL dataset format.

        Args:
            task_type: Type of task being performed
            target_cwe: Target CWE for cwe_specific task
            is_use_call_graph: Whether to include call graph context
            max_samples: Maximum number of samples to load

        Returns:
            SampleCollection containing loaded samples
        """
        samples: list[BenchmarkSample] = []
        max_samples = kwargs.get("limit", None)
        task_type = kwargs.get("task_type", TaskType.BINARY_VULNERABILITY)
        target_cwe = kwargs.get("target_cwe", None)

        with open(self.source_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and len(samples) >= max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    # Convert each item to samples based on task type
                    jitvul_samples = self._convert_jitvul_item_to_samples(
                        item, line_num, task_type, target_cwe, is_use_call_graph
                    )
                    samples.extend(jitvul_samples)
                except json.JSONDecodeError as e:
                    self.__logger.warning(
                        f"Skipping invalid JSON at line {line_num}: {e}"
                    )
                    continue
                except Exception as e:
                    self.__logger.warning(f"Error processing line {line_num}: {e}")
                    continue

        self.__logger.info(f"Loaded {len(samples)} samples from raw JitVul dataset")
        return SampleCollection(samples)

    def _convert_jitvul_item_to_samples(
        self,
        item: dict[str, Any],
        line_num: int,
        task_type: TaskType,
        target_cwe: str | None,
        is_use_call_graph: bool,
    ) -> list[BenchmarkSample]:
        """
        Convert a single JitVul item into BenchmarkSample objects.

        Args:
            item: Raw JitVul data item
            line_num: Line number for ID generation
            task_type: Type of task being performed
            target_cwe: Target CWE for cwe_specific task
            is_use_call_graph: Whether to include call graph context

        Returns:
            list containing appropriate samples based on task type
        """
        samples: list[BenchmarkSample] = []

        # Extract basic information
        vuln_func = item.get("vulnerable_function_body", "")
        non_vuln_func = item.get("non_vulnerable_function_body", "")
        cwe = CWEType(item.get("cwe", ""))
        project = item.get("project", "unknown")
        func_hash = item.get("func_hash", "")

        if not vuln_func.strip() or not non_vuln_func.strip():
            return samples

        # Extract common metadata
        base_metadata: dict[str, Any] = {
            "project": project,
            "cwe": cwe,
            "function_hash": func_hash,
            "source": "jitvul",
            "line_number": line_num,
        }
        cwe_severity = CWEType(cwe).get_cwe_severity()

        # Create samples based on task type
        if task_type == TaskType.BINARY_VULNERABILITY:
            # Create both vulnerable and non-vulnerable samples
            vuln_sample = BenchmarkSample(
                id=f"jitvul_{line_num}_vulnerable",
                code=self._augment_code_with_context(
                    vuln_func, item, is_use_call_graph
                ),
                label=1,
                metadata={
                    **base_metadata,
                    "function_type": "vulnerable",
                    "original_cwe": cwe,
                },
                cwe_types=[cwe] if cwe else [],
                severity=cwe_severity,
            )

            non_vuln_sample = BenchmarkSample(
                id=f"jitvul_{line_num}_non_vulnerable",
                code=self._augment_code_with_context(
                    non_vuln_func, item, is_use_call_graph
                ),
                label=0,
                metadata={**base_metadata, "function_type": "non_vulnerable"},
                cwe_types=[],
                severity=None,
            )

            samples.extend([vuln_sample, non_vuln_sample])

        elif task_type == TaskType.MULTICLASS_VULNERABILITY:
            # Only include vulnerable samples with CWE labels
            if cwe:
                vuln_sample = BenchmarkSample(
                    id=f"jitvul_{line_num}_vulnerable",
                    code=self._augment_code_with_context(
                        vuln_func, item, is_use_call_graph
                    ),
                    label=cwe,
                    metadata={**base_metadata, "function_type": "vulnerable"},
                    cwe_types=[cwe],
                    severity=cwe_severity,
                )
                samples.append(vuln_sample)

        elif task_type == TaskType.BINARY_CWE_SPECIFIC:
            # Filter for specific CWE type
            if target_cwe and target_cwe in cwe:
                # Include both vulnerable (positive) and non-vulnerable (negative) for this CWE
                vuln_sample = BenchmarkSample(
                    id=f"jitvul_{line_num}_vulnerable",
                    code=self._augment_code_with_context(
                        vuln_func, item, is_use_call_graph
                    ),
                    label=1,
                    metadata={
                        **base_metadata,
                        "function_type": "vulnerable",
                        "target_cwe": target_cwe,
                    },
                    cwe_types=[cwe] if cwe else [],
                    severity=cwe_severity,
                )

                non_vuln_sample = BenchmarkSample(
                    id=f"jitvul_{line_num}_non_vulnerable",
                    code=self._augment_code_with_context(
                        non_vuln_func, item, is_use_call_graph
                    ),
                    label=0,
                    metadata={
                        **base_metadata,
                        "function_type": "non_vulnerable",
                        "target_cwe": target_cwe,
                    },
                    cwe_types=[],
                    severity=None,
                )

                samples.extend([vuln_sample, non_vuln_sample])

        return samples


class JitVulDatasetLoaderFramework(BaseModel):
    """Framework-compatible wrapper for JitVul dataset loader."""

    loader: JitVulDatasetLoader

    def load_dataset(
        self,
        dataset_path: str,
        max_samples: int | None = None,
        task_type: TaskType | None = None,
        target_cwe: str | None = None,
        is_use_call_graph: bool = True,
    ) -> list[BenchmarkSample]:
        """
        Load dataset compatible with benchmark framework.

        Args:
            dataset_path: Path to the dataset file.
            max_samples: Maximum number of samples to load.
            task_type: Benchmark task type override.
            target_cwe: Target CWE for cwe_specific task.
            is_use_call_graph: Whether to include call graph context.

        Returns:
            list of BenchmarkSample objects.
        """
        self.loader.source_path = Path(dataset_path)
        resolved_task_type: TaskType = task_type or TaskType.BINARY_VULNERABILITY

        samples: SampleCollection = self.loader.load_dataset(
            is_use_call_graph=is_use_call_graph,
            task_type=resolved_task_type,
            target_cwe=target_cwe,
            limit=max_samples,
        )

        return list(samples)

    def _augment_code_with_context(
        self, code: str, item: dict[str, Any], use_call_graph: bool
    ) -> str:
        """
        Augment code with call graph context if available.

        Args:
            code: Original function code
            item: JitVul data item
            use_call_graph: Whether to add call graph context

        Returns:
            Augmented code string
        """
        if not use_call_graph or "call_graph" not in item:
            return code

        call_graph = item.get("call_graph", [])
        if not call_graph:
            return code

        # Add call graph context as comments
        context_lines = ["// Call graph context:"]
        for func_name in call_graph[:5]:  # Limit to first 5 for token efficiency
            context_lines.append(f"// - {func_name}")

        context = "\n".join(context_lines) + "\n\n"
        return context + code

    def get_dataset_stats(self, data_file: Path) -> dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.

        Args:
            data_file: Path to the dataset file

        Returns:
            dictionary containing dataset statistics
        """
        stats: dict[str, Any] = {
            "total_items": 0,
            "cwe_distribution": defaultdict(int),
            "project_distribution": defaultdict(int),
            "severity_distribution": defaultdict(int),
            "has_call_graph": 0,
            "average_function_length": {"vulnerable": 0, "non_vulnerable": 0},
        }

        vuln_lengths: list[int] = []
        non_vuln_lengths: list[int] = []

        try:
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                        stats["total_items"] += 1

                        # CWE distribution
                        cwe = item.get("cwe", ["Unknown"])[0]
                        stats["cwe_distribution"][cwe] += 1

                        # Project distribution
                        project = item.get("project", "Unknown")
                        stats["project_distribution"][project] += 1

                        # Severity distribution
                        severity = CWEType(cwe).get_cwe_severity()
                        stats["severity_distribution"][severity] += 1

                        # Call graph presence
                        if "call_graph" in item and item["call_graph"]:
                            stats["has_call_graph"] += 1

                        # Function lengths
                        vuln_func = item.get("vulnerable_function", "")
                        non_vuln_func = item.get("non_vulnerable_function", "")
                        vuln_lengths.append(len(vuln_func))
                        non_vuln_lengths.append(len(non_vuln_func))

                    except json.JSONDecodeError:
                        continue

        except Exception:
            self.__logger.exception("Error generating statistics")
            return {}

        # Calculate averages
        if vuln_lengths:
            stats["average_function_length"]["vulnerable"] = sum(vuln_lengths) / len(
                vuln_lengths
            )
        if non_vuln_lengths:
            stats["average_function_length"]["non_vulnerable"] = sum(
                non_vuln_lengths
            ) / len(non_vuln_lengths)

        # Convert defaultdicts to regular dicts
        stats["cwe_distribution"] = dict(stats["cwe_distribution"])
        stats["project_distribution"] = dict(stats["project_distribution"])
        stats["severity_distribution"] = dict(stats["severity_distribution"])

        return stats

#!/usr/bin/env python3
"""
JitVul Dataset Loader

This module provides dataset loading functionality for the JitVul benchmark,
which contains real-world vulnerability data with function pairs (vulnerable vs. non-vulnerable)
and associated call graph information.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Unpack

from pydantic import PrivateAttr

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, CWEType, SampleCollection
from datasets.loaders.base import DatasetLoadParams, IDatasetLoader


class JitVulDatasetLoader(IDatasetLoader):
    """Framework-compatible dataset loader for JitVul."""

    source_path: Path
    __logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__)
    )

    @staticmethod
    def _normalize_cwe(raw_cwe: Any) -> str:
        """Normalize raw CWE values from JitVul records to a single CWE-* string."""
        if raw_cwe is None:
            return ""

        if isinstance(raw_cwe, (list, tuple, set)):
            for item in raw_cwe:
                normalized_item = JitVulDatasetLoader._normalize_cwe(item)
                if normalized_item:
                    return normalized_item
            return ""

        raw_cwe_string: str = str(raw_cwe).strip()
        if not raw_cwe_string:
            return ""

        cwe_match = re.search(r"CWE-\d+", raw_cwe_string.upper())
        return cwe_match.group(0) if cwe_match else ""

    def get_dataset_stats(self, data_file: Path | str | None = None) -> dict[str, Any]:
        """Compute basic dataset statistics for JitVul source data."""
        source_path = Path(data_file) if data_file is not None else self.source_path
        cwe_distribution: dict[str, int] = {}
        total_items = 0

        with open(source_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue

                try:
                    item: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_items += 1
                cwe: str = self._normalize_cwe(item.get("cwe"))
                if cwe:
                    cwe_distribution[cwe] = cwe_distribution.get(cwe, 0) + 1

        return {
            "total_items": total_items,
            "cwe_distribution": cwe_distribution,
        }

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
        max_samples = kwargs.get("max_samples", kwargs.get("limit", None))
        task_type = kwargs.get("task_type", TaskType.BINARY_VULNERABILITY)
        if isinstance(task_type, str) and not isinstance(task_type, TaskType):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                pass

        if isinstance(task_type, str) and not isinstance(task_type, TaskType):
            task_type_mapping: dict[str, TaskType] = {
                "binary": TaskType.BINARY_VULNERABILITY,
                "multiclass": TaskType.MULTICLASS_VULNERABILITY,
                "cwe_specific": TaskType.BINARY_CWE_SPECIFIC,
            }
            task_type = task_type_mapping.get(task_type, TaskType.BINARY_VULNERABILITY)
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
        cwe = CWEType(self._normalize_cwe(item.get("cwe", "")))
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

    def _augment_code_with_context(
        self,
        code: str,
        item: dict[str, Any],
        is_use_call_graph: bool,
    ) -> str:
        """
        Augment code with call graph context if available.

        Args:
            code: Source code snippet
            item: Raw JitVul item with optional call graph data
            is_use_call_graph: Whether to prepend call graph context

        Returns:
            str: Code with optional call graph context
        """
        if is_use_call_graph and "call_graph" in item:
            call_graph = item.get("call_graph")
            if call_graph:
                code = f"// Call graph context:\n// {call_graph}\n\n{code}"

        return code

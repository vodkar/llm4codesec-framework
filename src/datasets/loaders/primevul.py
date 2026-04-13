#!/usr/bin/env python3
"""
PrimeVul Dataset Loader

Loads and processes the PrimeVul benchmark dataset (Ding et al., 2024 —
"Vulnerability Detection with Code Language Models: How Far Are We?").

PrimeVul provides function-level C/C++ vulnerability data.  Each vulnerability
instance is identified by a shared ``idx`` value.  The same ``idx`` appears
twice: once with ``target=1`` (vulnerable version) and once with ``target=0``
(the patched/non-vulnerable version).  A single CVE can span multiple functions,
so multiple rows may share the same ``idx``.

The loader therefore uses a **two-pass grouped strategy**:

1. First pass  – collect all JSONL rows into groups keyed by ``idx``.
2. Second pass – for each group create at most two
   :class:`~benchmark.models.BenchmarkSample` objects:
   one from the vulnerable function(s) and one from the non-vulnerable
   function(s).  When a group contains several functions with the same
   target, their code is concatenated with a clear separator so the model
   sees the full vulnerability context.

Expected JSONL fields:
    idx             – shared integer / string key that identifies a
                      vulnerable/non-vulnerable pair (same value for both rows)
    project         – source project name (e.g. "ffmpeg", "linux")
    commit_id       – git commit hash where the vulnerability was fixed
    commit_message  – commit message  (optional)
    target          – 0 (non-vulnerable / patched) or 1 (vulnerable)
    func            – function source code
    func_hash       – function-level hash  (optional)
    cwe             – list of CWE strings e.g. ["CWE-125"] (empty for target=0)
    cve             – CVE identifier  (optional)
    cve_desc        – CVE description  (optional)
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Unpack

from pydantic import PrivateAttr

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, Dataset, DatasetMetadata, SampleCollection
from datasets.loaders.base import DatasetLoadParams, IDatasetLoader

_FUNC_SEPARATOR = "\n\n/* ---- next function ---- */\n\n"


class PrimeVulDatasetLoader(IDatasetLoader):
    """Loads and processes the PrimeVul JSONL benchmark dataset."""

    source_path: Path
    __logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__)
    )

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_cwe(raw: Any) -> str:
        """Return a canonical CWE-NNN string or empty string."""
        if raw is None:
            return ""
        if isinstance(raw, (list, tuple)):
            for item in raw:
                result = PrimeVulDatasetLoader._normalize_cwe(item)
                if result:
                    return result
            return ""
        text = str(raw).strip().upper()
        if not text:
            return ""
        match = re.search(r"CWE-\d+", text)
        return match.group(0) if match else ""

    @staticmethod
    def _normalize_cwe_list(raw: Any) -> list[str]:
        """Return all canonical CWE-NNN strings found in *raw* (deduplicated)."""
        if raw is None:
            return []
        if not isinstance(raw, list):
            raw = [raw]
        seen: list[str] = []
        for item in raw:
            cwe = PrimeVulDatasetLoader._normalize_cwe(item)
            if cwe and cwe not in seen:
                seen.append(cwe)
        return seen

    # ------------------------------------------------------------------ #
    # first pass: read + group by idx                                      #
    # ------------------------------------------------------------------ #

    def _read_groups(
        self,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """
        Read the entire JSONL source and group rows by ``idx``.

        Returns a mapping::

            {
                "<idx>": {
                    "vuln": [row, ...],   # target == 1
                    "safe": [row, ...],   # target == 0
                }
            }

        Insertion order is preserved so that samples maintain file order.
        """
        groups: dict[str, dict[str, list[dict[str, Any]]]] = {}

        with open(self.source_path, "r", encoding="utf-8") as fh:
            for line_num, raw_line in enumerate(fh, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                try:
                    item: dict[str, Any] = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    self.__logger.warning(
                        "Skipping invalid JSON at line %d: %s", line_num, exc
                    )
                    continue

                idx_key: str = str(item.get("idx", f"line{line_num}"))
                target: int = int(item.get("target", 0))

                if idx_key not in groups:
                    groups[idx_key] = {"vuln": [], "safe": []}

                bucket = "vuln" if target == 1 else "safe"
                groups[idx_key][bucket].append(item)

        return groups

    # ------------------------------------------------------------------ #
    # second pass: group → samples                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _aggregate_code(rows: list[dict[str, Any]]) -> str:
        """Join the ``func`` fields from *rows* (multiple = same vulnerability)."""
        parts = [r.get("func", "").strip() for r in rows if r.get("func", "").strip()]
        return _FUNC_SEPARATOR.join(parts)

    @staticmethod
    def _aggregate_metadata(
        rows: list[dict[str, Any]],
        idx_key: str,
        cwe_number: int = 0,
    ) -> dict[str, Any]:
        """Build metadata from the first row in *rows* (representative).

        ``cwe_number`` is the integer CWE identifier consumed by
        :class:`~benchmark.models.Dataset` to count vulnerable vs safe samples.
        Pass the numeric part of the primary CWE for vulnerable samples and
        leave it at 0 for safe samples.
        """
        first = rows[0] if rows else {}
        return {
            "idx": idx_key,
            "project": first.get("project", "unknown"),
            "commit_id": first.get("commit_id", ""),
            "commit_message": first.get("commit_message", ""),
            "func_hash": first.get("func_hash", ""),
            "cve": first.get("cve", ""),
            "source": "primevul",
            "function_count": len(rows),
            "cwe_number": cwe_number,
        }

    @staticmethod
    def _aggregate_cwes(rows: list[dict[str, Any]]) -> list[str]:
        """Union of all CWE types across *rows*."""
        seen: list[str] = []
        for row in rows:
            for cwe in PrimeVulDatasetLoader._normalize_cwe_list(row.get("cwe", [])):
                if cwe not in seen:
                    seen.append(cwe)
        return seen

    def _group_to_samples(
        self,
        idx_key: str,
        group: dict[str, list[dict[str, Any]]],
        task_type: TaskType,
        target_cwe: str,
    ) -> list[BenchmarkSample]:
        """Convert one *group* (idx-keyed) into BenchmarkSample objects."""
        samples: list[BenchmarkSample] = []

        vuln_rows = group["vuln"]
        safe_rows = group["safe"]

        vuln_code = self._aggregate_code(vuln_rows)
        safe_code = self._aggregate_code(safe_rows)

        vuln_cwes = self._aggregate_cwes(vuln_rows)
        primary_cwe = vuln_cwes[0] if vuln_cwes else ""

        # Extract the integer part of the primary CWE (e.g. "CWE-125" → 125).
        # Dataset.model_post_init uses metadata["cwe_number"] > 0 to count
        # vulnerable vs safe samples, so this must be set correctly.
        cwe_match = re.search(r"\d+", primary_cwe)
        vuln_cwe_number: int = int(cwe_match.group()) if cwe_match else 0

        vuln_meta = self._aggregate_metadata(vuln_rows, idx_key, cwe_number=vuln_cwe_number)
        safe_meta = self._aggregate_metadata(safe_rows, idx_key, cwe_number=0)

        # ---------- BINARY_VULNERABILITY -------------------------------- #
        if task_type == TaskType.BINARY_VULNERABILITY:
            if vuln_code:
                samples.append(
                    BenchmarkSample(
                        id=f"primevul_{idx_key}_vuln",
                        code=vuln_code,
                        label=1,
                        metadata={**vuln_meta, "is_vulnerable": True},
                        cwe_types=vuln_cwes,
                        severity=None,
                    )
                )
            if safe_code:
                samples.append(
                    BenchmarkSample(
                        id=f"primevul_{idx_key}_safe",
                        code=safe_code,
                        label=0,
                        metadata={**safe_meta, "is_vulnerable": False},
                        cwe_types=[],
                        severity=None,
                    )
                )

        # ---------- MULTICLASS_VULNERABILITY ---------------------------- #
        elif task_type == TaskType.MULTICLASS_VULNERABILITY:
            if vuln_code:
                mc_label: str = primary_cwe if primary_cwe else "UNKNOWN"
                samples.append(
                    BenchmarkSample(
                        id=f"primevul_{idx_key}_vuln",
                        code=vuln_code,
                        label=mc_label,
                        metadata={**vuln_meta, "is_vulnerable": True},
                        cwe_types=vuln_cwes,
                        severity=None,
                    )
                )
            if safe_code:
                samples.append(
                    BenchmarkSample(
                        id=f"primevul_{idx_key}_safe",
                        code=safe_code,
                        label="NONE",
                        metadata={**safe_meta, "is_vulnerable": False},
                        cwe_types=[],
                        severity=None,
                    )
                )

        # ---------- BINARY_CWE_SPECIFIC --------------------------------- #
        elif task_type == TaskType.BINARY_CWE_SPECIFIC:
            # Vulnerable sample only when it matches the target CWE
            if vuln_code and target_cwe in vuln_cwes:
                samples.append(
                    BenchmarkSample(
                        id=f"primevul_{idx_key}_vuln",
                        code=vuln_code,
                        label=1,
                        metadata={
                            **vuln_meta,
                            "is_vulnerable": True,
                            "target_cwe": target_cwe,
                        },
                        cwe_types=vuln_cwes,
                        severity=None,
                    )
                )
            # Non-vulnerable sample always acts as a negative example
            if safe_code:
                samples.append(
                    BenchmarkSample(
                        id=f"primevul_{idx_key}_safe",
                        code=safe_code,
                        label=0,
                        metadata={
                            **safe_meta,
                            "is_vulnerable": False,
                            "target_cwe": target_cwe,
                        },
                        cwe_types=[],
                        severity=None,
                    )
                )

        else:
            self.__logger.warning(
                "Unsupported task_type=%s, skipping group %s", task_type, idx_key
            )

        return samples

    # ------------------------------------------------------------------ #
    # public API                                                           #
    # ------------------------------------------------------------------ #

    def load_dataset(self, **kwargs: Unpack[DatasetLoadParams]) -> SampleCollection:
        """
        Load PrimeVul JSONL and return a :class:`SampleCollection`.

        Rows are first grouped by ``idx``, then each group is converted to at
        most two samples (one vulnerable, one non-vulnerable).

        Parameters (via *kwargs*):
            task_type   – TaskType; default ``BINARY_VULNERABILITY``
            target_cwe  – required for ``BINARY_CWE_SPECIFIC``
            limit       – maximum number of vulnerability *groups* to include
                          (each group produces up to 2 samples)
        """
        task_type: TaskType = kwargs.get("task_type", TaskType.BINARY_VULNERABILITY)
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        target_cwe: str = (kwargs.get("target_cwe") or "").strip().upper()
        limit: int | None = kwargs.get("limit")

        if task_type == TaskType.BINARY_CWE_SPECIFIC and not target_cwe:
            raise ValueError(
                "target_cwe must be specified for BINARY_CWE_SPECIFIC tasks"
            )

        groups = self._read_groups()
        samples: list[BenchmarkSample] = []

        for count, (idx_key, group) in enumerate(groups.items()):
            if limit is not None and count >= limit:
                break
            new = self._group_to_samples(idx_key, group, task_type, target_cwe)
            samples.extend(new)

        self.__logger.info(
            "Loaded %d samples from %d groups (PrimeVul at %s)",
            len(samples),
            min(len(groups), limit) if limit else len(groups),
            self.source_path,
        )
        return SampleCollection(samples)

    def create_dataset_json(
        self,
        output_path: str,
        **kwargs: Unpack[DatasetLoadParams],
    ) -> None:
        """
        Write a processed benchmark JSON file compatible with the framework.

        Args:
            output_path: Destination path for the JSON file.
            **kwargs:    Forwarded to :meth:`load_dataset`.
        """
        task_type: TaskType = kwargs.get("task_type", TaskType.BINARY_VULNERABILITY)
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        target_cwe: str | None = kwargs.get("target_cwe")

        samples = self.load_dataset(**kwargs)

        dataset = Dataset(
            metadata=DatasetMetadata(
                name="PrimeVul-Benchmark",
                version="1.0",
                task_type=task_type,
                programming_language="C",
                cwe_type=target_cwe,
            ),
            samples=samples,
        )

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "w", encoding="utf-8") as fh:
            json.dump(dataset.model_dump(), fh, indent=2, ensure_ascii=False)

        self.__logger.info(
            "Created dataset JSON with %d samples at %s", len(samples), output_path
        )

    def get_dataset_stats(self) -> dict[str, Any]:
        """Compute statistics at the vulnerability-group level."""
        groups = self._read_groups()

        total_groups = len(groups)
        groups_with_vuln = sum(1 for g in groups.values() if g["vuln"])
        groups_with_safe = sum(1 for g in groups.values() if g["safe"])
        cwe_dist: dict[str, int] = defaultdict(int)

        for group in groups.values():
            for cwe in self._aggregate_cwes(group["vuln"]):
                cwe_dist[cwe] += 1

        return {
            "total_groups": total_groups,
            "groups_with_vulnerable": groups_with_vuln,
            "groups_with_safe": groups_with_safe,
            "cwe_distribution": dict(cwe_dist),
        }

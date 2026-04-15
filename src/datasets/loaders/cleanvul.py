#!/usr/bin/env python3
"""
CleanVul Dataset Loader

Loads and processes the CleanVul benchmark dataset (Li et al., 2022 —
"VulSifter: Towards High-Quality Vulnerability Datasets by Detecting
Vulnerability-Fixing Changes in Code Commits").

CleanVul provides function-level vulnerability data across six programming
languages (C, C++, Java, Python, JavaScript, C#).  Each row in the source CSV
represents a **before/after function pair** extracted from a vulnerability-fixing
commit (VFC):

    func_before – vulnerable version of the function  (label = 1)
    func_after  – patched/non-vulnerable version      (label = 0)

A single commit can touch multiple files and functions, so multiple rows often
share the same ``commit_url``.  The loader therefore uses a **two-pass grouped
strategy**:

1. First pass  – read the CSV and group rows by ``commit_url``.
2. Second pass – for each commit group produce at most two
   :class:`~benchmark.models.BenchmarkSample` objects: one from the
   vulnerable function(s) (``func_before``) and one from the patched
   function(s) (``func_after``).  Multiple functions are concatenated with
   a separator so the model sees the full vulnerability context.

Because every row contains both a ``func_before`` *and* a ``func_after``,
the resulting dataset is inherently balanced (equal numbers of vulnerable
and non-vulnerable samples).

CSV columns:
    func_before         – vulnerable function code
    func_after          – patched function code
    commit_msg          – commit message
    commit_url          – URL identifying the commit (grouping key)
    cve_id              – CVE identifier (often empty)
    cwe_id              – CWE identifier (often empty)
    file_name           – changed source file
    vulnerability_score – VulSifter confidence score (integer 0–4)
    extension           – file extension (c, cpp, java, py, js, cs)
    is_test             – bool; rows where this is truthy are skipped
    date                – commit date (often empty)
"""

import csv
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

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    "c":    "C",
    "cpp":  "C++",
    "java": "Java",
    "py":   "Python",
    "js":   "JavaScript",
    "cs":   "C#",
}


class CleanVulDatasetLoader(IDatasetLoader):
    """Loads and processes the CleanVul CSV benchmark dataset."""

    source_path: Path
    programming_language: str | None = None
    """If set, only rows whose ``extension`` maps to this language are loaded.
    Values should match :data:`EXTENSION_TO_LANGUAGE` values, e.g. ``"C"``
    or ``"Python"``.  ``None`` means all languages."""

    __logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__)
    )

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_cwe(raw: Any) -> str:
        """Return a canonical ``CWE-NNN`` string, or empty string."""
        if raw is None:
            return ""
        text = str(raw).strip().upper()
        if not text:
            return ""
        match = re.search(r"CWE-\d+", text)
        return match.group(0) if match else ""

    @staticmethod
    def _is_test_row(value: Any) -> bool:
        """Return True when the ``is_test`` column value is truthy."""
        return str(value).strip().lower() in {"true", "1", "yes"}

    # ------------------------------------------------------------------ #
    # first pass: read CSV and group by commit_url                         #
    # ------------------------------------------------------------------ #

    def _read_groups(
        self,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """
        Read the entire CSV source and group rows by ``commit_url``.

        Returns a mapping::

            {
                "<commit_url>": {
                    "vuln": [row, ...],   # func_before entries
                    "safe": [row, ...],   # func_after entries
                }
            }

        Rows are skipped when:
        - ``is_test`` is truthy
        - ``commit_url`` is empty
        - ``extension`` is not in :data:`EXTENSION_TO_LANGUAGE`
        - language filter is set and row's language does not match
        - both ``func_before`` and ``func_after`` are empty

        Both ``func_before`` and ``func_after`` from the same row are stored
        in the *same* group under their respective buckets.
        """
        groups: dict[str, dict[str, list[dict[str, Any]]]] = {}

        with open(self.source_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row_num, row in enumerate(reader, start=2):  # row 1 = header
                # --- skip test files ---
                if self._is_test_row(row.get("is_test", "")):
                    continue

                # --- skip unknown extensions ---
                ext = row.get("extension", "").strip().lower()
                if ext not in EXTENSION_TO_LANGUAGE:
                    continue

                # --- optional language filter ---
                lang = EXTENSION_TO_LANGUAGE[ext]
                if self.programming_language is not None and lang != self.programming_language:
                    continue

                # --- skip empty commit_url ---
                commit_url = row.get("commit_url", "").strip()
                if not commit_url:
                    self.__logger.debug("Skipping row %d: empty commit_url", row_num)
                    continue

                func_before = row.get("func_before", "").strip()
                func_after = row.get("func_after", "").strip()

                if not func_before and not func_after:
                    continue

                if commit_url not in groups:
                    groups[commit_url] = {"vuln": [], "safe": []}

                # Each row contributes to both vuln and safe buckets
                if func_before:
                    groups[commit_url]["vuln"].append(dict(row))
                if func_after:
                    groups[commit_url]["safe"].append(dict(row))

        return groups

    # ------------------------------------------------------------------ #
    # second pass: group → samples                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _aggregate_code(rows: list[dict[str, Any]], field: str) -> str:
        """Join *field* values from *rows* with a clear separator."""
        parts = [r.get(field, "").strip() for r in rows if r.get(field, "").strip()]
        return _FUNC_SEPARATOR.join(parts)

    @staticmethod
    def _aggregate_metadata(
        rows: list[dict[str, Any]],
        commit_url: str,
        cwe_number: int = 0,
    ) -> dict[str, Any]:
        """Build sample metadata from the first representative row.

        ``cwe_number`` must be > 0 for vulnerable samples so that
        :class:`~benchmark.models.Dataset` correctly counts them in
        ``vulnerable_samples``.  Use 1 as a placeholder when no real CWE
        is available.
        """
        first = rows[0] if rows else {}
        ext = first.get("extension", "").strip().lower()
        return {
            "commit_url": commit_url,
            "cve_id": first.get("cve_id", ""),
            "cwe_number": cwe_number,
            "vulnerability_score": first.get("vulnerability_score", ""),
            "extension": ext,
            "programming_language": EXTENSION_TO_LANGUAGE.get(ext, "Unknown"),
            "function_count": len(rows),
            "source": "cleanvul",
        }

    def _group_to_samples(
        self,
        commit_url: str,
        group: dict[str, list[dict[str, Any]]],
        task_type: TaskType,
        target_cwe: str,
    ) -> list[BenchmarkSample]:
        """Convert one commit *group* into :class:`BenchmarkSample` objects."""
        samples: list[BenchmarkSample] = []

        vuln_rows = group["vuln"]
        safe_rows = group["safe"]

        vuln_code = self._aggregate_code(vuln_rows, "func_before")
        safe_code = self._aggregate_code(safe_rows, "func_after")

        # Resolve CWE from the first row that has a non-empty cwe_id.
        raw_cwe = next(
            (r.get("cwe_id", "") for r in vuln_rows if r.get("cwe_id", "").strip()),
            "",
        )
        cwe_str = self._normalize_cwe(raw_cwe)

        # cwe_number must be > 0 for Dataset.model_post_init to count
        # this sample as vulnerable.  Fall back to 1 when no CWE is known.
        if cwe_str:
            m = re.search(r"\d+", cwe_str)
            cwe_number: int = int(m.group()) if m else 1
        else:
            cwe_number = 1  # placeholder — confirms sample is vulnerable

        vuln_cwe_types = [cwe_str] if cwe_str else []

        # Derive a short, stable sample ID from the commit URL.
        # We take the last path segment (the commit SHA) if present.
        sha_match = re.search(r"/commit/([0-9a-f]{7,})", commit_url)
        group_id = sha_match.group(1) if sha_match else re.sub(r"[^\w]", "_", commit_url)[-40:]

        vuln_meta = self._aggregate_metadata(vuln_rows, commit_url, cwe_number=cwe_number)
        safe_meta = self._aggregate_metadata(safe_rows, commit_url, cwe_number=0)

        # ---------- BINARY_VULNERABILITY -------------------------------- #
        if task_type == TaskType.BINARY_VULNERABILITY:
            if vuln_code:
                samples.append(
                    BenchmarkSample(
                        id=f"cleanvul_{group_id}_vuln",
                        code=vuln_code,
                        label=1,
                        metadata={**vuln_meta, "is_vulnerable": True},
                        cwe_types=vuln_cwe_types,
                        severity=None,
                    )
                )
            if safe_code:
                samples.append(
                    BenchmarkSample(
                        id=f"cleanvul_{group_id}_safe",
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
                mc_label: str = cwe_str if cwe_str else "UNKNOWN"
                samples.append(
                    BenchmarkSample(
                        id=f"cleanvul_{group_id}_vuln",
                        code=vuln_code,
                        label=mc_label,
                        metadata={**vuln_meta, "is_vulnerable": True},
                        cwe_types=vuln_cwe_types,
                        severity=None,
                    )
                )
            if safe_code:
                samples.append(
                    BenchmarkSample(
                        id=f"cleanvul_{group_id}_safe",
                        code=safe_code,
                        label="NONE",
                        metadata={**safe_meta, "is_vulnerable": False},
                        cwe_types=[],
                        severity=None,
                    )
                )

        # ---------- BINARY_CWE_SPECIFIC --------------------------------- #
        elif task_type == TaskType.BINARY_CWE_SPECIFIC:
            if vuln_code and target_cwe in vuln_cwe_types:
                samples.append(
                    BenchmarkSample(
                        id=f"cleanvul_{group_id}_vuln",
                        code=vuln_code,
                        label=1,
                        metadata={
                            **vuln_meta,
                            "is_vulnerable": True,
                            "target_cwe": target_cwe,
                        },
                        cwe_types=vuln_cwe_types,
                        severity=None,
                    )
                )
            if safe_code:
                samples.append(
                    BenchmarkSample(
                        id=f"cleanvul_{group_id}_safe",
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
                "Unsupported task_type=%s, skipping group %s", task_type, group_id
            )

        return samples

    # ------------------------------------------------------------------ #
    # public API                                                           #
    # ------------------------------------------------------------------ #

    def load_dataset(self, **kwargs: Unpack[DatasetLoadParams]) -> SampleCollection:
        """
        Load CleanVul CSV and return a :class:`SampleCollection`.

        Rows are first grouped by ``commit_url``, then each group is converted
        to at most two samples (one vulnerable, one non-vulnerable).

        Parameters (via *kwargs*):
            task_type   – :class:`~benchmark.enums.TaskType`;
                          default ``BINARY_VULNERABILITY``
            target_cwe  – required for ``BINARY_CWE_SPECIFIC``
            limit       – maximum number of commit *groups* to include;
                          each group produces up to 2 samples
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

        for count, (commit_url, group) in enumerate(groups.items()):
            if limit is not None and count >= limit:
                break
            samples.extend(self._group_to_samples(commit_url, group, task_type, target_cwe))

        n_groups = min(len(groups), limit) if limit else len(groups)
        lang_label = self.programming_language or "all languages"
        self.__logger.info(
            "Loaded %d samples from %d groups (CleanVul %s at %s)",
            len(samples),
            n_groups,
            lang_label,
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
        lang = self.programming_language or "Multi"

        samples = self.load_dataset(**kwargs)

        dataset = Dataset(
            metadata=DatasetMetadata(
                name="CleanVul-Benchmark",
                version="1.0",
                task_type=task_type,
                programming_language=lang,
                cwe_type=target_cwe,
            ),
            samples=samples,
        )

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_path_obj, "w", encoding="utf-8") as fh:
            json.dump(dataset.model_dump(), fh, indent=2, ensure_ascii=False)

        self.__logger.info(
            "Created dataset JSON with %d samples at %s", len(samples), output_path
        )

    def get_dataset_stats(self) -> dict[str, Any]:
        """Compute statistics at the commit-group level."""
        groups = self._read_groups()

        total_groups = len(groups)
        groups_with_vuln = sum(1 for g in groups.values() if g["vuln"])
        groups_with_safe = sum(1 for g in groups.values() if g["safe"])

        cwe_dist: dict[str, int] = defaultdict(int)
        ext_dist: dict[str, int] = defaultdict(int)

        for group in groups.values():
            for row in group["vuln"]:
                cwe = self._normalize_cwe(row.get("cwe_id", ""))
                if cwe:
                    cwe_dist[cwe] += 1
                ext = row.get("extension", "").strip().lower()
                if ext:
                    ext_dist[ext] += 1

        return {
            "total_groups": total_groups,
            "groups_with_vulnerable": groups_with_vuln,
            "groups_with_safe": groups_with_safe,
            "cwe_distribution": dict(cwe_dist),
            "extension_distribution": dict(ext_dist),
        }

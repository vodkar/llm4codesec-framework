#!/usr/bin/env python3
"""
ContextAssembler Dataset Preparation Script

Creates two processed datasets for the ContextAssembler comparison experiment:

1. ``context_assembler_binary.json``
   Normalised version of the ContextAssembler benchmark
   (benchmarks/ContextAssembler/cvefixes_context_benchmark.json).
   Contains Python code samples with binary labels (1 = vulnerable, 0 = safe).

2. ``cvefixes_python_matched.json``  (requires --database-path)
   Python file-level samples from the CVEFixes SQLite database filtered to
   the same CVE IDs that appear in the ContextAssembler benchmark.
   For each code change both the **pre-fix** (vulnerable, label=1) and the
   **post-fix** (safe, label=0) versions are included, allowing a direct
   apples-to-apples label-balanced comparison with dataset 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, Dataset, DatasetMetadata, SampleCollection
from datasets.loaders.context_assembler import ContextAssemblerDatasetLoader
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)

_DEFAULT_SOURCE: str = "benchmarks/ContextAssembler/cvefixes_context_benchmark.json"
_DEFAULT_OUTPUT_DIR: str = "datasets_processed/context_assembler"


# ---------------------------------------------------------------------------
# ContextAssembler → processed JSON
# ---------------------------------------------------------------------------


def create_context_assembler_dataset(
    source_path: Path,
    output_dir: Path,
    sample_limit: int | None = None,
) -> None:
    """
    Write the normalised ContextAssembler binary dataset to disk.

    Args:
        source_path: Path to the raw ContextAssembler JSON benchmark file.
        output_dir: Directory to write the processed output into.
        sample_limit: Optional cap on the number of output samples (for testing).
    """
    output_path = output_dir / "context_assembler_binary.json"
    _LOGGER.info("Creating ContextAssembler binary dataset → %s", output_path)

    loader = ContextAssemblerDatasetLoader(source_path=source_path)
    loader.create_dataset_json(
        output_path=str(output_path),
        task_type=TaskType.BINARY_VULNERABILITY,
        limit=sample_limit,
    )

    _LOGGER.info("ContextAssembler binary dataset written to %s", output_path)


# ---------------------------------------------------------------------------
# CVEFixes matched dataset
# ---------------------------------------------------------------------------


def _extract_matching_cve_ids(source_path: Path) -> dict[str, set[int]]:
    """
    Read the ContextAssembler JSON and return a mapping from each CVE ID to the
    set of labels (0 / 1) that appear for it in the source benchmark.

    Storing labels per CVE lets the matched CVEFixes dataset produce exactly the
    same (CVE ID, label) pairs as ContextAssembler — no more, no fewer — so
    the two datasets stay perfectly size-matched.

    Args:
        source_path: Path to the ContextAssembler benchmark JSON.

    Returns:
        Mapping ``{cve_id: {label, ...}}``  (e.g.
        ``{"CVE-2022-24798": {1}, "CVE-2022-21734": {0, 1}, ...}``).
    """
    with open(source_path, "r", encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)

    cve_label_map: dict[str, set[int]] = defaultdict(set)
    for sample in payload.get("samples", []):
        cve_id: str = sample.get("metadata", {}).get("CVEFixes-Number", "").strip()
        label_raw = sample.get("label")
        if cve_id and label_raw is not None:
            try:
                cve_label_map[cve_id].add(int(label_raw))
            except (ValueError, TypeError):
                pass

    _LOGGER.info(
        "Extracted %d unique CVE IDs (%d label-pairs) from ContextAssembler source",
        len(cve_label_map),
        sum(len(v) for v in cve_label_map.values()),
    )
    return dict(cve_label_map)


def _build_matched_samples(
    database_path: Path,
    cve_label_map: dict[str, set[int]],
) -> list[BenchmarkSample]:
    """
    Query the CVEFixes SQLite database for Python file-level changes whose CVE
    ID appears in *cve_label_map*.

    All file changes belonging to the same CVE are joined into a single code
    block with ``# filename`` comment separators.  For each CVE **only the
    labels present in ContextAssembler** are emitted:

    * ``label=1`` – all ``code_before`` snippets joined (vulnerable version)
    * ``label=0`` – all ``code_after``  snippets joined (safe/patched version)

    Most CVEs produce one sample (whichever label ContextAssembler holds); the
    16 CVEs that appear with both labels produce two.  This keeps the matched
    dataset exactly size-matched to ContextAssembler.

    Args:
        database_path: Path to the CVEFixes SQLite database file.
        cve_label_map: Mapping ``{cve_id: set_of_labels}`` from
            :func:`_extract_matching_cve_ids`.

    Returns:
        List of ``BenchmarkSample`` objects.
    """
    if not cve_label_map:
        _LOGGER.warning("No CVE IDs provided; matched dataset will be empty.")
        return []

    cve_ids: set[str] = set(cve_label_map.keys())
    query: str = """
    SELECT
        cv.cve_id,
        cv.description,
        cv.published_date,
        cv.severity,
        f.filename,
        f.code_before,
        f.code_after,
        c.hash AS commit_hash,
        fx.repo_url,
        cc.cwe_id
    FROM cve cv
    JOIN fixes fx     ON cv.cve_id   = fx.cve_id
    JOIN commits c    ON fx.hash     = c.hash
    JOIN file_change f ON c.hash     = f.hash
    LEFT JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
    WHERE f.programming_language = 'Python'
    AND   f.code_before IS NOT NULL
    AND   f.code_after  IS NOT NULL
    AND   LENGTH(f.code_before) > 50
    AND   LENGTH(f.code_after)  > 50
    """

    placeholders: str = ",".join("?" * len(cve_ids))
    query += f" AND cv.cve_id IN ({placeholders})"

    try:
        conn: sqlite3.Connection = sqlite3.connect(str(database_path), timeout=10)
    except sqlite3.Error:
        _LOGGER.exception("Failed to connect to CVEFixes database: %s", database_path)
        raise

    try:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(query, sorted(cve_ids))
        rows: list[tuple[Any, ...]] = cursor.fetchall()
    finally:
        conn.close()

    _LOGGER.info(
        "Retrieved %d CVEFixes Python file-change rows for %d CVE IDs",
        len(rows),
        len(cve_ids),
    )

    # Group all rows by cve_id so each vulnerability becomes one BenchmarkSample.
    # Preserve dict-insertion order to keep files in stable order per CVE.
    grouped: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)

    samples: list[BenchmarkSample] = []

    for cve_id, cve_rows in grouped.items():
        # Labels that ContextAssembler holds for this CVE — emit only those.
        desired_labels: set[int] = cve_label_map.get(cve_id, set())
        if not desired_labels:
            _LOGGER.debug("CVE %s found in DB but not in label map — skipping", cve_id)
            continue

        # Pull CVE-level metadata from the first row (same for all rows of this CVE)
        _, description, published_date, severity, *_ = cve_rows[0]

        # Resolve cwe from first row that has a usable value
        cwe_str: str | None = None
        cwe_number_int: int = 0
        for row in cve_rows:
            raw_cwe_val = row[9]  # cc.cwe_id
            if raw_cwe_val and not str(raw_cwe_val).startswith("NVD-CWE-"):
                raw_cwe: str = str(raw_cwe_val).strip()
                cwe_str = f"CWE-{raw_cwe}" if raw_cwe.isdigit() else raw_cwe.upper()
                if raw_cwe.isdigit():
                    cwe_number_int = int(raw_cwe)
                break

        # Join all file snippets with "# filename" separators
        before_parts: list[str] = []
        after_parts: list[str] = []
        for row in cve_rows:
            _, _, _, _, filename, code_before, code_after, *_ = row
            header: str = f"# {filename}"
            before_parts.append(f"{header}\n{code_before}")
            after_parts.append(f"{header}\n{code_after}")

        code_vulnerable: str = "\n\n".join(before_parts)
        code_safe: str = "\n\n".join(after_parts)

        base_meta: dict[str, Any] = {
            "cve_id": cve_id,
            "description": description,
            "published_date": published_date,
            "severity": severity,
            "file_count": len(cve_rows),
            "cwe_id": cve_rows[0][9],
            "cwe_number": cwe_number_int,
            "source": "CVEFixes-matched",
        }

        if 1 in desired_labels:
            # Vulnerable sample (all before-fix files, grouped by CVE)
            samples.append(
                BenchmarkSample(
                    id=f"cvefixes-matched-{cve_id}-vuln",
                    code=code_vulnerable,
                    label=1,
                    metadata={**base_meta, "version": "before_fix"},
                    cwe_types=[cwe_str] if cwe_str else [],
                    severity=severity if isinstance(severity, str) else None,
                )
            )

        if 0 in desired_labels:
            # Safe sample (all after-fix files, grouped by CVE)
            samples.append(
                BenchmarkSample(
                    id=f"cvefixes-matched-{cve_id}-safe",
                    code=code_safe,
                    label=0,
                    metadata={**base_meta, "version": "after_fix"},
                    cwe_types=[cwe_str] if cwe_str else [],
                    severity=severity if isinstance(severity, str) else None,
                )
            )

    _LOGGER.info(
        "Built %d matched CVEFixes samples from %d CVEs (label-matched to ContextAssembler)",
        len(samples),
        len(grouped),
    )
    return samples


def create_cvefixes_matched_dataset(
    source_path: Path,
    database_path: Path,
    output_dir: Path,
) -> None:
    """
    Build and write the CVEFixes Python matched dataset.

    Args:
        source_path: ContextAssembler benchmark JSON (to extract CVE IDs).
        database_path: CVEFixes SQLite database.
        output_dir: Directory to write the processed output into.
    """
    output_path = output_dir / "cvefixes_python_matched.json"
    _LOGGER.info("Creating CVEFixes Python matched dataset → %s", output_path)

    cve_label_map: dict[str, set[int]] = _extract_matching_cve_ids(source_path)
    samples: list[BenchmarkSample] = _build_matched_samples(
        database_path, cve_label_map
    )

    dataset = Dataset(
        metadata=DatasetMetadata(
            name="CVEFixes-Python-Matched-ContextAssembler",
            version="1.0",
            task_type=TaskType.BINARY_VULNERABILITY,
            programming_language="Python",
            change_level="file",
        ),
        samples=SampleCollection(samples),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(dataset.model_dump(), fh, indent=2, ensure_ascii=False)

    _LOGGER.info(
        "CVEFixes matched dataset written to %s (%d samples)", output_path, len(samples)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point for the ContextAssembler dataset preparation script."""
    parser = argparse.ArgumentParser(
        description="Prepare ContextAssembler benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create only the ContextAssembler binary dataset (no DB required)
  python src/entrypoints/loaders/run_setup_context_assembler_datasets.py

  # Also create the matched CVEFixes dataset for comparison
  python src/entrypoints/loaders/run_setup_context_assembler_datasets.py \\
      --database-path datasets_processed/cvefixes/CVEfixes.db

  # Custom paths and sample limit (for quick testing)
  python src/entrypoints/loaders/run_setup_context_assembler_datasets.py \\
      --source benchmarks/ContextAssembler/cvefixes_context_benchmark.json \\
      --output-dir datasets_processed/context_assembler \\
      --sample-limit 20 \\
      --verbose
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        default=_DEFAULT_SOURCE,
        help=f"Path to ContextAssembler benchmark JSON (default: {_DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--database-path",
        type=str,
        default=None,
        help=(
            "Path to CVEFixes SQLite database. "
            "When provided, also creates the matched CVEFixes Python dataset."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory for processed datasets (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit samples in context_assembler_binary.json (useful for testing)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    source_path = Path(args.source)
    if not source_path.exists():
        parser.error(f"ContextAssembler source file not found: {source_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        create_context_assembler_dataset(
            source_path=source_path,
            output_dir=output_dir,
            sample_limit=args.sample_limit,
        )

        if args.database_path is not None:
            db_path = Path(args.database_path)
            if not db_path.exists():
                parser.error(f"CVEFixes database not found: {db_path}")
            create_cvefixes_matched_dataset(
                source_path=source_path,
                database_path=db_path,
                output_dir=output_dir,
            )
        else:
            _LOGGER.info(
                "Skipping CVEFixes matched dataset (no --database-path provided). "
                "Run with --database-path to also create cvefixes_python_matched.json."
            )

        _LOGGER.info("Dataset preparation completed. Output directory: %s", output_dir)
        return 0

    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user.")
        return 1
    except Exception:
        _LOGGER.exception("Dataset preparation failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
ContextAssembler Dataset Preparation Script

Creates two processed datasets for the ContextAssembler comparison experiment:

1. ``context_assembler_binary.json``
    Normalised version of the CleanVul ContextAssembler benchmark
    (benchmarks/context-assembler-dataset/cleanvul_compare_rankings/
    cleanvul_context_benchmark.json).
    Contains Python code samples with binary labels (1 = vulnerable, 0 = safe).

2. ``cleanvul_python_matched.json``
    Function-level CleanVul samples extracted from
    ``cleanvul_entries.json`` in the same benchmark directory.
    Each sample keeps the original vulnerable/safe label and metadata from the
    benchmark source, so no external database is required.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, Dataset, DatasetMetadata, SampleCollection
from datasets.loaders.context_assembler import ContextAssemblerDatasetLoader
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)

_DEFAULT_SOURCE_DIR: str = "benchmarks/context-assembler-dataset/cleanvul_compare_rankings"
_DEFAULT_OUTPUT_DIR: str = "datasets_processed/context_assembler"
_CONTEXT_BENCHMARK_FILENAME: str = "cleanvul_context_benchmark.json"
_CLEANVUL_ENTRIES_FILENAME: str = "cleanvul_entries.json"


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
# CleanVul matched dataset
# ---------------------------------------------------------------------------


def _normalize_cwe_id(raw_cwe: Any) -> str | None:
    """Normalize a CleanVul CWE identifier to ``CWE-NNN`` form."""
    if raw_cwe is None:
        return None

    value: str = str(raw_cwe).strip()
    if not value or value.lower() == "null":
        return None
    if value.upper().startswith("CWE-"):
        suffix = value[4:]
    else:
        suffix = value

    if suffix.isdigit():
        return f"CWE-{suffix}"
    return None


def _extract_cleanvul_cwe_data(entry: dict[str, Any]) -> tuple[list[str], int]:
    """Return normalized CWE strings and the first numeric CWE number."""
    raw_cwes: list[Any] = []
    cwe_ids = cast(list[Any] | None, entry.get("cwe_ids"))
    if isinstance(cwe_ids, list):
        raw_cwes.extend(cwe_ids)

    singular_cwe = entry.get("cwe_id")
    if singular_cwe is not None:
        raw_cwes.append(singular_cwe)

    cwe_types: list[str] = []
    seen: set[str] = set()
    for raw_cwe in raw_cwes:
        normalized_cwe = _normalize_cwe_id(raw_cwe)
        if normalized_cwe is None or normalized_cwe in seen:
            continue
        seen.add(normalized_cwe)
        cwe_types.append(normalized_cwe)

    cwe_number: int = 0
    if cwe_types:
        numeric_part: str = cwe_types[0].removeprefix("CWE-")
        if numeric_part.isdigit():
            cwe_number = int(numeric_part)

    return cwe_types, cwe_number


def _build_cleanvul_matched_samples(
    entries: list[dict[str, Any]],
    sample_limit: int | None = None,
) -> list[BenchmarkSample]:
    """Convert CleanVul entry records into processed benchmark samples."""
    samples: list[BenchmarkSample] = []

    for entry in entries:
        code: str = str(entry.get("func_code") or "").strip()
        if not code:
            _LOGGER.warning(
                "Skipping CleanVul entry without func_code: %s",
                entry.get("sample_id", "<unknown>"),
            )
            continue

        cwe_types, cwe_number = _extract_cleanvul_cwe_data(entry)
        is_vulnerable: bool = bool(entry.get("is_vulnerable"))
        sample_id: str = str(entry.get("sample_id") or f"cleanvul-{len(samples) + 1}")

        samples.append(
            BenchmarkSample(
                id=sample_id,
                code=code,
                label=1 if is_vulnerable else 0,
                metadata={
                    "sample_id": sample_id,
                    "cve_id": str(entry.get("cve_id") or ""),
                    "description": str(entry.get("commit_msg") or ""),
                    "cwe_number": cwe_number,
                    "commit_url": str(entry.get("commit_url") or ""),
                    "repo_url": str(entry.get("repo_url") or ""),
                    "fix_hash": str(entry.get("fix_hash") or ""),
                    "file_name": str(entry.get("file_name") or ""),
                    "vulnerability_score": entry.get("vulnerability_score"),
                    "source": "CleanVul-matched",
                },
                cwe_types=cwe_types,
                severity="unknown",
            )
        )

        if sample_limit is not None and len(samples) >= sample_limit:
            break

    _LOGGER.info("Built %d CleanVul matched samples from JSON entries", len(samples))
    return samples


def create_cleanvul_matched_dataset(
    entries_path: Path,
    output_dir: Path,
    sample_limit: int | None = None,
) -> None:
    """Build and write the CleanVul matched dataset from JSON entries."""
    output_path = output_dir / "cleanvul_python_matched.json"
    _LOGGER.info("Creating CleanVul matched dataset → %s", output_path)

    with open(entries_path, "r", encoding="utf-8") as fh:
        payload: Any = json.load(fh)

    if not isinstance(payload, list):
        raise ValueError(
            f"Expected a list of CleanVul entries in {entries_path}, got {type(payload).__name__}"
        )

    payload_list = cast(list[object], payload)
    entries: list[dict[str, Any]] = [
        cast(dict[str, Any], entry) for entry in payload_list if isinstance(entry, dict)
    ]

    samples: list[BenchmarkSample] = _build_cleanvul_matched_samples(
        entries,
        sample_limit=sample_limit,
    )

    dataset = Dataset(
        metadata=DatasetMetadata(
            name="CleanVul-Python-Matched-ContextAssembler",
            version="1.0",
            task_type=TaskType.BINARY_VULNERABILITY,
            programming_language="Python",
            change_level="function",
        ),
        samples=SampleCollection(samples),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(dataset.model_dump(), fh, indent=2, ensure_ascii=False)

    _LOGGER.info(
        "CleanVul matched dataset written to %s (%d samples)",
        output_path,
        len(samples),
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
  # Create the processed CleanVul-backed ContextAssembler datasets
  python src/entrypoints/loaders/run_setup_context_assembler_datasets.py

  # Custom source/output paths and sample limit (for quick testing)
  python src/entrypoints/loaders/run_setup_context_assembler_datasets.py \\
      --source-dir benchmarks/context-assembler-dataset/cleanvul_compare_rankings \
      --output-dir datasets_processed/context_assembler \\
      --sample-limit 20 \\
      --verbose
        """,
    )

    parser.add_argument(
        "--source-dir",
        type=str,
        default=_DEFAULT_SOURCE_DIR,
        help=(
            "Directory containing cleanvul_context_benchmark.json and "
            f"cleanvul_entries.json (default: {_DEFAULT_SOURCE_DIR})"
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

    source_dir = Path(args.source_dir)
    if not source_dir.exists() or not source_dir.is_dir():
        parser.error(f"ContextAssembler source directory not found: {source_dir}")

    source_path = source_dir / _CONTEXT_BENCHMARK_FILENAME
    if not source_path.exists():
        parser.error(f"ContextAssembler source file not found: {source_path}")

    cleanvul_entries_path = source_dir / _CLEANVUL_ENTRIES_FILENAME
    if not cleanvul_entries_path.exists():
        parser.error(f"CleanVul entries file not found: {cleanvul_entries_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        create_context_assembler_dataset(
            source_path=source_path,
            output_dir=output_dir,
            sample_limit=args.sample_limit,
        )
        create_cleanvul_matched_dataset(
            entries_path=cleanvul_entries_path,
            output_dir=output_dir,
            sample_limit=args.sample_limit,
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

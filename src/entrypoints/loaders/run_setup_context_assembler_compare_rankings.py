#!/usr/bin/env python3
"""Prepare processed datasets for the ContextAssembler compare-rankings study.

This entrypoint normalizes the associated ContextAssembler ranking variants from
``benchmarks/context-assembler-dataset/compare_rankings`` into framework-ready
processed JSON files under a dedicated output directory.

The ``cvefixes_unassociated.json`` file is intentionally excluded because it is
not part of the counted comparison set for this experiment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

SRC_DIR: Final[Path] = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, Dataset, DatasetMetadata, SampleCollection
from datasets.loaders.context_assembler import ContextAssemblerDatasetLoader
from logging_tools import setup_logging

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

DEFAULT_SOURCE_DIR: Final[str] = "benchmarks/context-assembler-dataset/compare_rankings"
DEFAULT_OUTPUT_DIR: Final[str] = "datasets_processed/context_assembler_compare_rankings"

DATASET_VARIANTS: Final[dict[str, tuple[str, str]]] = {
    "cvefixes_context_benchmark.json": (
        "context_assembler_compare_current.json",
        "ContextAssembler Compare Rankings - Current",
    ),
    "cvefixes_context_benchmark_depth_repeats_context.json": (
        "context_assembler_compare_depth_repeats_context.json",
        "ContextAssembler Compare Rankings - Depth Repeats Context",
    ),
    "cvefixes_context_benchmark_dummy.json": (
        "context_assembler_compare_dummy.json",
        "ContextAssembler Compare Rankings - Dummy",
    ),
    "cvefixes_context_benchmark_multiplicative_boost.json": (
        "context_assembler_compare_multiplicative_boost.json",
        "ContextAssembler Compare Rankings - Multiplicative Boost",
    ),
    "cvefixes_context_benchmark_random_picking.json": (
        "context_assembler_compare_random_picking.json",
        "ContextAssembler Compare Rankings - Random Picking",
    ),
    # "cvefixes_context_benchmark_security_first.json": (
    #     "context_assembler_compare_security_first.json",
    #     "ContextAssembler Compare Rankings - Security First",
    # ),
    # "cvefixes_context_benchmark_security_score_only.json": (
    #     "context_assembler_compare_security_score_only.json",
    #     "ContextAssembler Compare Rankings - Security Score Only",
    # ),
}

EXCLUDED_SOURCE_FILES: Final[set[str]] = {"cvefixes_unassociated.json"}
SMOKE_TEST_OUTPUT_NAME: Final[str] = "context_assembler_compare_smoke_test.json"
SMOKE_TEST_DATASET_NAME: Final[str] = "ContextAssembler Compare Rankings - Smoke Test"


def _write_processed_dataset(
    source_path: Path,
    output_path: Path,
    dataset_name: str,
    sample_limit: int | None,
) -> None:
    """Normalize one ranking-variant dataset into processed JSON.

    Args:
        source_path: Raw ranking-variant JSON file.
        output_path: Processed output file path.
        dataset_name: Human-readable dataset name stored in metadata.
        sample_limit: Optional cap on loaded samples for quick validation runs.
    """
    loader: ContextAssemblerDatasetLoader = ContextAssemblerDatasetLoader(
        source_path=source_path
    )
    samples: SampleCollection = loader.load_dataset(limit=sample_limit)

    dataset: Dataset = Dataset(
        metadata=DatasetMetadata(
            name=dataset_name,
            version="1.0",
            task_type=TaskType.BINARY_VULNERABILITY,
            programming_language="Python",
            change_level="file",
        ),
        samples=samples,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(dataset.model_dump(), output_file, indent=2, ensure_ascii=False)

    LOGGER.info("Wrote %d samples to %s", len(samples), output_path)


def build_compare_rankings_datasets(
    source_dir: Path,
    output_dir: Path,
    sample_limit: int | None = None,
) -> list[Path]:
    """Build processed datasets for every associated compare-rankings variant.

    Args:
        source_dir: Directory containing raw compare-rankings JSON files.
        output_dir: Output directory for processed dataset JSON files.
        sample_limit: Optional cap on samples per dataset.

    Returns:
        List of written processed dataset paths.

    Raises:
        FileNotFoundError: If the source directory or any expected source file is
            missing.
    """
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(
            f"Compare-rankings source directory not found: {source_dir}"
        )

    written_files: list[Path] = []
    discovered_files: set[str] = {
        path.name
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix == ".json"
    }

    for excluded_name in sorted(EXCLUDED_SOURCE_FILES):
        if excluded_name in discovered_files:
            LOGGER.info("Skipping excluded compare-rankings file: %s", excluded_name)

    missing_expected_files: list[str] = sorted(
        set(DATASET_VARIANTS.keys()) - discovered_files
    )
    if missing_expected_files:
        raise FileNotFoundError(
            "Missing compare-rankings source files: "
            + ", ".join(missing_expected_files)
        )

    for source_name, (output_name, dataset_name) in DATASET_VARIANTS.items():
        source_path: Path = source_dir / source_name
        output_path: Path = output_dir / output_name
        LOGGER.info("Processing %s -> %s", source_path, output_path)
        _write_processed_dataset(
            source_path=source_path,
            output_path=output_path,
            dataset_name=dataset_name,
            sample_limit=sample_limit,
        )
        written_files.append(output_path)

    current_source_path: Path = source_dir / "cvefixes_context_benchmark.json"
    smoke_output_path: Path = output_dir / SMOKE_TEST_OUTPUT_NAME
    _write_smoke_test_dataset(
        source_path=current_source_path,
        output_path=smoke_output_path,
        sample_limit=sample_limit,
    )
    written_files.append(smoke_output_path)

    return written_files


def _write_smoke_test_dataset(
    source_path: Path,
    output_path: Path,
    sample_limit: int | None,
) -> None:
    """Write a tiny smoke-test dataset using the shortest available sample.

    Args:
        source_path: Raw baseline compare-rankings dataset.
        output_path: Processed smoke-test output file.
        sample_limit: Optional cap applied during source loading.
    """
    loader: ContextAssemblerDatasetLoader = ContextAssemblerDatasetLoader(
        source_path=source_path
    )
    samples: SampleCollection = loader.load_dataset(limit=sample_limit)
    if len(samples) == 0:
        raise ValueError("Cannot create smoke-test dataset from an empty sample set")

    shortest_sample: BenchmarkSample = min(samples, key=lambda sample: len(sample.code))
    smoke_samples: SampleCollection = SampleCollection.model_validate([shortest_sample])

    dataset: Dataset = Dataset(
        metadata=DatasetMetadata(
            name=SMOKE_TEST_DATASET_NAME,
            version="1.0",
            task_type=TaskType.BINARY_VULNERABILITY,
            programming_language="Python",
            change_level="file",
        ),
        samples=smoke_samples,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(dataset.model_dump(), output_file, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Wrote smoke-test dataset with sample %s to %s",
        shortest_sample.id,
        output_path,
    )


def main() -> int:
    """Run the compare-rankings dataset preparation workflow."""
    parser = argparse.ArgumentParser(
        description="Prepare processed datasets for the ContextAssembler compare-rankings study.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory containing compare-rankings JSON files (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for processed datasets (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit the number of samples written per processed dataset.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        source_dir: Path = Path(args.source_dir)
        output_dir: Path = Path(args.output_dir)
        written_files: list[Path] = build_compare_rankings_datasets(
            source_dir=source_dir,
            output_dir=output_dir,
            sample_limit=args.sample_limit,
        )
        LOGGER.info(
            "Prepared %d compare-rankings datasets in %s",
            len(written_files),
            output_dir,
        )
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 1
    except Exception:
        LOGGER.exception("Compare-rankings dataset preparation failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

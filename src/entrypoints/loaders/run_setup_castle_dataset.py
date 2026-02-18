#!/usr/bin/env python3
"""
CASTLE Dataset Setup Script

This script processes the CASTLE benchmark source files and creates
structured JSON datasets for use with the LLM benchmark framework.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from benchmark.enums import TaskType
from benchmark.models import Dataset, DatasetMetadata
from datasets.loaders.castle_dataset_loader import (
    CastleDatasetLoader,
)
from logging_tools import setup_logging


def validate_castle_source(source_dir: Path) -> bool:
    """Validate that CASTLE source directory exists and has expected structure."""
    logger: logging.Logger = logging.getLogger(__name__)

    source_path = Path(source_dir)

    if not source_path.exists():
        logger.error(f"CASTLE source directory not found: {source_path}")
        return False

    if not source_path.is_dir():
        logger.error(f"CASTLE source path is not a directory: {source_path}")
        return False

    # Check for some expected CWE directories
    expected_cwes = ["125", "190", "476", "787"]
    found_cwes: list[str] = []

    for cwe in expected_cwes:
        cwe_dir = source_path / cwe
        if cwe_dir.exists() and cwe_dir.is_dir():
            found_cwes.append(cwe)

    if not found_cwes:
        logger.error(f"No expected CWE directories found in {source_path}")
        logger.error(f"Expected directories: {expected_cwes}")
        return False

    logger.info(f"Found CWE directories: {found_cwes}")
    return True


def create_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def create_binary_dataset(loader: CastleDatasetLoader, output_dir: Path) -> None:
    """Create binary classification dataset."""
    logger: logging.Logger = logging.getLogger(__name__)

    logger.info("Creating binary classification dataset...")

    output_file = output_dir / "castle_binary.json"
    loader.create_dataset_json(
        str(output_file), task_type=TaskType.BINARY_VULNERABILITY
    )

    logger.info(f"Binary dataset created: {output_file}")


def create_multiclass_dataset(loader: CastleDatasetLoader, output_dir: Path) -> None:
    """Create multi-class classification dataset."""
    logger: logging.Logger = logging.getLogger(__name__)

    logger.info("Creating multi-class classification dataset...")

    output_file = output_dir / "castle_multiclass.json"
    loader.create_dataset_json(
        str(output_file), task_type=TaskType.MULTICLASS_VULNERABILITY
    )

    logger.info(f"Multi-class dataset created: {output_file}")


def create_cwe_specific_datasets(
    loader: CastleDatasetLoader,
    output_dir: Path,
) -> None:
    """Create CWE-specific datasets."""
    logger: logging.Logger = logging.getLogger(__name__)

    logger.info("Creating CWE-specific datasets...")

    # Load all samples first
    all_samples = loader.load_dataset(task_type=TaskType.BINARY_CWE_SPECIFIC)
    available_cwes = all_samples.available_cwes

    logger.info(f"Available CWEs in dataset: {available_cwes}")

    for cwe in available_cwes:
        logger.info(f"Processing {cwe}...")

        # Filter samples for this CWE
        cwe_samples = all_samples.filter_by_cwe(cwe)

        if not cwe_samples:
            logger.warning(f"No samples found for {cwe}")
            continue

        # Count vulnerable vs safe samples
        vulnerable_count = sum(1 for s in cwe_samples if s.label == 1)
        safe_count = len(cwe_samples) - vulnerable_count

        logger.info(f"{cwe}: {vulnerable_count} vulnerable, {safe_count} safe samples")

        # Create dataset structure
        cwe_number = cwe.split("-")[1]
        dataset = Dataset(
            metadata=DatasetMetadata(
                name=f"CASTLE-Benchmark-{cwe}",
                version="1.2",
                task_type="binary_cwe_specific",
                cwe_type=cwe,
                total_samples=len(cwe_samples),
                vulnerable_samples=vulnerable_count,
                safe_samples=safe_count,
                programming_language="C",
                change_level="file",
            ),
            samples=cwe_samples,
        )

        # Save dataset
        output_file = output_dir / f"castle_cwe_{cwe_number}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"{cwe} dataset created: {output_file}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Setup CASTLE benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--source-dir",
        default="benchmarks/CASTLE-Source/dataset",
        help="Path to CASTLE source dataset directory",
    )

    parser.add_argument(
        "--output-dir",
        default="datasets_processed/castle",
        help="Output directory for processed datasets",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger: logging.Logger = logging.getLogger(__name__)

    logger.info("CASTLE Dataset Setup")
    logger.info("=" * 50)

    source_dir = Path(args.source_dir)

    # Validate CASTLE source
    if not validate_castle_source(source_dir):
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize dataset loader
    loader: CastleDatasetLoader = CastleDatasetLoader(source_dir=source_dir)

    create_binary_dataset(loader, output_dir)

    create_multiclass_dataset(loader, output_dir)

    create_cwe_specific_datasets(loader, output_dir)

    logger.info("Dataset setup completed successfully!")
    logger.info(f"Processed datasets saved to: {output_dir}")

    # Print summary
    dataset_files = list(output_dir.glob("*.json"))
    logger.info(f"Created {len(dataset_files)} dataset files:")
    for file in sorted(dataset_files):
        logger.info(f"  - {file.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
VulBench Data Processor

This script processes raw VulBench data and creates JSON datasets
for use with the VulBench benchmark runner.
"""

import json
import logging
from pathlib import Path
from typing import Any

from datasets.loaders.vulbench_dataset_loader import VulBenchDatasetLoader


def process_vulbench_datasets() -> None:
    """Process all VulBench datasets and create JSON files."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger: logging.Logger = logging.getLogger(__name__)

    # Initialize loader
    loader: VulBenchDatasetLoader = VulBenchDatasetLoader()

    # Define datasets to process
    datasets: dict[str, str] = {
        "d2a": "benchmarks/VulBench/data/d2a",
        "ctf": "benchmarks/VulBench/data/ctf",
        "magma": "benchmarks/VulBench/data/magma",
        "big_vul": "benchmarks/VulBench/data/big-vul",
        "devign": "benchmarks/VulBench/data/devign",
    }

    # Output directory
    output_dir: Path = Path("datasets_processed/vulbench")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    for dataset_name, data_path in datasets.items():
        logger.info(f"Processing {dataset_name} dataset...")

        # Check if data directory exists
        if not Path(data_path).exists():
            logger.warning(f"Data directory not found: {data_path}")
            continue

        try:
            # Create binary classification dataset
            binary_output = output_dir / f"vulbench_binary_{dataset_name}.json"
            loader.create_dataset_from_vulbench_data(
                vulbench_data_dir=data_path,
                output_path=str(binary_output),
                dataset_name=dataset_name,
                task_type="binary",
            )

            # Create multiclass classification dataset
            multiclass_output = output_dir / f"vulbench_multiclass_{dataset_name}.json"
            loader.create_dataset_from_vulbench_data(
                vulbench_data_dir=data_path,
                output_path=str(multiclass_output),
                dataset_name=dataset_name,
                task_type="multiclass",
            )

            # Generate and save statistics
            stats: dict[str, Any] = loader.get_dataset_stats(str(binary_output))
            stats_file = output_dir / f"vulbench_{dataset_name}_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            logger.info(f"Completed processing {dataset_name}")
            logger.info(f"  Binary dataset: {binary_output}")
            logger.info(f"  Multiclass dataset: {multiclass_output}")
            logger.info(f"  Statistics: {stats_file}")

        except Exception:
            logger.exception("Error processing %s", dataset_name)
            continue

    logger.info("VulBench data processing completed!")


if __name__ == "__main__":
    process_vulbench_datasets()

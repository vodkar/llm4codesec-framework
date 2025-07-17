#!/usr/bin/env python3
"""
VulDetectBench Data Processing Script

This script processes the raw VulDetectBench dataset files into structured JSON format
compatible with the benchmark framework.
"""

import argparse
import json
import logging
from pathlib import Path

from datasets.loaders.vuldetectbench_dataset_loader import VulDetectBenchDatasetLoader


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def process_vuldetectbench_data(
    input_dir: str = "benchmarks/VulDetectBench/dataset/test",
    output_dir: str = "datasets_processed/vuldetectbench",
    tasks: list = None,
) -> None:
    """
    Process VulDetectBench data for all tasks.

    Args:
        input_dir: Directory containing raw VulDetectBench data
        output_dir: Directory to save processed datasets
        tasks: List of tasks to process (default: all 5 tasks)
    """
    logger = logging.getLogger(__name__)

    if tasks is None:
        tasks = ["task1", "task2", "task3", "task4", "task5"]

    # Initialize the dataset loader
    loader = VulDetectBenchDatasetLoader()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing VulDetectBench data from: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Process each task
    for task in tasks:
        logger.info(f"Processing {task}...")

        try:
            # Define input and output paths
            input_file = input_path / f"{task}_code.jsonl"
            output_file = output_path / f"vuldetectbench_{task}.json"

            if not input_file.exists():
                logger.warning(f"Input file not found: {input_file}")
                continue

            # Process the dataset
            loader.create_dataset_from_vuldetectbench_data(
                str(input_path), str(output_file), task
            )

            # Generate and save statistics
            stats = loader.get_dataset_stats(str(output_file))
            stats_file = output_path / f"vuldetectbench_{task}_stats.json"

            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ {task} completed: {stats['total_samples']} samples")
            logger.info(f"   Dataset: {output_file}")
            logger.info(f"   Stats: {stats_file}")

        except Exception as e:
            logger.error(f"❌ Failed to process {task}: {e}")
            continue

    # Create summary statistics
    create_summary_stats(output_dir, tasks)


def create_summary_stats(output_dir: str, tasks: list) -> None:
    """Create summary statistics across all tasks."""
    logger = logging.getLogger(__name__)

    summary = {
        "dataset": "VulDetectBench",
        "processing_date": str(Path().resolve()),
        "tasks": {},
        "total_samples": 0,
        "task_descriptions": {
            "task1": "Binary vulnerability existence detection (YES/NO)",
            "task2": "Multi-choice vulnerability type inference",
            "task3": "Key objects and functions identification",
            "task4": "Root cause location identification",
            "task5": "Trigger point location identification",
        },
    }

    output_path = Path(output_dir)

    for task in tasks:
        stats_file = output_path / f"vuldetectbench_{task}_stats.json"

        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    task_stats = json.load(f)

                summary["tasks"][task] = {
                    "total_samples": task_stats.get("total_samples", 0),
                    "task_description": summary["task_descriptions"][task],
                    "cwe_distribution": task_stats.get("cwe_distribution", {}),
                    "average_code_length": task_stats.get("average_code_length", 0),
                }

                summary["total_samples"] += task_stats.get("total_samples", 0)

            except Exception as e:
                logger.warning(f"Could not read stats for {task}: {e}")

    # Save summary
    summary_file = output_path / "vuldetectbench_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"📊 Summary statistics saved to: {summary_file}")
    logger.info(f"📊 Total samples across all tasks: {summary['total_samples']}")


def validate_processed_data(output_dir: str, tasks: list[str]) -> bool:
    """Validate that processed data is correct."""
    logger = logging.getLogger(__name__)
    logger.info("Validating processed data...")

    output_path = Path(output_dir)
    all_valid = True

    for task in tasks:
        dataset_file = output_path / f"vuldetectbench_{task}.json"

        if not dataset_file.exists():
            logger.error(f"❌ Missing dataset file: {dataset_file}")
            all_valid = False
            continue

        try:
            with open(dataset_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check structure
            if "metadata" not in data or "samples" not in data:
                logger.error(f"❌ Invalid structure in {dataset_file}")
                all_valid = False
                continue

            samples = data["samples"]
            if not samples:
                logger.warning(f"⚠️  No samples in {dataset_file}")
                continue

            # Check sample structure
            sample = samples[0]
            required_fields = ["id", "code", "label"]
            missing_fields = [field for field in required_fields if field not in sample]

            if missing_fields:
                logger.error(f"❌ Missing fields in {dataset_file}: {missing_fields}")
                all_valid = False
                continue

            logger.info(f"✅ {task}: {len(samples)} samples validated")

        except Exception as e:
            logger.error(f"❌ Error validating {dataset_file}: {e}")
            all_valid = False

    if all_valid:
        logger.info("✅ All datasets validated successfully")
    else:
        logger.error("❌ Some datasets failed validation")

    return all_valid


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Process VulDetectBench raw data into structured datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tasks
  python process_vuldetectbench_data.py
  
  # Process specific tasks
  python process_vuldetectbench_data.py --tasks task1 task2
  
  # Custom input/output directories
  python process_vuldetectbench_data.py --input benchmarks/VulDetectBench/dataset/test --output datasets_processed/vuldetectbench
  
  # Validate processed data
  python process_vuldetectbench_data.py --validate-only
        """,
    )

    parser.add_argument(
        "--input",
        default="benchmarks/VulDetectBench/dataset/test",
        help="Input directory containing VulDetectBench raw data",
    )

    parser.add_argument(
        "--output",
        default="datasets_processed/vuldetectbench",
        help="Output directory for processed datasets",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["task1", "task2", "task3", "task4", "task5"],
        default=["task1", "task2", "task3", "task4", "task5"],
        help="Tasks to process",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing processed data",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        if args.validate_only:
            # Only validate existing data
            success = validate_processed_data(args.output, args.tasks)
            return 0 if success else 1
        else:
            # Process data
            process_vuldetectbench_data(args.input, args.output, args.tasks)

            # Validate processed data
            validate_processed_data(args.output, args.tasks)

            logger.info("🎉 VulDetectBench data processing completed successfully!")
            return 0

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

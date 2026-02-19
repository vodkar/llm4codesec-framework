#!/usr/bin/env python3
"""
JitVul Dataset Setup Script

This script processes the JitVul benchmark source files and creates
structured JSON datasets for use with the LLM benchmark framework.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

from benchmark.enums import TaskType
from datasets.loaders.jitvul import JitVulDatasetLoader


def tuple_encoder(obj: Any) -> Any:
    """Encode tuples for JSON serialization."""
    if isinstance(obj, tuple):
        tuple_obj: tuple[Any, ...] = cast(tuple[Any, ...], obj)
        tuple_items: list[Any] = list(tuple_obj)
        return {"__tuple__": True, "items": tuple_items}
    return obj


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def create_binary_dataset(
    loader: JitVulDatasetLoader,
    data_file: Path,
    output_dir: Path,
    max_samples: int | None = None,
) -> None:
    """Create binary classification dataset."""
    logger = logging.getLogger(__name__)

    logger.info("Creating binary classification dataset...")

    # Load samples
    samples = loader.load_dataset(
        task_type=TaskType.BINARY_VULNERABILITY,
        limit=max_samples,
    )

    # Create dataset dictionary
    dataset_dict: dict[str, Any] = {
        "metadata": {
            "name": "JitVul-Binary-Benchmark",
            "version": "1.0",
            "task_type": "binary",
            "total_samples": len(samples),
            "description": "JitVul dataset for binary vulnerability detection",
            "source": "JitVul",
        },
        "samples": [],
    }

    # Convert samples to dict format
    for sample in samples:
        sample_dict: dict[str, Any] = {
            "id": sample.id,
            "code": sample.code,
            "label": sample.label,
            "cwe_type": sample.cwe_types,
            "severity": sample.severity,
            "metadata": sample.metadata,
        }
        dataset_dict["samples"].append(sample_dict)

    # Save dataset
    output_file = output_dir / "jitvul_binary.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Binary dataset created: {output_file}")
    logger.info(f"Total samples: {len(samples)}")


def create_multiclass_dataset(
    loader: JitVulDatasetLoader,
    data_file: Path,
    output_dir: Path,
    max_samples: int | None = None,
) -> None:
    """Create multiclass classification dataset."""
    logger = logging.getLogger(__name__)

    logger.info("Creating multiclass classification dataset...")

    # Load samples
    samples = loader.load_dataset(
        task_type=TaskType.MULTICLASS_VULNERABILITY,
        limit=max_samples,
    )

    # Calculate CWE distribution
    cwe_counts: dict[str, int] = {}
    for sample in samples:
        cwes = sample.cwe_types or ["UNKNOWN"]
        for cwe in cwes:
            cwe_counts[cwe] = cwe_counts.get(cwe, 0) + 1

    # Create dataset dictionary
    dataset_dict: dict[str, Any] = {
        "metadata": {
            "name": "JitVul-Multiclass-Benchmark",
            "version": "1.0",
            "task_type": "multiclass",
            "total_samples": len(samples),
            "cwe_distribution": cwe_counts,
            "description": "JitVul dataset for multiclass CWE classification",
            "source": "JitVul",
        },
        "samples": [],
    }

    # Convert samples to dict format
    for sample in samples:
        sample_dict: dict[str, Any] = {
            "id": sample.id,
            "code": sample.code,
            "label": sample.label,
            "cwe_type": sample.cwe_types,
            "severity": sample.severity,
            "metadata": sample.metadata,
        }
        dataset_dict["samples"].append(sample_dict)

    # Save dataset
    output_file = output_dir / "jitvul_multiclass.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False, default=tuple_encoder)

    logger.info(f"Multiclass dataset created: {output_file}")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"CWE distribution: {cwe_counts}")


def create_cwe_specific_datasets(
    loader: JitVulDatasetLoader,
    data_file: Path,
    output_dir: Path,
    target_cwes: list[str] | None = None,
    max_samples: int | None = None,
) -> None:
    """Create CWE-specific datasets."""
    logger = logging.getLogger(__name__)

    logger.info("Creating CWE-specific datasets...")

    # First get dataset statistics to identify available CWEs
    stats = loader.get_dataset_stats(data_file)
    available_cwes = list(stats.get("cwe_distribution", {}).keys())

    logger.info(f"Available CWEs in dataset: {available_cwes}")

    # Use specified CWEs or all available ones
    cwes_to_process = target_cwes if target_cwes else available_cwes

    for cwe in cwes_to_process:
        if cwe not in available_cwes:
            logger.warning(f"CWE {cwe} not found in dataset, skipping...")
            continue

        logger.info(f"Processing {cwe}...")

        # Load samples for this specific CWE
        samples = loader.load_dataset(
            task_type=TaskType.BINARY_CWE_SPECIFIC,
            target_cwe=cwe,
            limit=max_samples,
        )

        if not samples:
            logger.warning(f"No samples found for {cwe}, skipping...")
            continue

        # Create dataset dictionary
        dataset_dict: dict[str, Any] = {
            "metadata": {
                "name": f"JitVul-{cwe}-Benchmark",
                "version": "1.0",
                "task_type": "cwe_specific",
                "target_cwe": cwe,
                "total_samples": len(samples),
                "description": f"JitVul dataset for {cwe} detection",
                "source": "JitVul",
            },
            "samples": [],
        }

        # Convert samples to dict format
        for sample in samples:
            sample_dict: dict[str, Any] = {
                "id": sample.id,
                "code": sample.code,
                "label": sample.label,
                "cwe_type": sample.cwe_types,
                "severity": sample.severity,
                "metadata": sample.metadata,
            }
            dataset_dict["samples"].append(sample_dict)

        # Save dataset
        cwe_safe = cwe.replace("-", "_").lower()
        output_file = output_dir / f"jitvul_{cwe_safe}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"{cwe} dataset created: {output_file}")
        logger.info(f"Total samples: {len(samples)}")


def generate_dataset_summary(output_dir: Path) -> None:
    """Generate a summary of all created datasets."""
    logger: logging.Logger = logging.getLogger(__name__)

    logger.info("Generating dataset summary...")

    summary: dict[str, Any] = {
        "created_datasets": [],
        "total_files": 0,
        "generation_timestamp": json.dumps(Path().resolve(), default=str),
    }

    # Find all JSON files in output directory
    json_files = list(output_dir.glob("jitvul_*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            file_info: dict[str, Any] = {
                "filename": json_file.name,
                "task_type": data.get("metadata", {}).get("task_type", "unknown"),
                "total_samples": data.get("metadata", {}).get("total_samples", 0),
                "target_cwe": data.get("metadata", {}).get("target_cwe"),
                "description": data.get("metadata", {}).get("description", ""),
            }
            summary["created_datasets"].append(file_info)

        except Exception:
            logger.exception("Could not process %s", json_file)

    summary["total_files"] = len(summary["created_datasets"])

    # Save summary
    summary_file = output_dir / "jitvul_datasets_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset summary created: {summary_file}")
    logger.info(f"Total datasets created: {summary['total_files']}")


def main() -> None:
    """Main entry point for JitVul dataset setup."""
    parser = argparse.ArgumentParser(
        description="Setup JitVul datasets for benchmarking"
    )

    # Input/Output configuration
    parser.add_argument("--data-file", required=True, help="Path to JitVul data file")
    parser.add_argument(
        "--output-dir",
        default="datasets_processed/jitvul",
        help="Output directory for processed datasets",
    )

    # Dataset types to create
    parser.add_argument(
        "--binary", action="store_true", help="Create binary classification dataset"
    )
    parser.add_argument(
        "--multiclass",
        action="store_true",
        help="Create multiclass classification dataset",
    )
    parser.add_argument(
        "--cwe-specific", action="store_true", help="Create CWE-specific datasets"
    )
    parser.add_argument("--all", action="store_true", help="Create all dataset types")

    # CWE-specific options
    parser.add_argument(
        "--target-cwes",
        nargs="+",
        help="Specific CWEs to process (for CWE-specific datasets)",
    )

    # Sample limiting
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples per dataset"
    )

    # JitVul source configuration
    parser.add_argument(
        "--source-dir",
        default="benchmarks/JitVul/data",
        help="JitVul source data directory",
    )

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger: logging.Logger = logging.getLogger(__name__)

    try:
        # Validate data file exists
        data_file_path: Path = Path(args.data_file)
        if not data_file_path.exists():
            logger.error(f"Data file not found: {args.data_file}")
            sys.exit(1)

        # Create output directory
        output_dir: Path = create_output_directory(args.output_dir)
        logger.info(f"Output directory: {output_dir}")

        # Initialize dataset loader
        loader: JitVulDatasetLoader = JitVulDatasetLoader(source_path=data_file_path)

        # Determine which datasets to create
        create_binary = args.binary or args.all
        create_multiclass = args.multiclass or args.all
        create_cwe = args.cwe_specific or args.all

        if not any([create_binary, create_multiclass, create_cwe]):
            logger.warning("No dataset types specified. Use --all or specific flags.")
            create_binary = True  # Default to binary

        # Create datasets
        if create_binary:
            create_binary_dataset(loader, data_file_path, output_dir, args.max_samples)

        if create_multiclass:
            create_multiclass_dataset(
                loader, data_file_path, output_dir, args.max_samples
            )

        if create_cwe:
            create_cwe_specific_datasets(
                loader, data_file_path, output_dir, args.target_cwes, args.max_samples
            )

        # Generate summary
        generate_dataset_summary(output_dir)

        logger.info("JitVul dataset setup completed successfully!")

    except Exception:
        logger.exception("JitVul dataset setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

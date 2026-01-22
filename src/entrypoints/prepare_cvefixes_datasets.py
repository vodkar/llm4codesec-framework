#!/usr/bin/env python3
"""
CVEFixes Dataset Preparation Script

This script prepares CVEFixes datasets in the same format as CASTLE datasets
for use with the benchmark framework.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from datasets.loaders.cvefixes_dataset_loader import CVEFixesDatasetLoader


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("cvefixes_preparation.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def prepare_binary_datasets(
    loader: CVEFixesDatasetLoader,
    output_dir: Path,
    programming_language: str = "C",
    sample_limit: int | None = None,
) -> None:
    """Prepare binary classification datasets."""

    # File-level binary dataset
    file_output = (
        output_dir / f"cvefixes_binary_{programming_language.lower()}_file.json"
    )
    logging.info(f"Creating binary file-level dataset: {file_output}")
    loader.create_dataset_json(
        output_path=str(file_output),
        task_type="binary",
        programming_language=programming_language,
        change_level="file",
        limit=sample_limit,
    )

    # Method-level binary dataset
    method_output = (
        output_dir / f"cvefixes_binary_{programming_language.lower()}_method.json"
    )
    logging.info(f"Creating binary method-level dataset: {method_output}")
    loader.create_dataset_json(
        output_path=str(method_output),
        task_type="binary",
        programming_language=programming_language,
        change_level="method",
        limit=sample_limit,
    )


def prepare_multiclass_datasets(
    loader: CVEFixesDatasetLoader,
    output_dir: Path,
    programming_language: str = "C",
    sample_limit: int | None = None,
) -> None:
    """Prepare multiclass classification datasets."""

    # File-level multiclass dataset
    file_output = (
        output_dir / f"cvefixes_multiclass_{programming_language.lower()}_file.json"
    )
    logging.info(f"Creating multiclass file-level dataset: {file_output}")
    loader.create_dataset_json(
        output_path=str(file_output),
        task_type="multiclass",
        programming_language=programming_language,
        change_level="file",
        limit=sample_limit,
    )

    # Method-level multiclass dataset
    method_output = (
        output_dir / f"cvefixes_multiclass_{programming_language.lower()}_method.json"
    )
    logging.info(f"Creating multiclass method-level dataset: {method_output}")
    loader.create_dataset_json(
        output_path=str(method_output),
        task_type="multiclass",
        programming_language=programming_language,
        change_level="method",
        limit=sample_limit,
    )


def prepare_cwe_specific_datasets(
    loader: CVEFixesDatasetLoader,
    output_dir: Path,
    programming_language: str = "C",
    cwe_types: list[str] | None = None,
    sample_limit: int | None = None,
) -> None:
    """Prepare CWE-specific binary classification datasets."""

    if not cwe_types:
        # Common CWE types found in CVEFixes
        cwe_types = [
            "CWE-119",
            "CWE-120",
            "CWE-125",
            "CWE-190",
            "CWE-476",
            "CWE-787",
            "CWE-401",
            "CWE-416",
            "CWE-20",
            "CWE-79",
            "CWE-89",
            "CWE-22",
        ]

    for cwe_type in cwe_types:
        cwe_number = cwe_type.replace("CWE-", "")
        output_file = output_dir / f"cvefixes_cwe_{cwe_number}.json"

        logging.info(f"Creating {cwe_type}-specific dataset: {output_file}")

        try:
            loader.create_dataset_json(
                output_path=str(output_file),
                task_type=cwe_type,
                programming_language=programming_language,
                change_level="file",
                limit=sample_limit,
            )
        except Exception:
            logging.exception("Error creating dataset for %s", cwe_type)


def analyze_database(loader: CVEFixesDatasetLoader, output_dir: Path) -> None:
    """Analyze the CVEFixes database and create statistics report."""

    logging.info("Analyzing CVEFixes database...")
    stats = loader.get_database_statistics()

    # Save statistics
    stats_file = output_dir / "cvefixes_database_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logging.info(f"Database statistics saved to: {stats_file}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("CVEFixes Database Statistics")
    print(f"{'=' * 60}")
    print(f"Total CVEs: {stats.get('cve_count', 0):,}")
    print(f"Total Commits: {stats.get('commits_count', 0):,}")
    print(f"Total File Changes: {stats.get('file_change_count', 0):,}")
    print(f"Total Method Changes: {stats.get('method_change_count', 0):,}")

    print("\nTop Programming Languages:")
    for lang, count in list(stats.get("programming_languages", {}).items())[:10]:
        print(f"  {lang}: {count:,}")

    print("\nTop CWE Types:")
    for cwe, count in list(stats.get("cwe_distribution", {}).items())[:10]:
        print(f"  {cwe}: {count:,}")

    print("\nSeverity Distribution:")
    for severity, count in stats.get("severity_distribution", {}).items():
        print(f"  {severity}: {count:,}")

    print(f"{'=' * 60}\n")


def main() -> int:
    """Main entry point for CVEFixes dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare CVEFixes datasets for benchmark framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare all dataset types for C language
  python src/entrypoints/prepare_cvefixes_datasets.py --database-path datasets_processed/cvefixes/CVEfixes.db

  # Prepare only binary datasets with sample limit
  python src/entrypoints/prepare_cvefixes_datasets.py --database-path datasets_processed/cvefixes/CVEfixes.db --dataset-types binary --sample-limit 1000

  # Prepare datasets for multiple languages
  python src/entrypoints/prepare_cvefixes_datasets.py --database-path datasets_processed/cvefixes/CVEfixes.db --languages C Java

  # Just analyze the database
  python src/entrypoints/prepare_cvefixes_datasets.py --database-path datasets_processed/cvefixes/CVEfixes.db --analyze-only
        """,
    )

    # Required arguments
    parser.add_argument(
        "--database-path",
        type=str,
        required=True,
        help="Path to CVEFixes SQLite database",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets_processed/cvefixes",
        help="Output directory for processed datasets (default: datasets_processed/cvefixes)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        choices=["binary", "multiclass", "cwe_specific", "all"],
        default=["all"],
        help="Types of datasets to prepare (default: all)",
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=["C"],
        help="Programming languages to process (default: C)",
    )

    parser.add_argument(
        "--cwe-types",
        nargs="+",
        help="Specific CWE types for cwe_specific datasets (e.g., CWE-119 CWE-120)",
    )

    # Sampling configuration
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Limit number of samples per dataset (for testing)",
    )

    # Analysis options
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze database, don't create datasets",
    )

    parser.add_argument(
        "--skip-analysis", action="store_true", help="Skip database analysis"
    )

    # System options
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output files"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    try:
        # Validate database path
        database_path = Path(args.database_path)
        if not database_path.exists():
            parser.error(f"Database file not found: {args.database_path}")

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize loader
        loader = CVEFixesDatasetLoader(str(database_path))

        # Analyze database if requested
        if not args.skip_analysis:
            analyze_database(loader, output_dir)

        # Exit if only analysis was requested
        if args.analyze_only:
            return 0

        # Expand dataset types
        dataset_types = args.dataset_types
        if "all" in dataset_types:
            dataset_types = ["binary", "multiclass", "cwe_specific"]

        # Process each language
        for language in args.languages:
            logging.info(f"Processing datasets for {language}...")

            # Prepare binary datasets
            if "binary" in dataset_types:
                prepare_binary_datasets(loader, output_dir, language, args.sample_limit)

            # Prepare multiclass datasets
            if "multiclass" in dataset_types:
                prepare_multiclass_datasets(
                    loader, output_dir, language, args.sample_limit
                )

            # Prepare CWE-specific datasets
            if "cwe_specific" in dataset_types:
                prepare_cwe_specific_datasets(
                    loader, output_dir, language, args.cwe_types, args.sample_limit
                )

        logging.info("Dataset preparation completed successfully!")
        print(f"\nDatasets created in: {output_dir}")

        return 0

    except KeyboardInterrupt:
        logging.info("Dataset preparation interrupted by user")
        return 1
    except Exception:
        logging.exception("Error during dataset preparation")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

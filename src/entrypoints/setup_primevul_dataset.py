#!/usr/bin/env python3
"""
PrimeVul Dataset Setup Script

Generates processed JSON benchmark files from a raw PrimeVul JSONL source file.
Run this once before executing PrimeVul experiments.

Usage (inside Docker):
    python src/entrypoints/setup_primevul_dataset.py \
        --source benchmarks/PrimeVul/primevul.jsonl \
        --output-dir datasets_processed/primevul

To use a real PrimeVul split (train/valid/test), pass the JSONL path directly.
"""

import argparse
import logging
import sys
from pathlib import Path

from benchmark.enums import TaskType
from datasets.loaders.primevul import PrimeVulDatasetLoader
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate processed PrimeVul benchmark JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        default="benchmarks/PrimeVul/primevul.jsonl",
        help="Path to PrimeVul JSONL source file",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets_processed/primevul",
        help="Output directory for processed JSON files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples per dataset (useful for testing)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logging.getLogger().setLevel(args.log_level)

    source_path = Path(args.source)
    if not source_path.exists():
        _LOGGER.error("Source file not found: %s", source_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = PrimeVulDatasetLoader(source_path=source_path)

    stats = loader.get_dataset_stats()
    _LOGGER.info(
        "Source stats — groups=%d  with_vuln=%d  with_safe=%d  CWEs=%s",
        stats["total_groups"],
        stats["groups_with_vulnerable"],
        stats["groups_with_safe"],
        list(stats["cwe_distribution"].keys()),
    )

    # ---- binary classification ---------------------------------------- #
    _LOGGER.info("Generating binary dataset …")
    loader.create_dataset_json(
        str(output_dir / "primevul_binary.json"),
        task_type=TaskType.BINARY_VULNERABILITY,
        limit=args.limit,
    )

    # ---- multiclass classification ------------------------------------- #
    _LOGGER.info("Generating multiclass dataset …")
    loader.create_dataset_json(
        str(output_dir / "primevul_multiclass.json"),
        task_type=TaskType.MULTICLASS_VULNERABILITY,
        limit=args.limit,
    )

    # ---- CWE-specific datasets ----------------------------------------- #
    target_cwes = [
        "CWE-119",
        "CWE-120",
        "CWE-125",
        "CWE-190",
        "CWE-476",
        "CWE-787",
    ]
    for cwe in target_cwes:
        cwe_slug = cwe.replace("-", "_").lower()
        out_path = output_dir / f"primevul_{cwe_slug}.json"
        _LOGGER.info("Generating %s dataset …", cwe)
        loader.create_dataset_json(
            str(out_path),
            task_type=TaskType.BINARY_CWE_SPECIFIC,
            target_cwe=cwe,
            limit=args.limit,
        )

    _LOGGER.info("All PrimeVul datasets written to %s", output_dir)


if __name__ == "__main__":
    main()

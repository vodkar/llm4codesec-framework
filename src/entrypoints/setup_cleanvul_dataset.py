#!/usr/bin/env python3
"""
CleanVul Dataset Setup Script

Generates processed JSON benchmark files from CleanVul CSV source files.
Run this once before executing CleanVul experiments.

Usage (inside Docker):
    python entrypoints/setup_cleanvul_dataset.py \\
        --source-score4 benchmarks/CleanVul/vulnerability_score_4.csv \\
        --source-score3 benchmarks/CleanVul/vulnerability_score_3.csv \\
        --output-dir datasets_processed/cleanvul
"""

import argparse
import logging
from pathlib import Path

from benchmark.enums import TaskType
from datasets.loaders.cleanvul import EXTENSION_TO_LANGUAGE, CleanVulDatasetLoader
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate processed CleanVul benchmark JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-score4",
        default="benchmarks/CleanVul/vulnerability_score_4.csv",
        help="Path to vulnerability_score_4.csv (score=4 only, highest quality)",
    )
    parser.add_argument(
        "--source-score3",
        default="benchmarks/CleanVul/vulnerability_score_3.csv",
        help="Path to vulnerability_score_3.csv (score>=3, larger dataset)",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets_processed/cleanvul",
        help="Output directory for processed JSON files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of groups per dataset (useful for testing)",
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("score4", Path(args.source_score4)),
        ("score3", Path(args.source_score3)),
    ]

    for score_label, source_path in sources:
        if not source_path.exists():
            _LOGGER.warning("Source not found, skipping %s: %s", score_label, source_path)
            continue

        _LOGGER.info("Processing %s from %s …", score_label, source_path)

        # Quick stats across all languages
        stats_loader = CleanVulDatasetLoader(source_path=source_path)
        stats = stats_loader.get_dataset_stats()
        _LOGGER.info(
            "%s stats — groups=%d  with_vuln=%d  with_safe=%d  "
            "CWEs=%d  extensions=%s",
            score_label,
            stats["total_groups"],
            stats["groups_with_vulnerable"],
            stats["groups_with_safe"],
            len(stats["cwe_distribution"]),
            stats["extension_distribution"],
        )

        score_dir = output_dir / score_label
        score_dir.mkdir(parents=True, exist_ok=True)

        for ext, lang in EXTENSION_TO_LANGUAGE.items():
            loader = CleanVulDatasetLoader(
                source_path=source_path,
                programming_language=lang,
            )

            # binary classification
            out_path = score_dir / f"cleanvul_{ext}_binary.json"
            _LOGGER.info("  Generating %s / %s binary …", score_label, ext)
            loader.create_dataset_json(
                str(out_path),
                task_type=TaskType.BINARY_VULNERABILITY,
                limit=args.limit,
            )

            # multiclass (CWE identification)
            out_path = score_dir / f"cleanvul_{ext}_multiclass.json"
            _LOGGER.info("  Generating %s / %s multiclass …", score_label, ext)
            loader.create_dataset_json(
                str(out_path),
                task_type=TaskType.MULTICLASS_VULNERABILITY,
                limit=args.limit,
            )

    _LOGGER.info("All CleanVul datasets written to %s", output_dir)


if __name__ == "__main__":
    main()

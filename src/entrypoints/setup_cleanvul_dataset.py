#!/usr/bin/env python3
"""
CleanVul Dataset Setup Script

Generates processed JSON benchmark files from a single CleanVul CSV source.
Supports optional filtering by vulnerability score and by per-sample token
count (used to build context-size buckets).

Usage (host, with token buckets):
    PYTHONPATH=src uv run python src/entrypoints/setup_cleanvul_dataset.py \\
        --source benchmarks/CleanVul/vulnerability_score_2.csv \\
        --output-dir datasets_processed/cleanvul/context_sizes \\
        --languages py --tasks binary \\
        --min-tokens 1000 --max-tokens 2000 --name-suffix _1k_2k

Usage (inside Docker, standard per-language generation):
    python entrypoints/setup_cleanvul_dataset.py \\
        --source benchmarks/CleanVul/vulnerability_score_4.csv \\
        --output-dir datasets_processed/cleanvul/score4
"""

import argparse
import logging
from pathlib import Path

from benchmark.enums import TaskType
from datasets.loaders.cleanvul import EXTENSION_TO_LANGUAGE, CleanVulDatasetLoader
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)

_TASKS = {
    "binary": [("binary", TaskType.BINARY_VULNERABILITY)],
    "multiclass": [("multiclass", TaskType.MULTICLASS_VULNERABILITY)],
    "both": [
        ("binary", TaskType.BINARY_VULNERABILITY),
        ("multiclass", TaskType.MULTICLASS_VULNERABILITY),
    ],
}

_DEFAULT_TOKENIZER = "google/gemma-4-12B-it-qat-w4a16-ct"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate processed CleanVul benchmark JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a CleanVul CSV (with a vulnerability_score column)",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets_processed/cleanvul",
        help="Output directory for processed JSON files",
    )
    parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated extensions to generate (e.g. 'py' or 'c,cpp'). "
        "Default: all supported languages.",
    )
    parser.add_argument(
        "--tasks",
        choices=list(_TASKS),
        default="both",
        help="Which task variants to generate (default: both)",
    )
    parser.add_argument("--min-score", type=int, default=None)
    parser.add_argument("--max-score", type=int, default=None)
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--tokenizer",
        default=_DEFAULT_TOKENIZER,
        help="HF tokenizer id for token counting (default: %(default)s)",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Suffix appended to output filenames (e.g. '_1k_2k')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of commit groups scanned per language",
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

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source CSV not found: {source}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.languages:
        exts = [e.strip().lower() for e in args.languages.split(",") if e.strip()]
    else:
        exts = list(EXTENSION_TO_LANGUAGE)

    unknown = [e for e in exts if e not in EXTENSION_TO_LANGUAGE]
    if unknown:
        raise ValueError(f"Unknown language extensions: {unknown}")

    for ext in exts:
        lang = EXTENSION_TO_LANGUAGE[ext]
        loader = CleanVulDatasetLoader(
            source_path=source,
            programming_language=lang,
            min_score=args.min_score,
            max_score=args.max_score,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            tokenizer_id=args.tokenizer,
        )
        for task_label, task_type in _TASKS[args.tasks]:
            out_path = output_dir / f"cleanvul_{ext}_{task_label}{args.name_suffix}.json"
            _LOGGER.info("Generating %s (%s) …", out_path.name, lang)
            loader.create_dataset_json(
                str(out_path),
                task_type=task_type,
                limit=args.limit,
            )

    _LOGGER.info("All CleanVul datasets written to %s", output_dir)


if __name__ == "__main__":
    main()

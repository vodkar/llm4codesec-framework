#!/usr/bin/env python3
"""Prepare processed datasets for the ContextAssembler compare-rankings study.

This entrypoint normalizes the associated CleanVul-backed ContextAssembler
ranking variants from
``benchmarks/context-assembler-dataset/cleanvul_compare_rankings`` into
framework-ready processed JSON files under a dedicated output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Final

SRC_DIR: Final[Path] = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, Dataset, DatasetMetadata, SampleCollection
from datasets.loaders.context_assembler import ContextAssemblerDatasetLoader
from logging_tools import setup_logging

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

DEFAULT_SOURCE_DIR: Final[str] = (
    "benchmarks/context-assembler-dataset/cleanvul_compare_rankings"
)
DEFAULT_OUTPUT_DIR: Final[str] = "datasets_processed/context_assembler_compare_rankings"

DATASET_VARIANTS: Final[dict[str, tuple[str, str]]] = {
    "cleanvul_context_benchmark.json": (
        "context_assembler_compare_current.json",
        "ContextAssembler Compare Rankings - Current",
    ),
    "cleanvul_context_benchmark_depth_repeats_context.json": (
        "context_assembler_compare_depth_repeats_context.json",
        "ContextAssembler Compare Rankings - Depth Repeats Context",
    ),
    "cleanvul_context_benchmark_dummy.json": (
        "context_assembler_compare_dummy.json",
        "ContextAssembler Compare Rankings - Dummy",
    ),
    "cleanvul_context_benchmark_multiplicative_amplification.json": (
        "context_assembler_compare_multiplicative_amplification.json",
        "ContextAssembler Compare Rankings - Multiplicative amplification",
    ),
    "cleanvul_context_benchmark_random_picking.json": (
        "context_assembler_compare_random_picking.json",
        "ContextAssembler Compare Rankings - Random Picking",
    ),
    "cleanvul_context_benchmark_current_default.json": (
        "context_assembler_compare_current_default.json",
        "ContextAssembler Compare Rankings - Current (Default, Manual Coefficients)",
    ),
    "cleanvul_context_benchmark_multiplicative_amplification_default.json": (
        "context_assembler_compare_multiplicative_amplification_default.json",
        "ContextAssembler Compare Rankings - Multiplicative amplification (Default, Manual Coefficients)",
    ),
    "cleanvul_context_benchmark_cpg_structural.json": (
        "context_assembler_compare_cpg_structural.json",
        "ContextAssembler Compare Rankings - CPG Structural",
    ),
    "cleanvul_context_benchmark_evidence_budgeted.json": (
        "context_assembler_compare_evidence_budgeted.json",
        "ContextAssembler Compare Rankings - Evidence Budgeted",
    ),
}

# Optional variants holding the last (final) hyperparameter-tuning values. They
# are only present in the top-level compare-rankings folder, so they are
# processed when found and silently skipped otherwise (e.g. for context-size
# sub-folders that do not ship them).
OPTIONAL_DATASET_VARIANTS: Final[dict[str, tuple[str, str]]] = {
    "cleanvul_context_benchmark_current_last.json": (
        "context_assembler_compare_current_last.json",
        "ContextAssembler Compare Rankings - Current (Last Tuning Values)",
    ),
    "cleanvul_context_benchmark_cpg_structural_last.json": (
        "context_assembler_compare_cpg_structural_last.json",
        "ContextAssembler Compare Rankings - CPG Structural (Last Tuning Values)",
    ),
    "cleanvul_context_benchmark_evidence_budgeted_last.json": (
        "context_assembler_compare_evidence_budgeted_last.json",
        "ContextAssembler Compare Rankings - Evidence Budgeted (Last Tuning Values)",
    ),
    "cleanvul_context_benchmark_multiplicative_amplification_last.json": (
        "context_assembler_compare_multiplicative_amplification_last.json",
        "ContextAssembler Compare Rankings - Multiplicative amplification (Last Tuning Values)",
    ),
}

ENTRIES_SOURCE_NAME: Final[str] = "cleanvul_entries.json"
EXCLUDED_SOURCE_FILES: Final[set[str]] = {ENTRIES_SOURCE_NAME}
SMOKE_TEST_OUTPUT_NAME: Final[str] = "context_assembler_compare_smoke_test.json"
SMOKE_TEST_DATASET_NAME: Final[str] = "ContextAssembler Compare Rankings - Smoke Test"
# Function-only baseline: the raw CleanVul function of every compare-rankings
# sample, without any assembled context. It is validated against this ranking
# variant so both datasets pair sample-for-sample.
FUNCTION_ONLY_REFERENCE_SOURCE_NAME: Final[str] = (
    "cleanvul_context_benchmark_cpg_structural.json"
)
FUNCTION_ONLY_OUTPUT_NAME: Final[str] = (
    "context_assembler_compare_cleanvul_function_only.json"
)
FUNCTION_ONLY_DATASET_NAME: Final[str] = (
    "ContextAssembler Compare Rankings - CleanVul Function Only (No Context)"
)


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

    for source_name, (output_name, dataset_name) in OPTIONAL_DATASET_VARIANTS.items():
        if source_name not in discovered_files:
            LOGGER.info(
                "Skipping optional compare-rankings variant (not present): %s",
                source_name,
            )
            continue
        source_path = source_dir / source_name
        output_path = output_dir / output_name
        LOGGER.info("Processing %s -> %s", source_path, output_path)
        _write_processed_dataset(
            source_path=source_path,
            output_path=output_path,
            dataset_name=dataset_name,
            sample_limit=sample_limit,
        )
        written_files.append(output_path)

    if ENTRIES_SOURCE_NAME in discovered_files:
        function_only_output_path: Path = output_dir / FUNCTION_ONLY_OUTPUT_NAME
        write_function_only_dataset(
            entries_path=source_dir / ENTRIES_SOURCE_NAME,
            reference_path=source_dir / FUNCTION_ONLY_REFERENCE_SOURCE_NAME,
            output_path=function_only_output_path,
            sample_limit=sample_limit,
        )
        written_files.append(function_only_output_path)
    else:
        LOGGER.info(
            "Skipping function-only baseline (not present): %s", ENTRIES_SOURCE_NAME
        )

    current_source_path: Path = source_dir / "cleanvul_context_benchmark.json"
    smoke_output_path: Path = output_dir / SMOKE_TEST_OUTPUT_NAME
    _write_smoke_test_dataset(
        source_path=current_source_path,
        output_path=smoke_output_path,
        sample_limit=sample_limit,
    )
    written_files.append(smoke_output_path)

    return written_files


def write_function_only_dataset(
    entries_path: Path,
    reference_path: Path,
    output_path: Path,
    sample_limit: int | None,
) -> None:
    """Write the raw CleanVul functions as a baseline paired with a ranking variant.

    Args:
        entries_path: ``cleanvul_entries.json`` holding the raw function per sample id.
        reference_path: Raw ranking-variant JSON the baseline must pair with.
        output_path: Processed output file path.
        sample_limit: Optional cap on written samples, applied in reference order.

    Raises:
        ValueError: If an entry is missing or disagrees with the reference sample
            on label or commit. Sample ids are re-randomized per dataset
            generation, so a stale entries file reuses ids for other commits.
    """
    with entries_path.open("r", encoding="utf-8") as entries_file:
        entries: list[dict[str, Any]] = json.load(entries_file)
    with reference_path.open("r", encoding="utf-8") as reference_file:
        reference_samples: list[dict[str, Any]] = json.load(reference_file).get(
            "samples", []
        )

    entries_by_id: dict[str, dict[str, Any]] = {
        entry["sample_id"]: entry for entry in entries
    }
    if sample_limit is not None:
        reference_samples = reference_samples[:sample_limit]

    samples: list[BenchmarkSample] = []
    for reference in reference_samples:
        sample_id: str = reference["id"]
        entry: dict[str, Any] | None = entries_by_id.get(sample_id)
        if entry is None:
            raise ValueError(f"No CleanVul entry for compare-rankings sample {sample_id}")

        label: int = 1 if entry["is_vulnerable"] else 0
        reference_meta: dict[str, Any] = reference.get("metadata", {})
        reference_commit_url: str | None = reference_meta.get(
            "commit_url"
        ) or reference_meta.get("CleanVul-CommitUrl")
        if label != int(reference["label"]) or (
            reference_commit_url and reference_commit_url != entry["commit_url"]
        ):
            raise ValueError(
                f"CleanVul entry for sample {sample_id} does not match {reference_path.name} "
                f"(label {label} vs {reference['label']}, commit {entry['commit_url']} "
                f"vs {reference_commit_url}); the entries file is stale"
            )

        samples.append(
            BenchmarkSample(
                id=sample_id,
                code=entry["func_code"],
                label=label,
                metadata={
                    "cve_id": entry.get("cve_id", ""),
                    "description": entry.get("commit_msg", ""),
                    "cwe_number": 0,
                    "source": "CleanVul-FunctionOnly",
                    "commit_url": entry["commit_url"],
                },
                cwe_types=[],
            )
        )

    dataset: Dataset = Dataset(
        metadata=DatasetMetadata(
            name=FUNCTION_ONLY_DATASET_NAME,
            version="1.0",
            task_type=TaskType.BINARY_VULNERABILITY,
            programming_language="Python",
            change_level="function",
        ),
        samples=SampleCollection.model_validate(samples),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(dataset.model_dump(), output_file, indent=2, ensure_ascii=False)

    LOGGER.info("Wrote %d function-only samples to %s", len(samples), output_path)


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

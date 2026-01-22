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
from typing import Any

from benchmark.models import BenchmarkSample
from datasets.loaders.castle_dataset_loader import (
    CastleDatasetLoader,
    filter_by_cwe,
    get_available_cwes,
)


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


def create_binary_dataset(loader: CastleDatasetLoader, output_dir: Path) -> None:
    """Create binary classification dataset."""
    logger: logging.Logger = logging.getLogger(__name__)

    logger.info("Creating binary classification dataset...")

    output_file = output_dir / "castle_binary.json"
    loader.create_dataset_json(str(output_file), task_type="binary")

    logger.info(f"Binary dataset created: {output_file}")


def create_multiclass_dataset(loader: CastleDatasetLoader, output_dir: Path) -> None:
    """Create multi-class classification dataset."""
    logger: logging.Logger = logging.getLogger(__name__)

    logger.info("Creating multi-class classification dataset...")

    output_file = output_dir / "castle_multiclass.json"
    loader.create_dataset_json(str(output_file), task_type="multiclass")

    logger.info(f"Multi-class dataset created: {output_file}")


def create_cwe_specific_datasets(
    loader: CastleDatasetLoader,
    output_dir: Path,
    target_cwes: list[str] | None = None,
) -> None:
    """Create CWE-specific datasets."""
    logger: logging.Logger = logging.getLogger(__name__)

    logger.info("Creating CWE-specific datasets...")

    # Load all samples first
    all_samples: list[BenchmarkSample] = loader.transform_dataset()
    available_cwes = get_available_cwes(all_samples)

    logger.info(f"Available CWEs in dataset: {available_cwes}")

    # Use specified CWEs or all available ones
    cwes_to_process = target_cwes if target_cwes else available_cwes

    for cwe in cwes_to_process:
        if cwe not in available_cwes:
            logger.warning(f"CWE {cwe} not found in dataset, skipping...")
            continue

        logger.info(f"Processing {cwe}...")

        # Filter samples for this CWE
        cwe_samples = filter_by_cwe(all_samples, cwe)

        if not cwe_samples:
            logger.warning(f"No samples found for {cwe}")
            continue

        # Count vulnerable vs safe samples
        vulnerable_count = sum(1 for s in cwe_samples if s.label == 1)
        safe_count = len(cwe_samples) - vulnerable_count

        logger.info(f"{cwe}: {vulnerable_count} vulnerable, {safe_count} safe samples")

        # Create dataset structure
        cwe_number = cwe.split("-")[1]
        dataset_dict: dict[str, Any] = {
            "metadata": {
                "name": f"CASTLE-Benchmark-{cwe}",
                "version": "1.2",
                "task_type": "binary_cwe_specific",
                "cwe_type": cwe,
                "total_samples": len(cwe_samples),
                "vulnerable_samples": vulnerable_count,
                "safe_samples": safe_count,
            },
            "samples": [],
        }

        # Convert samples to dict format
        for sample in cwe_samples:
            sample_dict = {
                "id": sample.id,
                "code": sample.code,
                "label": sample.label,
                "cwe_type": sample.cwe_types,
                "severity": sample.severity,
                "metadata": sample.metadata,
            }
            dataset_dict["samples"].append(sample_dict)

        # Save dataset
        output_file = output_dir / f"castle_cwe_{cwe_number}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"{cwe} dataset created: {output_file}")


def update_gitignore() -> None:
    """Update .gitignore to include processed datasets."""
    logger: logging.Logger = logging.getLogger(__name__)

    gitignore_path = Path(".gitignore")

    if not gitignore_path.exists():
        logger.warning(".gitignore not found, creating new one")
        gitignore_content = ""
    else:
        with open(gitignore_path, "r", encoding="utf-8") as f:
            gitignore_content = f.read()

    # Entries to add
    entries_to_add = [
        "# Processed datasets (generated)",
        "datasets_processed/",
        "*.dataset.json",
        "castle_processed/",
    ]

    # Check which entries are missing
    missing_entries: list[str] = []
    for entry in entries_to_add:
        if entry not in gitignore_content:
            missing_entries.append(entry)

    if missing_entries:
        # Add missing entries
        if gitignore_content and not gitignore_content.endswith("\n"):
            gitignore_content += "\n"

        gitignore_content += "\n" + "\n".join(missing_entries) + "\n"

        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(gitignore_content)

        logger.info(f"Updated .gitignore with {len(missing_entries)} new entries")
    else:
        logger.info(".gitignore already contains required entries")


def create_sample_prompts(output_dir: Path) -> None:
    """Create sample prompt files for experimentation."""
    logger: logging.Logger = logging.getLogger(__name__)

    prompts_dir = output_dir / "sample_prompts"
    prompts_dir.mkdir(exist_ok=True)

    # Basic security prompt
    basic_prompt = {
        "name": "Basic Security Analysis",
        "description": "Simple, direct security analysis prompt",
        "system_prompt": "You are an expert security analyst. Analyze the provided code and determine if it contains security vulnerabilities. Respond with only 'VULNERABLE' or 'SAFE'.",
        "user_prompt": "Analyze this code for security vulnerabilities:\n\n{code}",
        "example_usage": "For general vulnerability detection across all code samples",
    }

    # CWE-focused prompt
    cwe_prompt = {
        "name": "CWE-Focused Analysis",
        "description": "Prompt focused on specific CWE detection",
        "system_prompt": "You are a security expert specializing in {cwe_type} vulnerabilities. Analyze the code specifically for {cwe_type} patterns. Respond with only 'VULNERABLE' or 'SAFE'.",
        "user_prompt": "Check this code for {cwe_type} vulnerabilities:\n\n{code}",
        "example_usage": "For CWE-specific vulnerability detection tasks",
    }

    # Detailed analysis prompt
    detailed_prompt = {
        "name": "Detailed Security Analysis",
        "description": "Comprehensive analysis with specific guidelines",
        "system_prompt": "You are a senior cybersecurity expert. Perform thorough static analysis checking for: buffer overflows, memory corruption, input validation issues, bounds checking, integer overflow/underflow, race conditions, null pointer dereferences. Respond with only 'VULNERABLE' or 'SAFE'.",
        "user_prompt": "Please analyze the following C code for security vulnerabilities:\n\n```c\n{code}\n```",
        "example_usage": "For detailed vulnerability analysis with specific patterns",
    }

    # Save prompt files
    prompts: dict[str, dict[str, str]] = {
        "basic_security": basic_prompt,
        "cwe_focused": cwe_prompt,
        "detailed_analysis": detailed_prompt,
    }

    for prompt_name, prompt_data in prompts.items():
        prompt_file = prompts_dir / f"{prompt_name}.json"
        with open(prompt_file, "w", encoding="utf-8") as f:
            json.dump(prompt_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Created sample prompt: {prompt_file}")

    # Create README for prompts
    readme_content = """# Sample Prompts for CASTLE Benchmark

This directory contains sample prompt configurations for experimenting with different analysis approaches.

## Available Prompts

- **basic_security.json**: Simple, direct security analysis
- **cwe_focused.json**: CWE-specific vulnerability detection  
- **detailed_analysis.json**: Comprehensive analysis with specific guidelines

## Usage

These prompts can be used as templates in the experiment configuration file or loaded directly for custom experiments.

## Customization

Feel free to modify these prompts or create new ones based on your experimental needs. Key considerations:

1. **Response Format**: Ensure prompts request only "VULNERABLE" or "SAFE" responses
2. **Specificity**: Tailor prompts to the specific vulnerability types you're testing
3. **Context**: Provide appropriate context for the analysis task
4. **Instructions**: Include clear, specific instructions for the analysis approach
"""

    readme_file = prompts_dir / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info(f"Created prompts README: {readme_file}")


def validate_castle_source(source_dir: str) -> bool:
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
        "--cwes", nargs="+", help="Specific CWEs to process (e.g., CWE-125 CWE-190)"
    )

    parser.add_argument(
        "--skip-binary",
        action="store_true",
        help="Skip creating binary classification dataset",
    )

    parser.add_argument(
        "--skip-multiclass",
        action="store_true",
        help="Skip creating multi-class classification dataset",
    )

    parser.add_argument(
        "--skip-cwe-specific",
        action="store_true",
        help="Skip creating CWE-specific datasets",
    )

    parser.add_argument(
        "--create-prompts", action="store_true", help="Create sample prompt files"
    )

    parser.add_argument(
        "--update-gitignore",
        action="store_true",
        help="Update .gitignore with dataset entries",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger: logging.Logger = logging.getLogger(__name__)

    try:
        logger.info("CASTLE Dataset Setup")
        logger.info("=" * 50)

        # Validate CASTLE source
        if not validate_castle_source(args.source_dir):
            sys.exit(1)

        # Create output directory
        output_dir = create_output_directory(args.output_dir)
        logger.info(f"Output directory: {output_dir}")

        # Initialize dataset loader
        loader: CastleDatasetLoader = CastleDatasetLoader()
        loader.source_dir = Path(args.source_dir)

        # Create datasets
        if not args.skip_binary:
            create_binary_dataset(loader, output_dir)

        if not args.skip_multiclass:
            create_multiclass_dataset(loader, output_dir)

        if not args.skip_cwe_specific:
            create_cwe_specific_datasets(loader, output_dir, args.cwes)

        # Create sample prompts
        if args.create_prompts:
            create_sample_prompts(output_dir)

        # Update gitignore
        if args.update_gitignore:
            update_gitignore()

        logger.info("Dataset setup completed successfully!")
        logger.info(f"Processed datasets saved to: {output_dir}")

        # Print summary
        dataset_files = list(output_dir.glob("*.json"))
        logger.info(f"Created {len(dataset_files)} dataset files:")
        for file in sorted(dataset_files):
            logger.info(f"  - {file.name}")

    except Exception:
        logger.exception("Dataset setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Data processing utilities for the LLM benchmark framework.

Utilities for converting datasets, analyzing results, and preparing data for benchmarking.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetConverter:
    """Convert datasets between different formats."""

    @staticmethod
    def csv_to_json(
        csv_path: str,
        output_path: str,
        code_column: str = "code",
        label_column: str = "label",
        id_column: str | None = None,
        cwe_column: str | None = None,
    ) -> None:
        """
        Convert CSV dataset to JSON format.

        Args:
            csv_path (str): Path to input CSV file
            output_path (str): Path to output JSON file
            code_column (str): Name of code column
            label_column (str): Name of label column
            id_column (Optional[str]): Name of ID column
            cwe_column (Optional[str]): Name of CWE type column
        """
        df: pd.DataFrame = pd.read_csv(csv_path)

        samples: list[dict[str, Any]] = []
        for idx, row in df.iterrows():
            sample: dict[str, Any] = {
                "id": row[id_column]
                if id_column and id_column in df.columns
                else f"sample_{idx:06d}",
                "code": row[code_column],
                "label": int(row[label_column])
                if str(row[label_column]).isdigit()
                else row[label_column],
                "metadata": {
                    col: row[col]
                    for col in df.columns
                    if col not in [code_column, label_column, id_column, cwe_column]
                },
            }

            if cwe_column and cwe_column in df.columns and pd.notna(row[cwe_column]):
                sample["cwe_type"] = row[cwe_column]
            else:
                sample["cwe_type"] = None

            samples.append(sample)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        print(f"Converted {len(samples)} samples from {csv_path} to {output_path}")

    @staticmethod
    def json_to_csv(json_path: str, output_path: str) -> None:
        """Convert JSON dataset to CSV format."""
        with open(json_path, "r", encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)

        # Flatten the data
        rows: list[dict[str, Any]] = []
        for item in data:
            row: dict[str, Any] = {
                "id": item["id"],
                "code": item["code"],
                "label": item["label"],
                "cwe_type": item.get("cwe_type"),
                "severity": item.get("severity"),
            }

            # Add metadata fields
            if "metadata" in item:
                for key, value in item["metadata"].items():
                    row[f"meta_{key}"] = value

            rows.append(row)

        df: pd.DataFrame = pd.DataFrame(rows)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"Converted {len(rows)} samples from {json_path} to {output_path}")


class DatasetAnalyzer:
    """Analyze dataset characteristics and statistics."""

    def __init__(self, dataset_path: str):
        self.dataset_path: str = dataset_path
        self.data: list[dict[str, Any]] = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        """Load dataset from file."""
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_basic_stats(self) -> dict[str, Any]:
        """Get basic dataset statistics."""
        df: pd.DataFrame = pd.DataFrame(self.data)

        stats: dict[str, Any] = {
            "total_samples": len(self.data),
            "unique_labels": list(df["label"].unique()),
            "label_distribution": df["label"].value_counts().to_dict(),
            "avg_code_length": df["code"].str.len().mean(),
            "code_length_stats": {
                "min": df["code"].str.len().min(),
                "max": df["code"].str.len().max(),
                "median": df["code"].str.len().median(),
                "std": df["code"].str.len().std(),
            },
        }

        # CWE analysis if available
        if "cwe_type" in df.columns:
            cwe_counts: pd.Series = df["cwe_type"].value_counts(dropna=False)
            stats["cwe_distribution"] = cwe_counts.to_dict()

        # Language analysis if available in metadata
        languages: list[str] = []
        for item in self.data:
            if "metadata" in item and "language" in item["metadata"]:
                languages.append(item["metadata"]["language"])

        if languages:
            lang_counts: pd.Series = pd.Series(languages).value_counts()
            stats["language_distribution"] = lang_counts.to_dict()

        return stats

    def analyze_class_balance(self) -> dict[str, dict[str, float] | float]:
        """Analyze class balance in the dataset."""
        df: pd.DataFrame = pd.DataFrame(self.data)
        label_counts: pd.Series = df["label"].value_counts()
        total: int = len(df)

        balance_info: dict[str, dict[str, float] | float] = {}
        for label, count in label_counts.items():
            balance_info[str(label)] = {
                "count": float(count),
                "percentage": (float(count) / float(total)) * 100.0,
            }

        # Calculate imbalance ratio
        max_class: float = float(label_counts.max())
        min_class: float = float(label_counts.min())
        balance_info["imbalance_ratio"] = (
            max_class / min_class if min_class > 0 else float("inf")
        )

        return balance_info

    def find_duplicates(self) -> list[dict[str, Any]]:
        """Find duplicate code samples."""
        df: pd.DataFrame = pd.DataFrame(self.data)

        # Find duplicates based on code
        duplicates: pd.DataFrame = df[df.duplicated(subset=["code"], keep=False)]

        duplicate_groups: list[dict[str, Any]] = []
        for code in duplicates["code"].unique():
            group: pd.DataFrame = df[df["code"] == code]
            duplicate_groups.append(
                {
                    "code_snippet": code[:100] + "..." if len(code) > 100 else code,
                    "count": len(group),
                    "samples": group[["id", "label"]].to_dict("records"),
                }
            )

        return duplicate_groups

    def generate_report(self, output_path: str | None = None) -> dict[str, Any]:
        """Generate comprehensive dataset analysis report."""
        report: dict[str, Any] = {
            "dataset_path": self.dataset_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "basic_stats": self.get_basic_stats(),
            "class_balance": self.analyze_class_balance(),
            "duplicates": self.find_duplicates(),
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            print(f"Analysis report saved to: {output_path}")

        return report


class ResultsAnalyzer:
    """Analyze benchmark results and generate insights."""

    def __init__(self, results_dir: str):
        self.results_dir: Path = Path(results_dir)
        self.reports: list[dict[str, Any]] = self._load_all_reports()

    def _load_all_reports(self) -> list[dict[str, Any]]:
        """Load all benchmark report files."""
        reports: list[dict[str, Any]] = []
        for report_file in self.results_dir.glob("benchmark_report_*.json"):
            with open(report_file, "r", encoding="utf-8") as f:
                reports.append(json.load(f))
        return reports

    def compare_models(self) -> pd.DataFrame:
        """Compare performance across different models."""
        comparison_data: list[dict[str, Any]] = []

        for report in self.reports:
            info = report["benchmark_info"]
            metrics = report["metrics"]

            row: dict[str, Any] = {
                "model_name": info["model_name"],
                "task_type": info["task_type"],
                "total_samples": info["total_samples"],
                "accuracy": metrics.get("accuracy", 0),
                "processing_time": info["total_time_seconds"],
            }

            # Add binary classification metrics if available
            if "f1_score" in metrics:
                row.update(
                    {
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "true_positives": metrics["true_positives"],
                        "false_positives": metrics["false_positives"],
                        "false_negatives": metrics["false_negatives"],
                    }
                )

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def analyze_error_patterns(self) -> dict[str, Any]:
        """Analyze error patterns across models."""
        error_analysis: dict[str, Any] = {}

        for report in self.reports:
            model_name = report["benchmark_info"]["model_name"]
            predictions = report["predictions"]

            errors: list[dict[str, Any]] = []
            for pred in predictions:
                if pred["predicted_label"] != pred["true_label"]:
                    errors.append(
                        {
                            "sample_id": pred["sample_id"],
                            "true_label": pred["true_label"],
                            "predicted_label": pred["predicted_label"],
                            "response_text": pred["response_text"][
                                :200
                            ],  # Truncate for readability
                        }
                    )

            error_analysis[model_name] = {
                "total_errors": len(errors),
                "error_rate": len(errors) / len(predictions) if predictions else 0,
                "sample_errors": errors[:10],  # First 10 errors for inspection
            }

        return error_analysis

    def generate_comparison_report(self, output_path: str) -> None:
        """Generate a comprehensive comparison report."""
        comparison_df: pd.DataFrame = self.compare_models()
        error_analysis: dict[str, Any] = self.analyze_error_patterns()

        report: dict[str, Any] = {
            "comparison_timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.reports),
            "model_comparison": comparison_df.to_dict("records"),
            "performance_summary": {
                "best_accuracy": {
                    "model": comparison_df.loc[
                        comparison_df["accuracy"].idxmax(), "model_name"
                    ],
                    "score": comparison_df["accuracy"].max(),
                },
                "fastest_model": {
                    "model": comparison_df.loc[
                        comparison_df["processing_time"].idxmin(), "model_name"
                    ],
                    "time": comparison_df["processing_time"].min(),
                },
            },
            "error_analysis": error_analysis,
        }

        if "f1_score" in comparison_df.columns:
            report["performance_summary"]["best_f1"] = {
                "model": comparison_df.loc[
                    comparison_df["f1_score"].idxmax(), "model_name"
                ],
                "score": comparison_df["f1_score"].max(),
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"Comparison report saved to: {output_path}")


def main() -> None:
    """Command line interface for data processing utilities."""
    parser = argparse.ArgumentParser(
        description="Data processing utilities for LLM benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert dataset formats")
    convert_parser.add_argument("input_file", help="Input file path")
    convert_parser.add_argument("output_file", help="Output file path")
    convert_parser.add_argument(
        "--format",
        choices=["csv2json", "json2csv"],
        required=True,
        help="Conversion format",
    )
    convert_parser.add_argument(
        "--code-column", default="code", help="Code column name for CSV"
    )
    convert_parser.add_argument(
        "--label-column", default="label", help="Label column name for CSV"
    )
    convert_parser.add_argument("--id-column", help="ID column name for CSV")
    convert_parser.add_argument("--cwe-column", help="CWE column name for CSV")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze dataset")
    analyze_parser.add_argument("dataset_path", help="Path to dataset JSON file")
    analyze_parser.add_argument("--output", help="Output file for analysis report")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument(
        "results_dir", help="Directory containing benchmark results"
    )
    compare_parser.add_argument(
        "--output",
        default="./comparison_report.json",
        help="Output file for comparison report",
    )

    args = parser.parse_args()

    if args.command == "convert":
        if args.format == "csv2json":
            DatasetConverter.csv_to_json(
                args.input_file,
                args.output_file,
                args.code_column,
                args.label_column,
                args.id_column,
                args.cwe_column,
            )
        elif args.format == "json2csv":
            DatasetConverter.json_to_csv(args.input_file, args.output_file)

    elif args.command == "analyze":
        dataset_analyzer: DatasetAnalyzer = DatasetAnalyzer(args.dataset_path)
        report: dict[str, Any] = dataset_analyzer.generate_report(args.output)

        # Print summary to console
        print("\nDataset Analysis Summary:")
        print(f"Total samples: {report['basic_stats']['total_samples']}")
        print(f"Labels: {report['basic_stats']['unique_labels']}")
        print(f"Label distribution: {report['basic_stats']['label_distribution']}")
        print(
            f"Average code length: {report['basic_stats']['avg_code_length']:.1f} characters"
        )

        if report["duplicates"]:
            print(f"Found {len(report['duplicates'])} duplicate groups")

    elif args.command == "compare":
        results_analyzer: ResultsAnalyzer = ResultsAnalyzer(args.results_dir)
        results_analyzer.generate_comparison_report(args.output)

        # Print summary
        comparison_df: pd.DataFrame = results_analyzer.compare_models()
        print("\nModel Comparison Summary:")
        print(
            comparison_df[["model_name", "task_type", "accuracy"]].to_string(
                index=False
            )
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

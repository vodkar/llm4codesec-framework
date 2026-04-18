from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NotRequired, TypedDict, cast


FINAL_ANSWER_MARKER: str = "[[FINAL_ANSWER: "
DEFAULT_RESULTS_DIR: Path = Path("results")


class InferenceData(TypedDict, total=False):
    responses: list[object]


class Prediction(TypedDict):
    sample_id: NotRequired[object]
    inference_data: NotRequired[InferenceData]


class BenchmarkReport(TypedDict):
    predictions: NotRequired[list[object]]


def get_mapping_value(mapping: dict[object, object], key: str) -> object | None:
    value: object | None = mapping.get(key)
    return value


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Scan benchmark_report_*.json files and print report path plus sample id "
            "for predictions whose responses do not contain the final-answer marker."
        )
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory to scan recursively. Defaults to ./results.",
    )
    return parser.parse_args()


def iter_report_paths(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("benchmark_report_*.json"))


def load_json_file(report_path: Path) -> BenchmarkReport:
    with report_path.open("r", encoding="utf-8") as report_file:
        raw_data: object = json.load(report_file)

    if not isinstance(raw_data, dict):
        return BenchmarkReport()

    raw_mapping: dict[object, object] = cast(dict[object, object], raw_data)
    predictions: object | None = get_mapping_value(raw_mapping, "predictions")
    if isinstance(predictions, list):
        return BenchmarkReport(predictions=cast(list[object], predictions))

    return BenchmarkReport()


def prediction_has_final_answer_marker(prediction: Prediction) -> bool:
    inference_data: InferenceData | None = prediction.get("inference_data")
    if not isinstance(inference_data, dict):
        return False

    responses: list[object] | None = inference_data.get("responses")
    if not isinstance(responses, list):
        return False

    for response in responses:
        if isinstance(response, str) and FINAL_ANSWER_MARKER in response:
            return True

    return False


def find_missing_markers(report_path: Path) -> tuple[list[str], int]:
    report_data: BenchmarkReport = load_json_file(report_path)
    predictions: list[object] | None = report_data.get("predictions")
    if not isinstance(predictions, list):
        return [], 0

    missing_sample_ids: list[str] = []
    for raw_prediction in predictions:
        if not isinstance(raw_prediction, dict):
            continue

        prediction_mapping: dict[object, object] = cast(dict[object, object], raw_prediction)
        sample_id_value: object | None = get_mapping_value(prediction_mapping, "sample_id")
        inference_data_value: object | None = get_mapping_value(prediction_mapping, "inference_data")

        prediction: Prediction = Prediction(sample_id=sample_id_value)
        if isinstance(inference_data_value, dict):
            prediction["inference_data"] = InferenceData()

            inference_mapping: dict[object, object] = cast(dict[object, object], inference_data_value)
            responses_value: object | None = get_mapping_value(inference_mapping, "responses")
            if isinstance(responses_value, list):
                prediction["inference_data"]["responses"] = responses_value

        sample_id: object | None = prediction.get("sample_id")
        sample_id_str: str = str(sample_id) if sample_id is not None else "<missing-sample-id>"
        if not prediction_has_final_answer_marker(prediction):
            missing_sample_ids.append(sample_id_str)

    return missing_sample_ids, len(predictions)


def main() -> int:
    args: argparse.Namespace = parse_args()
    results_dir: Path = args.results_dir.resolve()

    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}", file=sys.stderr)
        return 1

    if not results_dir.is_dir():
        print(f"Path is not a directory: {results_dir}", file=sys.stderr)
        return 1

    exit_code: int = 0
    scanned_reports: int = 0
    reports_with_missing_answers: int = 0
    total_missing_answers: int = 0
    total_predictions_scanned: int = 0
    for report_path in iter_report_paths(results_dir):
        scanned_reports += 1
        try:
            missing_sample_ids, prediction_count = find_missing_markers(report_path)
        except json.JSONDecodeError:
            print(f"Invalid JSON: {report_path}", file=sys.stderr)
            exit_code = 1
            continue
        except OSError:
            print(f"Failed to read: {report_path}", file=sys.stderr)
            exit_code = 1
            continue

        total_predictions_scanned += prediction_count

        if missing_sample_ids:
            reports_with_missing_answers += 1
            total_missing_answers += len(missing_sample_ids)

        for sample_id in missing_sample_ids:
            try:
                print(f"{report_path}\t{sample_id}")
            except BrokenPipeError:
                return 0

    missing_answer_percentage: float = 0.0
    if total_predictions_scanned > 0:
        missing_answer_percentage = (total_missing_answers / total_predictions_scanned) * 100.0

    print(
        (
            "Summary: "
            f"scanned_reports={scanned_reports}, "
            f"reports_with_missing_answers={reports_with_missing_answers}, "
            f"total_missing_answers={total_missing_answers}, "
            f"total_predictions_scanned={total_predictions_scanned}, "
            f"missing_answer_percentage={missing_answer_percentage:.2f}%"
        ),
        file=sys.stderr,
    )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
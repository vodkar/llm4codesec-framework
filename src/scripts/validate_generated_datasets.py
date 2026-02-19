import json
from pathlib import Path


def main() -> None:
    roots = [
        Path("datasets_processed/castle"),
        Path("datasets_processed/cvefixes"),
        Path("datasets_processed/jitvul"),
        Path("datasets_processed/vulbench"),
        Path("datasets_processed/vuldetectbench"),
        Path("datasets_processed/vulnerabilitydetection"),
    ]

    problems: list[str] = []
    summary: list[tuple[str, int, list[str]]] = []

    for root in roots:
        if not root.exists():
            problems.append(f"Missing output dir: {root}")
            continue

        for file_path in sorted(root.glob("*.json")):
            if (
                file_path.name.endswith("_stats.json")
                or file_path.name.endswith("_statistics.json")
                or file_path.name.endswith("_summary.json")
            ):
                continue

            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as exc:
                problems.append(f"{file_path}: invalid JSON: {exc}")
                continue

            if not isinstance(data, dict) or "samples" not in data:
                problems.append(f"{file_path}: missing top-level samples")
                continue

            samples = data.get("samples", [])
            if not isinstance(samples, list):
                problems.append(f"{file_path}: samples is not list")
                continue

            if len(samples) == 0:
                problems.append(f"{file_path}: empty samples")
                continue

            missing_required = 0
            bad_code = 0
            bad_cwe = 0

            for sample in samples[:2000]:
                if not isinstance(sample, dict):
                    missing_required += 1
                    continue

                for key in ("id", "code", "label"):
                    if key not in sample:
                        missing_required += 1
                        break

                code = sample.get("code")
                if not isinstance(code, str) or not code.strip():
                    bad_code += 1

                cwe_types = sample.get("cwe_types")
                if isinstance(cwe_types, list):
                    for cwe in cwe_types:
                        cwe_str = str(cwe)
                        if cwe_str != "SAFE" and not cwe_str.startswith("CWE-"):
                            bad_cwe += 1

            labels = [
                sample.get("label") for sample in samples if isinstance(sample, dict)
            ]
            unique_labels = sorted({str(label) for label in labels})[:8]
            summary.append((str(file_path), len(samples), unique_labels))

            if missing_required:
                problems.append(
                    f"{file_path}: {missing_required} samples missing id/code/label"
                )
            if bad_code:
                problems.append(f"{file_path}: {bad_code} samples with empty code")
            if bad_cwe:
                problems.append(f"{file_path}: {bad_cwe} invalid cwe_types entries")

            if "vuldetectbench_task1" in file_path.name:
                non_binary = [label for label in labels if label not in (0, 1)]
                if non_binary:
                    problems.append(f"{file_path}: task1 has non-binary labels")

            if "cvefixes_cwe_" in file_path.name:
                int_labels = {label for label in labels if isinstance(label, int)}
                if not ({0, 1} <= int_labels):
                    problems.append(
                        f"{file_path}: CWE-specific file does not contain both 0 and 1 labels"
                    )

    print("=== SUMMARY ===")
    for file_name, count, labels in summary[:80]:
        print(f"{file_name} | samples={count} | labels={labels}")
    if len(summary) > 80:
        print(f"... ({len(summary) - 80} additional files omitted)")

    print("\n=== PROBLEMS ===")
    if problems:
        for problem in problems:
            print(f"- {problem}")
    else:
        print("No structural problems found.")


if __name__ == "__main__":
    main()

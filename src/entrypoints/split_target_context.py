#!/usr/bin/env python3
"""
Build a target/context-split dataset from an llm_scanner context dataset.

Each output sample's ``code`` is the function under analysis taken verbatim
(comments included) from the function-only dataset, and ``context`` is the
context snippet with the target's lines removed. The prompt renders the context
after the target as reference-only code (see ``benchmark.target_context``).

Root static findings are resolved against the original context snippet and
stored as ``root_static_findings``; ``metadata.target_line_coverage`` records the
fraction of target lines found (and removed) in the context snippet.

Samples are matched on ``(metadata.source_row_ids, label)``. Every context sample
must match exactly one function-only sample, otherwise nothing is written.

Usage (host):
    PYTHONPATH=src uv run python src/entrypoints/split_target_context.py \\
        --context-dataset benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json \\
        --function-dataset benchmarks/context-assembler-dataset/cleanvul_python_matched.json \\
        --output datasets_processed/context_assembler/cpg_structural_target_split.json
"""

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

from benchmark.static_findings import root_findings_from_raw
from benchmark.target_context import remove_target_lines
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)

SampleKey = tuple[tuple[int, ...], int]


def _sample_key(sample: dict[str, Any]) -> SampleKey:
    """Join key that survives dataset rebuilds (sample ids do not)."""
    row_ids: list[int] | None = sample.get("metadata", {}).get("source_row_ids")
    if not row_ids:
        raise ValueError(f"Sample {sample.get('id')} has no metadata.source_row_ids")
    return tuple(int(row_id) for row_id in row_ids), int(sample["label"])


def split_target_context(
    context_payload: dict[str, Any], function_payload: dict[str, Any]
) -> dict[str, Any]:
    """Return ``context_payload`` with every sample split into target ``code`` and ``context``.

    Raises:
        ValueError: On a duplicate function-only key or a context sample with no
            matching function-only sample.
    """
    function_by_key: dict[SampleKey, dict[str, Any]] = {}
    for function_sample in function_payload["samples"]:
        key: SampleKey = _sample_key(function_sample)
        if key in function_by_key:
            raise ValueError(f"Duplicate function-only key {key}")
        function_by_key[key] = function_sample

    samples: list[dict[str, Any]] = []
    for context_sample in context_payload["samples"]:
        key = _sample_key(context_sample)
        function_sample = function_by_key.get(key)
        if function_sample is None:
            raise ValueError(
                f"No function-only sample for key {key} (context {context_sample.get('id')})"
            )
        target: str = function_sample["code"]
        context, coverage = remove_target_lines(context_sample["code"], target)
        split: dict[str, Any] = {
            key_: value
            for key_, value in context_sample.items()
            if key_ not in ("code", "static_findings")
        }
        split["code"] = target
        split["context"] = context
        split["metadata"] = {**context_sample["metadata"], "target_line_coverage": coverage}
        raw_findings: list[dict[str, Any]] | None = context_sample.get("static_findings")
        if raw_findings is not None:
            findings = root_findings_from_raw(context_sample["code"], raw_findings)
            split["root_static_findings"] = [f.model_dump(mode="json") for f in findings]
        samples.append(split)
    return {**context_payload, "samples": samples}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--context-dataset", type=Path, required=True, help="llm_scanner context dataset JSON"
    )
    parser.add_argument(
        "--function-dataset", type=Path, required=True,
        help="Function-only dataset JSON supplying the target code",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output dataset JSON")
    args = parser.parse_args(argv)

    context_payload: dict[str, Any] = json.loads(args.context_dataset.read_text(encoding="utf-8"))
    function_payload: dict[str, Any] = json.loads(
        args.function_dataset.read_text(encoding="utf-8")
    )
    result: dict[str, Any] = split_target_context(context_payload, function_payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    coverages: list[float] = [s["metadata"]["target_line_coverage"] for s in result["samples"]]
    _LOGGER.info(
        "Wrote %d samples to %s (target line coverage: mean %.3f, full %d, below 0.9 %d; "
        "empty context %d)",
        len(coverages), args.output, statistics.mean(coverages),
        sum(c == 1.0 for c in coverages), sum(c < 0.9 for c in coverages),
        sum(not s["context"] for s in result["samples"]),
    )


if __name__ == "__main__":
    setup_logging()
    main()

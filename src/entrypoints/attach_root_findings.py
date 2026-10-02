#!/usr/bin/env python3
"""
Attach root static findings to a function-only dataset.

Copies the ``is_root`` static-analysis findings of a context dataset (built by
llm_scanner with static findings) onto the matching samples of a function-only
dataset, as ``root_static_findings``. The findings are stored for analysis only;
they reach a prompt only when the dataset config sets ``render_root_findings``.

Samples are matched on ``(metadata.source_row_ids, label)``. Every target sample
must match exactly one source sample, otherwise nothing is written.

With ``--as-root`` (for a v2 source with a ``roots`` field) the function-only code
also becomes a single root, labelled with the source roots' file paths, so the
baseline renders in the same CODE UNDER ANALYSIS layout as the v2 context datasets.

Usage (host):
    PYTHONPATH=src uv run python src/entrypoints/attach_root_findings.py \\
        --target benchmarks/context-assembler-dataset/cleanvul_python_matched.json \\
        --findings-source benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json \\
        --output datasets_processed/context_assembler/cleanvul_python_matched_root_findings.json

    PYTHONPATH=src uv run python src/entrypoints/attach_root_findings.py --as-root \\
        --target benchmarks/context-assembler-dataset/v2/cleanvul_python_matched.json \\
        --findings-source benchmarks/context-assembler-dataset/v2/context_assembler_cpg_structural.json \\
        --output datasets_processed/context_assembler/v2/cleanvul_python_matched_root_findings.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from benchmark.static_findings import root_findings_from_raw
from logging_tools import setup_logging

_LOGGER = logging.getLogger(__name__)

SampleKey = tuple[tuple[int, ...], int]


def _sample_key(sample: dict[str, Any]) -> SampleKey:
    """Join key that survives dataset rebuilds (sample ids do not)."""
    row_ids: list[int] | None = sample.get("metadata", {}).get("source_row_ids")
    if not row_ids:
        raise ValueError(f"Sample {sample.get('id')} has no metadata.source_row_ids")
    return tuple(int(row_id) for row_id in row_ids), int(sample["label"])


def _single_root(code: str, source: dict[str, Any]) -> dict[str, Any]:
    """Wrap function-only ``code`` as one root, labelled with the source roots' files."""
    raw_roots: list[dict[str, Any]] | None = source.get("roots")
    if not raw_roots:
        raise ValueError(f"Findings-source sample {source.get('id')} has no roots")
    file_paths: list[str] = list(dict.fromkeys(str(root["file_path"]) for root in raw_roots))
    return {"file_path": ", ".join(file_paths), "code": code}


def attach_root_findings(
    target_payload: dict[str, Any], source_payload: dict[str, Any], as_root: bool = False
) -> dict[str, Any]:
    """Return ``target_payload`` with ``root_static_findings`` added to every sample.

    With ``as_root`` each sample also gets a single-root ``roots`` field.

    Raises:
        ValueError: On a duplicate source key, a source sample without
            ``static_findings`` (or, with ``as_root``, without ``roots``), or a
            target sample with no matching source.
    """
    source_by_key: dict[SampleKey, dict[str, Any]] = {}
    for source in source_payload["samples"]:
        key: SampleKey = _sample_key(source)
        if key in source_by_key:
            raise ValueError(f"Duplicate findings-source key {key}")
        if source.get("static_findings") is None:
            raise ValueError(f"Findings-source sample {source.get('id')} has no static_findings")
        source_by_key[key] = source

    samples: list[dict[str, Any]] = []
    for target in target_payload["samples"]:
        key = _sample_key(target)
        source = source_by_key.get(key)
        if source is None:
            raise ValueError(f"No findings-source sample for key {key} (target {target.get('id')})")
        findings = root_findings_from_raw(source["code"], source["static_findings"])
        sample: dict[str, Any] = {
            **target,
            "root_static_findings": [f.model_dump(mode="json") for f in findings],
        }
        if as_root:
            sample["roots"] = [_single_root(target["code"], source)]
        samples.append(sample)
    return {**target_payload, "samples": samples}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--target", type=Path, required=True, help="Function-only dataset JSON")
    parser.add_argument(
        "--findings-source", type=Path, required=True,
        help="Context dataset JSON carrying static_findings",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output dataset JSON")
    parser.add_argument(
        "--as-root", action="store_true",
        help="Also wrap the code as a single root (v2 root/context prompt layout)",
    )
    args = parser.parse_args(argv)

    target_payload: dict[str, Any] = json.loads(args.target.read_text(encoding="utf-8"))
    source_payload: dict[str, Any] = json.loads(args.findings_source.read_text(encoding="utf-8"))
    result: dict[str, Any] = attach_root_findings(target_payload, source_payload, as_root=args.as_root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with_findings: int = sum(bool(s["root_static_findings"]) for s in result["samples"])
    _LOGGER.info(
        "Wrote %d samples (%d with root findings) to %s",
        len(result["samples"]), with_findings, args.output,
    )


if __name__ == "__main__":
    setup_logging()
    main()

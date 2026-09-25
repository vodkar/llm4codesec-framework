"""Plain-python checks: root static findings extraction and prompt rendering.

Run: PYTHONPATH=src uv run python tests/test_root_static_findings.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.static_findings import (
    ROOT_FINDINGS_PLACEHOLDER,
    RootStaticFinding,
    render_root_findings_block,
    root_findings_from_raw,
)

CODE = "def f(cmd):\n    x = 1\n    subprocess.call(cmd, shell=True)\n    assert x"


def _raw(**overrides) -> dict:
    finding = {
        "tool": "bandit",
        "rule_id": "B602",
        "cwe_id": 78,
        "severity": "HIGH",
        "message": "subprocess call with shell=True identified, security issue.",
        "file_path": "pkg/mod.py",
        "repo_line": 42,
        "snippet_line": 3,
        "is_root": True,
    }
    finding.update(overrides)
    return finding


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_placeholder_constant() -> None:
    assert ROOT_FINDINGS_PLACEHOLDER == "{root_static_findings}"


def test_keeps_only_root_findings_sorted_with_flagged_line() -> None:
    raw = [
        _raw(tool="dlint", rule_id="DUO116", snippet_line=3),
        _raw(snippet_line=4, rule_id="B101", severity="LOW", cwe_id=703),
        _raw(is_root=False),
        {k: v for k, v in _raw().items() if k != "is_root"},
        _raw(),
    ]
    findings = root_findings_from_raw(CODE, raw)
    assert [(f.tool, f.rule_id) for f in findings] == [
        ("bandit", "B602"),
        ("dlint", "DUO116"),
        ("bandit", "B101"),
    ]
    assert findings[0].flagged_line == "subprocess.call(cmd, shell=True)"
    assert findings[2].flagged_line == "assert x"
    assert findings[0].repo_line == 42
    assert findings[0].file_path == "pkg/mod.py"


def test_snippet_line_out_of_range_raises() -> None:
    _raises(ValueError, lambda: root_findings_from_raw(CODE, [_raw(snippet_line=5)]))
    _raises(ValueError, lambda: root_findings_from_raw(CODE, [_raw(snippet_line=0)]))


def test_empty_input_gives_empty_list() -> None:
    assert root_findings_from_raw(CODE, []) == []


def test_render_with_findings() -> None:
    findings = [
        RootStaticFinding(
            tool="bandit", rule_id="B602", cwe_id=78, severity="HIGH",
            message="subprocess call with shell=True identified, security issue.",
            file_path="a.py", repo_line=1, flagged_line="subprocess.call(cmd, shell=True)",
        ),
        RootStaticFinding(
            tool="dlint", rule_id="DUO116", message="use of \"shell=True\" is insecure",
            file_path="a.py", repo_line=1, flagged_line="subprocess.call(cmd, shell=True)",
        ),
    ]
    assert render_root_findings_block(findings) == (
        "\n\nStatic analyzer findings in the function under analysis "
        "(automated tools; may be false positives):\n"
        "1. [bandit B602 | CWE-78 | HIGH] subprocess call with shell=True identified, security issue.\n"
        "   Flagged line: subprocess.call(cmd, shell=True)\n"
        "2. [dlint DUO116] use of \"shell=True\" is insecure\n"
        "   Flagged line: subprocess.call(cmd, shell=True)"
    )


def test_render_without_findings() -> None:
    assert render_root_findings_block([]) == (
        "\n\nStatic analyzer findings in the function under analysis: none reported."
    )


def test_render_collapses_multiline_messages() -> None:
    finding = RootStaticFinding(
        tool="semgrep", rule_id="python.x", message="Line one.\n   Line   two.\n",
        file_path="a.py", repo_line=1, flagged_line="x = 1",
    )
    assert "1. [semgrep python.x] Line one. Line two.\n" in render_root_findings_block([finding])


from benchmark.models import BenchmarkSample


def _sample_dict(**extra) -> dict:
    return {"id": "s1", "code": CODE, "label": 1, "metadata": {"source_row_ids": [7]}, **extra}


def test_sample_extracts_root_findings_from_static_findings() -> None:
    sample = BenchmarkSample.model_validate(
        _sample_dict(static_findings=[_raw(), _raw(is_root=False)], source_map=[])
    )
    assert sample.root_static_findings is not None
    assert [f.rule_id for f in sample.root_static_findings] == ["B602"]


def test_sample_without_static_findings_has_none() -> None:
    assert BenchmarkSample.model_validate(_sample_dict()).root_static_findings is None
    assert (
        BenchmarkSample.model_validate(_sample_dict(static_findings=None)).root_static_findings
        is None
    )


def test_sample_with_no_root_findings_has_empty_list() -> None:
    sample = BenchmarkSample.model_validate(_sample_dict(static_findings=[_raw(is_root=False)]))
    assert sample.root_static_findings == []


def test_sample_loads_precomputed_root_findings() -> None:
    precomputed = RootStaticFinding(
        tool="bandit", rule_id="B101", message="m", file_path="a.py",
        repo_line=3, flagged_line="assert x",
    ).model_dump(mode="json")
    sample = BenchmarkSample.model_validate(_sample_dict(root_static_findings=[precomputed]))
    assert sample.root_static_findings == [RootStaticFinding.model_validate(precomputed)]


def test_real_context_dataset_root_findings() -> None:
    import json

    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/context-assembler-dataset/context_assembler_cpg_structural.json"
    )
    if not path.exists():
        print(f"SKIP: {path} missing")
        return
    samples = [
        BenchmarkSample.model_validate(raw)
        for raw in json.loads(path.read_text())["samples"]
    ]
    assert len(samples) == 732
    assert sum(len(s.root_static_findings or []) for s in samples) == 626
    assert sum(bool(s.root_static_findings) for s in samples) == 183


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")

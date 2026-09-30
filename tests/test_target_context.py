"""Plain-python checks: context samples split into a commented target plus reference context,
and function-only prompts stay unchanged.

Run: PYTHONPATH=src uv run python tests/test_target_context.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.benchmark_runner import user_template_values
from benchmark.models import BenchmarkSample
from benchmark.target_context import (
    remove_target_lines,
    render_context_block,
    strip_python_comments,
    top_level_definitions,
)
from entrypoints.split_target_context import main, split_target_context

TARGET = (
    "def f(cmd):\n"
    "    # only called with trusted input\n"
    "    args = [cmd]  # list form\n"
    "\n"
    "    subprocess.call(args)\n"
    "\n"
    "def g(x):\n"
    "    y = x + 1\n"
    "    return y"
)
# llm_scanner renders without comments or blank lines, and not in target order.
CONTEXT = (
    "import subprocess\n"
    "def g(x):\n"
    "    y = x + 1\n"
    "    return y\n"
    "def caller(request):\n"
    "    f(request.args['cmd'])\n"
    "def f(cmd):\n"
    "    args = [cmd]\n"
    "    subprocess.call(args)"
)


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_strip_comments_keeps_hash_in_strings() -> None:
    lines = strip_python_comments("x = '#not a comment'  # comment")
    assert lines == ["x = '#not a comment'  "]


def test_strip_comments_falls_back_on_untokenizable_code() -> None:
    assert strip_python_comments("def f(:\n    '''open") == ["def f(:", "    '''open"]


def test_definitions_keep_decorators_with_their_function() -> None:
    chunks = top_level_definitions("@a\n@b\ndef f():\n    pass\nclass C:\n    x = 1")
    assert chunks == [["@a", "@b", "def f():", "pass"], ["class C:", "x = 1"]]


def test_remove_target_lines_handles_reordered_definitions() -> None:
    remaining, coverage = remove_target_lines(CONTEXT, TARGET)
    assert coverage == 1.0
    assert remaining == "import subprocess\ndef caller(request):\n    f(request.args['cmd'])"


def test_remove_target_lines_ignores_single_line_matches() -> None:
    remaining, coverage = remove_target_lines("def h():\n    return y", "def g(x):\n    return y")
    assert remaining == "def h():\n    return y"
    assert coverage == 0.0


def test_render_context_block_empty() -> None:
    assert render_context_block(None) == ""
    assert render_context_block("  \n") == ""


def test_template_values_without_context_unchanged() -> None:
    sample = BenchmarkSample(id="1", code=TARGET, label=1, metadata={})
    assert user_template_values(sample, render_root_findings=False)["code"] == TARGET


def test_template_values_render_context_after_target() -> None:
    sample = BenchmarkSample(id="1", code=TARGET, label=1, metadata={}, context="def caller(): ...")
    code = user_template_values(sample, render_root_findings=False)["code"]
    assert code.startswith(TARGET + "\n\n")
    assert "for reference only" in code
    assert code.endswith("\n\ndef caller(): ...")


def _context_sample(row_ids: list[int], label: int) -> dict:
    finding = {
        "tool": "bandit", "rule_id": "B603", "cwe_id": 78, "severity": "LOW",
        "message": "subprocess", "file_path": "a.py", "repo_line": 9,
        "snippet_line": 9, "is_root": True,
    }
    return {"id": f"ctx-{row_ids}-{label}", "code": CONTEXT, "label": label,
            "metadata": {"source_row_ids": row_ids}, "static_findings": [finding],
            "source_map": []}


def _function_sample(row_ids: list[int], label: int) -> dict:
    return {"id": f"fo-{row_ids}-{label}", "code": TARGET, "label": label,
            "metadata": {"source_row_ids": row_ids}}


def test_split_sample_fields() -> None:
    result = split_target_context(
        {"metadata": {"name": "m"}, "samples": [_context_sample([1], 1)]},
        {"samples": [_function_sample([1], 1)]},
    )
    sample = result["samples"][0]
    assert result["metadata"] == {"name": "m"}
    assert sample["code"] == TARGET
    assert sample["context"].startswith("import subprocess")
    assert "static_findings" not in sample
    assert sample["root_static_findings"][0]["flagged_line"] == "subprocess.call(args)"
    assert sample["metadata"]["target_line_coverage"] == 1.0
    loaded = BenchmarkSample(**sample)
    assert loaded.context == sample["context"] and len(loaded.root_static_findings) == 1


def test_split_requires_function_sample() -> None:
    _raises(ValueError, lambda: split_target_context(
        {"samples": [_context_sample([1], 1)]}, {"samples": [_function_sample([1], 0)]}
    ))


def test_split_rejects_duplicate_function_key() -> None:
    _raises(ValueError, lambda: split_target_context(
        {"samples": [_context_sample([1], 1)]},
        {"samples": [_function_sample([1], 1), _function_sample([1], 1)]},
    ))


def test_cli_writes_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "c.json").write_text(json.dumps({"samples": [_context_sample([1], 1)]}))
        (tmp_path / "f.json").write_text(json.dumps({"samples": [_function_sample([1], 1)]}))
        output = tmp_path / "out" / "o.json"
        main(["--context-dataset", str(tmp_path / "c.json"),
              "--function-dataset", str(tmp_path / "f.json"), "--output", str(output)])
        written = json.loads(output.read_text())
        assert written["samples"][0]["code"] == TARGET


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")

"""Plain-python checks: function-only samples get the context dataset's root findings.

Run: PYTHONPATH=src uv run python tests/test_attach_root_findings.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.models import BenchmarkSample
from entrypoints.attach_root_findings import attach_root_findings, main

SOURCE_CODE = "ctx = 1\ndef f(cmd):\n    subprocess.call(cmd, shell=True)"


def _finding(is_root: bool = True) -> dict:
    return {
        "tool": "bandit", "rule_id": "B602", "cwe_id": 78, "severity": "HIGH",
        "message": "shell=True", "file_path": "a.py", "repo_line": 10,
        "snippet_line": 3, "is_root": is_root,
    }


def _source(row_ids: list[int], label: int, findings: list[dict]) -> dict:
    return {"id": f"src-{row_ids}-{label}", "code": SOURCE_CODE, "label": label,
            "metadata": {"source_row_ids": row_ids}, "static_findings": findings}


def _target(row_ids: list[int], label: int) -> dict:
    return {"id": f"fo-{row_ids}-{label}", "code": "def f(cmd): ...", "label": label,
            "metadata": {"source_row_ids": row_ids, "commit_url": "u"}}


def _raises(exc_type: type[BaseException], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_attaches_root_findings_by_row_ids_and_label() -> None:
    source = {"metadata": {}, "samples": [
        _source([1], 1, [_finding(), _finding(is_root=False)]),
        _source([1], 0, []),
    ]}
    target = {"metadata": {"name": "fo"}, "samples": [_target([1], 1), _target([1], 0)]}
    result = attach_root_findings(target, source)
    assert result["metadata"] == {"name": "fo"}
    vuln, fixed = result["samples"]
    assert vuln["code"] == "def f(cmd): ..."
    assert vuln["metadata"]["commit_url"] == "u"
    assert [f["rule_id"] for f in vuln["root_static_findings"]] == ["B602"]
    assert vuln["root_static_findings"][0]["flagged_line"] == "subprocess.call(cmd, shell=True)"
    assert fixed["root_static_findings"] == []
    loaded = BenchmarkSample.model_validate(vuln)
    assert loaded.root_static_findings is not None and loaded.root_static_findings[0].repo_line == 10


def test_unmatched_target_raises() -> None:
    source = {"samples": [_source([1], 1, [])]}
    _raises(ValueError, lambda: attach_root_findings({"samples": [_target([2], 1)]}, source))


def test_duplicate_source_key_raises() -> None:
    source = {"samples": [_source([1], 1, []), _source([1], 1, [])]}
    _raises(ValueError, lambda: attach_root_findings({"samples": [_target([1], 1)]}, source))


def test_source_without_static_findings_raises() -> None:
    bare = _source([1], 1, [])
    del bare["static_findings"]
    _raises(ValueError, lambda: attach_root_findings({"samples": [_target([1], 1)]}, {"samples": [bare]}))


def test_cli_writes_nothing_on_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "t.json").write_text(json.dumps({"samples": [_target([2], 1)]}))
        (tmp_path / "s.json").write_text(json.dumps({"samples": [_source([1], 1, [])]}))
        output = tmp_path / "out" / "o.json"
        _raises(ValueError, lambda: main([
            "--target", str(tmp_path / "t.json"), "--findings-source", str(tmp_path / "s.json"),
            "--output", str(output),
        ]))
        assert not output.exists()


def test_cli_writes_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "t.json").write_text(json.dumps({"samples": [_target([1], 1)]}))
        (tmp_path / "s.json").write_text(json.dumps({"samples": [_source([1], 1, [_finding()])]}))
        output = tmp_path / "out" / "o.json"
        main(["--target", str(tmp_path / "t.json"), "--findings-source", str(tmp_path / "s.json"),
              "--output", str(output)])
        written = json.loads(output.read_text())
        assert len(written["samples"][0]["root_static_findings"]) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")

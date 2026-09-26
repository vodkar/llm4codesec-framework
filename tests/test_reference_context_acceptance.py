"""Plain-python checks for the reference-context oracle's acceptance checks.

Run: PYTHONPATH=src uv run python tests/test_reference_context_acceptance.py
"""

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from analysis.reference_context.acceptance import (
    CheckResult,
    budget_check,
    leakage_probe,
    mismatched_control,
    reproducibility_check,
    run_all,
    sanity_check,
    symmetry_check,
)
from analysis.reference_context.inputs import AnalysisInputs, Condition, ItemRecord
from analysis.reference_context.metrics import (
    ClusterBootstrap,
    condition_table,
    decomposition,
    length_stats,
    stratum_tables,
    stratum_tables_with_redraws,
)
from analysis.reference_context.report import (
    coverage_reconciliation,
    draw_shortfall,
    write_report,
)
from analysis.reference_context.run_config import (
    FrameworkSection,
    Pins,
    ReferenceRunConfig,
)

_VALID_SANITY_PATTERN: list[tuple[int, float]] = [
    (1, 0.9),
    (0, 0.1),
    (1, 0.3),
    (0, 0.7),
]


def _item(
    pair: str,
    label: int,
    score: float,
    repo: str,
    cond: Condition = Condition.CPG,
    stratum: str = "r100",
    prompt_tokens: int = 10,
    token_count: int = 10,
    context_token_count: int | None = None,
) -> ItemRecord:
    """Build a test ``ItemRecord``.

    ``context_token_count`` defaults to ``token_count`` when not given
    explicitly, so callers that only care about ``token_count`` (leakage
    probe, sanity, budget checks) keep a sensible non-target-token value
    without having to pass both.
    """

    return ItemRecord(
        item_id=f"{pair}_{'vuln' if label else 'safe'}",
        pair_id=pair,
        condition=cond,
        label=label,
        repo_url=repo,
        score=score,
        predicted_label=int(score > 0.5),
        success=True,
        prompt_tokens=prompt_tokens,
        token_count=token_count,
        context_token_count=(
            token_count if context_token_count is None else context_token_count
        ),
        budget=10,
        node_count=1,
        file_count=1,
        symbol_count=1,
        symbol_names_hash="h",
        underfill=False,
        stratum=stratum,
    )


def _inputs(items: list[ItemRecord]) -> AnalysisInputs:
    return AnalysisInputs(
        items=items,
        pair_ids=sorted({i.pair_id for i in items}),
        coverage={},
        manifest={},
        inference_exclusions={},
    )


def _synthetic_pairs(
    repos: int, pairs_per_repo: int, seed: int, cond: Condition
) -> list[ItemRecord]:
    rng = random.Random(seed)
    items: list[ItemRecord] = []
    for repo_index in range(repos):
        for pair_index in range(pairs_per_repo):
            pair = f"{repo_index:03d}{pair_index:03d}aaaaaa"
            repo = f"repo{repo_index}"
            items.append(_item(pair, 1, rng.uniform(0.3, 1.0), repo, cond))
            items.append(_item(pair, 0, rng.uniform(0.0, 0.7), repo, cond))
    return items


def _valid_probe_items(
    cond: Condition, repos: int = 4, per_repo: int = 3
) -> list[ItemRecord]:
    """Constant-feature items for one condition: always passes the leakage probe."""

    items: list[ItemRecord] = []
    for repo_index in range(repos):
        for pair_index in range(per_repo):
            pair = f"{cond.value}{repo_index:02d}{pair_index:02d}aaa"
            repo = f"repo{repo_index}"
            items.append(_item(pair, 1, 0.5, repo, cond))
            items.append(_item(pair, 0, 0.5, repo, cond))
    return items


def _sanity_pattern_items(cond: Condition) -> list[ItemRecord]:
    """A small two-class, both-endpoints-agree pattern for one condition."""

    return [
        _item(f"{cond.value}{i:03d}", label, score, "r", cond)
        for i, (label, score) in enumerate(_VALID_SANITY_PATTERN)
    ]


def test_leakage_probe_detects_injected_leak_and_passes_shuffled() -> None:
    rng = random.Random(0)
    leaky, clean = [], []
    for i in range(80):
        for label in (0, 1):
            pair, repo = f"{i:012d}", f"r{i % 10}"
            leak_tokens = 100 + 50 * label + rng.randint(0, 5)
            clean_tokens = 100 + rng.randint(0, 50)
            for cond in Condition:
                leaky.append(
                    _item(pair, label, 0.5, repo, cond).model_copy(
                        update={"token_count": leak_tokens}
                    )
                )
                clean.append(
                    _item(pair, label, 0.5, repo, cond).model_copy(
                        update={"token_count": clean_tokens}
                    )
                )
    assert leakage_probe(_inputs(leaky), seed=0).passed is False
    assert leakage_probe(_inputs(clean), seed=0).passed is True


def test_leakage_probe_constant_features_get_auc_half_without_fitting() -> None:
    items = [
        _item(f"{i:012d}", label, 0.5, f"r{i % 5}", cond)
        for cond in Condition
        for i in range(20)
        for label in (0, 1)
    ]

    def _raise_fit(self: LogisticRegression, *args: object, **kwargs: object) -> None:
        raise AssertionError(
            "LogisticRegression.fit must not be called for constant features"
        )

    original_fit = LogisticRegression.fit
    LogisticRegression.fit = _raise_fit  # type: ignore[method-assign]
    try:
        result = leakage_probe(_inputs(items), seed=0)
    finally:
        LogisticRegression.fit = original_fit  # type: ignore[method-assign]

    assert result.passed is True
    assert result.value == 0.5
    for cond in Condition:
        assert result.detail["per_condition"][cond.value] == 0.5


def test_leakage_probe_records_none_and_fails_with_fewer_than_two_repos() -> None:
    items = [
        _item(f"{i:012d}", label, 0.5, "only-repo")
        for i in range(5)
        for label in (0, 1)
    ]
    result = leakage_probe(_inputs(items), seed=0)
    assert result.passed is False
    assert result.value is None
    assert result.detail["per_condition"][Condition.CPG.value] is None
    assert "notes" in result.detail


def test_leakage_and_sanity_fail_on_empty_inputs() -> None:
    empty = _inputs([])

    leakage_result = leakage_probe(empty, seed=0)
    assert leakage_result.passed is False
    assert leakage_result.value is None
    assert list(leakage_result.detail["per_condition"]) == [c.value for c in Condition]
    assert all(v is None for v in leakage_result.detail["per_condition"].values())
    assert set(leakage_result.detail["notes"]) == {c.value for c in Condition}

    sanity_result = sanity_check(empty)
    assert sanity_result.passed is False
    assert sanity_result.value is None
    assert all(
        row["endpoint_precision"] is None
        for row in sanity_result.detail["per_condition"].values()
    )
    assert set(sanity_result.detail["notes"]) == {c.value for c in Condition}


def test_leakage_probe_missing_condition_is_noted_and_fails() -> None:
    items = [
        item
        for cond in Condition
        if cond is not Condition.MISMATCHED
        for item in _valid_probe_items(cond)
    ]
    result = leakage_probe(_inputs(items), seed=0)
    assert result.passed is False
    assert result.detail["per_condition"][Condition.MISMATCHED.value] is None
    assert Condition.MISMATCHED.value in result.detail["notes"]
    assert (
        result.detail["notes"][Condition.MISMATCHED.value]
        == "no items for this condition"
    )
    # Present, well-formed conditions still get a real (passing) value.
    assert result.detail["per_condition"][Condition.CPG.value] == 0.5


def test_leakage_probe_single_label_condition_fails_without_crash() -> None:
    single_label_items = [
        _item(f"single{repo:02d}{pair:02d}aaaa", 1, 0.5, f"repo{repo}", Condition.CPG)
        for repo in range(4)
        for pair in range(3)
    ]
    items = single_label_items + [
        item
        for cond in Condition
        if cond is not Condition.CPG
        for item in _valid_probe_items(cond)
    ]
    result = leakage_probe(_inputs(items), seed=0)
    assert result.passed is False
    assert result.detail["per_condition"][Condition.CPG.value] is None
    assert "single label" in result.detail["notes"][Condition.CPG.value]
    # Other conditions are unaffected and still individually reportable.
    assert result.detail["per_condition"][Condition.RANDOM.value] == 0.5


def test_leakage_probe_length_ranking_diagnostic_reported_without_failing() -> None:
    rng = random.Random(0)
    items: list[ItemRecord] = []
    for i in range(80):
        baseline = 50 + rng.randint(0, 450)
        repo = f"r{i % 10}"
        pair = f"{i:012d}"
        for cond in Condition:
            items.append(
                _item(pair, 1, 0.5, repo, cond).model_copy(
                    update={"token_count": baseline + 3}
                )
            )
            items.append(
                _item(pair, 0, 0.5, repo, cond).model_copy(
                    update={"token_count": baseline}
                )
            )
    result = leakage_probe(_inputs(items), seed=0)
    assert result.passed is True
    for cond in Condition:
        assert result.detail["length_ranking_accuracy"][cond.value] == 1.0
        assert result.detail["mean_token_delta"][cond.value] == 3.0
        auc = result.detail["per_condition"][cond.value]
        assert auc is not None and auc <= 0.55


def test_sanity_endpoint_equals_base_rate() -> None:
    items = [
        _item(f"{i:012d}", label, score, "r", cond)
        for cond in Condition
        for i, (label, score) in enumerate(_VALID_SANITY_PATTERN)
    ]
    assert sanity_check(_inputs(items)).passed


def test_sanity_check_flags_condition_that_cannot_be_validated() -> None:
    items = [_item(f"{i:012d}", 1, 0.5 + 0.01 * i, "r") for i in range(3)]
    result = sanity_check(_inputs(items))
    assert result.passed is False
    assert "notes" in result.detail


def test_sanity_check_missing_condition_is_noted_and_fails() -> None:
    items = [
        item
        for cond in Condition
        if cond is not Condition.RANDOM
        for item in _sanity_pattern_items(cond)
    ]
    result = sanity_check(_inputs(items))
    assert result.passed is False
    assert (
        result.detail["per_condition"][Condition.RANDOM.value]["endpoint_precision"]
        is None
    )
    assert Condition.RANDOM.value in result.detail["notes"]
    assert (
        result.detail["per_condition"][Condition.CPG.value]["endpoint_precision"]
        is not None
    )


def test_sanity_check_single_label_condition_fails_without_crash() -> None:
    single_label_items = [
        _item(f"{i:012d}", 1, 0.5 + 0.01 * i, "r", Condition.CPG) for i in range(3)
    ]
    items = single_label_items + [
        item
        for cond in Condition
        if cond is not Condition.CPG
        for item in _sanity_pattern_items(cond)
    ]
    result = sanity_check(_inputs(items))
    assert result.passed is False
    assert (
        result.detail["per_condition"][Condition.CPG.value]["endpoint_precision"]
        is None
    )
    assert Condition.CPG.value in result.detail["notes"]


def test_symmetry_flags_hash_mismatch() -> None:
    items = [
        _item("aaaaaaaaaaaa", 1, 0.5, "r", Condition.REFERENCE),
        _item("aaaaaaaaaaaa", 0, 0.5, "r", Condition.REFERENCE).model_copy(
            update={"symbol_names_hash": "other"}
        ),
    ]
    result = symmetry_check(_inputs(items))
    assert result.passed is False and result.detail["violators"] == ["aaaaaaaaaaaa"]


def test_symmetry_check_passes_when_symmetric_across_covered_conditions() -> None:
    covered = (Condition.REFERENCE, Condition.RANDOM, Condition.MISMATCHED)
    items = [
        _item(pair, label, 0.5, "r", cond)
        for pair in ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
        for cond in covered
        for label in (0, 1)
    ]
    result = symmetry_check(_inputs(items))
    assert result.value == 1.0
    assert result.passed is True
    assert result.detail["violators"] == []


def test_symmetry_check_flags_context_token_ratio_violation() -> None:
    items = [
        _item(
            "aaaaaaaaaaaa", 1, 0.5, "r", Condition.REFERENCE, context_token_count=100
        ),
        _item("aaaaaaaaaaaa", 0, 0.5, "r", Condition.REFERENCE, context_token_count=94),
    ]
    result = symmetry_check(_inputs(items))
    assert result.passed is False
    assert result.value == 0.0
    assert result.detail["violators"] == ["aaaaaaaaaaaa"]


def test_symmetry_check_ignores_target_length_uses_context_tokens() -> None:
    """Target lengths (``token_count``) differ a lot, but the non-target

    context tokens are equal on both halves: the pair is symmetric.
    """

    covered = (Condition.REFERENCE, Condition.RANDOM, Condition.MISMATCHED)
    items = [
        _item(
            "aaaaaaaaaaaa",
            label,
            0.5,
            "r",
            cond,
            token_count=500 if label else 50,
            context_token_count=20,
        )
        for cond in covered
        for label in (0, 1)
    ]
    result = symmetry_check(_inputs(items))
    assert result.passed is True
    assert result.detail["violators"] == []


def test_symmetry_check_missing_context_token_count_is_a_violator() -> None:
    vuln = _item("aaaaaaaaaaaa", 1, 0.5, "r", Condition.REFERENCE).model_copy(
        update={"context_token_count": None}
    )
    safe = _item("aaaaaaaaaaaa", 0, 0.5, "r", Condition.REFERENCE)
    result = symmetry_check(_inputs([vuln, safe]))
    assert result.passed is False
    assert result.detail["violators"] == ["aaaaaaaaaaaa"]
    assert (
        result.detail["notes"]["aaaaaaaaaaaa:reference"]
        == "missing context_token_count"
    )


def test_symmetry_check_both_context_tokens_zero_is_symmetric() -> None:
    covered = (Condition.REFERENCE, Condition.RANDOM, Condition.MISMATCHED)
    items = [
        _item("aaaaaaaaaaaa", label, 0.5, "r", cond, context_token_count=0)
        for cond in covered
        for label in (0, 1)
    ]
    result = symmetry_check(_inputs(items))
    assert result.passed is True
    assert result.detail["violators"] == []


def test_symmetry_check_missing_half_in_covered_condition_is_a_violator() -> None:
    covered = (Condition.REFERENCE, Condition.RANDOM, Condition.MISMATCHED)
    items = [
        _item(pair, label, 0.5, "r", cond)
        for pair in ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
        for cond in covered
        for label in (0, 1)
    ]
    # Drop only the "safe" half of one pair's RANDOM-condition entry; every
    # other (pair, condition) combination keeps both halves and is symmetric.
    items = [
        item
        for item in items
        if not (
            item.pair_id == "bbbbbbbbbbbb"
            and item.condition is Condition.RANDOM
            and item.label == 0
        )
    ]
    result = symmetry_check(_inputs(items))
    assert result.passed is False
    assert result.detail["violators"] == ["bbbbbbbbbbbb"]
    assert result.value == 5 / 6
    assert result.detail["n_checked"] == 6


def test_budget_check_median_deviation() -> None:
    cpg = [
        _item(f"{i:012d}", 1, 0.5, "r", Condition.CPG).model_copy(
            update={"token_count": 100}
        )
        for i in range(3)
    ]
    ref = [
        c.model_copy(update={"condition": Condition.REFERENCE, "token_count": t})
        for c, t in zip(cpg, (100, 97, 60))
    ]
    result = budget_check(_inputs(cpg + ref))
    assert result.value == 1.0 and result.passed
    assert result.detail["median_relative_shortfall"] == 0.03
    assert result.detail["n_violations"] == 0
    assert result.detail["violators"] == []


def _budget_pair(
    pair: str, cpg_tokens: int, ref_tokens: int, ref_context: int | None
) -> list[ItemRecord]:
    cpg = _item(pair, 1, 0.5, "r", Condition.CPG, token_count=cpg_tokens)
    ref = cpg.model_copy(
        update={
            "condition": Condition.REFERENCE,
            "token_count": ref_tokens,
            "context_token_count": ref_context,
        }
    )
    return [cpg, ref]


def test_budget_check_violation_only_when_reference_has_context() -> None:
    items = [
        *_budget_pair("aaaaaaaaaaaa", 100, 101, 5),
        *_budget_pair("bbbbbbbbbbbb", 100, 100, 5),
        *_budget_pair("cccccccccccc", 100, 50, 5),
    ]
    result = budget_check(_inputs(items))
    assert result.passed is False
    assert result.value == 2 / 3
    assert result.detail["n_violations"] == 1
    assert result.detail["violators"] == ["aaaaaaaaaaaa_vuln"]


def test_budget_check_targets_only_overshoot_passes() -> None:
    items = [
        *_budget_pair("aaaaaaaaaaaa", 100, 400, 0),
        *_budget_pair("bbbbbbbbbbbb", 100, 90, 10),
    ]
    result = budget_check(_inputs(items))
    assert result.passed is True
    assert result.value == 1.0
    assert result.detail["n_violations"] == 0


def test_budget_check_missing_context_count_overshoot_violates() -> None:
    result = budget_check(_inputs(_budget_pair("aaaaaaaaaaaa", 100, 101, None)))
    assert result.passed is False
    assert result.detail["violators"] == ["aaaaaaaaaaaa_vuln"]


def test_budget_check_underfill_does_not_fail() -> None:
    items = [
        *_budget_pair("aaaaaaaaaaaa", 100, 10, 3),
        *_budget_pair("bbbbbbbbbbbb", 100, 20, 0),
    ]
    items = [
        i.model_copy(update={"underfill": True})
        if i.condition == Condition.REFERENCE
        else i
        for i in items
    ]
    result = budget_check(_inputs(items))
    assert result.passed is True
    assert result.detail["underfill_rate"] == 1.0
    assert abs(result.detail["median_relative_shortfall"] - 0.85) < 1e-9


def test_budget_check_lists_at_most_50_violators() -> None:
    items = [item for i in range(60) for item in _budget_pair(f"{i:012d}", 10, 11, 1)]
    result = budget_check(_inputs(items))
    assert result.detail["n_violations"] == 60
    assert len(result.detail["violators"]) == 50
    assert result.value == 0.0


def test_budget_check_underfill_rate() -> None:
    cpg = [
        _item(f"{i:012d}", 1, 0.5, "r", Condition.CPG).model_copy(
            update={"token_count": 100}
        )
        for i in range(4)
    ]
    ref = [
        c.model_copy(
            update={
                "condition": Condition.REFERENCE,
                "token_count": 99,
                "underfill": i < 2,
            }
        )
        for i, c in enumerate(cpg)
    ]
    result = budget_check(_inputs(cpg + ref))
    assert result.detail["underfill_rate"] == 0.5
    assert result.passed is True


def test_budget_check_fails_when_reference_item_has_no_cpg_match() -> None:
    cpg = [
        _item(f"{i:012d}", 1, 0.5, "r", Condition.CPG).model_copy(
            update={"token_count": 100}
        )
        for i in range(3)
    ]
    ref = [
        c.model_copy(update={"condition": Condition.REFERENCE, "token_count": t})
        for c, t in zip(cpg, (100, 97, 60))
    ]
    extra_ref = _item("999999999999", 1, 0.5, "r", Condition.REFERENCE)
    result = budget_check(_inputs(cpg + ref + [extra_ref]))
    assert result.passed is False
    assert result.value == 1.0
    assert result.detail["n_unmatched_or_zero_cpg"] == 1
    assert "note" in result.detail


def test_budget_check_fails_when_cpg_token_count_is_zero() -> None:
    cpg = [
        _item("aaaaaaaaaaaa", 1, 0.5, "r", Condition.CPG).model_copy(
            update={"token_count": 0}
        )
    ]
    ref = [
        cpg[0].model_copy(update={"condition": Condition.REFERENCE, "token_count": 5})
    ]
    result = budget_check(_inputs(cpg + ref))
    assert result.passed is False
    assert result.value is None
    assert result.detail["n_unmatched_or_zero_cpg"] == 1


def test_mismatched_control_passes_when_identical() -> None:
    items = [
        _item(f"{i:012d}", label, 0.3 + 0.4 * label, f"r{i % 4}", Condition.MISMATCHED)
        for i in range(12)
        for label in (0, 1)
    ]
    none_items = [i.model_copy(update={"condition": Condition.NONE}) for i in items]
    boot = ClusterBootstrap(
        {Condition.MISMATCHED: items, Condition.NONE: none_items}, resamples=50, seed=3
    )
    result = mismatched_control(boot, tolerance=0.02)
    assert result.passed is True
    assert result.value == 0.0


def test_mismatched_control_fails_when_mismatched_diverges() -> None:
    none_items: list[ItemRecord] = []
    mismatched_items: list[ItemRecord] = []
    for repo_index in range(6):
        for pair_index in range(3):
            pair = f"{repo_index:03d}{pair_index:03d}aaaaaa"
            repo = f"repo{repo_index}"
            none_items.append(_item(pair, 1, 0.5, repo, Condition.NONE))
            none_items.append(_item(pair, 0, 0.5, repo, Condition.NONE))
            mismatched_items.append(_item(pair, 1, 1.0, repo, Condition.MISMATCHED))
            mismatched_items.append(_item(pair, 0, 0.0, repo, Condition.MISMATCHED))
    boot = ClusterBootstrap(
        {Condition.MISMATCHED: mismatched_items, Condition.NONE: none_items},
        resamples=200,
        seed=1,
    )
    result = mismatched_control(boot, tolerance=0.02)
    assert result.passed is False
    assert result.value == 0.5


def _pins() -> Pins:
    return Pins(llm_scanner_path=Path("/tmp/llm_scanner"), llm_scanner_git_sha="abc123")


def test_reproducibility_check_passes_when_everything_matches() -> None:
    manifest = {
        "llm_scanner_git_sha": "abc123",
        "llm_scanner_dirty": False,
        "config_sha256": "deadbeef",
    }
    result = reproducibility_check(
        manifest, _pins(), "deadbeef", "framework_sha_1", False
    )
    assert result.passed is True
    assert result.detail["framework_sha"] == "framework_sha_1"


def test_reproducibility_check_fails_on_sha_mismatch() -> None:
    manifest = {
        "llm_scanner_git_sha": "other",
        "llm_scanner_dirty": False,
        "config_sha256": "deadbeef",
    }
    result = reproducibility_check(
        manifest, _pins(), "deadbeef", "framework_sha_1", False
    )
    assert result.passed is False


def test_reproducibility_check_fails_on_dirty_flags() -> None:
    manifest = {
        "llm_scanner_git_sha": "abc123",
        "llm_scanner_dirty": True,
        "config_sha256": "deadbeef",
    }
    ok_manifest = {**manifest, "llm_scanner_dirty": False}
    assert (
        reproducibility_check(manifest, _pins(), "deadbeef", "fw", False).passed
        is False
    )
    assert (
        reproducibility_check(ok_manifest, _pins(), "deadbeef", "fw", True).passed
        is False
    )


def test_reproducibility_check_missing_manifest_key_does_not_raise() -> None:
    result = reproducibility_check({}, _pins(), "deadbeef", "fw", False)
    assert result.passed is False
    assert result.value is None
    assert "note" in result.detail


def test_run_all_returns_six_checks_in_spec_order() -> None:
    by_condition = {
        cond: _synthetic_pairs(6, 3, 42 + i, cond) for i, cond in enumerate(Condition)
    }
    items = [item for cond_items in by_condition.values() for item in cond_items]
    inputs = AnalysisInputs(
        items=items,
        pair_ids=sorted({i.pair_id for i in items}),
        coverage={},
        manifest={
            "llm_scanner_git_sha": "sha123",
            "llm_scanner_dirty": False,
            "config_sha256": "cfgsha",
        },
        inference_exclusions={},
    )
    bootstrap = ClusterBootstrap(by_condition, resamples=30, seed=1)
    config = ReferenceRunConfig(
        pins=Pins(llm_scanner_path=Path("/tmp/x"), llm_scanner_git_sha="sha123"),
        llm_scanner={},
        framework=FrameworkSection(
            experiments_config=Path("e.json"),
            datasets_config=Path("d.json"),
            results_dir=Path("r"),
            datasets_dir=Path("d"),
            bootstrap_seed=1,
            probe_seed=7,
        ),
    )
    results = run_all(inputs, bootstrap, config, "cfgsha", "fwsha", False)
    assert [r.name for r in results] == [
        "leakage_probe",
        "mismatched_control",
        "symmetry_check",
        "budget_check",
        "sanity_check",
        "reproducibility_check",
    ]
    assert results[5].passed is True


def _valid_report_setup() -> tuple[
    AnalysisInputs, ClusterBootstrap, ReferenceRunConfig
]:
    """Synthetic 5-condition inputs designed so every acceptance check passes.

    Every item shares constant leakage-probe features (so the probe short
    circuits to AUC 0.5), the same ``token_count``/``symbol_names_hash`` on
    both halves of a pair (so symmetry and budget hold exactly), and the
    MISMATCHED items are an exact copy of the NONE items (so the mismatched
    control's difference is exactly 0). The manifest matches the pins and
    config sha used by the caller.
    """

    none_items: list[ItemRecord] = _synthetic_pairs(6, 3, seed=100, cond=Condition.NONE)
    mismatched_items: list[ItemRecord] = [
        item.model_copy(update={"condition": Condition.MISMATCHED})
        for item in none_items
    ]
    cpg_items: list[ItemRecord] = _synthetic_pairs(6, 3, seed=200, cond=Condition.CPG)
    reference_items: list[ItemRecord] = [
        item.model_copy(update={"condition": Condition.REFERENCE}) for item in cpg_items
    ]
    random_items: list[ItemRecord] = _synthetic_pairs(
        6, 3, seed=300, cond=Condition.RANDOM
    )

    by_condition: dict[Condition, list[ItemRecord]] = {
        Condition.NONE: none_items,
        Condition.MISMATCHED: mismatched_items,
        Condition.CPG: cpg_items,
        Condition.REFERENCE: reference_items,
        Condition.RANDOM: random_items,
    }
    items: list[ItemRecord] = [
        item for group in by_condition.values() for item in group
    ]
    inputs = AnalysisInputs(
        items=items,
        pair_ids=sorted({item.pair_id for item in items}),
        coverage={
            "csv_rows": 100,
            "eligible_pairs": 90,
            "s1_survivors": 85,
            "s1_excluded": {"checkout_failed": 5},
            "s2_input_pairs": 85,
            "s2_excluded": {"donor_unavailable": 3},
            "evaluated_pairs": 82,
            "s2_mismatched_targets_only": 2,
        },
        manifest={
            "llm_scanner_git_sha": "sha123",
            "llm_scanner_dirty": False,
            "config_sha256": "cfgsha",
        },
        inference_exclusions={"filtered_by_token_limit": 1},
    )
    bootstrap = ClusterBootstrap(by_condition, resamples=30, seed=1)
    config = ReferenceRunConfig(
        pins=Pins(llm_scanner_path=Path("/tmp/x"), llm_scanner_git_sha="sha123"),
        llm_scanner={},
        framework=FrameworkSection(
            experiments_config=Path("e.json"),
            datasets_config=Path("d.json"),
            results_dir=Path("r"),
            datasets_dir=Path("d"),
            bootstrap_seed=1,
            probe_seed=7,
        ),
    )
    return inputs, bootstrap, config


def _write_report_for(
    inputs: AnalysisInputs, bootstrap: ClusterBootstrap, config: ReferenceRunConfig
) -> tuple[Path, list[CheckResult]]:
    tables = condition_table(inputs, bootstrap)
    strata = stratum_tables(
        inputs, config.framework.bootstrap_resamples, config.framework.bootstrap_seed
    )
    lengths = length_stats(inputs)
    decomp = decomposition(bootstrap)
    checks = run_all(inputs, bootstrap, config, "cfgsha", "fwsha", False)

    tmp_dir = Path(tempfile.mkdtemp())
    report_path = write_report(
        tmp_dir,
        inputs,
        tables,
        strata,
        lengths,
        decomp,
        checks,
        {"framework_sha": "fwsha"},
    )
    return report_path, checks


def test_write_report_valid_run_has_no_invalid_header() -> None:
    inputs, bootstrap, config = _valid_report_setup()
    report_path, checks = _write_report_for(inputs, bootstrap, config)
    assert all(check.passed for check in checks), [
        (c.name, c.passed, c.detail) for c in checks if not c.passed
    ]

    report_text = report_path.read_text(encoding="utf-8")
    assert not report_text.startswith("# RUN INVALID")
    assert report_text.startswith("## Coverage")

    out_dir = report_path.parent
    acceptance = json.loads((out_dir / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["run_valid"] is True
    assert len(acceptance["checks"]) == 6

    report_json = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report_json["evaluated_pairs_in_analysis"] == len(inputs.pair_ids)


def test_write_report_one_failing_check_has_invalid_header() -> None:
    inputs, bootstrap, config = _valid_report_setup()
    broken_manifest = {**inputs.manifest, "llm_scanner_git_sha": "wrong-sha"}
    inputs = inputs.model_copy(update={"manifest": broken_manifest})

    report_path, checks = _write_report_for(inputs, bootstrap, config)
    failed_names = [check.name for check in checks if not check.passed]
    assert failed_names == ["reproducibility_check"]

    report_text = report_path.read_text(encoding="utf-8")
    assert report_text.startswith("# RUN INVALID")
    assert "reproducibility_check" in report_text.splitlines()[2]

    out_dir = report_path.parent
    acceptance = json.loads((out_dir / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["run_valid"] is False


def test_draw_shortfall_counts_items_with_missing_draws() -> None:
    inputs, _, _ = _valid_report_setup()
    first = inputs.items[0]
    items = [
        item.model_copy(update={"scored_draws": 1 if item is first else 3})
        for item in inputs.items
    ]
    shortfall = draw_shortfall(
        inputs.model_copy(update={"items": items, "draws_per_item": 3})
    )
    assert shortfall["draws_per_item"] == 3
    per_condition = shortfall["per_condition"]
    assert isinstance(per_condition, dict)
    assert per_condition[first.condition.value] == 1
    assert sum(per_condition.values()) == 1
    unknown = draw_shortfall(inputs.model_copy(update={"draws_per_item": None}))
    assert set(unknown["per_condition"].values()) == {None}


def test_coverage_reconciliation_flags_mismatch() -> None:
    assert coverage_reconciliation({"evaluated_pairs": 5}, {"unscored": 1}, 4) == {
        "coverage_evaluated_pairs": 5,
        "analysed_plus_excluded": 5,
        "matches": True,
    }
    assert coverage_reconciliation({"evaluated_pairs": 9}, {}, 4)["matches"] is False
    assert coverage_reconciliation({}, {}, 4)["matches"] is None


def test_stratum_redraws_reported_per_stratum() -> None:
    inputs, _, config = _valid_report_setup()
    tables, redraws = stratum_tables_with_redraws(
        inputs, config.framework.bootstrap_resamples, config.framework.bootstrap_seed
    )
    assert list(tables) == list(redraws)
    bootstrapped = {stratum for stratum, count in redraws.items() if count is not None}
    assert bootstrapped, redraws
    assert all(isinstance(redraws[stratum], int) for stratum in bootstrapped)


if __name__ == "__main__":
    test_leakage_probe_detects_injected_leak_and_passes_shuffled()
    test_leakage_probe_constant_features_get_auc_half_without_fitting()
    test_leakage_probe_records_none_and_fails_with_fewer_than_two_repos()
    test_leakage_and_sanity_fail_on_empty_inputs()
    test_leakage_probe_missing_condition_is_noted_and_fails()
    test_leakage_probe_single_label_condition_fails_without_crash()
    test_leakage_probe_length_ranking_diagnostic_reported_without_failing()
    test_sanity_endpoint_equals_base_rate()
    test_sanity_check_flags_condition_that_cannot_be_validated()
    test_sanity_check_missing_condition_is_noted_and_fails()
    test_sanity_check_single_label_condition_fails_without_crash()
    test_symmetry_flags_hash_mismatch()
    test_symmetry_check_passes_when_symmetric_across_covered_conditions()
    test_symmetry_check_flags_context_token_ratio_violation()
    test_symmetry_check_ignores_target_length_uses_context_tokens()
    test_symmetry_check_missing_context_token_count_is_a_violator()
    test_symmetry_check_both_context_tokens_zero_is_symmetric()
    test_symmetry_check_missing_half_in_covered_condition_is_a_violator()
    test_budget_check_median_deviation()
    test_budget_check_underfill_rate()
    test_budget_check_fails_when_reference_item_has_no_cpg_match()
    test_budget_check_fails_when_cpg_token_count_is_zero()
    test_budget_check_violation_only_when_reference_has_context()
    test_budget_check_targets_only_overshoot_passes()
    test_budget_check_missing_context_count_overshoot_violates()
    test_budget_check_underfill_does_not_fail()
    test_budget_check_lists_at_most_50_violators()
    test_mismatched_control_passes_when_identical()
    test_mismatched_control_fails_when_mismatched_diverges()
    test_reproducibility_check_passes_when_everything_matches()
    test_reproducibility_check_fails_on_sha_mismatch()
    test_reproducibility_check_fails_on_dirty_flags()
    test_reproducibility_check_missing_manifest_key_does_not_raise()
    test_run_all_returns_six_checks_in_spec_order()
    test_write_report_valid_run_has_no_invalid_header()
    test_write_report_one_failing_check_has_invalid_header()
    test_draw_shortfall_counts_items_with_missing_draws()
    test_coverage_reconciliation_flags_mismatch()
    test_stratum_redraws_reported_per_stratum()
    print("ALL PASSED")

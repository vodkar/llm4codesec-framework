"""Plain-python checks for reference-context metrics and cluster bootstrap.

Run: PYTHONPATH=src uv run python tests/test_reference_context_metrics.py
"""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.reference_context.inputs import AnalysisInputs, Condition, ItemRecord
from analysis.reference_context.metrics import (
    ClusterBootstrap,
    ConfidenceInterval,
    MetricName,
    base_rate,
    condition_table,
    decomposition,
    length_stats,
    point_metrics,
    stratum_tables,
)
from benchmark.metrics_calculator import BinaryMetricsCalculator
from benchmark.models import PredictionResult


def _item(
    pair: str,
    label: int,
    score: float,
    repo: str,
    cond: Condition = Condition.CPG,
    stratum: str = "r100",
    prompt_tokens: int = 10,
    token_count: int = 10,
) -> ItemRecord:
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


def _all_conditions(items: list[ItemRecord]) -> list[ItemRecord]:
    return [i.model_copy(update={"condition": c}) for c in Condition for i in items]


def _synthetic_pairs(
    repos: int, pairs_per_repo: int, seed: int, stratum: str = "r100"
) -> list[ItemRecord]:
    rng = random.Random(seed)
    items: list[ItemRecord] = []
    for repo_index in range(repos):
        for pair_index in range(pairs_per_repo):
            pair = f"{stratum}{repo_index:03d}{pair_index:03d}"
            repo = f"{stratum}-repo{repo_index}"
            items.append(_item(pair, 1, rng.uniform(0.3, 1.0), repo, stratum=stratum))
            items.append(_item(pair, 0, rng.uniform(0.0, 0.7), repo, stratum=stratum))
    return items


def test_point_metrics_hand_computed() -> None:
    items = [
        _item("a", 1, 0.9, "r1"),
        _item("a", 0, 0.2, "r1"),
        _item("b", 1, 0.4, "r2"),
        _item("b", 0, 0.6, "r2"),
    ]
    metrics = point_metrics(items)
    assert metrics[MetricName.ROC_AUC] == 0.75
    assert metrics[MetricName.PAIRED_RANKING_ACCURACY] == 0.5


def test_point_metrics_pr_family_and_mcc_hand_computed() -> None:
    # Ranked: 0.9(1) 0.6(0) 0.4(1) 0.2(0); verdicts: TP, FP, FN, TN.
    items = [
        _item("a", 1, 0.9, "r1"),
        _item("a", 0, 0.2, "r1"),
        _item("b", 1, 0.4, "r2"),
        _item("b", 0, 0.6, "r2"),
    ]
    metrics = point_metrics(items)
    assert math.isclose(metrics[MetricName.PR_AUC], 0.5 * 1.0 + 0.5 * (2 / 3))
    assert metrics[MetricName.P_AT_R_50] == 1.0
    assert math.isclose(metrics[MetricName.P_AT_R_80], 2 / 3)
    assert math.isclose(metrics[MetricName.P_AT_R_90], 2 / 3)
    assert math.isclose(metrics[MetricName.P_AT_R_95], 2 / 3)
    assert metrics[MetricName.MCC] == 0.0
    assert set(metrics) == set(MetricName)


def test_point_metrics_single_class_yields_none() -> None:
    items = [_item("a", 1, 0.9, "r1"), _item("b", 1, 0.4, "r2")]
    metrics = point_metrics(items)
    assert all(value is None for value in metrics.values())


def test_point_metrics_rejects_duplicate_items() -> None:
    items = [_item("a", 1, 0.9, "r1"), _item("a", 0, 0.2, "r1")]
    try:
        point_metrics(items + items)
    except ValueError:
        return
    raise AssertionError("duplicate item ids must be rejected")


def test_paired_ranking_parity_with_framework_calculator() -> None:
    items = [
        _item("a", 1, 0.9, "r1"),
        _item("a", 0, 0.9, "r1"),
        _item("b", 1, 0.7, "r2"),
        _item("b", 0, 0.1, "r2"),
    ]
    preds = [
        PredictionResult(
            sample_id=i.item_id,
            predicted_label=i.predicted_label,
            true_label=i.label,
            confidence=None,
            binary_label_confidence=i.score,
            response_text="",
            processing_time=0.0,
            is_success=True,
            error_message=None,
        )
        for i in items
    ]
    framework = (
        BinaryMetricsCalculator().calculate(preds).summary["paired_ranking_accuracy"]
    )
    assert point_metrics(items)[MetricName.PAIRED_RANKING_ACCURACY] == framework


def test_all_metrics_parity_with_framework_calculator_on_random_data() -> None:
    items = _synthetic_pairs(repos=7, pairs_per_repo=6, seed=11)
    # Force some exact ties so the tie rule is exercised.
    items = [
        i.model_copy(update={"score": 0.5}) if i.pair_id.endswith(("000", "003")) else i
        for i in items
    ]
    preds = [
        PredictionResult(
            sample_id=i.item_id,
            predicted_label=i.predicted_label,
            true_label=i.label,
            confidence=None,
            binary_label_confidence=i.score,
            response_text="",
            processing_time=0.0,
            is_success=True,
            error_message=None,
        )
        for i in items
    ]
    summary = BinaryMetricsCalculator().calculate(preds).summary
    ours = point_metrics(items)
    keys = {
        MetricName.ROC_AUC: "roc_auc",
        MetricName.PR_AUC: "pr_auc",
        MetricName.PAIRED_RANKING_ACCURACY: "paired_ranking_accuracy",
        MetricName.MCC: "mcc",
        MetricName.P_AT_R_50: "precision_at_recall_50",
        MetricName.P_AT_R_80: "precision_at_recall_80",
        MetricName.P_AT_R_90: "precision_at_recall_90",
        MetricName.P_AT_R_95: "precision_at_recall_95",
    }
    for metric, key in keys.items():
        assert ours[metric] == summary[key], (metric, ours[metric], summary[key])


def test_cluster_bootstrap_covers_truth_and_widens_with_repo_correlation() -> None:
    rng = random.Random(0)
    items: list[ItemRecord] = []
    for repo_index in range(30):
        shift = rng.uniform(-0.3, 0.3)
        for pair_index in range(5):
            pair = f"{repo_index:03d}{pair_index:03d}aaaaaa"
            items.append(
                _item(
                    pair,
                    1,
                    min(1, max(0, 0.6 + shift + rng.gauss(0, 0.1))),
                    f"repo{repo_index}",
                )
            )
            items.append(
                _item(
                    pair,
                    0,
                    min(1, max(0, 0.4 + shift + rng.gauss(0, 0.1))),
                    f"repo{repo_index}",
                )
            )
    clustered = ClusterBootstrap({Condition.CPG: items}, resamples=400, seed=1)
    iid_items = [i.model_copy(update={"repo_url": i.pair_id}) for i in items]
    iid = ClusterBootstrap({Condition.CPG: iid_items}, resamples=400, seed=1)
    ci = clustered.interval(Condition.CPG, MetricName.ROC_AUC)
    ci_iid = iid.interval(Condition.CPG, MetricName.ROC_AUC)
    assert ci.low is not None and ci.high is not None and ci.low <= ci.point <= ci.high
    assert (ci.high - ci.low) > (ci_iid.high - ci_iid.low)


def test_paired_difference_of_identical_conditions_is_zero() -> None:
    items = [
        _item(f"{i:012d}", lab, 0.3 + 0.4 * lab, f"r{i % 4}")
        for i in range(12)
        for lab in (0, 1)
    ]
    same = [i.model_copy(update={"condition": Condition.NONE}) for i in items]
    boot = ClusterBootstrap(
        {Condition.CPG: items, Condition.NONE: same}, resamples=100, seed=3
    )
    diff = boot.difference(Condition.CPG, Condition.NONE, MetricName.ROC_AUC)
    assert diff.point == diff.low == diff.high == 0.0


def test_replicates_equal_point_metrics_on_explicit_resample() -> None:
    items = _synthetic_pairs(repos=5, pairs_per_repo=3, seed=5)
    boot = ClusterBootstrap({Condition.CPG: items}, resamples=20, seed=9)
    by_repo: dict[str, list[ItemRecord]] = {}
    for item in items:
        by_repo.setdefault(item.repo_url, []).append(item)
    for metric in MetricName:
        replicates = boot.replicates(Condition.CPG, metric)
        assert len(replicates) == 20
        for draw_index, repos in enumerate(boot.repo_draws):
            # Duplicated repos keep their pairs as distinct pairs (renamed copies).
            explicit = [
                i.model_copy(
                    update={
                        "pair_id": f"{copy}:{i.pair_id}",
                        "item_id": f"{copy}:{i.item_id}",
                    }
                )
                for copy, repo in enumerate(repos)
                for i in by_repo[repo]
            ]
            expected = point_metrics(explicit)[metric]
            got = replicates[draw_index]
            assert (got is None and expected is None) or math.isclose(got, expected), (
                metric,
                got,
                expected,
            )


def test_repo_draws_are_seeded_and_shared_across_conditions() -> None:
    items = _synthetic_pairs(repos=6, pairs_per_repo=2, seed=2)
    a = ClusterBootstrap({Condition.CPG: items}, resamples=30, seed=4)
    b = ClusterBootstrap(
        {
            Condition.CPG: items,
            Condition.NONE: [
                i.model_copy(update={"condition": Condition.NONE}) for i in items
            ],
        },
        resamples=30,
        seed=4,
    )
    assert a.repo_draws == b.repo_draws
    assert all(len(draw) == 6 for draw in a.repo_draws)
    c = ClusterBootstrap({Condition.CPG: items}, resamples=30, seed=5)
    assert c.repo_draws != a.repo_draws


def test_single_class_resamples_are_redrawn_and_counted() -> None:
    # Each repo holds one class only, so some draws of two repos are single-class.
    items = [_item("a", 1, 0.8, "r1"), _item("b", 0, 0.3, "r2")]
    boot = ClusterBootstrap({Condition.CPG: items}, resamples=50, seed=0)
    assert boot.redraws > 0
    assert all(
        value is not None
        for value in boot.replicates(Condition.CPG, MetricName.ROC_AUC)
    )
    assert all(len(set(draw)) == 2 for draw in boot.repo_draws)


def test_single_class_input_raises() -> None:
    items = [_item("a", 1, 0.8, "r1"), _item("b", 1, 0.3, "r2")]
    try:
        ClusterBootstrap({Condition.CPG: items}, resamples=5, seed=0)
    except ValueError:
        return
    raise AssertionError("single-class input cannot be bootstrapped")


def test_mismatched_repo_sets_raise() -> None:
    items = _synthetic_pairs(repos=3, pairs_per_repo=1, seed=1)
    other = [
        i.model_copy(update={"condition": Condition.NONE, "repo_url": "elsewhere"})
        for i in items
    ]
    try:
        ClusterBootstrap(
            {Condition.CPG: items, Condition.NONE: other}, resamples=5, seed=0
        )
    except ValueError:
        return
    raise AssertionError("conditions over different repositories cannot be paired")


def test_condition_table_shape_and_intervals() -> None:
    items = _all_conditions(_synthetic_pairs(repos=8, pairs_per_repo=3, seed=7))
    inputs = _inputs(items)
    boot = ClusterBootstrap(
        {c: [i for i in items if i.condition is c] for c in Condition},
        resamples=60,
        seed=1,
    )
    table = condition_table(inputs, boot)
    assert list(table) == list(Condition)
    for condition, row in table.items():
        assert list(row) == list(MetricName)
        ci = row[MetricName.ROC_AUC]
        assert isinstance(ci, ConfidenceInterval)
        assert (
            ci.point
            == point_metrics([i for i in items if i.condition is condition])[
                MetricName.ROC_AUC
            ]
        )
        assert ci.low is not None and ci.high is not None and ci.low <= ci.high


def test_stratum_tables_handle_sparse_strata() -> None:
    rich = _synthetic_pairs(repos=6, pairs_per_repo=2, seed=3, stratum="r100")
    one_repo = _synthetic_pairs(repos=1, pairs_per_repo=3, seed=4, stratum="r0")
    inputs = _inputs(_all_conditions(rich + one_repo))
    tables = stratum_tables(inputs, resamples=40, seed=2)
    assert list(tables) == ["no_reference", "r0", "r0_50", "r50_100", "r100"]
    empty = tables["no_reference"][Condition.CPG][MetricName.ROC_AUC]
    assert empty == ConfidenceInterval(None, None, None)
    single = tables["r0"][Condition.CPG][MetricName.ROC_AUC]
    assert single.point is not None and single.low is None and single.high is None
    full = tables["r100"][Condition.CPG][MetricName.ROC_AUC]
    assert full.point is not None and full.low is not None and full.high is not None
    # The per-stratum bootstrap only sees that stratum's items.
    expected = ClusterBootstrap(
        {
            c: [i for i in inputs.items if i.condition is c and i.stratum == "r100"]
            for c in Condition
        },
        resamples=40,
        seed=2,
    ).interval(Condition.CPG, MetricName.ROC_AUC)
    assert full == expected


def test_stratum_tables_reject_unknown_stratum() -> None:
    inputs = _inputs(
        _all_conditions(
            _synthetic_pairs(repos=2, pairs_per_repo=1, seed=3, stratum="bogus")
        )
    )
    try:
        stratum_tables(inputs, resamples=5, seed=0)
    except ValueError:
        return
    raise AssertionError("unknown strata must be rejected")


def test_length_stats() -> None:
    base = [
        _item("a", 1, 0.9, "r1", prompt_tokens=10, token_count=1),
        _item("a", 0, 0.1, "r1", prompt_tokens=20, token_count=2),
        _item("b", 1, 0.9, "r2", prompt_tokens=30, token_count=3),
        _item("b", 0, 0.1, "r2", prompt_tokens=40, token_count=4),
    ]
    stats = length_stats(_inputs(_all_conditions(base)))
    assert list(stats) == list(Condition)
    cpg = stats[Condition.CPG]
    assert cpg["prompt_tokens_mean"] == 25.0
    assert cpg["prompt_tokens_median"] == 25.0
    assert cpg["prompt_tokens_q1"] == 17.5
    assert cpg["prompt_tokens_q3"] == 32.5
    assert cpg["context_tokens_mean"] == 2.5
    assert cpg["context_tokens_q1"] == 1.75
    assert cpg["context_tokens_q3"] == 3.25


def test_base_rate() -> None:
    items = [
        _item("a", 1, 0.9, "r1"),
        _item("a", 0, 0.1, "r1"),
        _item("b", 1, 0.9, "r2"),
    ]
    assert math.isclose(base_rate(items), 2 / 3)
    try:
        base_rate([])
    except ValueError:
        return
    raise AssertionError("empty base rate must raise")


def test_decomposition() -> None:
    cpg = _synthetic_pairs(repos=8, pairs_per_repo=3, seed=13)
    reference = [
        i.model_copy(
            update={"condition": Condition.REFERENCE, "score": 0.2 + 0.6 * i.label}
        )
        for i in cpg
    ]
    boot = ClusterBootstrap(
        {Condition.CPG: cpg, Condition.REFERENCE: reference}, resamples=80, seed=6
    )
    result = decomposition(boot)
    assert list(result) == [
        MetricName.ROC_AUC,
        MetricName.PR_AUC,
        MetricName.PAIRED_RANKING_ACCURACY,
    ]
    for metric, parts in result.items():
        assert parts["retrieval_loss"] == boot.difference(
            Condition.REFERENCE, Condition.CPG, metric
        )
        ref = boot.interval(Condition.REFERENCE, metric)
        assert parts["reasoning_loss"] == ConfidenceInterval(
            1 - ref.point, 1 - ref.high, 1 - ref.low
        )
    roc = result[MetricName.ROC_AUC]
    assert roc["reasoning_loss"] == ConfidenceInterval(0.0, 0.0, 0.0)
    cpg_auc = point_metrics(cpg)[MetricName.ROC_AUC]
    assert math.isclose(roc["retrieval_loss"].point, 1.0 - cpg_auc)
    assert roc["retrieval_loss"].low > 0.0


def test_length_stats_missing_prompt_tokens_are_none_and_json_safe() -> None:
    base = [
        _item("a", 1, 0.9, "r1").model_copy(update={"prompt_tokens": None}),
        _item("a", 0, 0.1, "r1").model_copy(update={"prompt_tokens": None}),
    ]
    stats = length_stats(_inputs(_all_conditions(base)))
    cpg = stats[Condition.CPG]
    for key in ("mean", "median", "q1", "q3"):
        assert cpg[f"prompt_tokens_{key}"] is None
    assert cpg["context_tokens_mean"] == 10.0
    json.dumps({c.value: row for c, row in stats.items()}, allow_nan=False)


def test_interval_reports_dropped_replicates() -> None:
    # r2 holds two classes but no complete pair, so a resample drawing r2
    # twice is valid yet leaves paired ranking undefined (a None replicate).
    items = [
        _item("p", 1, 0.9, "r1"),
        _item("p", 0, 0.1, "r1"),
        _item("x", 1, 0.8, "r2"),
        _item("y", 0, 0.3, "r2"),
    ]
    boot = ClusterBootstrap({Condition.CPG: items}, resamples=200, seed=0)
    replicates = boot.replicates(Condition.CPG, MetricName.PAIRED_RANKING_ACCURACY)
    expected_dropped = sum(value is None for value in replicates)
    assert 0 < expected_dropped < 200
    ci = boot.interval(Condition.CPG, MetricName.PAIRED_RANKING_ACCURACY)
    assert ci.dropped == expected_dropped
    assert boot.interval(Condition.CPG, MetricName.ROC_AUC).dropped == 0
    diff = boot.difference(
        Condition.CPG, Condition.CPG, MetricName.PAIRED_RANKING_ACCURACY
    )
    assert diff.dropped == expected_dropped


def test_sparse_stratum_tables_have_full_key_shape() -> None:
    rich = _synthetic_pairs(repos=6, pairs_per_repo=2, seed=3, stratum="r100")
    one_repo_cpg_only = _synthetic_pairs(
        repos=1, pairs_per_repo=3, seed=4, stratum="r0"
    )
    rich_all = _all_conditions(rich)
    inputs = _inputs(rich_all + one_repo_cpg_only)
    tables = stratum_tables(inputs, resamples=20, seed=2)
    for stratum, table in tables.items():
        assert list(table) == list(Condition), stratum
        for row in table.values():
            assert list(row) == list(MetricName), stratum
    assert tables["r0"][Condition.NONE][MetricName.ROC_AUC] == ConfidenceInterval(
        None, None, None
    )
    assert tables["r0"][Condition.CPG][MetricName.ROC_AUC].point is not None


if __name__ == "__main__":
    test_point_metrics_hand_computed()
    test_point_metrics_pr_family_and_mcc_hand_computed()
    test_point_metrics_single_class_yields_none()
    test_point_metrics_rejects_duplicate_items()
    test_paired_ranking_parity_with_framework_calculator()
    test_all_metrics_parity_with_framework_calculator_on_random_data()
    test_cluster_bootstrap_covers_truth_and_widens_with_repo_correlation()
    test_paired_difference_of_identical_conditions_is_zero()
    test_replicates_equal_point_metrics_on_explicit_resample()
    test_repo_draws_are_seeded_and_shared_across_conditions()
    test_single_class_resamples_are_redrawn_and_counted()
    test_single_class_input_raises()
    test_mismatched_repo_sets_raise()
    test_condition_table_shape_and_intervals()
    test_stratum_tables_handle_sparse_strata()
    test_stratum_tables_reject_unknown_stratum()
    test_length_stats()
    test_base_rate()
    test_decomposition()
    test_length_stats_missing_prompt_tokens_are_none_and_json_safe()
    test_interval_reports_dropped_replicates()
    test_sparse_stratum_tables_have_full_key_shape()
    print("ALL PASSED")

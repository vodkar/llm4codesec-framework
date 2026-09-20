"""Plain-python checks for binary ranking metrics (ROC AUC, PR AUC, P@R), paired ranking and accuracy@coverage.

Run: PYTHONPATH=src uv run python tests/test_metrics_calculator.py
Expected values are hand-computed from the ranked label lists in each test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.metrics_calculator import BinaryMetricsCalculator
from benchmark.models import PredictionResult

_RANKING_KEYS = [
    "roc_auc",
    "pr_auc",
    "precision_at_recall_50",
    "precision_at_recall_80",
    "precision_at_recall_90",
    "precision_at_recall_95",
]


_COVERAGE_KEYS = [
    "accuracy_at_coverage_25",
    "accuracy_at_coverage_50",
    "accuracy_at_coverage_75",
]


def _pred(
    index,
    true_label,
    score=None,
    vote_counts=None,
    answer_probability=None,
    predicted_label=None,
    stated_confidence=None,
    self_validation_probability=None,
):
    if predicted_label is None:
        predicted_label = true_label if score is None else int(score > 0.5)
    return PredictionResult(
        sample_id=f"s{index}",
        predicted_label=predicted_label,
        true_label=true_label,
        confidence=None,
        binary_label_confidence=score,
        answer_probability=answer_probability,
        stated_confidence=stated_confidence,
        self_validation_probability=self_validation_probability,
        response_text="",
        processing_time=0.0,
        is_success=True,
        error_message=None,
        vote_counts=vote_counts or {},
    )


def _scored(labels_by_descending_score):
    """Predictions whose P(VULNERABLE) decreases along the given label list."""
    n = len(labels_by_descending_score)
    return [
        _pred(i, label, score=(n - i) / n)
        for i, label in enumerate(labels_by_descending_score)
    ]


def _by_confidence(correct_by_descending_probability, source="answer_probability"):
    """Predictions whose confidence score decreases along the given correctness list."""
    n = len(correct_by_descending_probability)
    return [
        _pred(
            i,
            true_label=1,
            predicted_label=1 if is_correct else 0,
            **{source: 0.5 + 0.5 * (n - i) / n},
        )
        for i, is_correct in enumerate(correct_by_descending_probability)
    ]


def _half(pair_key, is_vulnerable, score=None, vote_counts=None):
    """One half of a pre/post pair, with the loader's ``_vuln`` / ``_safe`` ID suffix."""
    pred = _pred(0, int(is_vulnerable), score=score, vote_counts=vote_counts)
    suffix = "vuln" if is_vulnerable else "safe"
    return pred.model_copy(update={"sample_id": f"cleanvul_{pair_key}_{suffix}"})


def _pair(pair_key, vuln_score, safe_score):
    return [_half(pair_key, True, vuln_score), _half(pair_key, False, safe_score)]


def _close(actual, expected):
    return actual is not None and abs(actual - expected) < 1e-9


def test_ranking_metrics_match_hand_computed_values():
    # Ranked labels: positives at ranks 1,2,3,6,8 of 10 (5 pos, 5 neg).
    # ROC AUC: pos-above-neg pairs = 5+5+5+3+2 = 20 of 25.
    # PR AUC (average precision): (1 + 1 + 1 + 4/6 + 5/8) / 5.
    # Precision at top-k: k=3 -> P=1, R=0.6; k=6 -> P=4/6, R=0.8; k=8 -> P=5/8, R=1.
    summary = BinaryMetricsCalculator().calculate(
        _scored([1, 1, 1, 0, 0, 1, 0, 1, 0, 0])
    ).summary
    assert _close(summary["roc_auc"], 0.8), summary
    assert _close(summary["pr_auc"], (3 + 4 / 6 + 5 / 8) / 5), summary
    assert _close(summary["precision_at_recall_50"], 1.0), summary
    assert _close(summary["precision_at_recall_80"], 4 / 6), summary
    assert _close(summary["precision_at_recall_90"], 5 / 8), summary
    assert _close(summary["precision_at_recall_95"], 5 / 8), summary
    print("test_ranking_metrics_match_hand_computed_values PASSED")


def test_inverted_ranking_gives_zero_roc_auc():
    # Every negative outranks every positive; at any recall > 0 the best
    # precision is reached only after all 2 negatives: 2 pos / 4 samples.
    summary = BinaryMetricsCalculator().calculate(_scored([0, 0, 1, 1])).summary
    assert _close(summary["roc_auc"], 0.0), summary
    assert _close(summary["precision_at_recall_95"], 0.5), summary
    print("test_inverted_ranking_gives_zero_roc_auc PASSED")


def test_vote_fraction_used_when_no_label_confidence():
    predictions = [
        _pred(0, 1, vote_counts={"1": 5}),
        _pred(1, 1, vote_counts={"1": 3, "0": 2}),
        _pred(2, 0, vote_counts={"1": 1, "0": 4}),
        _pred(3, 0, vote_counts={"0": 5}),
    ]
    result = BinaryMetricsCalculator().calculate(predictions)
    assert _close(result.summary["roc_auc"], 1.0), result.summary
    ranking = result.details["ranking_metrics"]
    assert ranking["score_source"] == "vote_fraction", ranking
    assert ranking["scored_samples"] == 4, ranking
    print("test_vote_fraction_used_when_no_label_confidence PASSED")


def test_unscored_predictions_are_excluded_and_counted():
    # The unscored sample is a misclassified positive; were it included with
    # any default score it would change the perfect ranking of the other four.
    predictions = _scored([1, 1, 0, 0]) + [_pred(4, 1)]
    result = BinaryMetricsCalculator().calculate(predictions)
    assert _close(result.summary["roc_auc"], 1.0), result.summary
    ranking = result.details["ranking_metrics"]
    assert ranking["score_source"] == "binary_label_confidence", ranking
    assert ranking["scored_samples"] == 4, ranking
    assert ranking["total_samples"] == 5, ranking
    print("test_unscored_predictions_are_excluded_and_counted PASSED")


def test_no_scores_yields_none_metrics():
    result = BinaryMetricsCalculator().calculate([_pred(0, 1), _pred(1, 0)])
    for key in _RANKING_KEYS:
        assert key in result.summary and result.summary[key] is None, (key, result.summary)
    assert result.details["ranking_metrics"]["scored_samples"] == 0, result.details
    assert result.summary["accuracy"] == 1.0, result.summary
    print("test_no_scores_yields_none_metrics PASSED")


def test_single_vote_is_not_treated_as_a_score():
    # The runner records vote_counts even when self_consistency_samples=1;
    # a lone hard label carries no ranking information.
    predictions = [
        _pred(0, 1, vote_counts={"1": 1}),
        _pred(1, 0, vote_counts={"0": 1}),
    ]
    result = BinaryMetricsCalculator().calculate(predictions)
    for key in _RANKING_KEYS:
        assert result.summary[key] is None, (key, result.summary)
    assert result.details["ranking_metrics"]["scored_samples"] == 0, result.details
    print("test_single_vote_is_not_treated_as_a_score PASSED")


def test_single_class_yields_none_metrics():
    result = BinaryMetricsCalculator().calculate(_scored([1, 1, 1]))
    for key in _RANKING_KEYS:
        assert key in result.summary and result.summary[key] is None, (key, result.summary)
    assert result.details["ranking_metrics"]["scored_samples"] == 3, result.details
    print("test_single_class_yields_none_metrics PASSED")


def test_accuracy_at_coverage_matches_hand_computed_values():
    # 8 samples ranked by answer probability; correct at ranks 1,2,4,6.
    # Top 2 -> 2/2, top 4 -> 3/4, top 6 -> 4/6. Overall accuracy is 4/8.
    result = BinaryMetricsCalculator().calculate(
        _by_confidence([True, True, False, True, False, True, False, False])
    )
    assert _close(result.summary["accuracy_at_coverage_25"], 1.0), result.summary
    assert _close(result.summary["accuracy_at_coverage_50"], 0.75), result.summary
    assert _close(result.summary["accuracy_at_coverage_75"], 4 / 6), result.summary
    assert _close(result.summary["accuracy"], 0.5), result.summary
    print("test_accuracy_at_coverage_matches_hand_computed_values PASSED")


def test_accuracy_at_coverage_ranks_by_probability_not_input_order():
    # Same samples as above fed least-confident first must give the same answer.
    predictions = _by_confidence([True, True, False, True, False, True, False, False])
    result = BinaryMetricsCalculator().calculate(predictions[::-1])
    assert _close(result.summary["accuracy_at_coverage_25"], 1.0), result.summary
    assert _close(result.summary["accuracy_at_coverage_50"], 0.75), result.summary
    print("test_accuracy_at_coverage_ranks_by_probability_not_input_order PASSED")


def test_coverage_selection_rounds_up():
    # 5 samples: 25% -> ceil(1.25) = 2, 50% -> ceil(2.5) = 3, 75% -> ceil(3.75) = 4.
    # Correct at ranks 1,3,4 -> 1/2, 2/3, 3/4.
    result = BinaryMetricsCalculator().calculate(
        _by_confidence([True, False, True, True, False])
    )
    assert _close(result.summary["accuracy_at_coverage_25"], 0.5), result.summary
    assert _close(result.summary["accuracy_at_coverage_50"], 2 / 3), result.summary
    assert _close(result.summary["accuracy_at_coverage_75"], 0.75), result.summary
    levels = result.details["selective_prediction"]["levels"]
    assert [level["selected_samples"] for level in levels] == [2, 3, 4], levels
    # Probabilities are 1.0, 0.9, 0.8, 0.7, 0.6 -> cutoffs at ranks 2, 3, 4.
    for level, expected in zip(levels, [0.9, 0.8, 0.7]):
        assert _close(level["min_answer_probability"], expected), levels
    print("test_coverage_selection_rounds_up PASSED")


def test_coverage_excludes_samples_without_answer_probability():
    # The two unscored samples are wrong; if ranked anywhere they would
    # lower at least one of the perfect accuracies below.
    predictions = _by_confidence([True, True, True, True]) + [
        _pred(4, true_label=1, predicted_label=0),
        _pred(5, true_label=1, predicted_label=0),
    ]
    result = BinaryMetricsCalculator().calculate(predictions)
    for key in _COVERAGE_KEYS:
        assert _close(result.summary[key], 1.0), (key, result.summary)
    selective = result.details["selective_prediction"]
    assert selective["scored_samples"] == 4, selective
    assert selective["total_samples"] == 6, selective
    print("test_coverage_excludes_samples_without_answer_probability PASSED")


def test_no_answer_probabilities_yields_none_coverage_metrics():
    result = BinaryMetricsCalculator().calculate([_pred(0, 1), _pred(1, 0)])
    for key in _COVERAGE_KEYS:
        assert key in result.summary and result.summary[key] is None, (key, result.summary)
    assert result.details["selective_prediction"]["scored_samples"] == 0, result.details
    print("test_no_answer_probabilities_yields_none_coverage_metrics PASSED")


def test_paired_ranking_matches_hand_computed_values():
    # 4 pairs: vuln above fixed in a and b, tied in c, below in d.
    # Accuracy: (2 wins + 0.5 * 1 tie) / 4. Margins: 0.7, 0.2, 0.0, -0.5 -> mean 0.1.
    predictions = (
        _pair("a", 0.9, 0.2) + _pair("b", 0.6, 0.4) + _pair("c", 0.5, 0.5) + _pair("d", 0.3, 0.8)
    )
    result = BinaryMetricsCalculator().calculate(predictions)
    assert _close(result.summary["paired_ranking_accuracy"], 0.625), result.summary
    assert _close(result.summary["paired_mean_score_margin"], 0.1), result.summary
    paired = result.details["paired_ranking"]
    assert paired["score_source"] == "binary_label_confidence", paired
    assert (paired["scored_pairs"], paired["wins"], paired["ties"], paired["losses"]) == (
        4,
        2,
        1,
        1,
    ), paired
    assert paired["unpaired_samples"] == 0, paired
    print("test_paired_ranking_matches_hand_computed_values PASSED")


def test_paired_ranking_ignores_between_function_variance():
    # The "hot" fixed function outscores the "cold" vulnerable one, so pooled ROC AUC
    # is 3/4, yet within each pair the vulnerable version wins.
    predictions = _pair("hot", 0.9, 0.8) + _pair("cold", 0.2, 0.1)
    summary = BinaryMetricsCalculator().calculate(predictions).summary
    assert _close(summary["roc_auc"], 0.75), summary
    assert _close(summary["paired_ranking_accuracy"], 1.0), summary
    print("test_paired_ranking_ignores_between_function_variance PASSED")


def test_paired_ranking_pairs_by_id_not_input_order():
    # Halves arrive interleaved and fixed-first; adjacent-position pairing would
    # compare a's fixed half with b's vulnerable half.
    predictions = [
        _half("a", False, 0.2),
        _half("b", True, 0.3),
        _half("a", True, 0.9),
        _half("b", False, 0.8),
    ]
    result = BinaryMetricsCalculator().calculate(predictions)
    assert _close(result.summary["paired_ranking_accuracy"], 0.5), result.summary
    assert _close(result.summary["paired_mean_score_margin"], 0.1), result.summary
    print("test_paired_ranking_pairs_by_id_not_input_order PASSED")


def test_unpaired_half_is_excluded_and_counted():
    # Token-range buckets can keep only one side of a commit.
    predictions = _pair("a", 0.9, 0.2) + [_half("b", True, 0.1), _half("c", False, 0.9)]
    result = BinaryMetricsCalculator().calculate(predictions)
    assert _close(result.summary["paired_ranking_accuracy"], 1.0), result.summary
    paired = result.details["paired_ranking"]
    assert paired["scored_pairs"] == 1, paired
    assert paired["unpaired_samples"] == 2, paired
    print("test_unpaired_half_is_excluded_and_counted PASSED")


def test_pair_with_an_unscored_half_is_excluded():
    # Pair b would be a loss if its missing score defaulted to anything >= 0.1.
    predictions = _pair("a", 0.9, 0.2) + [_half("b", True, 0.1), _half("b", False)]
    result = BinaryMetricsCalculator().calculate(predictions)
    assert _close(result.summary["paired_ranking_accuracy"], 1.0), result.summary
    paired = result.details["paired_ranking"]
    assert paired["scored_pairs"] == 1, paired
    assert paired["unpaired_samples"] == 0, paired
    print("test_pair_with_an_unscored_half_is_excluded PASSED")


def test_paired_ranking_falls_back_to_vote_fraction():
    # Vote fractions: a 5/5 vs 1/5 (win), b 2/5 vs 2/5 (tie) -> (1 + 0.5) / 2.
    predictions = [
        _half("a", True, vote_counts={"1": 5}),
        _half("a", False, vote_counts={"1": 1, "0": 4}),
        _half("b", True, vote_counts={"1": 2, "0": 3}),
        _half("b", False, vote_counts={"1": 2, "0": 3}),
    ]
    result = BinaryMetricsCalculator().calculate(predictions)
    assert _close(result.summary["paired_ranking_accuracy"], 0.75), result.summary
    assert _close(result.summary["paired_mean_score_margin"], 0.4), result.summary
    assert result.details["paired_ranking"]["score_source"] == "vote_fraction", result.details
    print("test_paired_ranking_falls_back_to_vote_fraction PASSED")


def test_ids_without_pair_suffix_yield_none_paired_metrics():
    result = BinaryMetricsCalculator().calculate(_scored([1, 0, 1, 0]))
    for key in ("paired_ranking_accuracy", "paired_mean_score_margin"):
        assert key in result.summary and result.summary[key] is None, (key, result.summary)
    paired = result.details["paired_ranking"]
    assert paired["scored_pairs"] == 0, paired
    assert paired["unpaired_samples"] == 4, paired
    print("test_ids_without_pair_suffix_yield_none_paired_metrics PASSED")


def test_optional_confidence_methods_get_their_own_coverage_metrics():
    # Same ranking as the answer-probability test: top 2 -> 2/2, top 4 -> 3/4, top 6 -> 4/6.
    correct = [True, True, False, True, False, True, False, False]
    for source in ("stated_confidence", "self_validation_probability"):
        result = BinaryMetricsCalculator().calculate(_by_confidence(correct, source=source))
        prefix = source.removesuffix("_probability")
        assert _close(result.summary[f"{prefix}_accuracy_at_coverage_25"], 1.0), result.summary
        assert _close(result.summary[f"{prefix}_accuracy_at_coverage_50"], 0.75), result.summary
        assert _close(result.summary[f"{prefix}_accuracy_at_coverage_75"], 4 / 6), result.summary
        assert result.details["confidence_methods"][prefix]["scored_samples"] == 8, result.details
        # Answer probability was not provided, so its own metrics stay empty.
        assert result.summary["accuracy_at_coverage_25"] is None, result.summary
    print("test_optional_confidence_methods_get_their_own_coverage_metrics PASSED")


def test_correctness_auroc_matches_hand_computed_value():
    # Correct at ranks 1,2,4,6; wrong at 3,5,7,8. Correct-above-wrong pairs:
    # 4 + 4 + 3 + 2 = 13 of 16.
    correct = [True, True, False, True, False, True, False, False]
    result = BinaryMetricsCalculator().calculate(_by_confidence(correct, source="stated_confidence"))
    assert _close(result.summary["stated_confidence_correctness_auroc"], 13 / 16), result.summary
    result = BinaryMetricsCalculator().calculate(_by_confidence(correct))
    assert _close(result.summary["answer_probability_correctness_auroc"], 13 / 16), result.summary
    print("test_correctness_auroc_matches_hand_computed_value PASSED")


def test_correctness_auroc_is_none_when_all_answers_are_correct():
    result = BinaryMetricsCalculator().calculate(
        _by_confidence([True, True, True], source="self_validation_probability")
    )
    assert result.summary["self_validation_correctness_auroc"] is None, result.summary
    assert _close(result.summary["self_validation_accuracy_at_coverage_50"], 1.0), result.summary
    print("test_correctness_auroc_is_none_when_all_answers_are_correct PASSED")


def test_disabled_confidence_methods_add_no_summary_keys():
    result = BinaryMetricsCalculator().calculate(_by_confidence([True, False]))
    unexpected = [k for k in result.summary if k.startswith(("stated_confidence", "self_validation"))]
    assert unexpected == [], unexpected
    assert result.details["confidence_methods"] == {}, result.details
    print("test_disabled_confidence_methods_add_no_summary_keys PASSED")


if __name__ == "__main__":
    test_paired_ranking_matches_hand_computed_values()
    test_paired_ranking_ignores_between_function_variance()
    test_paired_ranking_pairs_by_id_not_input_order()
    test_unpaired_half_is_excluded_and_counted()
    test_pair_with_an_unscored_half_is_excluded()
    test_paired_ranking_falls_back_to_vote_fraction()
    test_ids_without_pair_suffix_yield_none_paired_metrics()
    test_ranking_metrics_match_hand_computed_values()
    test_inverted_ranking_gives_zero_roc_auc()
    test_vote_fraction_used_when_no_label_confidence()
    test_unscored_predictions_are_excluded_and_counted()
    test_no_scores_yields_none_metrics()
    test_single_vote_is_not_treated_as_a_score()
    test_single_class_yields_none_metrics()
    test_accuracy_at_coverage_matches_hand_computed_values()
    test_accuracy_at_coverage_ranks_by_probability_not_input_order()
    test_coverage_selection_rounds_up()
    test_coverage_excludes_samples_without_answer_probability()
    test_no_answer_probabilities_yields_none_coverage_metrics()
    test_optional_confidence_methods_get_their_own_coverage_metrics()
    test_correctness_auroc_matches_hand_computed_value()
    test_correctness_auroc_is_none_when_all_answers_are_correct()
    test_disabled_confidence_methods_add_no_summary_keys()
    print("ALL PASSED")

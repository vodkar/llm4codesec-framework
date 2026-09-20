"""Plain-python checks for optional confidence methods (stated confidence, self-validation).

Run: PYTHONPATH=src uv run python tests/test_confidence_methods.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.benchmark_runner import _aggregate_draw_confidences
from benchmark.config import ExperimentConfig, ExperimentsPlanConfig
from benchmark.enums import ConfidenceMethod, TaskType
from benchmark.prompt_generator import DefaultPromptGenerator
from benchmark.response_parser import (
    BinaryResponseParser,
    extract_stated_confidence,
    has_explicit_binary_verdict,
)
from llm.self_validation import SELF_VALIDATION_PREFILL, build_self_validation_messages


def _close(actual, expected):
    return actual is not None and abs(actual - expected) < 1e-9


def _config(plan_overrides=None, model_overrides=None, task_type="binary_vulnerability"):
    return {
        "models": {
            "m": {
                "model_identifier": "org/model",
                "model_type": "qwen-3",
                "backend": "vllm",
                "max_output_tokens": 16,
                "batch_size": 1,
                "temperature": 0.0,
                "use_quantization": False,
                **(model_overrides or {}),
            }
        },
        # The dataset file only has to exist for config validation.
        "datasets": {"d": {"path": __file__, "task_type": task_type, "description": "d"}},
        "prompts": {"p": {"name": "p", "system_prompt": "sys", "user_prompt": "{code}"}},
        "output_settings": {"base_output_dir": "results/x"},
        "experiment_plans": {
            "plan": {"datasets": ["d"], "models": ["m"], "prompts": ["p"], **(plan_overrides or {})}
        },
    }


def _generator(confidence_methods):
    return DefaultPromptGenerator(
        system_prompt_template="sys",
        user_prompt_template="{code}",
        template_values={},
        task_type=TaskType.BINARY_VULNERABILITY,
        prompt_identifier="p",
        confidence_methods=confidence_methods,
    )


def test_plan_without_setting_has_no_confidence_methods():
    plan = ExperimentsPlanConfig.from_file(_config(), "plan")
    assert plan.experiments[0].confidence_methods == [], plan.experiments[0].confidence_methods
    print("test_plan_without_setting_has_no_confidence_methods PASSED")


def test_plan_setting_reaches_every_experiment():
    plan = ExperimentsPlanConfig.from_file(
        _config({"confidence_methods": ["stated_confidence", "self_validation"]}), "plan"
    )
    assert plan.experiments[0].confidence_methods == [
        ConfidenceMethod.STATED_CONFIDENCE,
        ConfidenceMethod.SELF_VALIDATION,
    ], plan.experiments[0].confidence_methods
    print("test_plan_setting_reaches_every_experiment PASSED")


def test_confidence_methods_survive_the_subprocess_rebuild():
    # Plan experiments are rebuilt in a child process from plain values.
    experiment = ExperimentConfig.from_file(
        config=_config(),
        model_key="m",
        dataset_key="d",
        prompt_key="p",
        experiment_name="plan",
        confidence_methods=["stated_confidence"],
    )
    assert experiment.confidence_methods == [ConfidenceMethod.STATED_CONFIDENCE]
    print("test_confidence_methods_survive_the_subprocess_rebuild PASSED")


def test_confidence_methods_are_rejected_for_multiclass_tasks():
    try:
        ExperimentsPlanConfig.from_file(
            _config({"confidence_methods": ["stated_confidence"]}, task_type="multiclass_vulnerability"),
            "plan",
        )
    except ValueError as error:
        assert "binary" in str(error), error
    else:
        raise AssertionError("multiclass task accepted confidence_methods")
    print("test_confidence_methods_are_rejected_for_multiclass_tasks PASSED")


def test_self_validation_is_rejected_for_backends_without_support():
    try:
        ExperimentsPlanConfig.from_file(
            _config({"confidence_methods": ["self_validation"]}, {"backend": "llama_cpp"}), "plan"
        )
    except ValueError as error:
        assert "vllm" in str(error), error
    else:
        raise AssertionError("llama_cpp backend accepted self_validation")
    print("test_self_validation_is_rejected_for_backends_without_support PASSED")


def test_default_contract_does_not_ask_for_confidence():
    prompt = _generator([]).get_user_prompt({"code": "int x;"})
    assert '{"is_vulnerable": true}' in prompt, prompt
    assert "confidence" not in prompt, prompt
    print("test_default_contract_does_not_ask_for_confidence PASSED")


def test_stated_confidence_contract_asks_for_a_digit():
    prompt = _generator([ConfidenceMethod.STATED_CONFIDENCE]).get_user_prompt({"code": "int x;"})
    assert '{"is_vulnerable": true, "confidence": <0-9>}' in prompt, prompt
    assert '{"is_vulnerable": false, "confidence": <0-9>}' in prompt, prompt
    print("test_stated_confidence_contract_asks_for_a_digit PASSED")


def test_self_validation_alone_keeps_the_default_contract():
    prompt = _generator([ConfidenceMethod.SELF_VALIDATION]).get_user_prompt({"code": "int x;"})
    assert "confidence" not in prompt, prompt
    print("test_self_validation_alone_keeps_the_default_contract PASSED")


def test_verdict_with_confidence_still_parses():
    response = 'analysis\n{"is_vulnerable": true, "confidence": 7}'
    assert BinaryResponseParser().parse_response(response) == 1
    assert _close(extract_stated_confidence(response), 7 / 9)
    print("test_verdict_with_confidence_still_parses PASSED")


def test_last_stated_confidence_wins_and_missing_is_none():
    response = '{"is_vulnerable": false, "confidence": 2}\nwait\n{"is_vulnerable": true, "confidence": 9}'
    assert _close(extract_stated_confidence(response), 1.0)
    assert extract_stated_confidence('{"is_vulnerable": true}') is None
    # Out-of-scale numbers are not a 0-9 rating.
    assert extract_stated_confidence('{"is_vulnerable": true, "confidence": 85}') is None
    print("test_last_stated_confidence_wins_and_missing_is_none PASSED")


def test_explicit_verdict_detection():
    assert has_explicit_binary_verdict('x\n{"is_vulnerable": false}')
    assert not has_explicit_binary_verdict("the code looks vulnerable but I ran out of tok")
    print("test_explicit_verdict_detection PASSED")


def test_draw_confidences_are_aggregated_towards_the_predicted_label():
    # Draws 1 and 2 back the predicted label (1) with 0.9 and 0.7; draw 3 backed
    # the other label with 0.8, i.e. 0.2 for the predicted one -> (0.9+0.7+0.2)/3.
    score = _aggregate_draw_confidences([1, 1, 0], 1, [0.9, 0.7, 0.8])
    assert _close(score, 0.6), score
    print("test_draw_confidences_are_aggregated_towards_the_predicted_label PASSED")


def test_unscored_draws_are_ignored_in_aggregation():
    assert _close(_aggregate_draw_confidences([1, 1, 0], 1, [None, 0.7, None]), 0.7)
    assert _aggregate_draw_confidences([1, 0], 1, [None, None]) is None
    print("test_unscored_draws_are_ignored_in_aggregation PASSED")


def test_self_validation_messages_replay_the_conversation():
    messages = build_self_validation_messages("sys", "user code", 'why\n{"is_vulnerable": true}')
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"], messages
    assert messages[1]["content"] == "user code"
    assert messages[2]["content"] == 'why\n{"is_vulnerable": true}'
    assert "verdict_is_correct" in messages[3]["content"], messages[3]
    assert SELF_VALIDATION_PREFILL == '{"verdict_is_correct":'
    print("test_self_validation_messages_replay_the_conversation PASSED")


if __name__ == "__main__":
    test_plan_without_setting_has_no_confidence_methods()
    test_plan_setting_reaches_every_experiment()
    test_confidence_methods_survive_the_subprocess_rebuild()
    test_confidence_methods_are_rejected_for_multiclass_tasks()
    test_self_validation_is_rejected_for_backends_without_support()
    test_default_contract_does_not_ask_for_confidence()
    test_stated_confidence_contract_asks_for_a_digit()
    test_self_validation_alone_keeps_the_default_contract()
    test_verdict_with_confidence_still_parses()
    test_last_stated_confidence_wins_and_missing_is_none()
    test_explicit_verdict_detection()
    test_draw_confidences_are_aggregated_towards_the_predicted_label()
    test_unscored_draws_are_ignored_in_aggregation()
    test_self_validation_messages_replay_the_conversation()
    print("ALL PASSED")

"""Plain-python checks for final-answer label probability extraction.

Run: PYTHONPATH=src uv run python tests/test_final_answer_logprobs.py
Token splits mirror Qwen3: "VULNERABLE" is multi-token and labels carry a leading space in context.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm.final_answer_logprobs import (
    compute_label_probability,
    compute_p_vulnerable,
    compute_stated_confidence,
    locate_final_answer_label,
    locate_stated_confidence,
    resolve_label_token_ids,
    resolve_self_validation_token_ids,
)

_MARKER_LABELS = {"VULNERABLE": ("VULNERABLE",), "SAFE": ("SAFE",)}


class _FakeTokenizer:
    """Greedy longest-match tokenizer over a fixed vocabulary."""

    def __init__(self, vocab):
        self.vocab = vocab

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            token = max((t for t in self.vocab if text.startswith(t)), key=len)
            ids.append(self.vocab.index(token))
            text = text[len(token):]
        return ids

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(self.vocab[i] for i in token_ids)


_QWEN_LIKE = _FakeTokenizer(
    ["why", "\n", "[[", "FINAL", "_ANS", "WER", ":", " V", "V", "UL", "NER", "ABLE", " SAFE", "SAFE", "]]"]
)
_JSON_VERDICT = _FakeTokenizer(
    ["why", "\n", '{"', "is", "_v", "ulnerable", '":', " true", " false", "true", "false", "True", "False", "}", " "]
)
_CONFIDENCE_VERDICT = _FakeTokenizer(
    ["why", "\n", '{"', "is", "_v", "ulnerable", '":', " true", " false", "true", "false", "True", "False",
     ",", ' "', "confidence", "}", " "] + [str(d) for d in range(10)] + [f" {d}" for d in range(10)]
)
_CHAR_LEVEL = _FakeTokenizer(sorted(set("why\n[]FINAL_ANSWER: VULNERABLESAFE")))


def _id(tokenizer, token):
    return tokenizer.vocab.index(token)


def _close(actual, expected):
    return actual is not None and abs(actual - expected) < 1e-9


def test_multi_token_label_resolves_to_first_token_variants():
    ids = resolve_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
    assert ids == {
        "VULNERABLE": frozenset({_id(_QWEN_LIKE, " V"), _id(_QWEN_LIKE, "V")}),
        "SAFE": frozenset({_id(_QWEN_LIKE, " SAFE"), _id(_QWEN_LIKE, "SAFE")}),
    }, ids
    print("test_multi_token_label_resolves_to_first_token_variants PASSED")


def test_first_tokens_shared_by_both_labels_are_dropped():
    # Char-level: " SAFE" and " VULNERABLE" both start with the space token,
    # which cannot discriminate the labels; only "S" and "V" remain.
    ids = resolve_label_token_ids(_CHAR_LEVEL, _MARKER_LABELS)
    assert ids == {
        "VULNERABLE": frozenset({_id(_CHAR_LEVEL, "V")}),
        "SAFE": frozenset({_id(_CHAR_LEVEL, "S")}),
    }, ids
    print("test_first_tokens_shared_by_both_labels_are_dropped PASSED")


def test_label_position_is_first_label_token_after_last_marker():
    text = "why\n[[FINAL_ANSWER: SAFE]]\n[[FINAL_ANSWER: VULNERABLE]]"
    token_ids = _QWEN_LIKE.encode(text)
    position, label_token_ids = locate_final_answer_label(_QWEN_LIKE, token_ids)
    assert _QWEN_LIKE.vocab[token_ids[position]] == " V", position
    assert position == len(token_ids) - 5, (position, token_ids)
    assert label_token_ids == resolve_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
    print("test_label_position_is_first_label_token_after_last_marker PASSED")


def test_label_position_skips_separate_whitespace_token():
    token_ids = _CHAR_LEVEL.encode("why\n[[FINAL_ANSWER: SAFE]]")
    position, _ = locate_final_answer_label(_CHAR_LEVEL, token_ids)
    assert _CHAR_LEVEL.vocab[token_ids[position]] == "S", position
    print("test_label_position_skips_separate_whitespace_token PASSED")


def test_no_marker_gives_no_position():
    token_ids = _QWEN_LIKE.encode("why\n")
    assert locate_final_answer_label(_QWEN_LIKE, token_ids) is None
    print("test_no_marker_gives_no_position PASSED")


def test_json_verdict_is_located_with_boolean_label_tokens():
    # The binary prompts ask for {"is_vulnerable": true|false}; an earlier quoted
    # verdict in the reasoning must not win over the final one.
    text = 'why{"is_vulnerable": false}\n{"is_vulnerable": true}'
    token_ids = _JSON_VERDICT.encode(text)
    position, label_token_ids = locate_final_answer_label(_JSON_VERDICT, token_ids)
    assert _JSON_VERDICT.vocab[token_ids[position]] == " true", position
    assert position == len(token_ids) - 2, (position, token_ids)
    assert label_token_ids == {
        "VULNERABLE": frozenset(_id(_JSON_VERDICT, t) for t in (" true", "true", "True")),
        "SAFE": frozenset(_id(_JSON_VERDICT, t) for t in (" false", "false", "False")),
    }, label_token_ids
    print("test_json_verdict_is_located_with_boolean_label_tokens PASSED")


def test_json_verdict_probability_end_to_end():
    token_ids = _JSON_VERDICT.encode('why\n{"is_vulnerable": false}')
    _, label_token_ids = locate_final_answer_label(_JSON_VERDICT, token_ids)
    logprobs = {
        _id(_JSON_VERDICT, " false"): math.log(0.9),
        _id(_JSON_VERDICT, " true"): math.log(0.1),
    }
    assert _close(compute_p_vulnerable(label_token_ids, logprobs), 0.1)
    print("test_json_verdict_probability_end_to_end PASSED")


def test_verdict_is_still_located_when_confidence_follows_it():
    token_ids = _CONFIDENCE_VERDICT.encode('why\n{"is_vulnerable": true, "confidence": 7}')
    position, _ = locate_final_answer_label(_CONFIDENCE_VERDICT, token_ids)
    assert _CONFIDENCE_VERDICT.vocab[token_ids[position]] == " true", position
    print("test_verdict_is_still_located_when_confidence_follows_it PASSED")


def test_stated_confidence_digit_is_located():
    token_ids = _CONFIDENCE_VERDICT.encode('why\n{"is_vulnerable": true, "confidence": 7}')
    position, digit_token_ids = locate_stated_confidence(_CONFIDENCE_VERDICT, token_ids)
    assert _CONFIDENCE_VERDICT.vocab[token_ids[position]] == " 7", position
    assert digit_token_ids[7] == frozenset(
        {_id(_CONFIDENCE_VERDICT, "7"), _id(_CONFIDENCE_VERDICT, " 7")}
    ), digit_token_ids
    assert sorted(digit_token_ids) == list(range(10)), digit_token_ids
    print("test_stated_confidence_digit_is_located PASSED")


def test_response_without_confidence_field_has_no_stated_confidence():
    token_ids = _CONFIDENCE_VERDICT.encode('why\n{"is_vulnerable": true}')
    assert locate_stated_confidence(_CONFIDENCE_VERDICT, token_ids) is None
    print("test_response_without_confidence_field_has_no_stated_confidence PASSED")


def test_stated_confidence_is_expected_digit_scaled_to_unit_interval():
    token_ids = _CONFIDENCE_VERDICT.encode('why\n{"is_vulnerable": true, "confidence": 9}')
    _, digit_token_ids = locate_stated_confidence(_CONFIDENCE_VERDICT, token_ids)
    # Mass 0.4 on 9, 0.4 on 7 (split over both spellings), 0.2 elsewhere:
    # E[digit] = (9 * 0.4 + 7 * 0.4) / 0.8 = 8 -> 8 / 9.
    logprobs = {
        _id(_CONFIDENCE_VERDICT, " 9"): math.log(0.4),
        _id(_CONFIDENCE_VERDICT, " 7"): math.log(0.3),
        _id(_CONFIDENCE_VERDICT, "7"): math.log(0.1),
        _id(_CONFIDENCE_VERDICT, "why"): math.log(0.2),
    }
    assert _close(compute_stated_confidence(digit_token_ids, logprobs), 8 / 9)
    print("test_stated_confidence_is_expected_digit_scaled_to_unit_interval PASSED")


def test_stated_confidence_without_digit_mass_is_none():
    token_ids = _CONFIDENCE_VERDICT.encode('why\n{"is_vulnerable": true, "confidence": 9}')
    _, digit_token_ids = locate_stated_confidence(_CONFIDENCE_VERDICT, token_ids)
    assert compute_stated_confidence(digit_token_ids, {_id(_CONFIDENCE_VERDICT, "why"): 0.0}) is None
    print("test_stated_confidence_without_digit_mass_is_none PASSED")


def test_self_validation_probability_of_true():
    ids = resolve_self_validation_token_ids(_JSON_VERDICT)
    logprobs = {
        _id(_JSON_VERDICT, " true"): math.log(0.6),
        _id(_JSON_VERDICT, " false"): math.log(0.2),
        _id(_JSON_VERDICT, "why"): math.log(0.2),
    }
    assert _close(compute_label_probability(ids, logprobs, True), 0.75)
    assert _close(compute_label_probability(ids, logprobs, False), 0.25)
    print("test_self_validation_probability_of_true PASSED")


def test_p_vulnerable_is_renormalized_over_both_labels():
    ids = resolve_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
    logprobs = {
        _id(_QWEN_LIKE, " V"): math.log(0.6),
        _id(_QWEN_LIKE, " SAFE"): math.log(0.2),
        _id(_QWEN_LIKE, "why"): math.log(0.2),
    }
    assert _close(compute_p_vulnerable(ids, logprobs), 0.75)
    print("test_p_vulnerable_is_renormalized_over_both_labels PASSED")


def test_label_variants_are_summed():
    ids = resolve_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
    logprobs = {
        _id(_QWEN_LIKE, " V"): math.log(0.3),
        _id(_QWEN_LIKE, "V"): math.log(0.1),
        _id(_QWEN_LIKE, " SAFE"): math.log(0.4),
    }
    assert _close(compute_p_vulnerable(ids, logprobs), 0.5)
    print("test_label_variants_are_summed PASSED")


def test_label_missing_from_top_k_counts_as_zero_mass():
    ids = resolve_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
    assert _close(compute_p_vulnerable(ids, {_id(_QWEN_LIKE, " SAFE"): math.log(0.9)}), 0.0)
    assert _close(compute_p_vulnerable(ids, {_id(_QWEN_LIKE, " V"): math.log(0.9)}), 1.0)
    print("test_label_missing_from_top_k_counts_as_zero_mass PASSED")


def test_neither_label_present_gives_none():
    ids = resolve_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
    assert compute_p_vulnerable(ids, {_id(_QWEN_LIKE, "why"): math.log(0.9)}) is None
    print("test_neither_label_present_gives_none PASSED")


if __name__ == "__main__":
    test_multi_token_label_resolves_to_first_token_variants()
    test_first_tokens_shared_by_both_labels_are_dropped()
    test_label_position_is_first_label_token_after_last_marker()
    test_label_position_skips_separate_whitespace_token()
    test_no_marker_gives_no_position()
    test_json_verdict_is_located_with_boolean_label_tokens()
    test_json_verdict_probability_end_to_end()
    test_verdict_is_still_located_when_confidence_follows_it()
    test_stated_confidence_digit_is_located()
    test_response_without_confidence_field_has_no_stated_confidence()
    test_stated_confidence_is_expected_digit_scaled_to_unit_interval()
    test_stated_confidence_without_digit_mass_is_none()
    test_self_validation_probability_of_true()
    test_p_vulnerable_is_renormalized_over_both_labels()
    test_label_variants_are_summed()
    test_label_missing_from_top_k_counts_as_zero_mass()
    test_neither_label_present_gives_none()
    print("ALL PASSED")

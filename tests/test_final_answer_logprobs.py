"""Plain-python checks for final-answer label probability extraction.

Run: PYTHONPATH=src uv run python tests/test_final_answer_logprobs.py
Token splits mirror Qwen3: "VULNERABLE" is multi-token and labels carry a leading space in context.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm.final_answer_logprobs import (
    compute_p_vulnerable,
    locate_final_answer_label,
    resolve_binary_label_token_ids,
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
_CHAR_LEVEL = _FakeTokenizer(sorted(set("why\n[]FINAL_ANSWER: VULNERABLESAFE")))


def _id(tokenizer, token):
    return tokenizer.vocab.index(token)


def _close(actual, expected):
    return actual is not None and abs(actual - expected) < 1e-9


def test_multi_token_label_resolves_to_first_token_variants():
    ids = resolve_binary_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
    assert ids == {
        "VULNERABLE": frozenset({_id(_QWEN_LIKE, " V"), _id(_QWEN_LIKE, "V")}),
        "SAFE": frozenset({_id(_QWEN_LIKE, " SAFE"), _id(_QWEN_LIKE, "SAFE")}),
    }, ids
    print("test_multi_token_label_resolves_to_first_token_variants PASSED")


def test_first_tokens_shared_by_both_labels_are_dropped():
    # Char-level: " SAFE" and " VULNERABLE" both start with the space token,
    # which cannot discriminate the labels; only "S" and "V" remain.
    ids = resolve_binary_label_token_ids(_CHAR_LEVEL, _MARKER_LABELS)
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
    assert label_token_ids == resolve_binary_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
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


def test_p_vulnerable_is_renormalized_over_both_labels():
    ids = resolve_binary_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
    logprobs = {
        _id(_QWEN_LIKE, " V"): math.log(0.6),
        _id(_QWEN_LIKE, " SAFE"): math.log(0.2),
        _id(_QWEN_LIKE, "why"): math.log(0.2),
    }
    assert _close(compute_p_vulnerable(ids, logprobs), 0.75)
    print("test_p_vulnerable_is_renormalized_over_both_labels PASSED")


def test_label_variants_are_summed():
    ids = resolve_binary_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
    logprobs = {
        _id(_QWEN_LIKE, " V"): math.log(0.3),
        _id(_QWEN_LIKE, "V"): math.log(0.1),
        _id(_QWEN_LIKE, " SAFE"): math.log(0.4),
    }
    assert _close(compute_p_vulnerable(ids, logprobs), 0.5)
    print("test_label_variants_are_summed PASSED")


def test_label_missing_from_top_k_counts_as_zero_mass():
    ids = resolve_binary_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
    assert _close(compute_p_vulnerable(ids, {_id(_QWEN_LIKE, " SAFE"): math.log(0.9)}), 0.0)
    assert _close(compute_p_vulnerable(ids, {_id(_QWEN_LIKE, " V"): math.log(0.9)}), 1.0)
    print("test_label_missing_from_top_k_counts_as_zero_mass PASSED")


def test_neither_label_present_gives_none():
    ids = resolve_binary_label_token_ids(_QWEN_LIKE, _MARKER_LABELS)
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
    test_p_vulnerable_is_renormalized_over_both_labels()
    test_label_variants_are_summed()
    test_label_missing_from_top_k_counts_as_zero_mass()
    test_neither_label_present_gives_none()
    print("ALL PASSED")

"""Plain-python checks: binary verdicts in the JSON variants LFM2.5 writes are parsed,
and the answer-probability lookup finds their label token.

Run: PYTHONPATH=src uv run python tests/test_verdict_formats.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark.response_parser import BinaryResponseParser, has_explicit_binary_verdict
from llm.final_answer_logprobs import locate_final_answer_label

PARSER = BinaryResponseParser()


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


_TOKENIZER = _FakeTokenizer(
    ["why", "\n", "{", "}", '"', "\\", "is", "_v", "ulnerable", "verdict", ":", " ",
     " true", " false", "true", "false", "True", "False", "final", "_line", ","]
)


def test_requested_format_still_parses() -> None:
    assert PARSER.parse_response('why\n{"is_vulnerable": true}') == 1
    assert PARSER.parse_response('why\n{"is_vulnerable": false}') == 0


def test_verdict_key_is_parsed_not_keyword_guessed() -> None:
    # The keyword fallback read "not vulnerable" as VULNERABLE before this key was accepted.
    response = 'The code is not vulnerable.\n{"reasoning": "...", "verdict": false}'
    assert PARSER.parse_response(response) == 0
    assert has_explicit_binary_verdict(response)


def test_renamed_keys_and_quoted_booleans() -> None:
    assert PARSER.parse_response('{"final_verdict": true}') == 1
    assert PARSER.parse_response('{"vulnerability_status": false}') == 0
    assert PARSER.parse_response('{"verdict": "true"}') == 1
    assert PARSER.parse_response('{"is_vulnerable": "false"}') == 0


def test_verdict_nested_in_json_string() -> None:
    response = '{"analysis": "vulnerable input", "final_line": "{\\"is_vulnerable\\": false}"}'
    assert PARSER.parse_response(response) == 0


def test_last_verdict_wins_across_formats() -> None:
    assert PARSER.parse_response('{"verdict": false}\n{"is_vulnerable": true}') == 1
    assert PARSER.parse_response('{"is_vulnerable": true}\n{"verdict": false}') == 0


def test_key_prefix_is_not_matched() -> None:
    # "is_vulnerable_input" is not a verdict key; the tail fallback decides instead.
    assert not has_explicit_binary_verdict('{"is_vulnerable_input": true}')


def test_verdict_marker_located_for_probability() -> None:
    token_ids = _TOKENIZER.encode('why\n{"verdict": false}')
    position, _ = locate_final_answer_label(_TOKENIZER, token_ids)
    assert _TOKENIZER.vocab[token_ids[position]] == " false", position


def test_escaped_marker_located_for_probability() -> None:
    token_ids = _TOKENIZER.encode('{"final_line": "{\\"is_vulnerable\\": true}"}')
    position, _ = locate_final_answer_label(_TOKENIZER, token_ids)
    assert _TOKENIZER.vocab[token_ids[position]] == " true", position


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASSED")

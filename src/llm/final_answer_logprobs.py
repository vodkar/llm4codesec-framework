"""Backend-agnostic extraction of P(VULNERABLE) at the final-answer label position."""

import math
from collections import Counter
from collections.abc import Hashable, Mapping
from typing import Final, Protocol

# Answer formats the binary parsers accept: (marker preceding the label, label spellings).
_ANSWER_FORMATS: Final[tuple[tuple[str, Mapping[str, tuple[str, ...]]], ...]] = (
    ('"is_vulnerable":', {"VULNERABLE": ("true", "True"), "SAFE": ("false", "False")}),
    ("[[FINAL_ANSWER:", {"VULNERABLE": ("VULNERABLE",), "SAFE": ("SAFE",)}),
)
_STATED_CONFIDENCE_MARKER: Final[str] = '"confidence":'
_STATED_CONFIDENCE_MAX: Final[int] = 9
_BOOLEAN_TEXTS: Final[Mapping[bool, tuple[str, ...]]] = {
    True: ("true", "True"),
    False: ("false", "False"),
}


class ITokenizer(Protocol):
    """Subset of the tokenizer interface needed to locate label tokens."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool = ...,
    ) -> str: ...


def resolve_label_token_ids[LabelT: Hashable](
    tokenizer: ITokenizer, label_texts: Mapping[LabelT, tuple[str, ...]]
) -> dict[LabelT, frozenset[int]] | None:
    """Resolve the first-token ids that begin each label.

    Labels can be multi-token ("VULNERABLE") and usually follow a space, so the
    first token of both the bare and the space-prefixed spelling is collected.
    Ids shared by several labels cannot discriminate them and are dropped;
    returns None when a label is left without any id.
    """
    first_token_ids: dict[LabelT, set[int]] = {}
    for label, texts in label_texts.items():
        ids: set[int] = set()
        for text in texts:
            for variant in (text, f" {text}"):
                token_ids: list[int] = tokenizer.encode(variant, add_special_tokens=False)
                if token_ids:
                    ids.add(token_ids[0])
        first_token_ids[label] = ids

    id_counts: Counter[int] = Counter(
        token_id for ids in first_token_ids.values() for token_id in ids
    )
    resolved: dict[LabelT, frozenset[int]] = {
        label: frozenset(token_id for token_id in ids if id_counts[token_id] == 1)
        for label, ids in first_token_ids.items()
    }
    if not all(resolved.values()):
        return None
    return resolved


def resolve_self_validation_token_ids(tokenizer: ITokenizer) -> dict[bool, frozenset[int]] | None:
    """Resolve the token ids that answer a true/false self-validation question."""
    return resolve_label_token_ids(tokenizer, _BOOLEAN_TEXTS)


def _locate_value_token_position(
    tokenizer: ITokenizer, token_ids: list[int], generated_text: str, marker_index: int, marker: str
) -> int | None:
    """Find the token index where the value following ``marker`` begins."""
    value_start_char_index: int = marker_index + len(marker)
    while (
        value_start_char_index < len(generated_text)
        and generated_text[value_start_char_index].isspace()
    ):
        value_start_char_index += 1

    if value_start_char_index >= len(generated_text):
        return None

    # The marker sits at the end of the response, so walk back from the last token
    # instead of re-decoding every prefix of a long reasoning trace.
    token_index: int = len(token_ids) - 1
    while token_index > 0:
        decoded_prefix: str = _decode(tokenizer, token_ids[:token_index])
        if len(decoded_prefix) <= value_start_char_index:
            break
        token_index -= 1
    return token_index


def _decode(tokenizer: ITokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def locate_final_answer_label(
    tokenizer: ITokenizer, token_ids: list[int]
) -> tuple[int, dict[str, frozenset[int]]] | None:
    """Find where the last final-answer label begins and which token ids spell each label.

    Returns (token index of the label's first token, label token ids), or None
    when the response carries no recognizable final answer.
    """
    if not token_ids:
        return None

    generated_text: str = _decode(tokenizer, token_ids)
    marker_index, marker, label_texts = max(
        ((generated_text.rfind(marker), marker, texts) for marker, texts in _ANSWER_FORMATS),
        key=lambda candidate: candidate[0],
    )
    if marker_index < 0:
        return None

    label_position: int | None = _locate_value_token_position(
        tokenizer, token_ids, generated_text, marker_index, marker
    )
    label_token_ids: dict[str, frozenset[int]] | None = resolve_label_token_ids(
        tokenizer, label_texts
    )
    if label_position is None or label_token_ids is None:
        return None
    return label_position, label_token_ids


def locate_stated_confidence(
    tokenizer: ITokenizer, token_ids: list[int]
) -> tuple[int, dict[int, frozenset[int]]] | None:
    """Find where the last stated 0-9 confidence digit begins and which token ids spell each digit."""
    if not token_ids:
        return None

    generated_text: str = _decode(tokenizer, token_ids)
    marker_index: int = generated_text.rfind(_STATED_CONFIDENCE_MARKER)
    if marker_index < 0:
        return None

    digit_position: int | None = _locate_value_token_position(
        tokenizer, token_ids, generated_text, marker_index, _STATED_CONFIDENCE_MARKER
    )
    digit_token_ids: dict[int, frozenset[int]] | None = resolve_label_token_ids(
        tokenizer, {digit: (str(digit),) for digit in range(_STATED_CONFIDENCE_MAX + 1)}
    )
    if digit_position is None or digit_token_ids is None:
        return None
    return digit_position, digit_token_ids


def _label_masses[LabelT: Hashable](
    label_token_ids: Mapping[LabelT, frozenset[int]],
    logprob_by_token_id: Mapping[int, float],
) -> dict[LabelT, float]:
    """Sum the probability mass of each label's tokens; absent tokens contribute zero."""
    return {
        label: sum(
            math.exp(logprob_by_token_id[token_id])
            for token_id in token_ids
            if token_id in logprob_by_token_id
        )
        for label, token_ids in label_token_ids.items()
    }


def compute_label_probability[LabelT: Hashable](
    label_token_ids: Mapping[LabelT, frozenset[int]],
    logprob_by_token_id: Mapping[int, float],
    label: LabelT,
) -> float | None:
    """Renormalize probability mass over the known labels and return ``label``'s share.

    A label whose tokens are absent from the returned top-k contributes zero
    mass; returns None when no label is present.
    """
    masses: dict[LabelT, float] = _label_masses(label_token_ids, logprob_by_token_id)
    denominator: float = sum(masses.values())
    if denominator == 0.0:
        return None
    return masses[label] / denominator


def compute_stated_confidence(
    digit_token_ids: Mapping[int, frozenset[int]],
    logprob_by_token_id: Mapping[int, float],
) -> float | None:
    """Return the expected stated 0-9 confidence digit scaled to [0, 1].

    The expectation over the digit distribution is continuous, unlike the sampled
    digit, which clusters on a few values. Returns None when no digit is present.
    """
    masses: dict[int, float] = _label_masses(digit_token_ids, logprob_by_token_id)
    denominator: float = sum(masses.values())
    if denominator == 0.0:
        return None
    expected_digit: float = sum(digit * mass for digit, mass in masses.items()) / denominator
    return expected_digit / _STATED_CONFIDENCE_MAX


def compute_p_vulnerable(
    label_token_ids: Mapping[str, frozenset[int]],
    logprob_by_token_id: Mapping[int, float],
) -> float | None:
    """Renormalize label-position probability mass over the two binary labels."""
    return compute_label_probability(label_token_ids, logprob_by_token_id, "VULNERABLE")

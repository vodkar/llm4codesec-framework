"""Backend-agnostic extraction of P(VULNERABLE) at the final-answer label position."""

import math
from collections.abc import Mapping
from typing import Final, Protocol

# Answer formats the binary parsers accept: (marker preceding the label, label spellings).
_ANSWER_FORMATS: Final[tuple[tuple[str, Mapping[str, tuple[str, ...]]], ...]] = (
    ('"is_vulnerable":', {"VULNERABLE": ("true", "True"), "SAFE": ("false", "False")}),
    ("[[FINAL_ANSWER:", {"VULNERABLE": ("VULNERABLE",), "SAFE": ("SAFE",)}),
)


class ITokenizer(Protocol):
    """Subset of the tokenizer interface needed to locate label tokens."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = ...,
        clean_up_tokenization_spaces: bool = ...,
    ) -> str: ...


def resolve_binary_label_token_ids(
    tokenizer: ITokenizer, label_texts: Mapping[str, tuple[str, ...]]
) -> dict[str, frozenset[int]] | None:
    """Resolve the first-token ids that begin each binary label.

    Labels can be multi-token ("VULNERABLE") and usually follow a space, so the
    first token of both the bare and the space-prefixed spelling is collected.
    Ids shared by both labels cannot discriminate them and are dropped; returns
    None when a label is left without any id.
    """
    first_token_ids: dict[str, set[int]] = {}
    for label, texts in label_texts.items():
        ids: set[int] = set()
        for text in texts:
            for variant in (text, f" {text}"):
                token_ids: list[int] = tokenizer.encode(variant, add_special_tokens=False)
                if token_ids:
                    ids.add(token_ids[0])
        first_token_ids[label] = ids

    shared_ids: set[int] = first_token_ids["VULNERABLE"] & first_token_ids["SAFE"]
    resolved: dict[str, frozenset[int]] = {
        label: frozenset(ids - shared_ids) for label, ids in first_token_ids.items()
    }
    if not all(resolved.values()):
        return None
    return resolved


def locate_final_answer_label(
    tokenizer: ITokenizer, token_ids: list[int]
) -> tuple[int, dict[str, frozenset[int]]] | None:
    """Find where the last final-answer label begins and which token ids spell each label.

    Returns (token index of the label's first token, label token ids), or None
    when the response carries no recognizable final answer.
    """
    if not token_ids:
        return None

    generated_text: str = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    marker_index, marker, label_texts = max(
        ((generated_text.rfind(marker), marker, texts) for marker, texts in _ANSWER_FORMATS),
        key=lambda candidate: candidate[0],
    )
    if marker_index < 0:
        return None

    label_start_char_index: int = marker_index + len(marker)
    while (
        label_start_char_index < len(generated_text)
        and generated_text[label_start_char_index].isspace()
    ):
        label_start_char_index += 1

    if label_start_char_index >= len(generated_text):
        return None

    label_token_ids: dict[str, frozenset[int]] | None = resolve_binary_label_token_ids(
        tokenizer, label_texts
    )
    if label_token_ids is None:
        return None

    # The marker sits at the end of the response, so walk back from the last token
    # instead of re-decoding every prefix of a long reasoning trace.
    token_index: int = len(token_ids) - 1
    while token_index > 0:
        decoded_prefix: str = tokenizer.decode(
            token_ids[:token_index],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(decoded_prefix) <= label_start_char_index:
            break
        token_index -= 1
    return token_index, label_token_ids


def compute_p_vulnerable(
    label_token_ids: Mapping[str, frozenset[int]],
    logprob_by_token_id: Mapping[int, float],
) -> float | None:
    """Renormalize label-position probability mass over the two binary labels.

    A label whose tokens are absent from the returned top-k contributes zero
    mass; returns None when neither label is present.
    """
    mass: dict[str, float] = {
        label: sum(
            math.exp(logprob_by_token_id[token_id])
            for token_id in token_ids
            if token_id in logprob_by_token_id
        )
        for label, token_ids in label_token_ids.items()
    }
    denominator: float = mass["VULNERABLE"] + mass["SAFE"]
    if denominator == 0.0:
        return None
    return mass["VULNERABLE"] / denominator

"""Deterministic sampling seeds so every condition sees identical draws per item."""

import hashlib
from typing import Final

_SEED_MODULUS: Final[int] = 2**31 - 1


def draw_seed(sampling_seed: int, sample_id: str, draw_index: int) -> int:
    """Derive the vLLM seed for one self-consistency draw of one sample.

    Depends only on the global seed, the sample id and the draw index, so the
    same item gets the same seeds in every condition regardless of dataset order.

    Args:
        sampling_seed: Global pinned seed.
        sample_id: Framework sample id (e.g. ``<pair_id>_vuln``).
        draw_index: 0-based self-consistency draw index.

    Returns:
        Seed in ``[0, 2**31 - 1)``.
    """

    digest: bytes = hashlib.sha256(f"{sampling_seed}|{sample_id}|{draw_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS

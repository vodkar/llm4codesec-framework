"""Write v2 (4096-token) and v2_6k (6144-token) subsets holding only the samples whose
context differs between the two budgets, plus their vulnerable/fixed partners.

Run: uv run python scripts/build_ctx_size_subsets.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "benchmarks" / "context-assembler-dataset"
OUT = REPO / "datasets_processed" / "context_assembler" / "v2_4k_vs_6k"


def contexts(sample: dict) -> list:
    return [root.get("context") for root in sample["roots"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for strategy in ("cpg_structural", "multiplicative_amplification"):
        small = json.loads((SRC / "v2" / f"context_assembler_{strategy}.json").read_text())
        large = json.loads((SRC / "v2_6k" / f"context_assembler_{strategy}.json").read_text())
        assert [s["id"] for s in small["samples"]] == [s["id"] for s in large["samples"]]
        changed_pairs = {
            str(a["metadata"]["source_row_ids"])
            for a, b in zip(small["samples"], large["samples"])
            if contexts(a) != contexts(b)
        }
        for size, data in (("4k", small), ("6k", large)):
            samples = [
                s for s in data["samples"] if str(s["metadata"]["source_row_ids"]) in changed_pairs
            ]
            meta = dict(data["metadata"], total_samples=len(samples))
            path = OUT / f"{strategy}_{size}_changed.json"
            path.write_text(json.dumps({"metadata": meta, "samples": samples}))
            print(path.relative_to(REPO), len(samples))


if __name__ == "__main__":
    main()

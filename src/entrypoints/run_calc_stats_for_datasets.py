import json
from pathlib import Path

datasets_path = Path("datasets_processed")

castle_dataset = datasets_path / "castle" / "castle_binary.json"
data = json.loads(castle_dataset.read_text())

stats = {}
for sample in data["samples"]:
    stats[sample["id"]] = {
        "code_len": len(sample["code"]),
    }

print("Average code length:", sum(s["code_len"] for s in stats.values()) / len(stats))
print(
    "Median code length:",
    sorted(s["code_len"] for s in stats.values())[len(stats) // 2],
)

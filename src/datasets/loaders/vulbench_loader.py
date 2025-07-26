from datasets.loaders.base import IDatasetLoader
from benchmark.models import BenchmarkSample


import pandas as pd


import json
from pathlib import Path


class VulBenchLoader(IDatasetLoader):
    """Loader for VulBench dataset format."""

    def load_dataset(self, path: Path) -> list[BenchmarkSample]:
        """
        Load VulBench dataset.

        Args:
            path (Path): Path to the dataset file

        Returns:
            list[BenchmarkSample]: Loaded samples
        """
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        samples: list[BenchmarkSample] = []

        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for i, item in enumerate(data):
                sample = BenchmarkSample(
                    id=item.get("id", f"sample_{i}"),
                    code=item["code"],
                    label=item["label"],
                    metadata=item.get("metadata", {}),
                    cwe_types=item.get("cwe_type"),
                    severity=item.get("severity"),
                )
                samples.append(sample)

        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            for idx, row in df.iterrows():
                sample = BenchmarkSample(
                    id=row.get("id", f"sample_{idx}"),
                    code=row["code"],
                    label=row["label"],
                    metadata={"source_file": row.get("source_file", "")},
                    cwe_types=row.get("cwe_type"),
                    severity=row.get("severity"),
                )
                samples.append(sample)

        return samples
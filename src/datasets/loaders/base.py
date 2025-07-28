import json
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from benchmark.models import BenchmarkSample


class IDatasetLoader(ABC, BaseModel):
    """Interface for dataset loading implementations."""

    @abstractmethod
    def load_dataset(self, path: Path) -> list[BenchmarkSample]:
        """Load dataset from the specified path."""
        ...


class JsonDatasetLoader(IDatasetLoader):
    """Concrete implementation of IDatasetLoader for JSON datasets."""

    def load_dataset(self, path: Path) -> list[BenchmarkSample]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples: list[BenchmarkSample] = []
        for sample_dict in data["samples"]:
            sample = BenchmarkSample(
                id=sample_dict["id"],
                code=sample_dict["code"],
                label=sample_dict["label"],
                metadata=sample_dict["metadata"],
                cwe_types=sample_dict.get("cwe_type"),
                severity=sample_dict.get("severity"),
            )
            samples.append(sample)

        return samples

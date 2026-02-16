import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, TypedDict, Unpack

from pydantic import BaseModel, ConfigDict

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, SampleCollection


class DatasetLoadParams(TypedDict, total=False):
    task_type: TaskType
    programming_language: str | None
    change_level: Literal["file", "method"] | None
    limit: int | None
    target_cwe: str | None


class IDatasetLoader(ABC, BaseModel):
    """Interface for dataset loading implementations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def load_dataset(self, **kwargs: Unpack[DatasetLoadParams]) -> SampleCollection:
        """Load dataset from the specified path."""
        ...


class JsonDatasetLoader:
    """Loader for processed JSON benchmark datasets."""

    def load_dataset(self, dataset_path: Path | str) -> list[BenchmarkSample]:
        """
        Load a processed benchmark dataset from JSON.

        Args:
            dataset_path: Path to the JSON dataset file.

        Returns:
            list of BenchmarkSample objects.
        """
        resolved_path: Path = Path(dataset_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {resolved_path}")

        with open(resolved_path, "r", encoding="utf-8") as handle:
            payload: dict[str, Any] = json.load(handle)

        raw_samples: list[dict[str, Any]] = list(payload.get("samples", []))
        return [BenchmarkSample.model_validate(sample) for sample in raw_samples]

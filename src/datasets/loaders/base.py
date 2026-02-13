from abc import ABC, abstractmethod
from typing import Literal, TypedDict, Unpack

from pydantic import BaseModel, ConfigDict

from benchmark.models import SampleCollection
from src.benchmark.enums import TaskType


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

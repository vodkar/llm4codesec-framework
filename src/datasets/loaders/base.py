from benchmark.models import BenchmarkSample


from abc import ABC, abstractmethod
from pathlib import Path


class IDatasetLoader(ABC):
    """Interface for dataset loading implementations."""

    @abstractmethod
    def load_dataset(self, path: Path) -> list[BenchmarkSample]:
        """Load dataset from the specified path."""
        ...
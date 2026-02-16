from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, RootModel, field_validator

from benchmark.enums import TaskType


class BenchmarkSample(BaseModel):
    """Data structure for a single benchmark sample."""

    id: str
    code: str
    label: int | str
    metadata: dict[str, Any]
    cwe_types: list[str] | None = None
    severity: str | None = None

    @field_validator("cwe_types", mode="before")
    @classmethod
    def validate_cwe_types(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        for cwe_type in v:
            if cwe_type != "SAFE" and not cwe_type.startswith("CWE-"):
                raise ValueError(f"Invalid CWE type format: {cwe_type}")
        return list(v)


class PredictionResult(BaseModel):
    """Data structure for model prediction results."""

    sample_id: str
    predicted_label: int | str
    true_label: int | str
    confidence: float | None
    response_text: str
    processing_time: float
    tokens_used: int | None = None


class SampleCollection(RootModel[list[BenchmarkSample]]):
    """Data structure for a collection of benchmark samples."""

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, index: int) -> BenchmarkSample:
        return self.root[index]

    def __iter__(self) -> Iterator[BenchmarkSample]:  # type: ignore
        return iter(self.root)

    @property
    def available_cwes(self) -> list[str]:
        """
        Get list of available CWE types in the dataset.

        Args:
            samples: list of benchmark samples

        Returns:
            list[str]: Available CWE types
        """
        cwes: set[str] = set()
        for sample in self.root:
            cwe_num = sample.metadata.get("cwe_number", 0)
            if cwe_num > 0:
                cwes.add(f"CWE-{cwe_num}")

        return sorted(list(cwes))

    def filter_by_cwe(self, target_cwe: str) -> "SampleCollection":
        """
        Filter samples by specific CWE type.

        Args:
            samples: list of benchmark samples
            target_cwe: Target CWE type (e.g., "CWE-125")

        Returns:
            "SampleCollection": Filtered samples
        """
        filtered: list[BenchmarkSample] = []
        for sample in self.root:
            sample_cwe = f"CWE-{sample.metadata.get('cwe_number', 0)}"
            if sample_cwe == target_cwe or sample.cwe_types == ["SAFE"]:
                # Create a copy with binary label for CWE-specific task
                sample_copy = BenchmarkSample(
                    id=sample.id,
                    code=sample.code,
                    label=1 if sample_cwe == target_cwe else 0,
                    metadata=sample.metadata.copy(),
                    cwe_types=sample.cwe_types,
                    severity=sample.severity,
                )
                filtered.append(sample_copy)

        return SampleCollection.model_validate(filtered)


class DatasetMetadata(BaseModel):
    """Metadata for the benchmark dataset."""

    name: str
    version: str
    task_type: TaskType
    total_samples: int = Field(default=0)
    vulnerable_samples: int = Field(default=0)
    safe_samples: int = Field(default=0)
    cwe_type: str | None = None
    cwe_distribution: dict[str, int] | None = None
    programming_language: str
    change_level: str | None = None


class Dataset(BaseModel):
    """Benchmark dataset structure."""

    metadata: DatasetMetadata
    samples: SampleCollection

    def model_post_init(self, context: Any) -> None:
        if self.metadata.cwe_distribution is None:
            cwe_counts: dict[str, int] = {}
            for sample in self.samples:
                cwe = sample.metadata.get("cwe_number", 0)
                cwe_key = f"CWE-{cwe}" if cwe > 0 else "SAFE"
                cwe_counts[cwe_key] = cwe_counts.get(cwe_key, 0) + 1

            self.metadata.cwe_distribution = cwe_counts

        self.metadata.total_samples = len(self.samples)
        self.metadata.vulnerable_samples = sum(
            1 for s in self.samples if s.metadata.get("cwe_number", 0) > 0
        )
        self.metadata.safe_samples = sum(
            1 for s in self.samples if s.metadata.get("cwe_number", 0) == 0
        )

        return super().model_post_init(context)


class CWEType(str):
    def get_cwe_severity(self) -> str:
        """
        Determine severity level for a CWE.

        Args:
            cwe: CWE identifier (string or list of strings, e.g., "CWE-125" or ["CWE-125"])

        Returns:
            Severity level (HIGH, MEDIUM, LOW)
        """
        # Handle both string and list formats
        if isinstance(self, list):
            if not self or not self[0]:
                return "LOW"
            single_cwe = self[0]
        else:
            single_cwe = self
        # Extract numeric part
        cwe_num = single_cwe.replace("CWE-", "")

        # High severity CWEs
        high_severity_cwes = {
            "78",  # OS Command Injection
            "79",  # Cross-site Scripting
            "89",  # SQL Injection
            "94",  # Code Injection
            "352",  # CSRF
            "434",  # Unrestricted Upload
            "611",  # XML External Entities
        }

        # Medium severity CWEs
        medium_severity_cwes = {
            "125",  # Out-of-bounds Read
            "190",  # Integer Overflow
            "787",  # Out-of-bounds Write
            "476",  # NULL Pointer Dereference
            "416",  # Use After Free
            "502",  # Deserialization
        }

        if cwe_num in high_severity_cwes:
            return "HIGH"
        elif cwe_num in medium_severity_cwes:
            return "MEDIUM"
        else:
            return "LOW"

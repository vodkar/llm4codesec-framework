#!/usr/bin/env python3
"""
CVEFixes Dataset Loader and Integration

This module provides functionality to load and process the CVEFixes benchmark dataset
for use with the LLM code security benchmark framework.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Unpack

from pydantic import PrivateAttr, field_validator

from benchmark.enums import TaskType
from benchmark.models import BenchmarkSample, Dataset, DatasetMetadata, SampleCollection
from datasets.loaders.base import DatasetLoadParams, IDatasetLoader


class CVEFixesDatasetLoader(IDatasetLoader):
    """Loads and processes CVEFixes benchmark dataset from SQLite database."""

    database_path: Path
    __logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger(__name__)
    )
    __conn: sqlite3.Connection | None = PrivateAttr(default=None)

    @field_validator("database_path")
    def validate_database_path(cls, v: Path) -> Path:
        if not v.exists() or not v.is_file():
            raise ValueError(f"Database file not found: {v}")
        return v

    def _create_connection(self) -> sqlite3.Connection:
        """Create a connection to the SQLite database."""
        try:
            return sqlite3.connect(str(self.database_path), timeout=10)
        except sqlite3.Error:
            self.__logger.exception("Error connecting to database")
            raise

    @staticmethod
    def _normalize_cwe_type(cwe_id: str | None) -> str:
        """Normalize raw CWE identifiers to benchmark-compatible format."""
        if not cwe_id:
            return "CWE-UNKNOWN"

        normalized_cwe_id: str = cwe_id.strip().upper()
        if normalized_cwe_id.startswith("CWE-"):
            return normalized_cwe_id

        if normalized_cwe_id.isdigit():
            return f"CWE-{normalized_cwe_id}"

        return "CWE-UNKNOWN"

    def _get_cwe_distribution(self) -> dict[str, int]:
        """Get distribution of CWE types in the database."""
        if not self.__conn:
            self.__conn = self._create_connection()

        query = """
        SELECT cc.cwe_id, COUNT(*) as count
        FROM cwe_classification cc
        GROUP BY cc.cwe_id
        ORDER BY count DESC
        """

        cursor = self.__conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()

        return {f"CWE-{cwe_id}": count for cwe_id, count in results if cwe_id}

    def _extract_file_level_data(
        self,
        programming_language: str = "C",
        limit: int | None = None,
        target_cve_ids: set[str] | None = None,
    ) -> list[
        tuple[
            str,
            str,
            str,
            float | None,
            str,
            str,
            str,
            str,
            int,
            int,
            str,
            str,
            str | None,
        ]
    ]:
        """
        Extract file-level vulnerability data from CVEFixes database.

        Args:
            programming_language: Filter by programming language
            limit: Maximum number of samples to extract

        Returns:
            list of tuples containing file-level data
        """
        if not self.__conn:
            self.__conn = self._create_connection()

        query = """
        SELECT 
            cv.cve_id,
            cv.description,
            cv.published_date,
            cv.severity,
            f.filename,
            f.programming_language,
            f.code_before,
            f.code_after,
            f.num_lines_added,
            f.num_lines_deleted,
            c.hash as commit_hash,
            fx.repo_url,
            cc.cwe_id
        FROM cve cv
        JOIN fixes fx ON cv.cve_id = fx.cve_id
        JOIN commits c ON fx.hash = c.hash
        JOIN file_change f ON c.hash = f.hash
        LEFT JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
        WHERE f.programming_language = ?
        AND f.code_before IS NOT NULL
        AND f.code_after IS NOT NULL
        AND LENGTH(f.code_before) > 50
        AND LENGTH(f.code_after) > 50
        """

        params: list[str | int] = [programming_language]

        if target_cve_ids:
            placeholders: str = ",".join("?" * len(target_cve_ids))
            query += f" AND cv.cve_id IN ({placeholders})"
            params.extend(sorted(target_cve_ids))

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.__conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def _extract_method_level_data(
        self,
        programming_language: str = "C",
        limit: int | None = None,
        target_cve_ids: set[str] | None = None,
    ) -> list[
        tuple[
            str,
            str,
            str,
            float | None,
            str,
            str,
            str,
            str,
            str,
            str,
            int,
            int,
            str,
            str,
            str | None,
        ]
    ]:
        """
        Extract method-level vulnerability data from CVEFixes database.

        Args:
            programming_language: Filter by programming language
            limit: Maximum number of samples to extract

        Returns:
            list of tuples containing method-level data
        """
        if not self.__conn:
            self.__conn = self._create_connection()

        query = """
        SELECT 
            cv.cve_id,
            cv.description,
            cv.published_date,
            cv.severity,
            f.filename,
            f.programming_language,
            m.name as method_name,
            m.signature,
            m.code,
            m.before_change,
            m.nloc,
            m.token_count,
            c.hash as commit_hash,
            fx.repo_url,
            cc.cwe_id
        FROM cve cv
        JOIN fixes fx ON cv.cve_id = fx.cve_id
        JOIN commits c ON fx.hash = c.hash
        JOIN file_change f ON c.hash = f.hash
        JOIN method_change m ON f.file_change_id = m.file_change_id
        LEFT JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
        WHERE f.programming_language = ?
        AND m.code IS NOT NULL
        AND m.before_change IS NOT NULL
        AND LENGTH(m.code) > 20
        AND LENGTH(m.before_change) > 20
        """

        params: list[str | int] = [programming_language]

        if target_cve_ids:
            placeholders: str = ",".join("?" * len(target_cve_ids))
            query += f" AND cv.cve_id IN ({placeholders})"
            params.extend(sorted(target_cve_ids))

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.__conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def _create_sample_from_file_data(
        self,
        data: tuple[
            str,
            str,
            str,
            float | None,
            str,
            str,
            str,
            str,
            int,
            int,
            str,
            str,
            str | None,
        ],
        index: int,
    ) -> BenchmarkSample:
        """Create a BenchmarkSample from file-level data."""
        (
            cve_id,
            description,
            published_date,
            severity,
            filename,
            programming_language,
            code_before,
            code_after,
            lines_added,
            lines_deleted,
            commit_hash,
            repo_url,
            cwe_id,
        ) = data

        # Create sample ID
        sample_id = f"{cve_id}_file_{index}"

        # Use vulnerable code (before fix) as the code to analyze
        code = code_before

        # Create metadata
        metadata: dict[str, Any] = {
            "cve_id": cve_id,
            "cwe_id": cwe_id,
            "severity": severity,
            "description": description,
            "published_date": published_date,
            "programming_language": programming_language,
            "filename": filename,
            "commit_hash": commit_hash,
            "repo_url": repo_url,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "change_type": "file",
            "code_after": code_after,  # Keep for reference
        }

        # Determine labels
        cwe_type: str = self._normalize_cwe_type(cwe_id)
        binary_label = 1  # All samples from CVEFixes are vulnerable by definition

        return BenchmarkSample(
            id=sample_id,
            code=code,
            label=binary_label,
            metadata=metadata,
            cwe_types=[cwe_type],
            severity=self._map_severity(severity)
            if isinstance(severity, (int, float))
            else severity,
        )

    def _create_sample_from_method_data(
        self,
        data: tuple[
            str,
            str,
            str,
            float | None,
            str,
            str,
            str,
            str,
            str,
            str,
            int,
            int,
            str,
            str,
            str | None,
        ],
        index: int,
    ) -> BenchmarkSample:
        """Create a BenchmarkSample from method-level data."""
        (
            cve_id,
            description,
            published_date,
            severity,
            filename,
            programming_language,
            method_name,
            signature,
            code,
            before_change,
            nloc,
            token_count,
            commit_hash,
            repo_url,
            cwe_id,
        ) = data

        # Create sample ID
        sample_id = f"{cve_id}_method_{index}"

        # Use vulnerable code (before change) as the code to analyze
        code_to_analyze = before_change

        # Create metadata
        metadata: dict[str, Any] = {
            "cve_id": cve_id,
            "cwe_id": cwe_id,
            "severity": severity,
            "description": description,
            "published_date": published_date,
            "programming_language": programming_language,
            "filename": filename,
            "method_name": method_name,
            "signature": signature,
            "nloc": nloc,
            "token_count": token_count,
            "commit_hash": commit_hash,
            "repo_url": repo_url,
            "change_type": "method",
            "code_after": code,  # Keep for reference
        }

        # Determine labels
        cwe_type: str = self._normalize_cwe_type(cwe_id)
        binary_label = 1  # All samples from CVEFixes are vulnerable by definition

        return BenchmarkSample(
            id=sample_id,
            code=code_to_analyze,
            label=binary_label,
            metadata=metadata,
            cwe_types=[cwe_type],
            severity=self._map_severity(severity)
            if isinstance(severity, (int, float))
            else severity,
        )

    def _map_severity(self, severity: float | None) -> str | None:
        """Map numeric CVSS severity to categorical severity."""
        if severity is None:
            return None

        if severity >= 9.0:
            return "CRITICAL"
        elif severity >= 7.0:
            return "HIGH"
        elif severity >= 4.0:
            return "MEDIUM"
        elif severity > 0.0:
            return "LOW"
        else:
            return "NONE"

    def load_dataset(self, **kwargs: Unpack[DatasetLoadParams]) -> SampleCollection:
        """
        Load CVEFixes dataset and convert to BenchmarkSample format.

        Args:
            task_type: Type of task (binary, multiclass, cwe_specific)
            programming_language: Programming language to filter by
            change_level: Level of change to analyze (file or method)
            limit: Maximum number of samples to load

        Returns:
            list of BenchmarkSample objects
        """
        samples: list[BenchmarkSample] = []
        change_level = kwargs.get("change_level") or "file"
        programming_language = kwargs.get("programming_language") or "C"
        limit = kwargs.get("limit", None)
        task_type = kwargs.get("task_type", TaskType.BINARY_VULNERABILITY)
        target_cwe = (kwargs.get("target_cwe") or "").upper()
        target_cve_ids: set[str] | None = kwargs.get("target_cve_ids")

        try:
            self.__conn = self._create_connection()

            if change_level == "file":
                file_data_rows = self._extract_file_level_data(
                    programming_language, limit, target_cve_ids
                )
                for i, file_row in enumerate(file_data_rows):
                    try:
                        sample = self._create_sample_from_file_data(file_row, i)

                        # Apply task-specific label adjustments
                        if task_type == TaskType.BINARY_VULNERABILITY:
                            sample.label = 1  # All CVEFixes samples are vulnerable
                        elif task_type == TaskType.MULTICLASS_VULNERABILITY:
                            if sample.cwe_types:
                                sample.label = sample.cwe_types[0]
                            else:
                                sample.label = "UNKNOWN"
                        elif task_type == TaskType.BINARY_CWE_SPECIFIC:
                            if not target_cwe:
                                raise ValueError(
                                    "target_cwe must be specified for CWE-specific tasks"
                                )

                            if sample.cwe_types:
                                sample.label = (
                                    1 if target_cwe in sample.cwe_types else 0
                                )
                            else:
                                sample.label = 0

                        samples.append(sample)

                    except Exception as e:
                        self.__logger.exception(
                            f"Error processing file sample {i}: {e}"
                        )
                        continue

            elif change_level == "method":
                method_data_rows = self._extract_method_level_data(
                    programming_language, limit, target_cve_ids
                )
                for i, method_row in enumerate(method_data_rows):
                    try:
                        sample = self._create_sample_from_method_data(method_row, i)

                        # Apply task-specific label adjustments
                        if task_type == TaskType.BINARY_VULNERABILITY:
                            sample.label = 1  # All CVEFixes samples are vulnerable
                        elif task_type == TaskType.MULTICLASS_VULNERABILITY:
                            if sample.cwe_types:
                                sample.label = sample.cwe_types[0]
                            else:
                                sample.label = "UNKNOWN"
                        elif task_type == TaskType.BINARY_CWE_SPECIFIC:
                            if not target_cwe:
                                raise ValueError(
                                    "target_cwe must be specified for CWE-specific tasks"
                                )
                            if sample.cwe_types:
                                sample.label = (
                                    1 if target_cwe in sample.cwe_types else 0
                                )
                            else:
                                sample.label = 0

                        samples.append(sample)

                    except Exception as e:
                        self.__logger.exception(
                            f"Error processing method sample {i}: {e}"
                        )
                        continue
            else:
                raise ValueError(f"Unsupported change_level: {change_level}")

        finally:
            if self.__conn:
                self.__conn.close()

        self.__logger.info(f"Loaded {len(samples)} samples from CVEFixes dataset")
        return SampleCollection(samples)

    def create_dataset_json(
        self,
        output_path: str,
        **kwargs: Unpack[DatasetLoadParams],
    ) -> None:
        """
        Create a JSON dataset file compatible with the benchmark framework.

        Args:
            output_path: Path to output JSON file
            task_type: Type of task classification
            programming_language: Programming language to filter by
            change_level: Level of change to analyze (file or method)
            limit: Maximum number of samples to include
        """
        task_type = kwargs.get("task_type", TaskType.BINARY_VULNERABILITY)
        programming_language = kwargs.get("programming_language") or "C"
        change_level = kwargs.get("change_level", "file")
        limit = kwargs.get("limit", None)
        target_cwe = kwargs.get("target_cwe")
        target_cve_ids: set[str] | None = kwargs.get("target_cve_ids")

        samples = self.load_dataset(
            task_type=task_type,
            programming_language=programming_language,
            change_level=change_level,
            limit=limit,
            target_cwe=target_cwe,
            target_cve_ids=target_cve_ids,
        )

        # Calculate statistics
        cwe_distribution: dict[str, int] = {}
        severity_distribution: dict[str, int] = {}

        for sample in samples:
            # CWE distribution
            cwe = (sample.cwe_types or ["UNKNOWN"])[0]
            cwe_distribution[cwe] = cwe_distribution.get(cwe, 0) + 1

            # Severity distribution
            severity = sample.severity or "UNKNOWN"
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1

        dataset = Dataset(
            metadata=DatasetMetadata(
                name="CVEFixes-Benchmark",
                version="1.0",
                task_type=task_type,
                programming_language=programming_language,
                change_level=change_level,
            ),
            samples=samples,
        )

        # Write to file
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "w", encoding="utf-8") as f:
            json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)

        self.__logger.info(
            f"Created dataset JSON with {len(samples)} samples at {output_path}"
        )

    def get_database_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the CVEFixes database."""
        if not self.__conn:
            self.__conn = self._create_connection()

        stats: dict[str, Any] = {}

        try:
            # Basic table counts
            tables = [
                "cve",
                "fixes",
                "commits",
                "repository",
                "file_change",
                "method_change",
            ]
            for table in tables:
                cursor = self.__conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                result = cursor.fetchone()
                stats[f"{table}_count"] = result[0] if result else 0

            # Programming language distribution
            cursor = self.__conn.cursor()
            cursor.execute("""
                SELECT programming_language, COUNT(*) as count
                FROM file_change
                WHERE programming_language IS NOT NULL
                GROUP BY programming_language
                ORDER BY count DESC
            """)
            stats["programming_languages"] = dict(cursor.fetchall())

            # CWE distribution
            stats["cwe_distribution"] = self._get_cwe_distribution()

            # Severity distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN severity >= 9.0 THEN 'CRITICAL'
                        WHEN severity >= 7.0 THEN 'HIGH'
                        WHEN severity >= 4.0 THEN 'MEDIUM'
                        WHEN severity > 0.0 THEN 'LOW'
                        ELSE 'NONE'
                    END as severity_category,
                    COUNT(*) as count
                FROM cve
                WHERE severity IS NOT NULL
                GROUP BY severity_category
                ORDER BY count DESC
            """)
            stats["severity_distribution"] = dict(cursor.fetchall())

        finally:
            if self.__conn:
                self.__conn.close()

        return stats

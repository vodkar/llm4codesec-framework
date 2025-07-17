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
from typing import Any, Dict, List, Optional, Tuple, Union

from benchmark.benchmark_framework import BenchmarkSample, DatasetLoader


class CVEFixesDatasetLoader:
    """Loads and processes CVEFixes benchmark dataset from SQLite database."""

    def __init__(self, database_path: str = "benchmarks/CVEfixes/Data/CVEfixes.db"):
        """
        Initialize the CVEFixes dataset loader.

        Args:
            database_path: Path to the CVEFixes SQLite database
        """
        self.database_path = Path(database_path)
        self.logger = logging.getLogger(__name__)
        self.conn: Optional[sqlite3.Connection] = None

        if not self.database_path.exists():
            raise FileNotFoundError(
                f"CVEFixes database not found at {database_path}. "
                "Please ensure the database is downloaded and placed in the correct location."
            )

    def _create_connection(self) -> sqlite3.Connection:
        """Create a connection to the SQLite database."""
        try:
            return sqlite3.connect(str(self.database_path), timeout=10)
        except sqlite3.Error as e:
            self.logger.exception(f"Error connecting to database: {e}")
            raise

    def _get_cwe_distribution(self) -> Dict[str, int]:
        """Get distribution of CWE types in the database."""
        if not self.conn:
            self.conn = self._create_connection()

        query = """
        SELECT cc.cwe_id, COUNT(*) as count
        FROM cwe_classification cc
        GROUP BY cc.cwe_id
        ORDER BY count DESC
        """

        cursor = self.conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()

        return {f"CWE-{cwe_id}": count for cwe_id, count in results if cwe_id}

    def _extract_file_level_data(
        self, programming_language: str = "C", limit: Optional[int] = None
    ) -> List[
        Tuple[
            str,
            str,
            str,
            Optional[float],
            str,
            str,
            str,
            str,
            int,
            int,
            str,
            str,
            Optional[str],
        ]
    ]:
        """
        Extract file-level vulnerability data from CVEFixes database.

        Args:
            programming_language: Filter by programming language
            limit: Maximum number of samples to extract

        Returns:
            List of tuples containing file-level data
        """
        if not self.conn:
            self.conn = self._create_connection()

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

        params: List[Union[str, int]] = [programming_language]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def _extract_method_level_data(
        self, programming_language: str = "C", limit: Optional[int] = None
    ) -> List[
        Tuple[
            str,
            str,
            str,
            Optional[float],
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
            Optional[str],
        ]
    ]:
        """
        Extract method-level vulnerability data from CVEFixes database.

        Args:
            programming_language: Filter by programming language
            limit: Maximum number of samples to extract

        Returns:
            List of tuples containing method-level data
        """
        if not self.conn:
            self.conn = self._create_connection()

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

        params: List[Union[str, int]] = [programming_language]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def _create_sample_from_file_data(
        self,
        data: Tuple[
            str,
            str,
            str,
            Optional[float],
            str,
            str,
            str,
            str,
            int,
            int,
            str,
            str,
            Optional[str],
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
        metadata: Dict[str, Any] = {
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
        cwe_type = cwe_id if cwe_id else "UNKNOWN"
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
        data: Tuple[
            str,
            str,
            str,
            Optional[float],
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
            Optional[str],
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
        metadata: Dict[str, Any] = {
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
        cwe_type = f"CWE-{cwe_id}" if cwe_id else "UNKNOWN"
        binary_label = 1  # All samples from CVEFixes are vulnerable by definition

        return BenchmarkSample(
            id=sample_id,
            code=code_to_analyze,
            label=binary_label,
            metadata=metadata,
            cwe=cwe_type,
            severity=self._map_severity(severity)
            if isinstance(severity, (int, float))
            else severity,
        )

    def _map_severity(self, severity: Optional[float]) -> Optional[str]:
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

    def load_dataset(
        self,
        task_type: str = "binary",
        programming_language: str = "C",
        change_level: str = "file",
        limit: Optional[int] = None,
    ) -> List[BenchmarkSample]:
        """
        Load CVEFixes dataset and convert to BenchmarkSample format.

        Args:
            task_type: Type of task (binary, multiclass, cwe_specific)
            programming_language: Programming language to filter by
            change_level: Level of change to analyze (file or method)
            limit: Maximum number of samples to load

        Returns:
            List of BenchmarkSample objects
        """
        samples: List[BenchmarkSample] = []

        try:
            self.conn = self._create_connection()

            if change_level == "file":
                data_rows = self._extract_file_level_data(programming_language, limit)
                for i, data in enumerate(data_rows):
                    try:
                        sample = self._create_sample_from_file_data(data, i)

                        # Apply task-specific label adjustments
                        if task_type == "binary":
                            sample.label = 1  # All CVEFixes samples are vulnerable
                        elif task_type == "multiclass":
                            if sample.cwe_types:
                                sample.label = sample.cwe_types[0]
                            else:
                                sample.label = "UNKNOWN"
                        elif task_type.startswith("CWE-"):
                            target_cwe = task_type.upper()
                            sample.label = 1 if target_cwe in sample.cwe_types else 0

                        samples.append(sample)

                    except Exception as e:
                        self.logger.exception(f"Error processing file sample {i}: {e}")
                        continue

            elif change_level == "method":
                data_rows = self._extract_method_level_data(programming_language, limit)
                for i, data in enumerate(data_rows):
                    try:
                        sample = self._create_sample_from_method_data(data, i)

                        # Apply task-specific label adjustments
                        if task_type == "binary":
                            sample.label = 1  # All CVEFixes samples are vulnerable
                        elif task_type == "multiclass":
                            if sample.cwe_types:
                                sample.label = sample.cwe_types
                            else:
                                sample.label = "UNKNOWN"
                        elif task_type.startswith("cwe_"):
                            target_cwe = task_type.upper()
                            sample.label = 1 if sample.cwe_types == target_cwe else 0

                        samples.append(sample)

                    except Exception as e:
                        self.logger.exception(
                            f"Error processing method sample {i}: {e}"
                        )
                        continue
            else:
                raise ValueError(f"Unsupported change_level: {change_level}")

        finally:
            if self.conn:
                self.conn.close()

        self.logger.info(f"Loaded {len(samples)} samples from CVEFixes dataset")
        return samples

    def create_dataset_json(
        self,
        output_path: str,
        task_type: str = "binary",
        programming_language: str = "C",
        change_level: str = "file",
        limit: Optional[int] = None,
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
        samples = self.load_dataset(
            task_type, programming_language, change_level, limit
        )

        # Calculate statistics
        cwe_distribution: Dict[str, int] = {}
        severity_distribution: Dict[str, int] = {}

        for sample in samples:
            # CWE distribution
            cwe = (sample.cwe_types or ["UNKNOWN"])[0]
            cwe_distribution[cwe] = cwe_distribution.get(cwe, 0) + 1

            # Severity distribution
            severity = sample.severity or "UNKNOWN"
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1

        # Create dataset dictionary
        dataset_dict: Dict[str, Any] = {
            "metadata": {
                "name": "CVEFixes-Benchmark",
                "version": "1.0",
                "task_type": task_type,
                "programming_language": programming_language,
                "change_level": change_level,
                "total_samples": len(samples),
                "vulnerable_samples": len(
                    samples
                ),  # All CVEFixes samples are vulnerable
                "cwe_distribution": cwe_distribution,
                "severity_distribution": severity_distribution,
            },
            "samples": [],
        }

        # Convert samples to dict format
        for sample in samples:
            sample_dict: Dict[str, Any] = {
                "id": sample.id,
                "code": sample.code,
                "label": sample.label,
                "cwe_type": sample.cwe_types,
                "severity": sample.severity,
                "metadata": sample.metadata,
            }
            dataset_dict["samples"].append(sample_dict)

        # Write to file
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "w", encoding="utf-8") as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"Created dataset JSON with {len(samples)} samples at {output_path}"
        )

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the CVEFixes database."""
        if not self.conn:
            self.conn = self._create_connection()

        stats: Dict[str, Any] = {}

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
                cursor = self.conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                result = cursor.fetchone()
                stats[f"{table}_count"] = result[0] if result else 0

            # Programming language distribution
            cursor = self.conn.cursor()
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
            if self.conn:
                self.conn.close()

        return stats


class CVEFixesJSONDatasetLoader(DatasetLoader):
    """Loads CVEFixes datasets from JSON files created by CVEFixesDatasetLoader."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, path: str) -> List[BenchmarkSample]:
        """Load dataset from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples: List[BenchmarkSample] = []
        for sample_data in data.get("samples", []):
            sample = BenchmarkSample(
                id=sample_data["id"],
                code=sample_data["code"],
                label=sample_data["label"],
                metadata=sample_data["metadata"],
                cwe_types=[sample_data.get("cwe_type")],
                severity=sample_data.get("severity"),
            )
            samples.append(sample)

        self.logger.info(f"Loaded {len(samples)} samples from {path}")
        return samples

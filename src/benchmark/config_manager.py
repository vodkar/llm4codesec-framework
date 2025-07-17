#!/usr/bin/env python3
"""
Configuration templates and utilities for the LLM benchmark framework.
"""

import json
from pathlib import Path
from typing import Any, Dict, Self

from benchmark.benchmark_framework import BenchmarkConfig, ModelType, TaskType


class BenchmarkConfigManager:
    """Manages benchmark configurations and templates."""

    # Pre-defined model configurations
    MODEL_CONFIGS: dict[str, dict[str, Any]] = {
        "llama2-7b": {
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "model_type": ModelType.LLAMA,
            "max_tokens": 512,
            "temperature": 0.1,
        },
        "qwen2.5-7b": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "model_type": ModelType.QWEN,
            "max_tokens": 512,
            "temperature": 0.1,
        },
        "deepseek-coder": {
            "model_name": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "model_type": ModelType.DEEPSEEK,
            "max_tokens": 512,
            "temperature": 0.1,
        },
        "codebert": {
            "model_name": "microsoft/codebert-base",
            "model_type": ModelType.CODEBERT,
            "max_tokens": 256,
            "temperature": 0.0,
        },
    }

    # Task configurations
    TASK_CONFIGS: dict[str, dict[str, Any]] = {
        "binary_vulnerability": {
            "task_type": TaskType.BINARY_VULNERABILITY,
            "description": "Binary classification: vulnerable vs safe code",
        },
        "cwe79_detection": {
            "task_type": TaskType.BINARY_CWE_SPECIFIC,
            "cwe_type": "CWE-79",
            "description": "Binary classification: CWE-79 (XSS) detection",
        },
        "cwe89_detection": {
            "task_type": TaskType.BINARY_CWE_SPECIFIC,
            "cwe_type": "CWE-89",
            "description": "Binary classification: CWE-89 (SQL Injection) detection",
        },
        "multiclass_vulnerability": {
            "task_type": TaskType.MULTICLASS_VULNERABILITY,
            "description": "Multi-class classification: vulnerability type identification",
        },
    }

    @classmethod
    def create_config(
        cls: type[Self],
        model_key: str,
        task_key: str,
        dataset_path: str,
        output_dir: str = "./results",
        **kwargs,
    ) -> BenchmarkConfig:
        """
        Create a benchmark configuration from templates.

        Args:
            model_key (str): Model configuration key
            task_key (str): Task configuration key
            dataset_path (str): Path to dataset
            output_dir (str): Output directory for results
            **kwargs: Additional configuration overrides

        Returns:
            BenchmarkConfig: Complete benchmark configuration
        """
        if model_key not in cls.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model key: {model_key}. Available: {list(cls.MODEL_CONFIGS.keys())}"
            )

        if task_key not in cls.TASK_CONFIGS:
            raise ValueError(
                f"Unknown task key: {task_key}. Available: {list(cls.TASK_CONFIGS.keys())}"
            )

        # Start with model config
        config_dict = cls.MODEL_CONFIGS[model_key].copy()

        # Add task config
        config_dict.update(cls.TASK_CONFIGS[task_key])

        # Add required parameters
        config_dict.update(
            {
                "dataset_path": dataset_path,
                "output_dir": output_dir,
                "batch_size": 1,
                "use_quantization": True,
            }
        )

        # Apply any overrides
        config_dict.update(kwargs)

        return BenchmarkConfig(**config_dict)

    @classmethod
    def save_config_template(self, filepath: str) -> None:
        """Save a configuration template file."""
        template = {
            "models": self.MODEL_CONFIGS,
            "tasks": self.TASK_CONFIGS,
            "example_configs": [
                {
                    "name": "llama2_binary_vulnerability",
                    "model": "llama2-7b",
                    "task": "binary_vulnerability",
                    "dataset": "./data/vulbench.json",
                    "description": "Llama2 7B on binary vulnerability detection",
                },
                {
                    "name": "qwen_cwe79_detection",
                    "model": "qwen2.5-7b",
                    "task": "cwe79_detection",
                    "dataset": "./data/xss_dataset.json",
                    "description": "Qwen 2.5 7B on CWE-79 (XSS) detection",
                },
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def list_available_configs(self) -> Dict[str, Any]:
        """List all available model and task configurations."""
        return {
            "models": {k: v["model_name"] for k, v in self.MODEL_CONFIGS.items()},
            "tasks": {k: v["description"] for k, v in self.TASK_CONFIGS.items()},
        }


def create_sample_dataset(output_path: str) -> None:
    """Create a sample dataset for testing the framework."""
    sample_data: list[dict[str, Any]] = [
        {
            "id": "sample_001",
            "code": """
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    return cursor.fetchone()
""",
            "label": 1,
            "cwe_type": "CWE-89",
            "severity": "high",
            "metadata": {"language": "python", "source": "synthetic"},
        },
        {
            "id": "sample_002",
            "code": """
def get_user_data(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()
""",
            "label": 0,
            "cwe_type": None,
            "severity": None,
            "metadata": {"language": "python", "source": "synthetic"},
        },
        {
            "id": "sample_003",
            "code": """
@app.route('/search')
def search():
    term = request.args.get('q')
    return f"<h1>Results for: {term}</h1>"
""",
            "label": 1,
            "cwe_type": "CWE-79",
            "severity": "medium",
            "metadata": {"language": "python", "source": "synthetic"},
        },
        {
            "id": "sample_004",
            "code": """
@app.route('/search')
def search():
    term = request.args.get('q', '')
    safe_term = html.escape(term)
    return f"<h1>Results for: {safe_term}</h1>"
""",
            "label": 0,
            "cwe_type": None,
            "severity": None,
            "metadata": {"language": "python", "source": "synthetic"},
        },
        {
            "id": "sample_005",
            "code": """
void copy_data(char* dest, char* src) {
    strcpy(dest, src);
}
""",
            "label": 1,
            "cwe_type": "CWE-120",
            "severity": "high",
            "metadata": {"language": "c", "source": "synthetic"},
        },
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"Sample dataset created at: {output_path}")


def main():
    """Example usage of configuration manager."""

    # Create sample dataset
    create_sample_dataset("./data/sample_dataset.json")

    # Save configuration template
    BenchmarkConfigManager.save_config_template("./config_template.json")

    # List available configurations
    available = BenchmarkConfigManager.list_available_configs()
    print("Available Models:", available["models"])
    print("Available Tasks:", available["tasks"])

    # Create a configuration
    config = BenchmarkConfigManager.create_config(
        model_key="qwen2.5-7b",
        task_key="binary_vulnerability",
        dataset_path="./data/sample_dataset.json",
        output_dir="./results/qwen_binary_test",
    )

    print(f"Created config for: {config.model_name} on {config.task_type.value}")


if __name__ == "__main__":
    main()

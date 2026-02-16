from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from benchmark.enums import ModelType, TaskType


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution."""

    model_name: str
    model_type: ModelType
    task_type: TaskType
    description: str
    dataset_path: Path
    output_dir: Path
    backend: Literal["hf", "vllm", "llama_cpp"]
    batch_size: int = 1
    max_tokens: int = 512
    temperature: float = 0.1
    use_quantization: bool = True
    is_thinking_enabled: bool = False
    cwe_type: str | None = None
    system_prompt_template: str
    user_prompt_template: str

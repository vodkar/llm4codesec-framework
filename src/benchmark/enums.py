from enum import Enum, StrEnum


class TaskType(Enum):
    """Enumeration of supported task types."""

    BINARY_VULNERABILITY = "binary_vulnerability"
    BINARY_CWE_SPECIFIC = "binary_cwe_specific"
    BINARY_VULNERABILITY_SPECIFIC = "binary_vulnerability_specific"
    MULTICLASS_VULNERABILITY = "multiclass_vulnerability"
    VULBENCH_MULTICLASS = "vulbench_multiclass"
    VULDETECTBENCH_SPECIFIC = "vuldetectbench_specific"


class ModelType(Enum):
    """Enumeration of supported model types."""

    # Qwen models
    QWEN_3 = "qwen-3"

    # DeepSeek models
    DEEPSEEK_CODER_V2 = "deepseek-coder-v2"
    DEEPSEEK_R1 = "deepseek-r1"

    # WizardCoder models
    WIZARDCODER = "wizardcoder"

    # New Llama models
    LLAMA_3 = "llama-3"
    LLAMA_4 = "llama-4"

    # Gemma models
    GEMMA_3 = "gemma-3"


class BackendFrameworks(StrEnum):
    HF = "hf"
    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"

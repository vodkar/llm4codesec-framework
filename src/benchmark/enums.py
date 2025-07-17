from enum import Enum


class TaskType(Enum):
    """Enumeration of supported task types."""

    BINARY_VULNERABILITY = "binary_vulnerability"
    BINARY_CWE_SPECIFIC = "binary_cwe_specific"
    BINARY_VULNERABILITY_SPECIFIC = "binary_vulnerability_specific"
    MULTICLASS_VULNERABILITY = "multiclass_vulnerability"


class ModelType(Enum):
    """Enumeration of supported model types."""

    # Legacy models
    LLAMA = "meta-llama/Llama-2-7b-chat-hf"
    QWEN = "Qwen/Qwen2.5-7B-Instruct"
    DEEPSEEK = "deepseek-ai/deepseek-coder-6.7b-instruct"
    CODEBERT = "microsoft/codebert-base"

    # New Qwen models
    QWEN3_4B = "Qwen/Qwen3-4B"
    QWEN3_30B_A3B = "Qwen/Qwen3-30B-A3B"

    # New DeepSeek models
    DEEPSEEK_CODER_V2_LITE_INSTRUCT = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    DEEPSEEK_R1_DISTILL_QWEN_1_5B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    DEEPSEEK_R1_DISTILL_QWEN_32B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    # WizardCoder models
    WIZARDCODER_PYTHON_34B = "WizardLMTeam/WizardCoder-Python-34B-V1.0"

    # New Llama models
    LLAMA_3_2_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"
    LLAMA_4_SCOUT_17B_INSTRUCT = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    # Gemma models
    GEMMA_3_1B_IT = "google/gemma-3-1b-it"
    GEMMA_3_27B_IT = "google/gemma-3-27b-it"

    # OpenCoder models
    OPENCODER_8B_INSTRUCT = "infly/OpenCoder-8B-Instruct"

    CUSTOM = "custom"

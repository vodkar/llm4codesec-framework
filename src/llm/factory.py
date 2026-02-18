import logging
from typing import Final

from benchmark.config import ExperimentConfig
from llm.llm import ILLMInference

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)


def create_llm_inference(config: ExperimentConfig) -> ILLMInference:
    """
    Create an LLM inference backend based on configuration.

    Args:
        config: Benchmark configuration including backend selection.

    Returns:
        ILLMInference: Initialized backend instance.
    """
    backend: str = config.backend.strip().lower()
    LOGGER.info("Selecting LLM backend: %s", backend)

    if backend == "hf":
        from llm.hugging_face import HuggingFaceLLM

        return HuggingFaceLLM(config)
    if backend == "vllm":
        from llm.vllm_backend import VllmLLM

        return VllmLLM(config)
    if backend == "llama_cpp":
        from llm.llama_cpp_backend import LlamaCppLLM

        return LlamaCppLLM(config)

    raise ValueError(f"Unsupported LLM backend: {config.backend}")

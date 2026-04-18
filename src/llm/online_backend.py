from typing import Final

from benchmark.config import ExperimentConfig
from llm.anthropic_backend import AnthropicLLM
from llm.llm import ILLMInference
from llm.openai_backend import OpenAILLM

_ANTHROPIC_PROVIDER: Final[str] = "anthropic"
_OPENAI_PROVIDER: Final[str] = "openai"


def resolve_online_provider(config: ExperimentConfig) -> str:
    """Resolve an upstream online provider from config or model identifier."""
    configured_provider: str | None = config.api_provider
    if configured_provider:
        normalized_provider: str = configured_provider.strip().lower()
        if normalized_provider in {_OPENAI_PROVIDER, _ANTHROPIC_PROVIDER}:
            return normalized_provider
        raise ValueError(
            f"Unsupported api_provider '{configured_provider}' for online backend"
        )

    model_name_lower: str = config.model_identifier.lower()
    if model_name_lower.startswith("claude"):
        return _ANTHROPIC_PROVIDER
    if (
        model_name_lower.startswith("gpt")
        or "chatgpt" in model_name_lower
    ):
        return _OPENAI_PROVIDER

    raise ValueError(
        "Unable to infer online provider from model_identifier. "
        "Set api_provider to 'openai' or 'anthropic'."
    )


def create_online_backend(config: ExperimentConfig) -> ILLMInference:
    """Create an explicit provider backend for legacy online configuration."""
    provider: str = resolve_online_provider(config)
    if provider == _OPENAI_PROVIDER:
        return OpenAILLM(config)
    return AnthropicLLM(config)

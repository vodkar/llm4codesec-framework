"""
LLM Code Security Benchmark Framework

A comprehensive framework for benchmarking Large Language Models on static code analysis tasks.
Supports binary and multi-class vulnerability detection with configurable models and datasets.
"""

import importlib.util
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from benchmark.enums import ModelType, TaskType
from benchmark.metrics_calculator import MetricsCalculator
from benchmark.models import BenchmarkSample, PredictionResult
from benchmark.response_parser import IResponseParser, ResponseParserFactory


def _is_flash_attention_available() -> bool:
    """Check if FlashAttention is available."""
    if importlib.util.find_spec("flash_attn"):
        logging.info("FlashAttention is available")
        return True
    else:
        logging.info(
            "FlashAttention is not available! You can install it to optimize an inference"
        )
        return False


def _is_flash_attention_supported() -> bool:
    """Check if a GPU supports FlashAttention 2."""
    major, minor = torch.cuda.get_device_capability(0)

    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    model_name: str
    model_type: ModelType
    task_type: TaskType
    description: str
    dataset_path: str
    output_dir: str
    batch_size: int = 1
    max_tokens: int = 512
    temperature: float = 0.1
    use_quantization: bool = True
    enable_thinking: bool = False
    cwe_type: str | None = None
    system_prompt_template: str | None = None
    user_prompt_template: str | None = None

    @classmethod
    def create_for_model(
        cls,
        model_type: ModelType,
        task_type: TaskType,
        dataset_path: str,
        output_dir: str,
        description: str | None = None,
        **kwargs,
    ) -> "BenchmarkConfig":
        """
        Factory method to create configuration for a specific model type.

        Args:
            model_type: The model type to use
            task_type: The task type to perform
            dataset_path: Path to the dataset
            output_dir: Output directory for results
            description: Optional description
            **kwargs: Additional configuration parameters

        Returns:
            BenchmarkConfig: Configured benchmark instance
        """
        if description is None:
            description = f"{model_type.name} on {task_type.value}"

        # Set model-specific defaults
        defaults = {
            "batch_size": 4,  # Increased default batch size for better GPU utilization
            "max_tokens": 512,
            "temperature": 0.1,
            "use_quantization": True,
        }

        # Override with model-specific settings
        if "llama-3.3-70b" in model_type.value.lower():
            defaults["batch_size"] = 1  # Large models need smaller batches
        elif "gemma-3-1b" in model_type.value.lower():
            defaults["use_quantization"] = (
                False  # Smaller model, no need for quantization
            )
            defaults["max_tokens"] = 1024
            defaults["batch_size"] = 8  # Smaller models can handle larger batches
        elif "opencoder" in model_type.value.lower():
            defaults["temperature"] = (
                0.0  # Code models often work better with deterministic output
            )

        # A100 40GB optimized settings
        if torch.cuda.is_available():
            gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )  # GB

            if gpu_memory >= 35:  # A100 40GB detected
                # Model-specific optimization for A100
                if any(size in model_type.value.lower() for size in ["70b", "72b"]):
                    defaults["use_quantization"] = True
                    defaults["quantization_type"] = (
                        "4bit"  # Still need 4-bit for 70B models
                    )
                    defaults["batch_size"] = 1  # Large models need batch size 1
                elif any(
                    size in model_type.value.lower() for size in ["30b", "32b", "34b"]
                ):
                    defaults["use_quantization"] = True
                    defaults["quantization_type"] = (
                        "8bit"  # 8-bit perfect for 30B+ models
                    )
                    defaults["batch_size"] = 2  # Medium models can do batch size 2
                elif any(
                    size in model_type.value.lower()
                    for size in ["7b", "8b", "13b", "17b"]
                ):
                    defaults["use_quantization"] = (
                        False  # No quantization needed for smaller models
                    )
                    defaults["torch_dtype"] = "bfloat16"  # Use native bfloat16 instead
                    defaults["batch_size"] = (
                        6  # Smaller models can handle larger batches
                    )
                elif any(
                    size in model_type.value.lower()
                    for size in ["1b", "1.5b", "3b", "4b"]
                ):
                    defaults["use_quantization"] = False
                    defaults["torch_dtype"] = "bfloat16"
                    defaults["batch_size"] = (
                        12  # Very small models can handle even larger batches
                    )
                else:
                    defaults["use_quantization"] = True
                    defaults["quantization_type"] = "8bit"  # Default to 8-bit
                    defaults["batch_size"] = 4  # Default batch size

        # Merge with user-provided kwargs
        config_params = {**defaults, **kwargs}

        return cls(
            model_name=model_type.value,
            model_type=model_type,
            task_type=task_type,
            description=description,
            dataset_path=dataset_path,
            output_dir=output_dir,
            **config_params,
        )

    @staticmethod
    def get_available_models() -> dict[str, list[str]]:
        """
        Get a dictionary of available models organized by family.

        Returns:
            dict[str, list[str]]: dictionary mapping model families to model names
        """
        return {
            "Llama": [
                ModelType.LLAMA.value,
                ModelType.LLAMA_3_2_3B_INSTRUCT.value,
                ModelType.LLAMA_4_SCOUT_17B_INSTRUCT.value,
            ],
            "Qwen": [
                ModelType.QWEN.value,
                ModelType.QWEN3_4B.value,
                ModelType.QWEN3_30B_A3B.value,
            ],
            "DeepSeek": [
                ModelType.DEEPSEEK.value,
                ModelType.DEEPSEEK_CODER_V2_LITE_INSTRUCT.value,
                ModelType.DEEPSEEK_R1_DISTILL_QWEN_1_5B.value,
                ModelType.DEEPSEEK_R1_DISTILL_QWEN_32B.value,
            ],
            "WizardCoder": [
                ModelType.WIZARDCODER_PYTHON_34B.value,
            ],
            "Gemma": [
                ModelType.GEMMA_3_1B_IT.value,
                ModelType.GEMMA_3_27B_IT.value,
            ],
            "OpenCoder": [
                ModelType.OPENCODER_8B_INSTRUCT.value,
            ],
            "Other": [
                ModelType.CODEBERT.value,
            ],
        }


class DatasetLoader(Protocol):
    """Protocol for dataset loading implementations."""

    def load_dataset(self, path: str) -> list[BenchmarkSample]:
        """Load dataset from the specified path."""
        ...


class VulBenchLoader:
    """Loader for VulBench dataset format."""

    def load_dataset(self, path: str) -> list[BenchmarkSample]:
        """
        Load VulBench dataset.

        Args:
            path (str): Path to the dataset file

        Returns:
            list[BenchmarkSample]: Loaded samples
        """
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        samples: list[BenchmarkSample] = []

        if data_path.suffix == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for i, item in enumerate(data):
                sample = BenchmarkSample(
                    id=item.get("id", f"sample_{i}"),
                    code=item["code"],
                    label=item["label"],
                    metadata=item.get("metadata", {}),
                    cwe_types=item.get("cwe_type"),
                    severity=item.get("severity"),
                )
                samples.append(sample)

        elif data_path.suffix == ".csv":
            df = pd.read_csv(data_path)
            for idx, row in df.iterrows():
                sample = BenchmarkSample(
                    id=row.get("id", f"sample_{idx}"),
                    code=row["code"],
                    label=row["label"],
                    metadata={"source_file": row.get("source_file", "")},
                    cwe_types=row.get("cwe_type"),
                    severity=row.get("severity"),
                )
                samples.append(sample)

        return samples


class PromptGenerator:
    """Generates prompts for different task types."""

    SYSTEM_PROMPTS = {
        TaskType.BINARY_VULNERABILITY: """You are an expert security analyst specializing in static code analysis. 
Your task is to analyze code snippets and determine if they contain security vulnerabilities.

Instructions:
- Analyze the provided code carefully
- Consider common vulnerability patterns (injection, buffer overflow, race conditions, etc.)
- Respond with only "VULNERABLE" or "SAFE" - no additional explanation
- Base your decision on concrete security risks, not coding style issues""",
        TaskType.BINARY_CWE_SPECIFIC: """You are an expert security analyst specializing in static code analysis.
Your task is to analyze code snippets and determine if they contain a specific type of vulnerability: {cwe_type}.

Instructions:
- Analyze the provided code for {cwe_type} vulnerabilities only
- Ignore other types of vulnerabilities
- Respond with only "VULNERABLE" or "SAFE" - no additional explanation
- Focus specifically on {cwe_type} patterns and indicators""",
        TaskType.MULTICLASS_VULNERABILITY: """You are an expert security analyst specializing in static code analysis.
Your task is to analyze code snippets and classify the type of vulnerability present.

Instructions:
- Analyze the provided code carefully
- If vulnerable, identify the primary vulnerability type from: CWE-79, CWE-89, CWE-120, CWE-190, CWE-476, CWE-787
- If no vulnerability is found, respond with "SAFE"
- Respond with only the vulnerability type (e.g., "CWE-79") or "SAFE" - no additional explanation""",
    }

    USER_PROMPTS = {
        TaskType.BINARY_VULNERABILITY: "Analyze this code for security vulnerabilities:\n\n{code}",
        TaskType.BINARY_CWE_SPECIFIC: "Analyze this code for {cwe_type} vulnerabilities:\n\n{code}",
        TaskType.MULTICLASS_VULNERABILITY: "Analyze this code and identify the vulnerability type:\n\n{code}",
    }

    def get_system_prompt(
        self, task_type: TaskType, cwe_type: str | None = None
    ) -> str:
        """Generate system prompt for the given task type."""
        prompt = self.SYSTEM_PROMPTS[task_type]
        if cwe_type and "{cwe_type}" in prompt:
            prompt = prompt.format(cwe_type=cwe_type)
        return prompt

    def get_user_prompt(
        self, task_type: TaskType, code: str, cwe_type: str | None = None
    ) -> str:
        """Generate user prompt for the given task type and code."""
        prompt = self.USER_PROMPTS[task_type]
        if cwe_type and "{cwe_type}" in prompt:
            prompt = prompt.format(code=code, cwe_type=cwe_type)
        else:
            prompt = prompt.format(code=code)
        return prompt


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, int, float]:
        """
        Generate response from the model.

        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt

        Returns:
            tuple[str, Optional[int]]: Response text and token count
        """
        pass

    @abstractmethod
    def generate_batch_responses(
        self, prompts: list[str]
    ) -> list[tuple[str, Optional[int]]]:
        """
        Generate responses for a batch of prompts.

        Args:
            prompts (list[str]): List of formatted prompts

        Returns:
            list[tuple[str, Optional[int]]]: List of (response_text, token_count) tuples
        """
        pass

    def generate_responses_batch_optimized(
        self, system_prompts: list[str], user_prompts: list[str]
    ) -> list[tuple[str, int, float]]:
        """
        Generate responses for multiple system/user prompt pairs with batch optimization.

        Args:
            system_prompts: List of system prompts
            user_prompts: List of user prompts (must be same length as system_prompts)

        Returns:
            List of (response_text, token_count) tuples
        """
        if len(system_prompts) != len(user_prompts):
            raise ValueError("system_prompts and user_prompts must have same length")

        # This is a default implementation - can be overridden by subclasses
        results: list[tuple[str, int, float]] = []
        for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
            result = self.generate_response(sys_prompt, user_prompt)
            results.append(result)
        return results

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up model resources."""
        pass


class HuggingFaceLLM(LLMInterface):
    """Hugging Face transformers-based LLM interface."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.pipeline = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        logging.info(f"Loading model: {self.config.model_name}")

        # Configure quantization if requested
        quantization_config = None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        attn_implementation = (
            "flash_attention_2"
            if _is_flash_attention_supported() and _is_flash_attention_available()
            else None
        )

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"Detected GPU memory: {gpu_memory:.1f}GB")

            if self.config.use_quantization:
                # Determine quantization type based on model size and GPU memory
                quantization_type = getattr(self.config, "quantization_type", "8bit")

                if quantization_type == "8bit" and gpu_memory >= 35:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=4.0,
                    )
                    logging.info("Using 8-bit quantization for A100")
                else:
                    # Fallback to 4-bit for very large models or smaller GPUs
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logging.info("Using 4-bit quantization")
            else:
                # No quantization - use optimal native precision
                torch_dtype = torch.bfloat16 if gpu_memory >= 35 else torch.float16
                logging.info(f"No quantization, using {torch_dtype}")

            if "gemma-3" in self.config.model_name:
                torch_dtype = torch.bfloat16
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="q4_0",  # match the model’s QAT quant type
                    bnb_4bit_compute_dtype=torch.bfloat16,  # fp16 compute
                    bnb_4bit_use_double_quant=True,
                )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="left",
                token=os.getenv("HF_TOKEN", None),
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                token=os.getenv("HF_TOKEN", None),
                attn_implementation=attn_implementation,
            )

            # Create text generation pipeline with batch optimization
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                return_full_text=False,
                batch_size=self.config.batch_size,  # Enable batch processing
                device_map="auto",
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            logging.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logging.exception(f"Failed to load model {self.config.model_name}: {e}")
            raise

    def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str, int, float]:
        """
        Generate response using the loaded model.

        Note: This method now uses batch processing internally for better GPU utilization.
        For processing multiple samples, consider using generate_batch_responses directly.
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        # Format the prompt based on model type
        formatted_prompt = self._format_prompt(system_prompt, user_prompt)

        try:
            # Use batch processing even for single requests for consistency
            batch_results = self.generate_batch_responses([formatted_prompt])
            return batch_results[0]

        except Exception as e:
            logging.exception(f"Error generating response: {e}")
            return f"ERROR: {str(e)}", None

    def generate_responses_batch_optimized(
        self, system_prompts: list[str], user_prompts: list[str]
    ) -> list[tuple[str, int, float]]:
        """
        Generate responses for multiple system/user prompt pairs with batch optimization.

        Args:
            system_prompts: List of system prompts
            user_prompts: List of user prompts (must be same length as system_prompts)

        Returns:
            List of (response_text, token_count) tuples
        """
        if len(system_prompts) != len(user_prompts):
            raise ValueError("system_prompts and user_prompts must have same length")

        # Format all prompts
        formatted_prompts = [
            self._format_prompt(sys_prompt, user_prompt)
            for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]

        return self.generate_batch_responses(formatted_prompts)

    def generate_batch_responses(
        self, prompts: list[str]
    ) -> list[tuple[str, int, float]]:
        """Generate responses for a batch of prompts."""
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        try:
            # Process in batches to manage memory
            batch_size = min(
                int(os.getenv("HARD_BATCH_SIZE", self.config.batch_size)), len(prompts)
            )
            results: list[tuple[str, int, float]] = []

            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]

                start = time.time()
                # Generate responses for the batch
                batch_responses = self.pipeline(
                    batch_prompts,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    batch_size=batch_size,
                )
                duration = time.time() - start

                # Process each response in the batch
                # Handle different response formats from Hugging Face pipeline
                if isinstance(batch_responses[0], list):
                    # If pipeline returns list of lists (batch processing)
                    for j, response_list in enumerate(batch_responses):
                        if response_list and isinstance(response_list[0], dict):
                            response_text = response_list[0]["generated_text"].strip()
                        else:
                            response_text = (
                                str(response_list[0]).strip() if response_list else ""
                            )

                        # Estimate token count
                        tokens = self.tokenizer.encode(batch_prompts[j] + response_text)
                        token_count = len(tokens)

                        results.append(
                            (response_text, token_count, duration / batch_size)
                        )
                else:
                    # If pipeline returns list of dicts (single responses)
                    for j, response in enumerate(batch_responses):
                        if isinstance(response, dict):
                            response_text = response["generated_text"].strip()
                        else:
                            response_text = str(response).strip()

                        # Estimate token count
                        tokens = self.tokenizer.encode(batch_prompts[j] + response_text)
                        token_count = len(tokens)

                        results.append(
                            (response_text, token_count, duration / batch_size)
                        )

                # Log progress
                logging.info(
                    f"Processed batch {i // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1}"
                )

            return results

        except Exception as e:
            logging.exception(f"Error generating batch responses: {e}")
            return [(f"ERROR: {str(e)}", 0, 0) for _ in prompts]

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format prompt using tokenizer's chat template."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")

        # Prepare messages in standard chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Use tokenizer's apply_chat_template method
            formatted_prompt = self.tokenizer.apply_chat_template(  # type: ignore
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking,
            )
            return formatted_prompt

        except Exception as e:
            # Fallback to generic format if chat template is not available
            logging.warning(
                f"Chat template not available for {self.config.model_name}, using fallback format: {e}"
            )
            return self._format_prompt_fallback(system_prompt, user_prompt)

    def _format_prompt_fallback(self, system_prompt: str, user_prompt: str) -> str:
        """Fallback prompt formatting for models without chat templates."""
        model_name_lower = self.config.model_name.lower()

        # Llama models (includes Llama-2, Llama-3.2, Llama-3.3)
        if "llama" in model_name_lower:
            if "llama-3" in model_name_lower:
                # Llama-3.x models use newer chat format
                return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                # Llama-2 format
                return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"

        # Qwen models (includes Qwen2.5, Qwen3)
        elif "qwen" in model_name_lower:
            if "qwen3" in model_name_lower:
                # Qwen3 models may use updated format
                return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                # Qwen2.5 and earlier format
                return f"<|system|>\n{system_prompt}<|endofsystem|>\n<|user|>\n{user_prompt}<|endofuser|>\n<|assistant|>\n"

        # DeepSeek models (includes original and V2, R1)
        elif "deepseek" in model_name_lower:
            if "r1" in model_name_lower:
                # DeepSeek-R1 models use specific format
                return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                # Standard DeepSeek format
                return f"### System:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"

        # WizardCoder models
        elif "wizard" in model_name_lower:
            return f"### Instruction:\n{system_prompt}\n\n### Input:\n{user_prompt}\n\n### Response:\n"

        # Gemma models
        elif "gemma" in model_name_lower:
            return f"<bos><start_of_turn>user\n{system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"

        # OpenCoder models
        elif "opencoder" in model_name_lower:
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        else:
            # Generic format for unknown models
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline
        torch.cuda.empty_cache()


class BenchmarkRunner:
    """Main benchmark execution class."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset_loader = VulBenchLoader()
        self.prompt_generator = PromptGenerator()
        self.response_parser = ResponseParserFactory.create_parser(config.task_type)
        self.metrics_calculator = MetricsCalculator.create_calculator(config.task_type)
        self.llm: Optional[LLMInterface] = None

        # Setup logging
        self._setup_logging()

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = Path(self.config.output_dir) / "benchmark.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def run_benchmark(self) -> dict[str, Any]:
        """
        Execute the complete benchmark.

        Returns:
            dict[str, Any]: Benchmark results
        """
        logging.info("Starting benchmark execution")
        start_time = time.time()

        try:
            # Load dataset
            logging.info(f"Loading dataset from: {self.config.dataset_path}")
            samples = self.dataset_loader.load_dataset(self.config.dataset_path)
            logging.info(f"Loaded {len(samples)} samples")

            # Initialize model
            logging.info("Initializing model")
            self.llm = HuggingFaceLLM(self.config)

            # Run predictions
            predictions = self._run_predictions(samples)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate(predictions)

            # Generate report
            report = self._generate_report(
                samples, predictions, metrics, time.time() - start_time
            )

            # Save results
            self._save_results(report)

            logging.info("Benchmark completed successfully")
            return report

        except Exception as e:
            logging.exception(f"Benchmark failed: {e}")
            raise
        finally:
            if self.llm:
                self.llm.cleanup()

    def _run_predictions(
        self, samples: list[BenchmarkSample]
    ) -> list[PredictionResult]:
        """Run model predictions on all samples using batch processing."""
        predictions: list[PredictionResult] = []

        system_prompt = self.prompt_generator.get_system_prompt(
            self.config.task_type, self.config.cwe_type
        )

        # Prepare all prompts in advance for batch processing
        logging.info("Preparing prompts for batch processing...")
        formatted_prompts = []
        for sample in samples:
            user_prompt = self.prompt_generator.get_user_prompt(
                self.config.task_type, sample.code, self.config.cwe_type
            )
            # Cast to HuggingFaceLLM to access the _format_prompt method
            if isinstance(self.llm, HuggingFaceLLM):
                formatted_prompt = self.llm._format_prompt(system_prompt, user_prompt)
            else:
                # Fallback for other LLM types
                formatted_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            formatted_prompts.append(formatted_prompt)

        # Process in batches
        logging.info(
            f"Processing {len(samples)} samples in batches of {self.config.batch_size}"
        )
        start_time = time.time()

        try:
            batch_responses = self.llm.generate_batch_responses(formatted_prompts)
            total_processing_time = time.time() - start_time
            avg_processing_time = total_processing_time / len(samples)

            # Process results
            for i, (sample, (response_text, tokens_used)) in enumerate(
                zip(samples, batch_responses)
            ):
                # Parse response
                predicted_label = self.response_parser.parse_response(response_text)

                prediction = PredictionResult(
                    sample_id=sample.id,
                    predicted_label=predicted_label,
                    true_label=sample.label
                    if isinstance(sample.label, int)
                    else self.response_parser.parse_response(sample.label),
                    confidence=None,  # Could be enhanced to extract confidence
                    response_text=response_text,
                    processing_time=avg_processing_time,  # Average time per sample
                    tokens_used=tokens_used,
                )

                predictions.append(prediction)

                # Log progress
                if (i + 1) % 50 == 0:
                    logging.info(f"Processed {i + 1}/{len(samples)} predictions")

        except Exception as e:
            logging.warning(
                f"Batch processing failed, falling back to sequential processing: {e}"
            )
            # Fallback to sequential processing if batch processing fails
            predictions = self._run_predictions_sequential(samples)

        logging.info(f"Completed all {len(samples)} predictions")
        return predictions

    def _run_predictions_sequential(
        self, samples: list[BenchmarkSample]
    ) -> list[PredictionResult]:
        """Fallback method for sequential prediction processing."""
        predictions: list[PredictionResult] = []

        system_prompt = self.prompt_generator.get_system_prompt(
            self.config.task_type, self.config.cwe_type
        )

        for i, sample in enumerate(samples):
            logging.info(f"Processing sample {i + 1}/{len(samples)}: {sample.id}")

            user_prompt = self.prompt_generator.get_user_prompt(
                self.config.task_type, sample.code, self.config.cwe_type
            )

            # Generate response
            start_time = time.time()
            response_text, tokens_used = self.llm.generate_response(
                system_prompt, user_prompt
            )
            processing_time = time.time() - start_time

            # Parse response
            predicted_label = self.response_parser.parse_response(response_text)

            prediction = PredictionResult(
                sample_id=sample.id,
                predicted_label=predicted_label,
                true_label=sample.label
                if isinstance(sample.label, int)
                else self.response_parser.parse_response(sample.label),
                confidence=None,  # Could be enhanced to extract confidence
                response_text=response_text,
                processing_time=processing_time,
                tokens_used=tokens_used,
            )

            predictions.append(prediction)

            # Log progress
            if (i + 1) % 10 == 0:
                logging.info(f"Completed {i + 1}/{len(samples)} predictions")

        return predictions

    def _generate_report(
        self,
        samples: list[BenchmarkSample],
        predictions: list[PredictionResult],
        metrics: dict[str, Any],
        total_time: float,
    ) -> dict[str, Any]:
        """Generate comprehensive benchmark report."""

        report = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "model_name": self.config.model_name,
                "model_type": self.config.model_type.value,
                "task_type": self.config.task_type.value,
                "dataset_path": self.config.dataset_path,
                "cwe_type": self.config.cwe_type,
                "total_samples": len(samples),
                "total_time_seconds": total_time,
            },
            "configuration": asdict(self.config),
            "metrics": metrics,
            "predictions": [asdict(pred) for pred in predictions],
            "sample_analysis": {
                "avg_processing_time": np.mean(
                    [p.processing_time for p in predictions]
                ),
                "total_tokens_used": sum(
                    p.tokens_used for p in predictions if p.tokens_used
                ),
                "error_count": len(
                    [p for p in predictions if "ERROR" in p.response_text]
                ),
            },
        }

        return report

    def _save_results(self, report: dict[str, Any]) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full report
        report_file = (
            Path(self.config.output_dir) / f"benchmark_report_{timestamp}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Save metrics summary
        metrics_file = (
            Path(self.config.output_dir) / f"metrics_summary_{timestamp}.json"
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(report["metrics"], f, indent=2, ensure_ascii=False, default=str)

        # Save predictions as CSV
        predictions_df = pd.DataFrame(
            [
                asdict(pred)
                for pred in [
                    PredictionResult(**pred_dict) for pred_dict in report["predictions"]
                ]
            ]
        )
        predictions_csv = Path(self.config.output_dir) / f"predictions_{timestamp}.csv"
        predictions_df.to_csv(predictions_csv, index=False)

        logging.info(f"Results saved to: {report_file}")

    @staticmethod
    def process_samples_with_batch_optimization(
        samples: list[BenchmarkSample],
        llm: LLMInterface,
        system_prompt: str,
        prompt_generator: PromptGenerator,
        response_parser: IResponseParser,
        config: BenchmarkConfig,
    ) -> list[PredictionResult]:
        """
        Process samples with batch optimization for better GPU utilization.

        This method can be used by benchmark runners to replace their sequential
        processing loops with efficient batch processing.

        Args:
            samples: List of benchmark samples to process
            llm: LLM interface instance
            system_prompt: System prompt to use
            prompt_generator: Prompt generator instance
            response_parser: Response parser instance
            config: Benchmark configuration

        Returns:
            List of prediction results
        """
        logging.info(f"Processing {len(samples)} samples with batch optimization")

        # Prepare all prompts for batch processing
        user_prompts = []
        system_prompts = []

        for sample in samples:
            # Handle custom user prompt template if provided
            if config.user_prompt_template:
                user_prompt = config.user_prompt_template.format(
                    code=sample.code, cwe_type=config.cwe_type
                )
            else:
                user_prompt = prompt_generator.get_user_prompt(
                    config.task_type, sample.code, config.cwe_type
                )

            # Handle custom system prompt template if provided
            if config.system_prompt_template:
                current_system_prompt = config.system_prompt_template.format(
                    cwe_type=config.cwe_type
                )
            else:
                current_system_prompt = system_prompt

            user_prompts.append(user_prompt)
            system_prompts.append(current_system_prompt)

        # Process in batches
        try:
            # Use batch processing
            batch_responses: list[tuple[str, int, float]] = (
                llm.generate_responses_batch_optimized(system_prompts, user_prompts)
            )

        except Exception as e:
            logging.warning(f"Batch processing failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            batch_responses: list[tuple[str, int, float]] = []
            for i, sample in enumerate(samples):
                # Handle custom user prompt template if provided
                if config.user_prompt_template:
                    user_prompt = config.user_prompt_template.format(
                        code=sample.code, cwe_type=config.cwe_type
                    )
                else:
                    user_prompt = prompt_generator.get_user_prompt(
                        config.task_type, sample.code, config.cwe_type
                    )

                # Handle custom system prompt template if provided
                current_system_prompt = (
                    config.system_prompt_template
                    if config.system_prompt_template
                    else system_prompt
                )

                response_text, tokens_used, processing_duration = llm.generate_response(
                    current_system_prompt, user_prompt
                )

                batch_responses.append(
                    (response_text, tokens_used, processing_duration)
                )

                if (i + 1) % 10 == 0:
                    logging.info(
                        f"Sequential processing: {i + 1}/{len(samples)} completed"
                    )

        # Process results
        predictions = []
        for i, (sample, (response_text, tokens_used, processing_duration)) in enumerate(
            zip(samples, batch_responses)
        ):
            # Parse response
            predicted_label = response_parser.parse_response(response_text)

            # Handle true label - might be int or string depending on dataset
            if isinstance(sample.label, int):
                true_label = sample.label
            else:
                true_label = response_parser.parse_response(str(sample.label))

            prediction = PredictionResult(
                sample_id=sample.id,
                predicted_label=predicted_label,
                true_label=true_label,
                confidence=None,
                response_text=response_text,
                processing_time=processing_duration,
                tokens_used=tokens_used,
            )

            predictions.append(prediction)

            # Log progress
            if (i + 1) % 50 == 0:
                logging.info(f"Processed {i + 1}/{len(samples)} predictions")

        logging.info(f"Completed processing all {len(samples)} samples")
        return predictions

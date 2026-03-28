import logging
import os
import time
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    pipeline,
)
from transformers.pipelines import Pipeline

from benchmark.config import ExperimentConfig
from flash_attention import is_flash_attention_available, is_flash_attention_supported
from llm.llm import ILLMInference


class HuggingFaceLLM(ILLMInference):
    """Hugging Face transformers-based LLM interface."""

    def __init__(self, config: ExperimentConfig):
        self.config: ExperimentConfig = config
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: AutoModelForCausalLM | None = None
        self.pipeline: Pipeline | None = None
        self.pad_token_id: int = 0
        self._warned_sampling_params: set[str] = set()

        self._load_model()

    def _parse_gguf_identifier(self, model_identifier: str) -> tuple[str, str] | None:
        """Parse hf:// GGUF URI into (repo_id, filename). Returns None if not GGUF."""
        if not model_identifier.startswith("hf://"):
            return None
        path = model_identifier[len("hf://"):]
        parts = path.split("/")
        if len(parts) < 3:
            return None
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return repo_id, filename

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        logging.info(f"Loading model: {self.config.model_identifier}")

        # Detect GGUF format and resolve identifiers
        gguf_info = self._parse_gguf_identifier(self.config.model_identifier)
        is_gguf = gguf_info is not None
        tokenizer_id = self.config.tokenizer_identifier or self.config.model_identifier

        # Configure quantization if requested
        quantization_config: BitsAndBytesConfig | None = None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        attn_implementation = (
            "flash_attention_2"
            if is_flash_attention_supported() and is_flash_attention_available()
            else None
        )

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"Detected GPU memory: {gpu_memory:.1f}GB")

            if is_gguf:
                # GGUF models are already quantized — skip BitsAndBytes
                torch_dtype = torch.float16
                attn_implementation = None
                logging.info("GGUF model detected: skipping BitsAndBytes quantization")
            elif self.config.use_quantization:
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

            if not is_gguf and "gemma-3" in self.config.model_identifier:
                torch_dtype = torch.bfloat16
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="q4_0",  # match the model’s QAT quant type
                    bnb_4bit_compute_dtype=torch.bfloat16,  # fp16 compute
                    bnb_4bit_use_double_quant=True,
                )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=True,
                padding_side="left",
                token=os.getenv("HF_TOKEN", None),
            )

            if self.tokenizer is None:
                raise RuntimeError("Tokenizer failed to load")

            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if is_gguf:
                repo_id, gguf_file = gguf_info
                logging.info(f"Loading GGUF model from repo={repo_id}, file={gguf_file}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    gguf_file=gguf_file,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    token=os.getenv("HF_TOKEN", None),
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_identifier,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    token=os.getenv("HF_TOKEN", None),
                    attn_implementation=attn_implementation,
                )

            if self.model is None:
                raise RuntimeError("Model failed to load")

            pad_token_id: int = (
                int(self.tokenizer.eos_token_id)
                if self.tokenizer.eos_token_id is not None
                else int(self.tokenizer.pad_token_id)
                if self.tokenizer.pad_token_id is not None
                else 0
            )
            self.pad_token_id = pad_token_id

            # Create text generation pipeline with batch optimization
            pipeline_fn: Any = pipeline
            self.pipeline = pipeline_fn(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
            )

            logging.info(f"Model loaded successfully on {self.device}")

        except Exception:
            logging.exception("Failed to load model %s", self.config.model_identifier)
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
            logging.exception("Error generating response")
            return f"ERROR: {str(e)}", 0, 0.0

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
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")

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
                batch_responses: Any = self.pipeline(
                    batch_prompts,
                    **self._build_generation_kwargs(batch_size=batch_size),
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
            logging.exception("Error generating batch responses")
            return [(f"ERROR: {str(e)}", 0, 0) for _ in prompts]

    def _build_generation_kwargs(
        self, batch_size: int
    ) -> dict[str, int | float | bool]:
        """Build generation kwargs for the transformers pipeline."""
        do_sample: bool = self.config.temperature > 0
        generation_kwargs: dict[str, int | float | bool] = {
            "max_new_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
            "do_sample": do_sample,
            "return_full_text": False,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.pad_token_id,
            "batch_size": batch_size,
        }

        optional_generation_values: dict[str, int | float | None] = {
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
        }
        for key, value in optional_generation_values.items():
            if value is not None:
                generation_kwargs[key] = value

        if self.config.min_p is not None:
            self._warn_unsupported_sampling_param(
                "min_p",
                "transformers pipeline does not expose min_p for this backend",
            )
        if self.config.presence_penalty is not None:
            self._warn_unsupported_sampling_param(
                "presence_penalty",
                "transformers pipeline does not expose presence_penalty",
            )

        return generation_kwargs

    def _warn_unsupported_sampling_param(self, param_name: str, reason: str) -> None:
        """Warn once when a configured sampling parameter is unsupported."""
        if param_name in self._warned_sampling_params:
            return

        logging.warning(
            "Ignoring sampling parameter '%s' for Hugging Face backend: %s",
            param_name,
            reason,
        )
        self._warned_sampling_params.add(param_name)

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
            apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
            if callable(apply_chat_template):
                formatted_prompt = apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    is_thinking_enabled=self.config.is_thinking_enabled,
                )
                return str(formatted_prompt)
            raise AttributeError("apply_chat_template is not available")

        except Exception:
            # Fallback to generic format if chat template is not available
            logging.warning(
                "Chat template not available for %s, using fallback format",
                self.config.model_identifier,
            )
            return self._format_prompt_fallback(system_prompt, user_prompt)

    def _format_prompt_fallback(self, system_prompt: str, user_prompt: str) -> str:
        """Fallback prompt formatting for models without chat templates."""
        model_name_lower = self.config.model_identifier.lower()

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

    def count_input_tokens(self, text: str) -> int:
        """Count tokenizer tokens for input text."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")

        return len(self.tokenizer.encode(text))

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline
        torch.cuda.empty_cache()

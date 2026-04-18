import logging
import math
import os
import time
from typing import Any, Final

from benchmark.enums import BinaryDecisionMode
import torch
import torch.nn.functional as F
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
from llm.llm import ILLMInference, InferenceResult

_BINARY_LABELS: Final[tuple[str, str]] = ("VULNERABLE", "SAFE")
_FINAL_ANSWER_PREFIX: Final[str] = "[[FINAL_ANSWER:"


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
        self._binary_label_token_ids: dict[str, int] | None = None
        self._did_warn_binary_label_tokenization: bool = False

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
    ) -> InferenceResult:
        """
        Generate response using the loaded model.

        Note: This method now uses batch processing internally for better GPU utilization.
        For processing multiple samples, consider using generate_batch_responses directly.
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        formatted_prompt = self._format_prompt(system_prompt, user_prompt)

        try:
            batch_results = self.generate_batch_responses([formatted_prompt])
            return batch_results[0]

        except Exception as e:
            logging.exception("Error generating response")
            return InferenceResult(response_text=f"ERROR: {str(e)}", tokens_used=0, duration=0.0)

    def generate_responses_batch_optimized(
        self, system_prompts: list[str], user_prompts: list[str]
    ) -> list[InferenceResult]:
        """
        Generate responses for multiple system/user prompt pairs with batch optimization.
        """
        if len(system_prompts) != len(user_prompts):
            raise ValueError("system_prompts and user_prompts must have same length")

        formatted_prompts = [
            self._format_prompt(sys_prompt, user_prompt)
            for sys_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]

        return self.generate_batch_responses(formatted_prompts)

    def generate_batch_responses(
        self, prompts: list[str]
    ) -> list[InferenceResult]:
        """Generate responses for a batch of prompts."""
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not loaded")

        try:
            batch_size = min(
                int(os.getenv("HARD_BATCH_SIZE", self.config.batch_size)), len(prompts)
            )

            if self.config.enable_logprobs:
                return self._generate_with_logprobs(prompts)

            results: list[InferenceResult] = []

            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]

                start = time.time()
                batch_responses: Any = self.pipeline(
                    batch_prompts,
                    **self._build_generation_kwargs(batch_size=batch_size),
                )
                duration = time.time() - start

                if isinstance(batch_responses[0], list):
                    for j, response_list in enumerate(batch_responses):
                        if response_list and isinstance(response_list[0], dict):
                            response_text = response_list[0]["generated_text"].strip()
                        else:
                            response_text = (
                                str(response_list[0]).strip() if response_list else ""
                            )
                        token_count = len(self.tokenizer.encode(batch_prompts[j] + response_text))
                        results.append(InferenceResult(
                            response_text=response_text,
                            tokens_used=token_count,
                            duration=duration / batch_size,
                        ))
                else:
                    for j, response in enumerate(batch_responses):
                        if isinstance(response, dict):
                            response_text = response["generated_text"].strip()
                        else:
                            response_text = str(response).strip()
                        token_count = len(self.tokenizer.encode(batch_prompts[j] + response_text))
                        results.append(InferenceResult(
                            response_text=response_text,
                            tokens_used=token_count,
                            duration=duration / batch_size,
                        ))

                logging.info(
                    f"Processed batch {i // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1}"
                )

            return results

        except Exception as e:
            logging.exception("Error generating batch responses")
            return [InferenceResult(response_text=f"ERROR: {str(e)}", tokens_used=0, duration=0.0) for _ in prompts]

    def _generate_with_logprobs(self, prompts: list[str]) -> list[InferenceResult]:
        """Generate responses using model.generate() directly to capture logprobs.

        Processes one prompt at a time to avoid padding complexity when extracting
        per-token log-probabilities from a batch.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer not loaded")

        do_sample: bool = self.config.temperature > 0
        results: list[InferenceResult] = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len: int = inputs["input_ids"].shape[1]

            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": self.config.max_output_tokens,
                "temperature": self.config.temperature,
                "do_sample": do_sample,
                "pad_token_id": self.pad_token_id,
                "eos_token_id": self.pad_token_id,
                "output_scores": True,
                "return_dict_in_generate": True,
            }
            if self.config.top_p is not None:
                generate_kwargs["top_p"] = self.config.top_p
            if self.config.top_k is not None:
                generate_kwargs["top_k"] = self.config.top_k
            if self.config.repetition_penalty is not None:
                generate_kwargs["repetition_penalty"] = self.config.repetition_penalty

            start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)
            duration = time.time() - start

            # Decode generated tokens (excluding the input)
            generated_ids = outputs.sequences[0, input_len:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            token_count = input_len + len(generated_ids)

            # Compute confidence: exp(mean(max_log_prob per generated token))
            # outputs.scores: tuple of (1, vocab_size) logit tensors, one per generated token
            max_lps: list[float] = [
                F.log_softmax(score[0], dim=-1).max().item()
                for score in outputs.scores
            ]
            confidence: float | None = math.exp(sum(max_lps) / len(max_lps)) if max_lps else None
            binary_label_confidence: float | None = None
            if self.config.binary_decision_mode == BinaryDecisionMode.FINAL_ANSWER_LOGPROBS:
                binary_label_confidence = self._compute_binary_label_confidence(
                    generated_ids.tolist(),
                    outputs.scores,
                )

            results.append(InferenceResult(
                response_text=response_text,
                tokens_used=token_count,
                duration=duration,
                confidence=confidence,
                binary_label_confidence=binary_label_confidence,
            ))

        return results

    def _compute_binary_label_confidence(
        self,
        generated_token_ids: list[int],
        scores: tuple[torch.Tensor, ...],
    ) -> float | None:
        """Compute final-answer-position P(VULNERABLE) from decoder scores."""
        label_token_ids: dict[str, int] | None = self._get_binary_label_token_ids()
        if label_token_ids is None or not scores:
            return None

        label_position: int | None = self._find_final_answer_label_token_position(
            generated_token_ids
        )
        if label_position is None or label_position >= len(scores):
            return None

        label_position_scores = F.log_softmax(scores[label_position][0], dim=-1)
        p_vulnerable: float = math.exp(
            label_position_scores[label_token_ids["VULNERABLE"]].item()
        )
        p_safe: float = math.exp(
            label_position_scores[label_token_ids["SAFE"]].item()
        )
        denominator: float = p_vulnerable + p_safe
        if denominator == 0.0:
            return None
        return p_vulnerable / denominator

    def _find_final_answer_label_token_position(self, token_ids: list[int]) -> int | None:
        """Find the token index where the final-answer label begins."""
        if self.tokenizer is None or not token_ids:
            return None

        generated_text: str = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        marker_index: int = generated_text.rfind(_FINAL_ANSWER_PREFIX)
        if marker_index < 0:
            return None

        label_start_char_index: int = marker_index + len(_FINAL_ANSWER_PREFIX)
        while (
            label_start_char_index < len(generated_text)
            and generated_text[label_start_char_index].isspace()
        ):
            label_start_char_index += 1

        if label_start_char_index >= len(generated_text):
            return None

        for token_index in range(len(token_ids)):
            decoded_prefix: str = self.tokenizer.decode(
                token_ids[: token_index + 1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(decoded_prefix) > label_start_char_index:
                return token_index

        return None

    def _get_binary_label_token_ids(self) -> dict[str, int] | None:
        """Resolve token ids for SAFE and VULNERABLE when both are single tokens."""
        if self._binary_label_token_ids is not None:
            return self._binary_label_token_ids

        if self.tokenizer is None:
            return None

        label_token_ids: dict[str, int] = {}
        for label in _BINARY_LABELS:
            token_ids: list[int] = self.tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) != 1:
                if not self._did_warn_binary_label_tokenization:
                    logging.warning(
                        "Skipping final-answer logprob binary calibration for %s: %s tokenizes to %s",
                        self.config.model_identifier,
                        label,
                        token_ids,
                    )
                    self._did_warn_binary_label_tokenization = True
                return None
            label_token_ids[label] = token_ids[0]

        self._binary_label_token_ids = label_token_ids
        return self._binary_label_token_ids

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

    def _uses_qwen3_chat_template(self) -> bool:
        """Return whether the configured tokenizer/model uses Qwen3 chat controls."""
        candidate_identifiers: tuple[str, ...] = (
            self.config.model_identifier,
            self.config.tokenizer_identifier or "",
            self.config.hf_config_path or "",
        )
        return any("qwen3" in identifier.lower() for identifier in candidate_identifiers)

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
                template_kwargs: dict[str, object] = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if self._uses_qwen3_chat_template():
                    template_kwargs["enable_thinking"] = self.config.is_thinking_enabled

                formatted_prompt = apply_chat_template(messages, **template_kwargs)
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

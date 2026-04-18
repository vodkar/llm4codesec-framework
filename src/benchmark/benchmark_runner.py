import time

from pydantic import BaseModel, ConfigDict

from benchmark.config import ExperimentConfig
from benchmark.metrics_calculator import MetricsCalculatorFactory
from benchmark.models import PredictionResult, SampleCollection
from benchmark.prompt_generator import IPromptGenerator, get_prompt_generator
from benchmark.response_parser import IResponseParser, ResponseParserFactory
from benchmark.results import BenchmarkRunResult
from datasets.loaders.base import JsonDatasetLoader
from llm.factory import create_llm_inference
from llm.llm import ILLMInference, InferenceResult
from logging_tools import get_logger
from benchmark.enums import BinaryDecisionMode, TaskType

_LOGGER = get_logger(__name__)

_ERRORS_THRESHOLD = 0.2  # If more than 20% of samples fail, fail the experiment
_CHAT_TEMPLATE_TOKEN_OVERHEAD = 200  # Safety margin for chat-template special tokens


def _is_binary_task(task_type: object) -> bool:
    """Return whether the configured task uses binary labels."""

    return task_type in {
        TaskType.BINARY_VULNERABILITY,
        TaskType.BINARY_CWE_SPECIFIC,
        TaskType.BINARY_VULNERABILITY_SPECIFIC,
    }


def _majority_vote(labels: list[int | str]) -> int | str:
    """Return the most frequent label, breaking ties by first occurrence."""
    if not labels:
        raise ValueError("Cannot vote on empty label list")
    counts: dict[int | str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    max_count = max(counts.values())
    for label in labels:
        if counts[label] == max_count:
            return label
    return labels[0]  # unreachable


class BenchmarkRunner(BaseModel):
    """Main benchmark execution class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ExperimentConfig
    dataset_loader: JsonDatasetLoader

    def run(self) -> BenchmarkRunResult:
        _LOGGER.info(f"Starting {self.config.experiment_name} benchmark execution")

        _LOGGER.info(
            f"Loading {self.config.dataset_name} dataset from: {self.config.dataset_path}"
        )
        samples = self.dataset_loader.load_dataset(
            self.config.dataset_path, self.config.sample_limit * 3 if self.config.sample_limit else None
        )
        _LOGGER.info(f"Loaded {len(samples)} samples")

        llm = create_llm_inference(self.config)
        try:
            prompt_generator = get_prompt_generator(
                self.config,
                template_values={"cwe": self.config.cwe_type}
                if self.config.cwe_type
                else {},
            )

            samples = self._filter_samples_by_token_limit(samples, llm, prompt_generator)
            samples = self._apply_sample_limit(samples)

            if len(samples) == 0:
                raise RuntimeError(
                    "No samples available after token filtering and sample limit application"
                )

            response_parser = ResponseParserFactory.create_parser(
                self.config.task_type,
            )

            start_time = time.time()
            predictions = self._process_samples_with_batch_optimization(
                samples, llm, prompt_generator, response_parser
            )
            total_time: float = time.time() - start_time
        finally:
            llm.cleanup()

        metrics_calculator = MetricsCalculatorFactory.create_calculator(
            self.config.task_type
        )
        metrics = metrics_calculator.calculate(predictions)

        # Generate results
        return BenchmarkRunResult(
            metrics=metrics,
            total_samples=len(samples),
            total_time=total_time,
            predictions=predictions,
        )

    def _filter_samples_by_token_limit(
        self,
        samples: SampleCollection,
        llm: ILLMInference,
        prompt_generator: IPromptGenerator,
    ) -> SampleCollection:
        """Filter samples whose full formatted prompt exceeds the model's input budget.

        Counts tokens on the complete prompt text (system + user with code substituted)
        and compares against context_length - max_output_tokens - chat template overhead.
        """
        context_len = self.config.model_context_length_tokens
        output_budget = self.config.max_output_tokens
        input_budget = context_len - output_budget - _CHAT_TEMPLATE_TOKEN_OVERHEAD

        if input_budget <= 0:
            _LOGGER.warning(
                "Input budget is %d (context_length=%d, max_output_tokens=%d, overhead=%d). "
                "Token filtering skipped — set context_length separately in model config.",
                input_budget,
                context_len,
                output_budget,
                _CHAT_TEMPLATE_TOKEN_OVERHEAD,
            )
            return samples

        filtered_samples = []
        for sample in samples:
            system_prompt = prompt_generator.get_system_prompt()
            user_prompt = prompt_generator.get_user_prompt({"code": sample.code})
            full_text = system_prompt + "\n" + user_prompt
            if llm.count_input_tokens(full_text) <= input_budget:
                filtered_samples.append(sample)

        removed_samples = len(samples) - len(filtered_samples)
        if removed_samples > 0:
            _LOGGER.info(
                "Filtered out %d samples exceeding input budget of %d tokens "
                "(context=%d, output_budget=%d, overhead=%d)",
                removed_samples,
                input_budget,
                context_len,
                output_budget,
                _CHAT_TEMPLATE_TOKEN_OVERHEAD,
            )
        _LOGGER.info("Samples after token filtering: %d", len(filtered_samples))

        return SampleCollection(filtered_samples)

    def _apply_sample_limit(self, samples: SampleCollection) -> SampleCollection:
        """Apply sample limit after token filtering."""
        sample_limit = self.config.sample_limit
        if sample_limit and sample_limit < len(samples):
            limited_samples = list(samples)[:sample_limit]
            _LOGGER.info(f"Limited to {sample_limit} samples")
            return SampleCollection(limited_samples)

        return samples

    def _process_samples_with_batch_optimization(
        self,
        samples: SampleCollection,
        llm: ILLMInference,
        prompt_generator: IPromptGenerator,
        response_parser: IResponseParser,
    ) -> list[PredictionResult]:
        """
        Process samples with batch optimization for better GPU utilization.

        When self_consistency_samples > 1, each sample's prompt is submitted N times.
        The final predicted label is chosen by majority vote across the N responses.
        """
        n: int = self.config.self_consistency_samples
        _LOGGER.info(
            "Processing %d samples (self_consistency_samples=%d, effective batch size=%d)",
            len(samples), n, len(samples) * n,
        )

        # Build expanded prompt lists: each sample repeated N times consecutively
        expanded_system_prompts: list[str] = []
        expanded_user_prompts: list[str] = []
        for sample in samples:
            sys_p = prompt_generator.get_system_prompt()
            usr_p = prompt_generator.get_user_prompt({"code": sample.code})
            for _ in range(n):
                expanded_system_prompts.append(sys_p)
                expanded_user_prompts.append(usr_p)

        batch_results: list[InferenceResult] = llm.generate_responses_batch_optimized(
            expanded_system_prompts, expanded_user_prompts
        )

        predictions: list[PredictionResult] = []
        errors_count = 0

        for i, sample in enumerate(samples):
            # Slice the N InferenceResults that belong to this sample
            group: list[InferenceResult] = batch_results[i * n : (i + 1) * n]
            all_texts: list[str] = [r.response_text for r in group]
            total_tokens: int = sum(r.tokens_used for r in group)
            avg_time: float = sum(r.duration for r in group) / max(n, 1)
            raw_confidences = [r.confidence for r in group if r.confidence is not None]
            confidence: float | None = (
                sum(raw_confidences) / len(raw_confidences) if raw_confidences else None
            )
            raw_binary_label_confidences: list[float] = [
                r.binary_label_confidence
                for r in group
                if r.binary_label_confidence is not None
            ]
            binary_label_confidence: float | None = (
                sum(raw_binary_label_confidences) / len(raw_binary_label_confidences)
                if raw_binary_label_confidences
                else None
            )

            try:
                parsed_labels: list[int | str] = [
                    response_parser.parse_response(t) for t in all_texts
                ]
                predicted_label: int | str
                binary_logprob_threshold: float | None = self.config.binary_logprob_threshold
                if (
                    _is_binary_task(self.config.task_type)
                    and self.config.binary_decision_mode == BinaryDecisionMode.FINAL_ANSWER_LOGPROBS
                    and binary_label_confidence is not None
                    and binary_logprob_threshold is not None
                ):
                    predicted_label = (
                        1
                        if binary_label_confidence > binary_logprob_threshold
                        else 0
                    )
                else:
                    predicted_label = _majority_vote(parsed_labels)
                vote_counts: dict[str, int] = {}
                for lbl in parsed_labels:
                    key = str(lbl)
                    vote_counts[key] = vote_counts.get(key, 0) + 1

                true_label: int | str
                if isinstance(sample.label, int):
                    true_label = sample.label
                else:
                    true_label = response_parser.parse_response(str(sample.label))

                prediction = PredictionResult(
                    sample_id=sample.id,
                    predicted_label=predicted_label,
                    true_label=true_label,
                    confidence=confidence,
                    binary_label_confidence=binary_label_confidence,
                    response_text=all_texts[0],
                    processing_time=avg_time,
                    tokens_used=total_tokens,
                    is_success=True,
                    error_message=None,
                    all_responses=all_texts,
                    vote_counts=vote_counts,
                )
            except Exception as e:
                _LOGGER.error(
                    f"Error processing sample ID {sample.id}: {e}\nResponse text: {all_texts[0] if all_texts else ''}"
                )
                prediction = PredictionResult(
                    sample_id=sample.id,
                    predicted_label="",
                    true_label=sample.label,
                    confidence=confidence,
                    binary_label_confidence=binary_label_confidence,
                    response_text=all_texts[0] if all_texts else "",
                    processing_time=avg_time,
                    tokens_used=total_tokens,
                    error_message=str(e),
                    is_success=False,
                    all_responses=all_texts,
                    vote_counts={},
                )
                errors_count += 1
                if errors_count / len(samples) > _ERRORS_THRESHOLD:
                    _LOGGER.error(
                        f"Error rate exceeded threshold of {_ERRORS_THRESHOLD:.0%} - aborting benchmark"
                    )
                    raise RuntimeError(
                        "Too many errors during benchmark execution - aborting"
                    ) from e

            predictions.append(prediction)

        _LOGGER.info(f"Completed processing all {len(samples)} samples")
        return predictions

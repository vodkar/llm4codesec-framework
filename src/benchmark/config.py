import json
import logging
from pathlib import Path

from llama_cpp import Any
from pydantic import BaseModel, PrivateAttr

from benchmark.enums import BackendFrameworks, ModelType, TaskType

_LOGGER = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for the model to benchmark."""

    model_name: str
    model_identifier: str
    model_type: ModelType
    batch_size: int
    max_tokens: int
    temperature: float
    use_quantization: bool
    backend: BackendFrameworks
    is_thinking_enabled: bool = False


class DatasetConfig(BaseModel):
    """Configuration for the dataset to benchmark on."""

    name: str
    description: str
    path: Path
    task_type: TaskType
    vulnerability_type: str | None = None
    cwe_type: str | None = None

    def model_post_init(self, context: Any) -> None:
        if not self.path.exists():
            _LOGGER.error(f"Missing dataset file: {self.path}")
            _LOGGER.error(
                "Run run_setup_castle_dataset.py first to create processed datasets"
            )
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        super().model_post_init(context)


class PromptConfig(BaseModel):
    """Configuration for the prompts used in benchmarking."""

    name: str
    slug: str
    system_prompt: str
    user_prompt: str


class OutputConfig(BaseModel):
    """Configuration for output settings of the benchmark results."""

    base_output_dir: Path
    include_timestamp: bool = True
    save_predictions: bool = True
    save_metrics: bool = True
    save_detailed_report: bool = True


class ExperimentConfig(BaseModel):
    """Configuration for single experiment execution."""

    model_type: ModelType
    task_type: TaskType
    description: str
    dataset_path: Path
    backend: BackendFrameworks
    experiment_name: str
    batch_size: int
    max_tokens: int
    temperature: float
    use_quantization: bool = True
    is_thinking_enabled: bool = False
    cwe_type: str | None = None
    system_prompt_template: str
    user_prompt_template: str
    sample_limit: int | None

    __model_config: ModelConfig = PrivateAttr()
    __dataset_config: DatasetConfig = PrivateAttr()
    __prompt_config: PromptConfig = PrivateAttr()
    __output_settings: OutputConfig = PrivateAttr()

    @classmethod
    def from_file(
        cls,
        config: Path,
        model_key: str,
        dataset_key: str,
        prompt_key: str,
        experiment_name: str,
        sample_limit: int | None = None,
    ) -> "ExperimentConfig":
        if not config.exists():
            raise FileNotFoundError(f"Config file not found: {config}")

        config_data = json.loads(config.read_text())

        # Extract relevant sections based on keys
        if model_key not in config_data["models"]:
            raise ValueError(f"Model key '{model_key}' not found in config")
        model_config_dict = config_data["models"][model_key]
        if dataset_key not in config_data["datasets"]:
            raise ValueError(f"Dataset key '{dataset_key}' not found in config")
        dataset_config_dict = config_data["datasets"][dataset_key]
        if prompt_key not in config_data["prompts"]:
            raise ValueError(f"Prompt key '{prompt_key}' not found in config")
        prompt_config_dict = config_data["prompts"][prompt_key]
        output_config_dict = config_data["output_settings"]

        _LOGGER.info(f"Received config for experiment: {experiment_name}")
        _LOGGER.info(
            f"Model: {model_config_dict['model_identifier']} ({model_config_dict['model_type']})"
        )
        _LOGGER.info(f"Dataset: {dataset_config_dict['description']}")
        _LOGGER.info(f"Prompt: {prompt_config_dict['name']}")

        # Validate and create config objects
        model_config = ModelConfig(**model_config_dict, model_name=model_key)
        dataset_config = DatasetConfig(**dataset_config_dict, name=dataset_key)
        prompt_config = PromptConfig(**prompt_config_dict, slug=prompt_key)
        output_config = OutputConfig(**output_config_dict)

        return cls.from_configs(
            model_config=model_config,
            dataset_config=dataset_config,
            prompt_config=prompt_config,
            output_settings=output_config,
            experiment_name=experiment_name,
            sample_limit=sample_limit,
        )

    @classmethod
    def from_configs(
        cls,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        prompt_config: PromptConfig,
        output_settings: OutputConfig,
        experiment_name: str,
        sample_limit: int | None = None,
    ) -> "ExperimentConfig":
        """Create a BenchmarkConfig from separate model, dataset, and prompt configs."""
        config = cls(
            model_type=model_config.model_type,
            task_type=dataset_config.task_type,
            description=dataset_config.description,
            dataset_path=dataset_config.path,
            backend=model_config.backend,
            experiment_name=experiment_name,
            batch_size=model_config.batch_size,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            use_quantization=model_config.use_quantization,
            is_thinking_enabled=model_config.is_thinking_enabled,
            cwe_type=dataset_config.cwe_type,
            system_prompt_template=prompt_config.system_prompt,
            user_prompt_template=prompt_config.user_prompt,
            sample_limit=sample_limit,
        )
        config.__model_config = model_config
        config.__dataset_config = dataset_config
        config.__prompt_config = prompt_config
        config.__output_settings = output_settings
        return config

    @property
    def output_dir(self) -> Path:
        """Generate the final output directory path based on settings."""
        output = (
            self.__output_settings.base_output_dir
            / self.experiment_name
            / self.__dataset_config.name
            / self.__model_config.model_name
            / self.__prompt_config.slug
        )

        return output

    @property
    def dataset_name(self) -> str:
        return self.__dataset_config.name

    @property
    def model_identifier(self) -> str:
        return self.__model_config.model_identifier

    @property
    def model_name(self) -> str:
        return self.__model_config.model_name


class ExperimentsPlanConfig(BaseModel):
    """Configuration for running a plan of experiments."""

    plan_name: str
    description: str
    experiments: list[ExperimentConfig]

    @classmethod
    def from_file(cls, config: Path, plan_name: str) -> "ExperimentsPlanConfig":
        if not config.exists():
            raise FileNotFoundError(f"Config file not found: {config}")

        config_data = json.loads(config.read_text())
        if "experiment_plans" not in config_data:
            raise ValueError("Config file missing 'experiment_plans' section")
        if plan_name not in config_data["experiment_plans"]:
            raise ValueError(f"Experiment plan '{plan_name}' not found in config")
        plan_config = config_data["experiment_plans"][plan_name]

        experiments: list[ExperimentConfig] = []
        for model_key in plan_config["models"]:
            for dataset_key in plan_config["datasets"]:
                for prompt_key in plan_config["prompts"]:
                    experiments.append(
                        ExperimentConfig.from_file(
                            config=config,
                            model_key=model_key,
                            dataset_key=dataset_key,
                            prompt_key=prompt_key,
                            experiment_name=plan_name,
                            sample_limit=plan_config.get("sample_limit"),
                        )
                    )

        return cls(
            experiments=experiments,
            plan_name=plan_name,
            description=plan_config.get("description", ""),
        )

    @staticmethod
    def list_plans(config: Path) -> list["ExperimentsPlanConfig"]:
        if not config.exists():
            raise FileNotFoundError(f"Config file not found: {config}")

        config_data = json.loads(config.read_text())
        if "experiment_plans" not in config_data:
            raise ValueError("Config file missing 'experiment_plans' section")
        plans: dict[str, dict[str, Any]] = config_data.get("experiment_plans", {})
        if not plans or not isinstance(plans, dict):
            raise ValueError(
                "Config file 'experiment_plans' section is empty or invalid"
            )

        return [
            ExperimentsPlanConfig.from_file(config, plan_name)
            for plan_name in plans.keys()
        ]

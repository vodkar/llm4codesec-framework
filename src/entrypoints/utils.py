from pathlib import Path

from benchmark.config import ExperimentsPlanConfig


def list_plans(config: Path):
    print("\nAvailable experiment plans:")
    for plan in ExperimentsPlanConfig.list_plans(config):
        print(f"  {plan.plan_name}: {plan.description}")
        print(
            f"    Datasets: {', '.join([exp.dataset_name for exp in plan.experiments])}"
        )
        print(f"    Models: {', '.join([exp.model_name for exp in plan.experiments])}")
        print(
            f"    Prompts: {', '.join([exp.system_prompt_template for exp in plan.experiments])}"
        )
        print(f"    Total experiments: {len(plan.experiments)}")
        print()

# VulBench Benchmark

## Overview

VulBench evaluates LLMs on C/C++ vulnerability detection across multiple source datasets (D2A, CTF, MAGMA, Big-Vul, Devign).
This benchmark now uses the unified CLI entrypoint.

## Core Components

- **Configuration**: `src/configs/vulbench_experiments.json`
- **Unified CLI**: `src/cli.py`
- **Dataset Loader**: `src/datasets/loaders/vulbench_dataset_loader.py`
- **Data Processor**: `src/scripts/process_vulbench_data.py`

## Datasets

VulBench provides binary and multiclass variants:

- `binary_d2a`, `binary_ctf`, `binary_magma`, `binary_big_vul`, `binary_devign`
- `multiclass_d2a`, `multiclass_ctf`, `multiclass_magma`, `multiclass_big_vul`, `multiclass_devign`
- vulnerability-specific binary subsets, for example `vulnerability_buffer_overflow`

## Prompt Strategies

### Binary

- `basic_security`
- `detailed_analysis`
- `context_aware`
- `step_by_step`

### Multiclass

- `multiclass_basic`
- `multiclass_detailed`
- `multiclass_comprehensive`

## CLI Usage

All commands run from the repository root:

```bash
PYTHONPATH=src python -m cli --help
```

### List available configurations and plans

```bash
PYTHONPATH=src python -m cli list-configs vulbench
PYTHONPATH=src python -m cli list-plans vulbench
```

### Run single experiments

```bash
# Binary classification
PYTHONPATH=src python -m cli run vulbench \
  --model qwen3-4b \
  --dataset binary_d2a \
  --prompt detailed_analysis

# Multiclass classification
PYTHONPATH=src python -m cli run vulbench \
  --model qwen3-4b \
  --dataset multiclass_d2a \
  --prompt multiclass_detailed
```

### Run experiment plans

```bash
# Quick checks
PYTHONPATH=src python -m cli run-plan vulbench --plan quick_test
PYTHONPATH=src python -m cli run-plan vulbench --plan multiclass_quick_test

# Prompt/model/dataset comparisons
PYTHONPATH=src python -m cli run-plan vulbench --plan prompt_comparison
PYTHONPATH=src python -m cli run-plan vulbench --plan model_comparison
PYTHONPATH=src python -m cli run-plan vulbench --plan binary_dataset_comparison

# Full evaluation
PYTHONPATH=src python -m cli run-plan vulbench --plan comprehensive_evaluation
```

### Common options

```bash
PYTHONPATH=src python -m cli run-plan vulbench \
  --plan prompt_comparison \
  --sample-limit 100 \
  --output-dir results/vulbench_test \
  --verbose
```

## Notes

- Plan/model/dataset/prompt keys come from `src/configs/vulbench_experiments.json`.
- Use `list-configs` and `list-plans` before running large experiments.

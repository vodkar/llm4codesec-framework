# JitVul Benchmark

## Overview

JitVul is integrated into the unified benchmark CLI and uses the same configuration model as CASTLE, CVEFixes, and VulBench.

## Core Components

- Configuration file: `src/configs/jitvul_experiments.json`
- Unified CLI: `src/cli.py`

## Prompt Strategies

### Binary

- `basic_security`
- `detailed_analysis`
- `cwe_focused`
- `context_aware`
- `step_by_step`

### Multiclass

- `multiclass_basic`
- `multiclass_detailed`
- `multiclass_comprehensive`

## Datasets

- `binary_all`
- `multiclass_all`
- `cwe_125`
- `cwe_190`
- `cwe_476`
- `cwe_787`

## Unified CLI Usage

### Run a single experiment

```bash
PYTHONPATH=src python -m cli run jitvul \
  --model qwen3-4b \
  --dataset binary_all \
  --prompt detailed_analysis
```

### Run a plan

```bash
PYTHONPATH=src python -m cli run-plan jitvul --plan quick_test
```

### List configurations

```bash
PYTHONPATH=src python -m cli list-configs jitvul
PYTHONPATH=src python -m cli list-plans jitvul
```

### Common options

```bash
PYTHONPATH=src python -m cli run-plan jitvul \
  --plan prompt_comparison \
  --output-dir results/jitvul_test \
  --log-level DEBUG
```

## Typical Plans

- `quick_test`
- `multiclass_quick_test`
- `prompt_comparison`
- `multiclass_prompt_comparison`
- `model_comparison`
- `multiclass_model_comparison`
- `small_models_binary`
- `small_models_multiclass`
- `large_models_binary`
- `large_models_multiclass`
- `cwe_specific_analysis`
- `comprehensive_evaluation`

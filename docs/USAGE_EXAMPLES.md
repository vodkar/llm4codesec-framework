# Unified CLI Usage Examples

## Overview

Use a single CLI entrypoint for experiments and plans across CASTLE, JitVul, CVEFixes, and VulBench.

```bash
PYTHONPATH=src python -m cli --help
```

## Quick Start

### List available configs and plans

```bash
PYTHONPATH=src python -m cli list-configs castle
PYTHONPATH=src python -m cli list-configs jitvul
PYTHONPATH=src python -m cli list-configs cvefixes
PYTHONPATH=src python -m cli list-configs vulbench

PYTHONPATH=src python -m cli list-plans castle
PYTHONPATH=src python -m cli list-plans jitvul
PYTHONPATH=src python -m cli list-plans cvefixes
PYTHONPATH=src python -m cli list-plans vulbench
```

### Run single experiments

```bash
PYTHONPATH=src python -m cli run jitvul \
  --model qwen3-4b \
  --dataset binary_all \
  --prompt basic_security \
  --sample-limit 50

PYTHONPATH=src python -m cli run cvefixes \
  --model deepseek-coder-v2-lite-16b \
  --dataset binary_c_file \
  --prompt detailed_analysis \
  --sample-limit 100

PYTHONPATH=src python -m cli run vulbench \
  --model qwen3-4b \
  --dataset binary_d2a \
  --prompt basic_security
```

### Run experiment plans

```bash
PYTHONPATH=src python -m cli run-plan castle --plan quick_test
PYTHONPATH=src python -m cli run-plan jitvul --plan quick_test
PYTHONPATH=src python -m cli run-plan cvefixes --plan quick_test
PYTHONPATH=src python -m cli run-plan vulbench --plan quick_test
```

### Advanced options

```bash
PYTHONPATH=src python -m cli run-plan jitvul \
  --plan prompt_comparison \
  --output-dir results/jitvul_prompt_study \
  --log-level DEBUG

PYTHONPATH=src python -m cli run cvefixes \
  --model gemma3-4b \
  --dataset cwe_125 \
  --prompt cwe_focused \
  --verbose

PYTHONPATH=src python -m cli run-plan jitvul \
  --config src/configs/jitvul_experiments.json \
  --plan comprehensive_evaluation

PYTHONPATH=src python -m cli run-plan castle \
  --config-dir src/configs/shared \
  --experiments-config src/configs/castle/experiments.json \
  --datasets-config src/configs/castle/datasets.json \
  --plan quick_test

PYTHONPATH=src python -m cli run castle \
  --config-dir src/configs/shared \
  --experiments-config src/configs/castle/experiments.json \
  --datasets-config src/configs/castle/datasets.json \
  --model qwen3-8b-gguf \
  --dataset binary_all \
  --prompt basic_security \
  --sample-limit 5
```

## Troubleshooting

1. **Config not found**

```bash
PYTHONPATH=src python -m cli list-configs castle
```

1. **Model/dataset/prompt key errors**

  Check available keys with `list-configs`.

1. **Dataset path errors**

  Ensure processed files exist under `datasets_processed/`.

1. **Import issues**

  Run from repo root and keep `PYTHONPATH=src`.

## Quick validation

```bash
PYTHONPATH=src python -m cli --help
PYTHONPATH=src python -m cli list-plans castle
PYTHONPATH=src python -m cli list-configs cvefixes
```

# Refactored Benchmark System - Usage Examples

## Overview

The refactored benchmark system provides a unified, configuration-based approach to running LLM security evaluations across CASTLE, JitVul, and CVEFixes datasets.

## Quick Start Commands

### 1. List Available Configurations

```bash
# List JitVul configurations
python src/entrypoints/run_jitvul_benchmark.py --list-configs

# List CVEFixes configurations  
python src/entrypoints/run_cvefixes_benchmark.py --list-configs

# List configurations via unified runner
python src/entrypoints/run_unified_benchmark.py jitvul --list-configs
python src/entrypoints/run_unified_benchmark.py cvefixes --list-configs
python src/entrypoints/run_unified_benchmark.py castle --list-configs
```

### 2. Run Single Experiments

```bash
# JitVul single experiment
python src/entrypoints/run_jitvul_benchmark.py \
  --model qwen3-4b \
  --dataset binary_all \
  --prompt basic_security \
  --sample-limit 50

# CVEFixes single experiment  
python src/entrypoints/run_cvefixes_benchmark.py \
  --model deepseek-coder-v2-lite-16b \
  --dataset binary_c_file \
  --prompt detailed_analysis \
  --sample-limit 100

# Using unified runner
python src/entrypoints/run_unified_benchmark.py jitvul \
  --model llama3.2-3B \
  --dataset binary_all \
  --prompt context_aware \
  --sample-limit 25
```

### 3. Run Experiment Plans

```bash
# Quick test plans for development
python src/entrypoints/run_jitvul_benchmark.py --plan quick_test
python src/entrypoints/run_cvefixes_benchmark.py --plan quick_test
python src/entrypoints/run_castle_benchmark.py --plan quick_test
python src/entrypoints/run_vulbench_benchmark.py --plan quick_test

# Model comparison studies
python src/entrypoints/run_jitvul_benchmark.py --plan model_comparison
python src/entrypoints/run_cvefixes_benchmark.py --plan model_comparison

# Comprehensive evaluations
python src/entrypoints/run_unified_benchmark.py jitvul --plan comprehensive_evaluation
python src/entrypoints/run_unified_benchmark.py cvefixes --plan comprehensive_evaluation
```

### 4. Advanced Options

```bash
# Custom output directory
python src/entrypoints/run_jitvul_benchmark.py \
  --plan prompt_comparison \
  --output-dir results/jitvul_prompt_study

# Enable verbose logging
python src/entrypoints/run_cvefixes_benchmark.py \
  --model gemma3-1b \
  --dataset cwe_125 \
  --prompt cwe_specific \
  --verbose

# Custom configuration file
python src/entrypoints/run_unified_benchmark.py jitvul \
  --config custom_jitvul_config.json \
  --plan my_custom_experiment
```

## Configuration Examples

### Adding New Models

Edit the configuration file to add new models:

```json
{
  "model_configurations": {
    "my_custom_model": {
      "model_name": "custom/my-new-model",
      "model_type": "CUSTOM",
      "config": {
        "max_tokens": 2048,
        "temperature": 0.0
      }
    }
  }
}
```

### Creating Custom Experiment Plans

```json
{
  "experiment_plans": {
    "my_custom_plan": {
      "description": "Custom evaluation with specific models and datasets",
      "datasets": ["binary_all", "cwe_125"],
      "models": ["qwen3-4b", "deepseek-coder-v2-lite-16b"],
      "prompts": ["basic_security", "detailed_analysis"],
      "sample_limit": 200
    }
  }
}
```

## Migration Guide

### From Old System to New System

**Old JitVul Command:**
```bash
python src/entrypoints/run_jitvul_benchmark.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --task-type binary_vulnerability \
  --dataset-path jitvul_data.json
```

**New JitVul Command:**
```bash
python src/entrypoints/run_jitvul_benchmark.py \
  --model qwen3-4b \
  --dataset binary_all \
  --prompt basic_security
```

**Old CVEFixes Command:**
```bash
python src/entrypoints/run_cvefixes_benchmark.py \
  --model-type qwen \
  --dataset-path cvefixes_data.json
```

**New CVEFixes Command:**
```bash
python src/entrypoints/run_cvefixes_benchmark.py \
  --model qwen3-4b \
  --dataset binary_c_file \
  --prompt basic_security
```

## Benefits of the New System

### 1. Consistency
- Same CLI arguments across all benchmarks
- Identical configuration structure
- Unified model definitions

### 2. Flexibility  
- Easy experiment plan definitions
- Mix and match models/datasets/prompts
- Configurable sample limits and output directories

### 3. Maintainability
- Single source of truth for configurations
- No hard-coded model names or paths
- Easy to add new models and experiments

### 4. Usability
- Single unified entry point option
- Clear configuration listing
- Helpful error messages and validation

## Troubleshooting

### Common Issues

1. **Configuration not found**
   ```bash
   # List available configurations
   python src/entrypoints/run_jitvul_benchmark.py --list-configs
   ```

2. **Model not recognized**
   - Check `model_configurations` section in config file
   - Ensure model key matches exactly

3. **Dataset path errors**
   - Verify dataset files exist in `datasets_processed/`
   - Check `dataset_path` in configuration

4. **Import errors**
   - Run from project root directory
   - Ensure Python path includes `src/` directory

### Quick Validation

```bash
# Test all systems are working
cd src/
python entrypoints/run_jitvul_benchmark.py --help
python entrypoints/run_cvefixes_benchmark.py --help  
python entrypoints/run_unified_benchmark.py --help
```

## Next Steps

1. **Run Development Tests**: Start with `--plan quick_test` for each dataset
2. **Customize Configurations**: Add your preferred models and experiment plans
3. **Scale Up**: Use comprehensive evaluation plans for full studies
4. **Extend Framework**: Add new datasets following the same pattern

---

**Ready to use!** The refactored system provides a powerful, flexible foundation for LLM security evaluation research.

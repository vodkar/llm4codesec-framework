# JitVul Benchmark - Configuration-Based Runner

## Overview

The JitVul benchmark provides comprehensive vulnerability detection evaluation using the JitVul dataset. The system has been unified with the CASTLE and CVEFixes benchmarks to use consistent configuration-based approaches and prompt strategies across all datasets.

## New Unified Configuration System

### Key Improvements ✅
- **Unified Configuration**: JSON-based experiment configuration following CASTLE/CVEFixes pattern
- **Consistent CLI**: Same command-line interface across all benchmarks (CASTLE, JitVul, CVEFixes)
- **Unified Prompts**: Standardized prompt strategies across all benchmarks
- **Separated Task Types**: Distinct experiment plans for binary and multiclass classification
- **Flexible Experiments**: Easy definition of model/dataset/prompt combinations
- **Single Entry Point**: All experiments configurable through JSON files
- **Model Synchronization**: Consistent model definitions across all datasets

### Core Components
- **Configuration File**: `src/configs/jitvul_experiments.json`
- **Unified Runner**: `src/entrypoints/run_jitvul_benchmark.py`
- **Universal Runner**: `src/entrypoints/run_unified_benchmark.py` (handles all datasets)
- **Dataset Loader**: `src/datasets/jitvul_dataset_loader.py`

## Configuration Structure

The JitVul configuration now follows the same structure as CASTLE and CVEFixes for consistency:

```json
{
  "experiment_metadata": {
    "name": "JitVul Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on JitVul benchmark with multiple prompt strategies",
    "version": "1.0",
    "dataset": "JitVul-Benchmark v1.0"
  },
  "dataset_configurations": {
    "binary_all": {
      "dataset_path": "datasets_processed/jitvul/jitvul_binary.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: all vulnerability types"
    },
    "multiclass_all": {
      "dataset_path": "datasets_processed/jitvul/jitvul_multiclass.json",
      "task_type": "multiclass_vulnerability", 
      "description": "Multi-class classification: vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "...",
      "user_prompt": "Analyze this code for security vulnerabilities:\n\n{code}"
    }
  }
}
```

## Available Prompt Strategies

### Binary Classification Prompts
- **`basic_security`**: Simple vulnerability detection analysis
- **`detailed_analysis`**: Comprehensive security analysis with CWE knowledge
- **`cwe_focused`**: Targeted analysis for specific CWE types (use with CWE-specific datasets)
- **`context_aware`**: Production-focused analysis considering real-world exploitation
- **`step_by_step`**: Methodical analysis following systematic steps

### Multiclass Classification Prompts
- **`multiclass_basic`**: Basic vulnerability type classification
- **`multiclass_detailed`**: Detailed CWE pattern analysis and classification
- **`multiclass_comprehensive`**: Comprehensive vulnerability classification for production systems

## Dataset Types

### Binary Classification
- **`binary_all`**: Vulnerable vs. safe classification across all vulnerability types

### Multiclass Classification
- **`multiclass_all`**: CWE type identification and classification

### CWE-Specific Detection
- **`cwe_125`**: Out-of-bounds Read detection
- **`cwe_190`**: Integer Overflow detection
- **`cwe_476`**: NULL Pointer Dereference detection
- **`cwe_787`**: Out-of-bounds Write detection

## Command Line Interface

### Individual Experiments
```bash
# Binary classification experiment
python src/entrypoints/run_jitvul_benchmark.py \
  --model qwen3-4b \
  --dataset binary_all \
  --prompt detailed_analysis

# Multiclass classification experiment  
python src/entrypoints/run_jitvul_benchmark.py \
  --model qwen3-4b \
  --dataset multiclass_all \
  --prompt multiclass_detailed

# CWE-specific analysis
python src/entrypoints/run_jitvul_benchmark.py \
  --model deepseek-coder-v2-lite-16b \
  --dataset cwe_125 \
  --prompt cwe_focused

# Using unified runner (handles all datasets)
python src/entrypoints/run_unified_benchmark.py \
  --dataset-type jitvul \
  --model qwen3-4b \
  --dataset binary_all \
  --prompt detailed_analysis
```

### Experiment Plans
```bash
# Quick testing
python src/entrypoints/run_jitvul_benchmark.py --plan quick_test
python src/entrypoints/run_jitvul_benchmark.py --plan multiclass_quick_test

# Prompt strategy comparison
python src/entrypoints/run_jitvul_benchmark.py --plan prompt_comparison
python src/entrypoints/run_jitvul_benchmark.py --plan multiclass_prompt_comparison

# Model size comparison
python src/entrypoints/run_jitvul_benchmark.py --plan model_comparison
python src/entrypoints/run_jitvul_benchmark.py --plan multiclass_model_comparison

# Model category evaluation
python src/entrypoints/run_jitvul_benchmark.py --plan small_models_binary
python src/entrypoints/run_jitvul_benchmark.py --plan small_models_multiclass
python src/entrypoints/run_jitvul_benchmark.py --plan large_models_binary
python src/entrypoints/run_jitvul_benchmark.py --plan large_models_multiclass

# CWE-specific analysis
python src/entrypoints/run_jitvul_benchmark.py --plan cwe_specific_analysis

# Comprehensive evaluation
python src/entrypoints/run_jitvul_benchmark.py --plan comprehensive_evaluation
```

### Common Options
```bash
# List available configurations
python src/entrypoints/run_jitvul_benchmark.py --list-configs

# Limit samples and set output directory
python src/entrypoints/run_jitvul_benchmark.py \
  --plan prompt_comparison \
  --sample-limit 100 \
  --output-dir results/jitvul_test

# Enable verbose logging
python src/entrypoints/run_jitvul_benchmark.py \
  --plan quick_test \
  --verbose
```

## Available Experiment Plans

### Quick Testing
- **`quick_test`**: Binary classification with 10 samples for development
- **`multiclass_quick_test`**: Multiclass classification with 10 samples for development

### Prompt Strategy Analysis
- **`prompt_comparison`**: Compare all binary classification prompts
- **`multiclass_prompt_comparison`**: Compare all multiclass classification prompts

### Model Performance Analysis
- **`model_comparison`**: Compare models on binary classification
- **`multiclass_model_comparison`**: Compare models on multiclass classification
- **`small_models_binary`**: Evaluate small models (≤4B parameters) on binary tasks
- **`small_models_multiclass`**: Evaluate small models on multiclass tasks
- **`large_models_binary`**: Evaluate large models (>4B parameters) on binary tasks
- **`large_models_multiclass`**: Evaluate large models on multiclass tasks

### Specialized Analysis
- **`cwe_specific_analysis`**: CWE-specific vulnerability detection
- **`comprehensive_evaluation`**: Full evaluation across all datasets and models

## Key Features

### 1. Unified Configuration System
- **Consistent Prompts**: Same prompt strategies as CASTLE and CVEFixes benchmarks
- **Task Separation**: Distinct prompts and experiments for binary vs. multiclass tasks
- **JSON Configuration**: All experiments defined in `jitvul_experiments.json`
- **Flexible Combinations**: Easy model/dataset/prompt combinations
- **Experiment Plans**: Predefined experimental setups

### 2. Enhanced Context Support
- **Call Graph Integration**: Function relationship context when available
- **Severity Classification**: Vulnerability severity determination
- **Rich Metadata**: Project info, commit details, function hashes

### 3. Comprehensive Evaluation
- **Standard Metrics**: AUC-ROC (primary), Accuracy, Precision, Recall, F1-score
- **Per-Class Analysis**: Individual CWE type performance  
- **Framework Integration**: Uses unified benchmark framework evaluation system
- **Error Analysis**: Detailed analysis of misclassifications

## Task Types

### Binary Vulnerability Detection
- **Purpose**: Determine if code contains any security vulnerability
- **Output**: `VULNERABLE` or `SAFE`
- **Datasets**: `binary_all`
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`

### Multiclass Vulnerability Classification  
- **Purpose**: Identify specific CWE type of vulnerability
- **Output**: Specific CWE identifier (e.g., `CWE-125`) or `SAFE`
- **Datasets**: `multiclass_all`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`

### CWE-Specific Detection
- **Purpose**: Detect specific types of vulnerabilities
- **Output**: `VULNERABLE` or `SAFE` for the specific CWE type
- **Datasets**: `cwe_125`, `cwe_787`
- **Prompts**: `cwe_focused`

---

**Status**: ✅ UNIFIED AND READY FOR USE  
**Last Updated**: June 2025  
**Configuration Aligned**: ✅ Castle, CVEFixes, JitVul

# VulBench Benchmark - Configuration-Based Runner

## Overview

The VulBench benchmark provides comprehensive vulnerability detection evaluation using the VulBench dataset containing multiple vulnerability datasets (D2A, CTF, MAGMA, Big-Vul, Devign). The system has been unified with the CASTLE, CVEFixes, and JitVul benchmarks to use consistent configuration-based approaches and prompt strategies across all datasets.

## Unified Configuration System

### Key Features ✅
- **Unified Configuration**: JSON-based experiment configuration following CASTLE/CVEFixes/JitVul pattern
- **Consistent CLI**: Same command-line interface across all benchmarks
- **C/C++-Specific Prompts**: Tailored prompts for C/C++ vulnerability detection
- **Separated Task Types**: Distinct experiment plans for binary and multiclass classification
- **Multiple Datasets**: Support for 5 VulBench sub-datasets with binary and multiclass variants
- **Flexible Experiments**: Easy definition of model/dataset/prompt combinations
- **Fixed Label Format**: Corrected label parsing and response format compatibility

### Core Components
- **Configuration File**: `src/configs/vulbench_experiments.json`
- **Benchmark Runner**: `src/entrypoints/run_vulbench_benchmark_new.py`
- **Unified Runner**: `src/entrypoints/run_unified_benchmark.py` (handles all datasets)
- **Dataset Loader**: `src/datasets/loaders/vulbench_dataset_loader.py` (updated for correct labels)
- **Data Processor**: `src/scripts/process_vulbench_data.py`

## VulBench Datasets

VulBench contains 5 sub-datasets, each available in both binary and multiclass variants:

| Dataset | Description | Binary Classification | Multiclass Classification |
|---------|-------------|----------------------|---------------------------|
| **D2A** | Detect-to-Assign dataset | Vulnerable vs Non-vulnerable | Vulnerability type classification |
| **CTF** | Capture The Flag challenges | Vulnerable vs Non-vulnerable | Vulnerability type classification |
| **MAGMA** | Generated vulnerabilities | Vulnerable vs Non-vulnerable | Vulnerability type classification |
| **Big-Vul** | Large-scale vulnerability dataset | Vulnerable vs Non-vulnerable | Vulnerability type classification |
| **Devign** | Graph-based vulnerability dataset | Vulnerable vs Non-vulnerable | Vulnerability type classification |

## Available Prompt Strategies

### Binary Classification Prompts
- **`basic_security`**: Simple C/C++ vulnerability detection analysis
- **`detailed_analysis`**: Comprehensive C/C++ security analysis with CWE knowledge
- **`context_aware`**: Production-focused C/C++ analysis considering real-world exploitation
- **`step_by_step`**: Methodical C/C++ analysis following systematic steps

### Multiclass Classification Prompts
- **`multiclass_basic`**: Basic C/C++ vulnerability type classification
- **`multiclass_detailed`**: Detailed C/C++ vulnerability pattern analysis and classification
- **`multiclass_comprehensive`**: Comprehensive C/C++ vulnerability classification for production systems

### Vulnerability Types in VulBench
- **Integer-Overflow**: Integer overflow/underflow vulnerabilities
- **Buffer-Overflow**: Buffer overflow/underflow vulnerabilities
- **Null-Pointer-Dereference**: NULL pointer dereference issues
- **Use-After-Free**: Use-after-free memory errors
- **Double-Free**: Double-free memory errors
- **Memory-Leak**: Memory leak issues
- **Format-String**: Format string vulnerabilities

## Command Line Interface

### Individual Experiments
```bash
# Binary classification experiment
python src/entrypoints/run_vulbench_benchmark_new.py \
  --model qwen3-4b \
  --dataset binary_d2a \
  --prompt detailed_analysis

# Multiclass classification experiment  
python src/entrypoints/run_vulbench_benchmark_new.py \
  --model qwen3-4b \
  --dataset multiclass_d2a \
  --prompt multiclass_detailed

# Using unified runner (handles all datasets)
python src/entrypoints/run_unified_benchmark.py \
  --dataset-type vulbench \
  --model qwen3-4b \
  --dataset binary_d2a \
  --prompt detailed_analysis
```

### Experiment Plans
```bash
# Quick testing
python src/entrypoints/run_vulbench_benchmark_new.py --plan quick_test
python src/entrypoints/run_vulbench_benchmark_new.py --plan multiclass_quick_test

# Prompt strategy comparison
python src/entrypoints/run_vulbench_benchmark_new.py --plan prompt_comparison
python src/entrypoints/run_vulbench_benchmark_new.py --plan multiclass_prompt_comparison

# Dataset comparison
python src/entrypoints/run_vulbench_benchmark_new.py --plan binary_dataset_comparison
python src/entrypoints/run_vulbench_benchmark_new.py --plan multiclass_dataset_comparison

# Model size comparison
python src/entrypoints/run_vulbench_benchmark_new.py --plan model_comparison
python src/entrypoints/run_vulbench_benchmark_new.py --plan multiclass_model_comparison

# Model category evaluation
python src/entrypoints/run_vulbench_benchmark_new.py --plan small_models_binary
python src/entrypoints/run_vulbench_benchmark_new.py --plan small_models_multiclass
python src/entrypoints/run_vulbench_benchmark_new.py --plan large_models_binary
python src/entrypoints/run_vulbench_benchmark_new.py --plan large_models_multiclass

# Comprehensive evaluation
python src/entrypoints/run_vulbench_benchmark_new.py --plan comprehensive_evaluation

# Vulnerability-specific analysis
python src/entrypoints/run_vulbench_benchmark_new.py --plan vulnerability_specific_analysis
python src/entrypoints/run_vulbench_benchmark_new.py --plan vulnerability_prompt_comparison
python src/entrypoints/run_vulbench_benchmark_new.py --plan small_models_vulnerability_specific
python src/entrypoints/run_vulbench_benchmark_new.py --plan large_models_vulnerability_specific
```

### Common Options
```bash
# List available configurations
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs

# Limit samples and set output directory
python src/entrypoints/run_vulbench_benchmark_new.py \
  --plan prompt_comparison \
  --sample-limit 100 \
  --output-dir results/vulbench_test

# Enable verbose logging
python src/entrypoints/run_vulbench_benchmark_new.py \
  --plan quick_test \
  --verbose
```

## Available Experiment Plans

### Quick Testing
- **`quick_test`**: Binary classification with 10 samples for development
- **`multiclass_quick_test`**: Multiclass classification with 10 samples for development

### Prompt Strategy Analysis
- **`prompt_comparison`**: Compare all binary classification prompts on D2A dataset
- **`multiclass_prompt_comparison`**: Compare all multiclass classification prompts on D2A dataset

### Dataset Performance Analysis
- **`binary_dataset_comparison`**: Compare performance across all binary datasets
- **`multiclass_dataset_comparison`**: Compare performance across all multiclass datasets

### Model Performance Analysis
- **`model_comparison`**: Compare models on binary classification
- **`multiclass_model_comparison`**: Compare models on multiclass classification
- **`small_models_binary`**: Evaluate small models (≤4B parameters) on binary tasks
- **`small_models_multiclass`**: Evaluate small models on multiclass tasks
- **`large_models_binary`**: Evaluate large models (>4B parameters) on binary tasks
- **`large_models_multiclass`**: Evaluate large models on multiclass tasks

### Comprehensive Analysis
- **`comprehensive_evaluation`**: Full evaluation across all datasets and models

### Vulnerability-Specific Analysis
- **`vulnerability_specific_analysis`**: Comprehensive vulnerability-specific detection evaluation
- **`vulnerability_prompt_comparison`**: Compare different vulnerability-specific prompt strategies  
- **`small_models_vulnerability_specific`**: Small models evaluation on vulnerability-specific detection
- **`large_models_vulnerability_specific`**: Large models evaluation on vulnerability-specific detection

### Vulnerability-Specific Analysis
- **`vulnerability_specific_analysis`**: Comprehensive vulnerability-specific detection evaluation
- **`vulnerability_prompt_comparison`**: Compare different vulnerability-specific prompt strategies  
- **`small_models_vulnerability_specific`**: Small models evaluation on vulnerability-specific detection
- **`large_models_vulnerability_specific`**: Large models evaluation on vulnerability-specific detection

## Configuration Structure

The VulBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulBench Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulBench benchmark",
    "version": "1.0",
    "dataset": "VulBench v1.0"
  },
  "dataset_configurations": {
    "binary_d2a": {
      "dataset_path": "datasets_processed/vulbench/vulbench_binary_d2a.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: D2A vulnerability detection"
    },
    "multiclass_big_vul": {
      "dataset_path": "datasets_processed/vulbench/vulbench_multiclass_big_vul.json",
      "task_type": "multiclass_vulnerability",
      "description": "Multi-class classification: Big-Vul vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "...",
      "user_prompt": "Analyze this C/C++ code for security vulnerabilities:\n\n{code}"
    },
    "multiclass_detailed": {
      "name": "Detailed Multiclass Vulnerability Analysis", 
      "system_prompt": "...",
      "user_prompt": "Analyze and classify the vulnerability type in this C/C++ code:\n\n```c\n{code}\n```"
    }
  },
  "model_configurations": {
    "qwen3-4b": {
      "model_name": "Qwen/Qwen3-4B",
      "model_type": "QWEN",
      "max_tokens": 512,
      "temperature": 0.1
      }
    }
  }
}
```

## Task Types

### Binary Vulnerability Detection
- **Purpose**: Determine if C/C++ code contains any security vulnerability
- **Output**: `VULNERABLE` or `SAFE`
- **Datasets**: `binary_d2a`, `binary_ctf`, `binary_magma`, `binary_big_vul`, `binary_devign`
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`

### Multiclass Vulnerability Classification  
- **Purpose**: Identify specific vulnerability type in C/C++ code
- **Output**: Specific vulnerability type (e.g., `Integer-Overflow`) or `SAFE`
- **Datasets**: `multiclass_d2a`, `multiclass_ctf`, `multiclass_magma`, `multiclass_big_vul`, `multiclass_devign`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`

## Key Features

### 1. Unified Configuration System
- **C/C++-Specific Prompts**: Tailored prompts for C/C++ vulnerability patterns
- **Task Separation**: Distinct prompts and experiments for binary vs. multiclass tasks
- **JSON Configuration**: All experiments defined in `vulbench_experiments.json`
- **Fixed Label Format**: Corrected binary labels (0/1) and multiclass response parsing
- **Experiment Plans**: Predefined experimental setups for different analysis needs

### 2. Enhanced VulBench Support
- **Multiple Datasets**: Support for all 5 VulBench sub-datasets
- **Vulnerability Types**: Proper handling of VulBench vulnerability type names
- **Response Parsing**: Updated parser to handle both CWE and VulBench formats
- **Label Consistency**: Fixed dataset loader to provide consistent integer labels for binary tasks

### 3. Comprehensive Evaluation
- **Standard Metrics**: AUC-ROC (primary), Accuracy, Precision, Recall, F1-score
- **Per-Dataset Analysis**: Individual performance analysis for each VulBench sub-dataset
- **Framework Integration**: Uses unified benchmark framework evaluation system
- **Error Analysis**: Detailed analysis of misclassifications

## Quick Start

### 1. Data Processing

First, process the raw VulBench data to create structured JSON datasets:

```bash
# Process VulBench data for all datasets
python src/scripts/process_vulbench_data.py

# Process specific dataset
python src/scripts/process_vulbench_data.py --dataset d2a

# Process both binary and multiclass variants
python src/scripts/process_vulbench_data.py --dataset big_vul --binary --multiclass
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vulbench --list-configs

# Using the direct VulBench runner
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs
```

### 3. Run Specific Experiments

```bash
# Run binary vulnerability detection on D2A dataset with GPT-4
python run_unified_benchmark.py vulbench \
  --plan quick_test \
  --model gpt-4-turbo \
  --dataset binary_d2a \
  --prompt vulnerability_detection_basic

# Run multiclass classification on Big-Vul with multiple models  
python run_unified_benchmark.py vulbench \
  --plan comprehensive_evaluation \
  --model qwen2.5-7b,claude-3.5-sonnet \
  --dataset multiclass_big_vul \
  --prompt vulnerability_classification_cwe
```

### 4. Run Full Experiment Plans

```bash
# Quick test across multiple datasets
python run_unified_benchmark.py vulbench --plan quick_test

# Comprehensive evaluation with all models and prompts
python run_unified_benchmark.py vulbench --plan comprehensive_evaluation

# Binary-only evaluation
python run_unified_benchmark.py vulbench --plan binary_classification_focus
```

## Available Experiment Plans

The configuration includes several pre-defined experiment plans:

### `quick_test`
- **Purpose**: Fast validation of setup
- **Datasets**: Binary D2A, Binary Big-Vul
- **Models**: GPT-4 Turbo, Qwen2.5-7B
- **Prompts**: Basic vulnerability detection
- **Runtime**: ~30 minutes

### `comprehensive_evaluation`
- **Purpose**: Full benchmark evaluation
- **Datasets**: All 10 datasets (5 binary + 5 multiclass)
- **Models**: All 11 configured models
- **Prompts**: All 5 prompt strategies
- **Runtime**: Several hours

### `binary_classification_focus`
- **Purpose**: Focus on binary vulnerability detection
- **Datasets**: All 5 binary datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5, CodeLlama
- **Prompts**: Basic and contextual detection
- **Runtime**: ~2 hours

### `multiclass_analysis`
- **Purpose**: Focus on vulnerability type classification
- **Datasets**: All 5 multiclass datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5
- **Prompts**: CWE-based and detailed classification
- **Runtime**: ~3 hours

### `model_comparison`
- **Purpose**: Compare different model families
- **Datasets**: Binary and multiclass D2A, Big-Vul
- **Models**: Representative models from each family
- **Prompts**: Standardized prompts for fair comparison
- **Runtime**: ~1.5 hours

### Vulnerability-Specific Analysis
- **`vulnerability_specific_analysis`**: Comprehensive vulnerability-specific detection evaluation
- **`vulnerability_prompt_comparison`**: Compare different vulnerability-specific prompt strategies  
- **`small_models_vulnerability_specific`**: Small models evaluation on vulnerability-specific detection
- **`large_models_vulnerability_specific`**: Large models evaluation on vulnerability-specific detection

## Configuration Structure

The VulBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulBench Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulBench benchmark",
    "version": "1.0",
    "dataset": "VulBench v1.0"
  },
  "dataset_configurations": {
    "binary_d2a": {
      "dataset_path": "datasets_processed/vulbench/vulbench_binary_d2a.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: D2A vulnerability detection"
    },
    "multiclass_big_vul": {
      "dataset_path": "datasets_processed/vulbench/vulbench_multiclass_big_vul.json",
      "task_type": "multiclass_vulnerability",
      "description": "Multi-class classification: Big-Vul vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "...",
      "user_prompt": "Analyze this C/C++ code for security vulnerabilities:\n\n{code}"
    },
    "multiclass_detailed": {
      "name": "Detailed Multiclass Vulnerability Analysis", 
      "system_prompt": "...",
      "user_prompt": "Analyze and classify the vulnerability type in this C/C++ code:\n\n```c\n{code}\n```"
    }
  },
  "model_configurations": {
    "qwen3-4b": {
      "model_name": "Qwen/Qwen3-4B",
      "model_type": "QWEN",
      "max_tokens": 512,
      "temperature": 0.1
      }
    }
  }
}
```

## Task Types

### Binary Vulnerability Detection
- **Purpose**: Determine if C/C++ code contains any security vulnerability
- **Output**: `VULNERABLE` or `SAFE`
- **Datasets**: `binary_d2a`, `binary_ctf`, `binary_magma`, `binary_big_vul`, `binary_devign`
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`

### Multiclass Vulnerability Classification  
- **Purpose**: Identify specific vulnerability type in C/C++ code
- **Output**: Specific vulnerability type (e.g., `Integer-Overflow`) or `SAFE`
- **Datasets**: `multiclass_d2a`, `multiclass_ctf`, `multiclass_magma`, `multiclass_big_vul`, `multiclass_devign`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`

## Key Features

### 1. Unified Configuration System
- **C/C++-Specific Prompts**: Tailored prompts for C/C++ vulnerability patterns
- **Task Separation**: Distinct prompts and experiments for binary vs. multiclass tasks
- **JSON Configuration**: All experiments defined in `vulbench_experiments.json`
- **Fixed Label Format**: Corrected binary labels (0/1) and multiclass response parsing
- **Experiment Plans**: Predefined experimental setups for different analysis needs

### 2. Enhanced VulBench Support
- **Multiple Datasets**: Support for all 5 VulBench sub-datasets
- **Vulnerability Types**: Proper handling of VulBench vulnerability type names
- **Response Parsing**: Updated parser to handle both CWE and VulBench formats
- **Label Consistency**: Fixed dataset loader to provide consistent integer labels for binary tasks

### 3. Comprehensive Evaluation
- **Standard Metrics**: AUC-ROC (primary), Accuracy, Precision, Recall, F1-score
- **Per-Dataset Analysis**: Individual performance analysis for each VulBench sub-dataset
- **Framework Integration**: Uses unified benchmark framework evaluation system
- **Error Analysis**: Detailed analysis of misclassifications

## Quick Start

### 1. Data Processing

First, process the raw VulBench data to create structured JSON datasets:

```bash
# Process VulBench data for all datasets
python src/scripts/process_vulbench_data.py

# Process specific dataset
python src/scripts/process_vulbench_data.py --dataset d2a

# Process both binary and multiclass variants
python src/scripts/process_vulbench_data.py --dataset big_vul --binary --multiclass
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vulbench --list-configs

# Using the direct VulBench runner
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs
```

### 3. Run Specific Experiments

```bash
# Run binary vulnerability detection on D2A dataset with GPT-4
python run_unified_benchmark.py vulbench \
  --plan quick_test \
  --model gpt-4-turbo \
  --dataset binary_d2a \
  --prompt vulnerability_detection_basic

# Run multiclass classification on Big-Vul with multiple models  
python run_unified_benchmark.py vulbench \
  --plan comprehensive_evaluation \
  --model qwen2.5-7b,claude-3.5-sonnet \
  --dataset multiclass_big_vul \
  --prompt vulnerability_classification_cwe
```

### 4. Run Full Experiment Plans

```bash
# Quick test across multiple datasets
python run_unified_benchmark.py vulbench --plan quick_test

# Comprehensive evaluation with all models and prompts
python run_unified_benchmark.py vulbench --plan comprehensive_evaluation

# Binary-only evaluation
python run_unified_benchmark.py vulbench --plan binary_classification_focus
```

## Available Experiment Plans

The configuration includes several pre-defined experiment plans:

### `quick_test`
- **Purpose**: Fast validation of setup
- **Datasets**: Binary D2A, Binary Big-Vul
- **Models**: GPT-4 Turbo, Qwen2.5-7B
- **Prompts**: Basic vulnerability detection
- **Runtime**: ~30 minutes

### `comprehensive_evaluation`
- **Purpose**: Full benchmark evaluation
- **Datasets**: All 10 datasets (5 binary + 5 multiclass)
- **Models**: All 11 configured models
- **Prompts**: All 5 prompt strategies
- **Runtime**: Several hours

### `binary_classification_focus`
- **Purpose**: Focus on binary vulnerability detection
- **Datasets**: All 5 binary datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5, CodeLlama
- **Prompts**: Basic and contextual detection
- **Runtime**: ~2 hours

### `multiclass_analysis`
- **Purpose**: Focus on vulnerability type classification
- **Datasets**: All 5 multiclass datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5
- **Prompts**: CWE-based and detailed classification
- **Runtime**: ~3 hours

### `model_comparison`
- **Purpose**: Compare different model families
- **Datasets**: Binary and multiclass D2A, Big-Vul
- **Models**: Representative models from each family
- **Prompts**: Standardized prompts for fair comparison
- **Runtime**: ~1.5 hours

### Vulnerability-Specific Analysis
- **`vulnerability_specific_analysis`**: Comprehensive vulnerability-specific detection evaluation
- **`vulnerability_prompt_comparison`**: Compare different vulnerability-specific prompt strategies  
- **`small_models_vulnerability_specific`**: Small models evaluation on vulnerability-specific detection
- **`large_models_vulnerability_specific`**: Large models evaluation on vulnerability-specific detection

## Configuration Structure

The VulBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulBench Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulBench benchmark",
    "version": "1.0",
    "dataset": "VulBench v1.0"
  },
  "dataset_configurations": {
    "binary_d2a": {
      "dataset_path": "datasets_processed/vulbench/vulbench_binary_d2a.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: D2A vulnerability detection"
    },
    "multiclass_big_vul": {
      "dataset_path": "datasets_processed/vulbench/vulbench_multiclass_big_vul.json",
      "task_type": "multiclass_vulnerability",
      "description": "Multi-class classification: Big-Vul vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "...",
      "user_prompt": "Analyze this C/C++ code for security vulnerabilities:\n\n{code}"
    },
    "multiclass_detailed": {
      "name": "Detailed Multiclass Vulnerability Analysis", 
      "system_prompt": "...",
      "user_prompt": "Analyze and classify the vulnerability type in this C/C++ code:\n\n```c\n{code}\n```"
    }
  },
  "model_configurations": {
    "qwen3-4b": {
      "model_name": "Qwen/Qwen3-4B",
      "model_type": "QWEN",
      "max_tokens": 512,
      "temperature": 0.1
      }
    }
  }
}
```

## Task Types

### Binary Vulnerability Detection
- **Purpose**: Determine if C/C++ code contains any security vulnerability
- **Output**: `VULNERABLE` or `SAFE`
- **Datasets**: `binary_d2a`, `binary_ctf`, `binary_magma`, `binary_big_vul`, `binary_devign`
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`

### Multiclass Vulnerability Classification  
- **Purpose**: Identify specific vulnerability type in C/C++ code
- **Output**: Specific vulnerability type (e.g., `Integer-Overflow`) or `SAFE`
- **Datasets**: `multiclass_d2a`, `multiclass_ctf`, `multiclass_magma`, `multiclass_big_vul`, `multiclass_devign`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`

## Key Features

### 1. Unified Configuration System
- **C/C++-Specific Prompts**: Tailored prompts for C/C++ vulnerability patterns
- **Task Separation**: Distinct prompts and experiments for binary vs. multiclass tasks
- **JSON Configuration**: All experiments defined in `vulbench_experiments.json`
- **Fixed Label Format**: Corrected binary labels (0/1) and multiclass response parsing
- **Experiment Plans**: Predefined experimental setups for different analysis needs

### 2. Enhanced VulBench Support
- **Multiple Datasets**: Support for all 5 VulBench sub-datasets
- **Vulnerability Types**: Proper handling of VulBench vulnerability type names
- **Response Parsing**: Updated parser to handle both CWE and VulBench formats
- **Label Consistency**: Fixed dataset loader to provide consistent integer labels for binary tasks

### 3. Comprehensive Evaluation
- **Standard Metrics**: AUC-ROC (primary), Accuracy, Precision, Recall, F1-score
- **Per-Dataset Analysis**: Individual performance analysis for each VulBench sub-dataset
- **Framework Integration**: Uses unified benchmark framework evaluation system
- **Error Analysis**: Detailed analysis of misclassifications

## Quick Start

### 1. Data Processing

First, process the raw VulBench data to create structured JSON datasets:

```bash
# Process VulBench data for all datasets
python src/scripts/process_vulbench_data.py

# Process specific dataset
python src/scripts/process_vulbench_data.py --dataset d2a

# Process both binary and multiclass variants
python src/scripts/process_vulbench_data.py --dataset big_vul --binary --multiclass
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vulbench --list-configs

# Using the direct VulBench runner
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs
```

### 3. Run Specific Experiments

```bash
# Run binary vulnerability detection on D2A dataset with GPT-4
python run_unified_benchmark.py vulbench \
  --plan quick_test \
  --model gpt-4-turbo \
  --dataset binary_d2a \
  --prompt vulnerability_detection_basic

# Run multiclass classification on Big-Vul with multiple models  
python run_unified_benchmark.py vulbench \
  --plan comprehensive_evaluation \
  --model qwen2.5-7b,claude-3.5-sonnet \
  --dataset multiclass_big_vul \
  --prompt vulnerability_classification_cwe
```

### 4. Run Full Experiment Plans

```bash
# Quick test across multiple datasets
python run_unified_benchmark.py vulbench --plan quick_test

# Comprehensive evaluation with all models and prompts
python run_unified_benchmark.py vulbench --plan comprehensive_evaluation

# Binary-only evaluation
python run_unified_benchmark.py vulbench --plan binary_classification_focus
```

## Available Experiment Plans

The configuration includes several pre-defined experiment plans:

### `quick_test`
- **Purpose**: Fast validation of setup
- **Datasets**: Binary D2A, Binary Big-Vul
- **Models**: GPT-4 Turbo, Qwen2.5-7B
- **Prompts**: Basic vulnerability detection
- **Runtime**: ~30 minutes

### `comprehensive_evaluation`
- **Purpose**: Full benchmark evaluation
- **Datasets**: All 10 datasets (5 binary + 5 multiclass)
- **Models**: All 11 configured models
- **Prompts**: All 5 prompt strategies
- **Runtime**: Several hours

### `binary_classification_focus`
- **Purpose**: Focus on binary vulnerability detection
- **Datasets**: All 5 binary datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5, CodeLlama
- **Prompts**: Basic and contextual detection
- **Runtime**: ~2 hours

### `multiclass_analysis`
- **Purpose**: Focus on vulnerability type classification
- **Datasets**: All 5 multiclass datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5
- **Prompts**: CWE-based and detailed classification
- **Runtime**: ~3 hours

### `model_comparison`
- **Purpose**: Compare different model families
- **Datasets**: Binary and multiclass D2A, Big-Vul
- **Models**: Representative models from each family
- **Prompts**: Standardized prompts for fair comparison
- **Runtime**: ~1.5 hours

### Vulnerability-Specific Analysis
- **`vulnerability_specific_analysis`**: Comprehensive vulnerability-specific detection evaluation
- **`vulnerability_prompt_comparison`**: Compare different vulnerability-specific prompt strategies  
- **`small_models_vulnerability_specific`**: Small models evaluation on vulnerability-specific detection
- **`large_models_vulnerability_specific`**: Large models evaluation on vulnerability-specific detection

## Configuration Structure

The VulBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulBench Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulBench benchmark",
    "version": "1.0",
    "dataset": "VulBench v1.0"
  },
  "dataset_configurations": {
    "binary_d2a": {
      "dataset_path": "datasets_processed/vulbench/vulbench_binary_d2a.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: D2A vulnerability detection"
    },
    "multiclass_big_vul": {
      "dataset_path": "datasets_processed/vulbench/vulbench_multiclass_big_vul.json",
      "task_type": "multiclass_vulnerability",
      "description": "Multi-class classification: Big-Vul vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "...",
      "user_prompt": "Analyze this C/C++ code for security vulnerabilities:\n\n{code}"
    },
    "multiclass_detailed": {
      "name": "Detailed Multiclass Vulnerability Analysis", 
      "system_prompt": "...",
      "user_prompt": "Analyze and classify the vulnerability type in this C/C++ code:\n\n```c\n{code}\n```"
    }
  },
  "model_configurations": {
    "qwen3-4b": {
      "model_name": "Qwen/Qwen3-4B",
      "model_type": "QWEN",
      "max_tokens": 512,
      "temperature": 0.1
      }
    }
  }
}
```

## Task Types

### Binary Vulnerability Detection
- **Purpose**: Determine if C/C++ code contains any security vulnerability
- **Output**: `VULNERABLE` or `SAFE`
- **Datasets**: `binary_d2a`, `binary_ctf`, `binary_magma`, `binary_big_vul`, `binary_devign`
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`

### Multiclass Vulnerability Classification  
- **Purpose**: Identify specific vulnerability type in C/C++ code
- **Output**: Specific vulnerability type (e.g., `Integer-Overflow`) or `SAFE`
- **Datasets**: `multiclass_d2a`, `multiclass_ctf`, `multiclass_magma`, `multiclass_big_vul`, `multiclass_devign`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`

## Key Features

### 1. Unified Configuration System
- **C/C++-Specific Prompts**: Tailored prompts for C/C++ vulnerability patterns
- **Task Separation**: Distinct prompts and experiments for binary vs. multiclass tasks
- **JSON Configuration**: All experiments defined in `vulbench_experiments.json`
- **Fixed Label Format**: Corrected binary labels (0/1) and multiclass response parsing
- **Experiment Plans**: Predefined experimental setups for different analysis needs

### 2. Enhanced VulBench Support
- **Multiple Datasets**: Support for all 5 VulBench sub-datasets
- **Vulnerability Types**: Proper handling of VulBench vulnerability type names
- **Response Parsing**: Updated parser to handle both CWE and VulBench formats
- **Label Consistency**: Fixed dataset loader to provide consistent integer labels for binary tasks

### 3. Comprehensive Evaluation
- **Standard Metrics**: AUC-ROC (primary), Accuracy, Precision, Recall, F1-score
- **Per-Dataset Analysis**: Individual performance analysis for each VulBench sub-dataset
- **Framework Integration**: Uses unified benchmark framework evaluation system
- **Error Analysis**: Detailed analysis of misclassifications

## Quick Start

### 1. Data Processing

First, process the raw VulBench data to create structured JSON datasets:

```bash
# Process VulBench data for all datasets
python src/scripts/process_vulbench_data.py

# Process specific dataset
python src/scripts/process_vulbench_data.py --dataset d2a

# Process both binary and multiclass variants
python src/scripts/process_vulbench_data.py --dataset big_vul --binary --multiclass
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vulbench --list-configs

# Using the direct VulBench runner
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs
```

### 3. Run Specific Experiments

```bash
# Run binary vulnerability detection on D2A dataset with GPT-4
python run_unified_benchmark.py vulbench \
  --plan quick_test \
  --model gpt-4-turbo \
  --dataset binary_d2a \
  --prompt vulnerability_detection_basic

# Run multiclass classification on Big-Vul with multiple models  
python run_unified_benchmark.py vulbench \
  --plan comprehensive_evaluation \
  --model qwen2.5-7b,claude-3.5-sonnet \
  --dataset multiclass_big_vul \
  --prompt vulnerability_classification_cwe
```

### 4. Run Full Experiment Plans

```bash
# Quick test across multiple datasets
python run_unified_benchmark.py vulbench --plan quick_test

# Comprehensive evaluation with all models and prompts
python run_unified_benchmark.py vulbench --plan comprehensive_evaluation

# Binary-only evaluation
python run_unified_benchmark.py vulbench --plan binary_classification_focus
```

## Available Experiment Plans

The configuration includes several pre-defined experiment plans:

### `quick_test`
- **Purpose**: Fast validation of setup
- **Datasets**: Binary D2A, Binary Big-Vul
- **Models**: GPT-4 Turbo, Qwen2.5-7B
- **Prompts**: Basic vulnerability detection
- **Runtime**: ~30 minutes

### `comprehensive_evaluation`
- **Purpose**: Full benchmark evaluation
- **Datasets**: All 10 datasets (5 binary + 5 multiclass)
- **Models**: All 11 configured models
- **Prompts**: All 5 prompt strategies
- **Runtime**: Several hours

### `binary_classification_focus`
- **Purpose**: Focus on binary vulnerability detection
- **Datasets**: All 5 binary datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5, CodeLlama
- **Prompts**: Basic and contextual detection
- **Runtime**: ~2 hours

### `multiclass_analysis`
- **Purpose**: Focus on vulnerability type classification
- **Datasets**: All 5 multiclass datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5
- **Prompts**: CWE-based and detailed classification
- **Runtime**: ~3 hours

### `model_comparison`
- **Purpose**: Compare different model families
- **Datasets**: Binary and multiclass D2A, Big-Vul
- **Models**: Representative models from each family
- **Prompts**: Standardized prompts for fair comparison
- **Runtime**: ~1.5 hours

### Vulnerability-Specific Analysis
- **`vulnerability_specific_analysis`**: Comprehensive vulnerability-specific detection evaluation
- **`vulnerability_prompt_comparison`**: Compare different vulnerability-specific prompt strategies  
- **`small_models_vulnerability_specific`**: Small models evaluation on vulnerability-specific detection
- **`large_models_vulnerability_specific`**: Large models evaluation on vulnerability-specific detection

## Configuration Structure

The VulBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulBench Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulBench benchmark",
    "version": "1.0",
    "dataset": "VulBench v1.0"
  },
  "dataset_configurations": {
    "binary_d2a": {
      "dataset_path": "datasets_processed/vulbench/vulbench_binary_d2a.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: D2A vulnerability detection"
    },
    "multiclass_big_vul": {
      "dataset_path": "datasets_processed/vulbench/vulbench_multiclass_big_vul.json",
      "task_type": "multiclass_vulnerability",
      "description": "Multi-class classification: Big-Vul vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "...",
      "user_prompt": "Analyze this C/C++ code for security vulnerabilities:\n\n{code}"
    },
    "multiclass_detailed": {
      "name": "Detailed Multiclass Vulnerability Analysis", 
      "system_prompt": "...",
      "user_prompt": "Analyze and classify the vulnerability type in this C/C++ code:\n\n```c\n{code}\n```"
    }
  },
  "model_configurations": {
    "qwen3-4b": {
      "model_name": "Qwen/Qwen3-4B",
      "model_type": "QWEN",
      "max_tokens": 512,
      "temperature": 0.1
      }
    }
  }
}
```

## Task Types

### Binary Vulnerability Detection
- **Purpose**: Determine if C/C++ code contains any security vulnerability
- **Output**: `VULNERABLE` or `SAFE`
- **Datasets**: `binary_d2a`, `binary_ctf`, `binary_magma`, `binary_big_vul`, `binary_devign`
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`

### Multiclass Vulnerability Classification  
- **Purpose**: Identify specific vulnerability type in C/C++ code
- **Output**: Specific vulnerability type (e.g., `Integer-Overflow`) or `SAFE`
- **Datasets**: `multiclass_d2a`, `multiclass_ctf`, `multiclass_magma`, `multiclass_big_vul`, `multiclass_devign`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`

## Key Features

### 1. Unified Configuration System
- **C/C++-Specific Prompts**: Tailored prompts for C/C++ vulnerability patterns
- **Task Separation**: Distinct prompts and experiments for binary vs. multiclass tasks
- **JSON Configuration**: All experiments defined in `vulbench_experiments.json`
- **Fixed Label Format**: Corrected binary labels (0/1) and multiclass response parsing
- **Experiment Plans**: Predefined experimental setups for different analysis needs

### 2. Enhanced VulBench Support
- **Multiple Datasets**: Support for all 5 VulBench sub-datasets
- **Vulnerability Types**: Proper handling of VulBench vulnerability type names
- **Response Parsing**: Updated parser to handle both CWE and VulBench formats
- **Label Consistency**: Fixed dataset loader to provide consistent integer labels for binary tasks

### 3. Comprehensive Evaluation
- **Standard Metrics**: AUC-ROC (primary), Accuracy, Precision, Recall, F1-score
- **Per-Dataset Analysis**: Individual performance analysis for each VulBench sub-dataset
- **Framework Integration**: Uses unified benchmark framework evaluation system
- **Error Analysis**: Detailed analysis of misclassifications

## Quick Start

### 1. Data Processing

First, process the raw VulBench data to create structured JSON datasets:

```bash
# Process VulBench data for all datasets
python src/scripts/process_vulbench_data.py

# Process specific dataset
python src/scripts/process_vulbench_data.py --dataset d2a

# Process both binary and multiclass variants
python src/scripts/process_vulbench_data.py --dataset big_vul --binary --multiclass
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vulbench --list-configs

# Using the direct VulBench runner
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs
```

### 3. Run Specific Experiments

```bash
# Run binary vulnerability detection on D2A dataset with GPT-4
python run_unified_benchmark.py vulbench \
  --plan quick_test \
  --model gpt-4-turbo \
  --dataset binary_d2a \
  --prompt vulnerability_detection_basic

# Run multiclass classification on Big-Vul with multiple models  
python run_unified_benchmark.py vulbench \
  --plan comprehensive_evaluation \
  --model qwen2.5-7b,claude-3.5-sonnet \
  --dataset multiclass_big_vul \
  --prompt vulnerability_classification_cwe
```

### 4. Run Full Experiment Plans

```bash
# Quick test across multiple datasets
python run_unified_benchmark.py vulbench --plan quick_test

# Comprehensive evaluation with all models and prompts
python run_unified_benchmark.py vulbench --plan comprehensive_evaluation

# Binary-only evaluation
python run_unified_benchmark.py vulbench --plan binary_classification_focus
```

## Available Experiment Plans

The configuration includes several pre-defined experiment plans:

### `quick_test`
- **Purpose**: Fast validation of setup
- **Datasets**: Binary D2A, Binary Big-Vul
- **Models**: GPT-4 Turbo, Qwen2.5-7B
- **Prompts**: Basic vulnerability detection
- **Runtime**: ~30 minutes

### `comprehensive_evaluation`
- **Purpose**: Full benchmark evaluation
- **Datasets**: All 10 datasets (5 binary + 5 multiclass)
- **Models**: All 11 configured models
- **Prompts**: All 5 prompt strategies
- **Runtime**: Several hours

### `binary_classification_focus`
- **Purpose**: Focus on binary vulnerability detection
- **Datasets**: All 5 binary datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5, CodeLlama
- **Prompts**: Basic and contextual detection
- **Runtime**: ~2 hours

### `multiclass_analysis`
- **Purpose**: Focus on vulnerability type classification
- **Datasets**: All 5 multiclass datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5
- **Prompts**: CWE-based and detailed classification
- **Runtime**: ~3 hours

### `model_comparison`
- **Purpose**: Compare different model families
- **Datasets**: Binary and multiclass D2A, Big-Vul
- **Models**: Representative models from each family
- **Prompts**: Standardized prompts for fair comparison
- **Runtime**: ~1.5 hours

### Vulnerability-Specific Analysis
- **`vulnerability_specific_analysis`**: Comprehensive vulnerability-specific detection evaluation
- **`vulnerability_prompt_comparison`**: Compare different vulnerability-specific prompt strategies  
- **`small_models_vulnerability_specific`**: Small models evaluation on vulnerability-specific detection
- **`large_models_vulnerability_specific`**: Large models evaluation on vulnerability-specific detection

## Configuration Structure

The VulBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulBench Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulBench benchmark",
    "version": "1.0",
    "dataset": "VulBench v1.0"
  },
  "dataset_configurations": {
    "binary_d2a": {
      "dataset_path": "datasets_processed/vulbench/vulbench_binary_d2a.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: D2A vulnerability detection"
    },
    "multiclass_big_vul": {
      "dataset_path": "datasets_processed/vulbench/vulbench_multiclass_big_vul.json",
      "task_type": "multiclass_vulnerability",
      "description": "Multi-class classification: Big-Vul vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "...",
      "user_prompt": "Analyze this C/C++ code for security vulnerabilities:\n\n{code}"
    },
    "multiclass_detailed": {
      "name": "Detailed Multiclass Vulnerability Analysis", 
      "system_prompt": "...",
      "user_prompt": "Analyze and classify the vulnerability type in this C/C++ code:\n\n```c\n{code}\n```"
    }
  },
  "model_configurations": {
    "qwen3-4b": {
      "model_name": "Qwen/Qwen3-4B",
      "model_type": "QWEN",
      "max_tokens": 512,
      "temperature": 0.1
      }
    }
  }
}
```

## Task Types

### Binary Vulnerability Detection
- **Purpose**: Determine if C/C++ code contains any security vulnerability
- **Output**: `VULNERABLE` or `SAFE`
- **Datasets**: `binary_d2a`, `binary_ctf`, `binary_magma`, `binary_big_vul`, `binary_devign`
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`

### Multiclass Vulnerability Classification  
- **Purpose**: Identify specific vulnerability type in C/C++ code
- **Output**: Specific vulnerability type (e.g., `Integer-Overflow`) or `SAFE`
- **Datasets**: `multiclass_d2a`, `multiclass_ctf`, `multiclass_magma`, `multiclass_big_vul`, `multiclass_devign`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`

## Key Features

### 1. Unified Configuration System
- **C/C++-Specific Prompts**: Tailored prompts for C/C++ vulnerability patterns
- **Task Separation**: Distinct prompts and experiments for binary vs. multiclass tasks
- **JSON Configuration**: All experiments defined in `vulbench_experiments.json`
- **Fixed Label Format**: Corrected binary labels (0/1) and multiclass response parsing
- **Experiment Plans**: Predefined experimental setups for different analysis needs

### 2. Enhanced VulBench Support
- **Multiple Datasets**: Support for all 5 VulBench sub-datasets
- **Vulnerability Types**: Proper handling of VulBench vulnerability type names
- **Response Parsing**: Updated parser to handle both CWE and VulBench formats
- **Label Consistency**: Fixed dataset loader to provide consistent integer labels for binary tasks

### 3. Comprehensive Evaluation
- **Standard Metrics**: AUC-ROC (primary), Accuracy, Precision, Recall, F1-score
- **Per-Dataset Analysis**: Individual performance analysis for each VulBench sub-dataset
- **Framework Integration**: Uses unified benchmark framework evaluation system
- **Error Analysis**: Detailed analysis of misclassifications

## Quick Start

### 1. Data Processing

First, process the raw VulBench data to create structured JSON datasets:

```bash
# Process VulBench data for all datasets
python src/scripts/process_vulbench_data.py

# Process specific dataset
python src/scripts/process_vulbench_data.py --dataset d2a

# Process both binary and multiclass variants
python src/scripts/process_vulbench_data.py --dataset big_vul --binary --multiclass
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vulbench --list-configs

# Using the direct VulBench runner
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs
```

### 3. Run Specific Experiments

```bash
# Run binary vulnerability detection on D2A dataset with GPT-4
python run_unified_benchmark.py vulbench \
  --plan quick_test \
  --model gpt-4-turbo \
  --dataset binary_d2a \
  --prompt vulnerability_detection_basic

# Run multiclass classification on Big-Vul with multiple models  
python run_unified_benchmark.py vulbench \
  --plan comprehensive_evaluation \
  --model qwen2.5-7b,claude-3.5-sonnet \
  --dataset multiclass_big_vul \
  --prompt vulnerability_classification_cwe
```

### 4. Run Full Experiment Plans

```bash
# Quick test across multiple datasets
python run_unified_benchmark.py vulbench --plan quick_test

# Comprehensive evaluation with all models and prompts
python run_unified_benchmark.py vulbench --plan comprehensive_evaluation

# Binary-only evaluation
python run_unified_benchmark.py vulbench --plan binary_classification_focus
```

## Available Experiment Plans

The configuration includes several pre-defined experiment plans:

### `quick_test`
- **Purpose**: Fast validation of setup
- **Datasets**: Binary D2A, Binary Big-Vul
- **Models**: GPT-4 Turbo, Qwen2.5-7B
- **Prompts**: Basic vulnerability detection
- **Runtime**: ~30 minutes

### `comprehensive_evaluation`
- **Purpose**: Full benchmark evaluation
- **Datasets**: All 10 datasets (5 binary + 5 multiclass)
- **Models**: All 11 configured models
- **Prompts**: All 5 prompt strategies
- **Runtime**: Several hours

### `binary_classification_focus`
- **Purpose**: Focus on binary vulnerability detection
- **Datasets**: All 5 binary datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5, CodeLlama
- **Prompts**: Basic and contextual detection
- **Runtime**: ~2 hours

### `multiclass_analysis`
- **Purpose**: Focus on vulnerability type classification
- **Datasets**: All 5 multiclass datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5
- **Prompts**: CWE-based and detailed classification
- **Runtime**: ~3 hours

### `model_comparison`
- **Purpose**: Compare different model families
- **Datasets**: Binary and multiclass D2A, Big-Vul
- **Models**: Representative models from each family
- **Prompts**: Standardized prompts for fair comparison
- **Runtime**: ~1.5 hours

### Vulnerability-Specific Analysis
- **`vulnerability_specific_analysis`**: Comprehensive vulnerability-specific detection evaluation
- **`vulnerability_prompt_comparison`**: Compare different vulnerability-specific prompt strategies  
- **`small_models_vulnerability_specific`**: Small models evaluation on vulnerability-specific detection
- **`large_models_vulnerability_specific`**: Large models evaluation on vulnerability-specific detection

## Configuration Structure

The VulBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulBench Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulBench benchmark",
    "version": "1.0",
    "dataset": "VulBench v1.0"
  },
  "dataset_configurations": {
    "binary_d2a": {
      "dataset_path": "datasets_processed/vulbench/vulbench_binary_d2a.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: D2A vulnerability detection"
    },
    "multiclass_big_vul": {
      "dataset_path": "datasets_processed/vulbench/vulbench_multiclass_big_vul.json",
      "task_type": "multiclass_vulnerability",
      "description": "Multi-class classification: Big-Vul vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "...",
      "user_prompt": "Analyze this C/C++ code for security vulnerabilities:\n\n{code}"
    },
    "multiclass_detailed": {
      "name": "Detailed Multiclass Vulnerability Analysis", 
      "system_prompt": "...",
      "user_prompt": "Analyze and classify the vulnerability type in this C/C++ code:\n\n```c\n{code}\n```"
    }
  },
  "model_configurations": {
    "qwen3-4b": {
      "model_name": "Qwen/Qwen3-4B",
      "model_type": "QWEN",
      "max_tokens": 512,
      "temperature": 0.1
      }
    }
  }
}
```

## Task Types

### Binary Vulnerability Detection
- **Purpose**: Determine if C/C++ code contains any security vulnerability
- **Output**: `VULNERABLE` or `SAFE`
- **Datasets**: `binary_d2a`, `binary_ctf`, `binary_magma`, `binary_big_vul`, `binary_devign`
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`

### Multiclass Vulnerability Classification  
- **Purpose**: Identify specific vulnerability type in C/C++ code
- **Output**: Specific vulnerability type (e.g., `Integer-Overflow`) or `SAFE`
- **Datasets**: `multiclass_d2a`, `multiclass_ctf`, `multiclass_magma`, `multiclass_big_vul`, `multiclass_devign`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`

## Key Features

### 1. Unified Configuration System
- **C/C++-Specific Prompts**: Tailored prompts for C/C++ vulnerability patterns
- **Task Separation**: Distinct prompts and experiments for binary vs. multiclass tasks
- **JSON Configuration**: All experiments defined in `vulbench_experiments.json`
- **Fixed Label Format**: Corrected binary labels (0/1) and multiclass response parsing
- **Experiment Plans**: Predefined experimental setups for different analysis needs

### 2. Enhanced VulBench Support
- **Multiple Datasets**: Support for all 5 VulBench sub-datasets
- **Vulnerability Types**: Proper handling of VulBench vulnerability type names
- **Response Parsing**: Updated parser to handle both CWE and VulBench formats
- **Label Consistency**: Fixed dataset loader to provide consistent integer labels for binary tasks

### 3. Comprehensive Evaluation
- **Standard Metrics**: AUC-ROC (primary), Accuracy, Precision, Recall, F1-score
- **Per-Dataset Analysis**: Individual performance analysis for each VulBench sub-dataset
- **Framework Integration**: Uses unified benchmark framework evaluation system
- **Error Analysis**: Detailed analysis of misclassifications

## Quick Start

### 1. Data Processing

First, process the raw VulBench data to create structured JSON datasets:

```bash
# Process VulBench data for all datasets
python src/scripts/process_vulbench_data.py

# Process specific dataset
python src/scripts/process_vulbench_data.py --dataset d2a

# Process both binary and multiclass variants
python src/scripts/process_vulbench_data.py --dataset big_vul --binary --multiclass
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vulbench --list-configs

# Using the direct VulBench runner
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs
```

### 3. Run Specific Experiments

```bash
# Run binary vulnerability detection on D2A dataset with GPT-4
python run_unified_benchmark.py vulbench \
  --plan quick_test \
  --model gpt-4-turbo \
  --dataset binary_d2a \
  --prompt vulnerability_detection_basic

# Run multiclass classification on Big-Vul with multiple models  
python run_unified_benchmark.py vulbench \
  --plan comprehensive_evaluation \
  --model qwen2.5-7b,claude-3.5-sonnet \
  --dataset multiclass_big_vul \
  --prompt vulnerability_classification_cwe
```

### 4. Run Full Experiment Plans

```bash
# Quick test across multiple datasets
python run_unified_benchmark.py vulbench --plan quick_test

# Comprehensive evaluation with all models and prompts
python run_unified_benchmark.py vulbench --plan comprehensive_evaluation

# Binary-only evaluation
python run_unified_benchmark.py vulbench --plan binary_classification_focus
```

## Available Experiment Plans

The configuration includes several pre-defined experiment plans:

### `quick_test`
- **Purpose**: Fast validation of setup
- **Datasets**: Binary D2A, Binary Big-Vul
- **Models**: GPT-4 Turbo, Qwen2.5-7B
- **Prompts**: Basic vulnerability detection
- **Runtime**: ~30 minutes

### `comprehensive_evaluation`
- **Purpose**: Full benchmark evaluation
- **Datasets**: All 10 datasets (5 binary + 5 multiclass)
- **Models**: All 11 configured models
- **Prompts**: All 5 prompt strategies
- **Runtime**: Several hours

### `binary_classification_focus`
- **Purpose**: Focus on binary vulnerability detection
- **Datasets**: All 5 binary datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5, CodeLlama
- **Prompts**: Basic and contextual detection
- **Runtime**: ~2 hours

### `multiclass_analysis`
- **Purpose**: Focus on vulnerability type classification
- **Datasets**: All 5 multiclass datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5
- **Prompts**: CWE-based and detailed classification
- **Runtime**: ~3 hours

### `model_comparison`
- **Purpose**: Compare different model families
- **Datasets**: Binary and multiclass D2A, Big-Vul
- **Models**: Representative models from each family
- **Prompts**: Standardized prompts for fair comparison
- **Runtime**: ~1.5 hours

### Vulnerability-Specific Analysis
- **`vulnerability_specific_analysis`**: Comprehensive vulnerability-specific detection evaluation
- **`vulnerability_prompt_comparison`**: Compare different vulnerability-specific prompt strategies  
- **`small_models_vulnerability_specific`**: Small models evaluation on vulnerability-specific detection
- **`large_models_vulnerability_specific`**: Large models evaluation on vulnerability-specific detection

## Configuration Structure

The VulBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulBench Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulBench benchmark",
    "version": "1.0",
    "dataset": "VulBench v1.0"
  },
  "dataset_configurations": {
    "binary_d2a": {
      "dataset_path": "datasets_processed/vulbench/vulbench_binary_d2a.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: D2A vulnerability detection"
    },
    "multiclass_big_vul": {
      "dataset_path": "datasets_processed/vulbench/vulbench_multiclass_big_vul.json",
      "task_type": "multiclass_vulnerability",
      "description": "Multi-class classification: Big-Vul vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "...",
      "user_prompt": "Analyze this C/C++ code for security vulnerabilities:\n\n{code}"
    },
    "multiclass_detailed": {
      "name": "Detailed Multiclass Vulnerability Analysis", 
      "system_prompt": "...",
      "user_prompt": "Analyze and classify the vulnerability type in this C/C++ code:\n\n```c\n{code}\n```"
    }
  },
  "model_configurations": {
    "qwen3-4b": {
      "model_name": "Qwen/Qwen3-4B",
      "model_type": "QWEN",
      "max_tokens": 512,
      "temperature": 0.1
      }
    }
  }
}
```

## Task Types

### Binary Vulnerability Detection
- **Purpose**: Determine if C/C++ code contains any security vulnerability
- **Output**: `VULNERABLE` or `SAFE`
- **Datasets**: `binary_d2a`, `binary_ctf`, `binary_magma`, `binary_big_vul`, `binary_devign`
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`

### Multiclass Vulnerability Classification  
- **Purpose**: Identify specific vulnerability type in C/C++ code
- **Output**: Specific vulnerability type (e.g., `Integer-Overflow`) or `SAFE`
- **Datasets**: `multiclass_d2a`, `multiclass_ctf`, `multiclass_magma`, `multiclass_big_vul`, `multiclass_devign`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`

## Key Features

### 1. Unified Configuration System
- **C/C++-Specific Prompts**: Tailored prompts for C/C++ vulnerability patterns
- **Task Separation**: Distinct prompts and experiments for binary vs. multiclass tasks
- **JSON Configuration**: All experiments defined in `vulbench_experiments.json`
- **Fixed Label Format**: Corrected binary labels (0/1) and multiclass response parsing
- **Experiment Plans**: Predefined experimental setups for different analysis needs

### 2. Enhanced VulBench Support
- **Multiple Datasets**: Support for all 5 VulBench sub-datasets
- **Vulnerability Types**: Proper handling of VulBench vulnerability type names
- **Response Parsing**: Updated parser to handle both CWE and VulBench formats
- **Label Consistency**: Fixed dataset loader to provide consistent integer labels for binary tasks

### 3. Comprehensive Evaluation
- **Standard Metrics**: AUC-ROC (primary), Accuracy, Precision, Recall, F1-score
- **Per-Dataset Analysis**: Individual performance analysis for each VulBench sub-dataset
- **Framework Integration**: Uses unified benchmark framework evaluation system
- **Error Analysis**: Detailed analysis of misclassifications

## Quick Start

### 1. Data Processing

First, process the raw VulBench data to create structured JSON datasets:

```bash
# Process VulBench data for all datasets
python src/scripts/process_vulbench_data.py

# Process specific dataset
python src/scripts/process_vulbench_data.py --dataset d2a

# Process both binary and multiclass variants
python src/scripts/process_vulbench_data.py --dataset big_vul --binary --multiclass
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vulbench --list-configs

# Using the direct VulBench runner
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs
```

### 3. Run Specific Experiments

```bash
# Run binary vulnerability detection on D2A dataset with GPT-4
python run_unified_benchmark.py vulbench \
  --plan quick_test \
  --model gpt-4-turbo \
  --dataset binary_d2a \
  --prompt vulnerability_detection_basic

# Run multiclass classification on Big-Vul with multiple models  
python run_unified_benchmark.py vulbench \
  --plan comprehensive_evaluation \
  --model qwen2.5-7b,claude-3.5-sonnet \
  --dataset multiclass_big_vul \
  --prompt vulnerability_classification_cwe
```

### 4. Run Full Experiment Plans

```bash
# Quick test across multiple datasets
python run_unified_benchmark.py vulbench --plan quick_test

# Comprehensive evaluation with all models and prompts
python run_unified_benchmark.py vulbench --plan comprehensive_evaluation

# Binary-only evaluation
python run_unified_benchmark.py vulbench --plan binary_classification_focus
```

## Available Experiment Plans

The configuration includes several pre-defined experiment plans:

### `quick_test`
- **Purpose**: Fast validation of setup
- **Datasets**: Binary D2A, Binary Big-Vul
- **Models**: GPT-4 Turbo, Qwen2.5-7B
- **Prompts**: Basic vulnerability detection
- **Runtime**: ~30 minutes

### `comprehensive_evaluation`
- **Purpose**: Full benchmark evaluation
- **Datasets**: All 10 datasets (5 binary + 5 multiclass)
- **Models**: All 11 configured models
- **Prompts**: All 5 prompt strategies
- **Runtime**: Several hours

### `binary_classification_focus`
- **Purpose**: Focus on binary vulnerability detection
- **Datasets**: All 5 binary datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5, CodeLlama
- **Prompts**: Basic and contextual detection
- **Runtime**: ~2 hours

### `multiclass_analysis`
- **Purpose**: Focus on vulnerability type classification
- **Datasets**: All 5 multiclass datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5
- **Prompts**: CWE-based and detailed classification
- **Runtime**: ~3 hours

### `model_comparison`
- **Purpose**: Compare different model families
- **Datasets**: Binary and multiclass D2A, Big-Vul
- **Models**: Representative models from each family
- **Prompts**: Standardized prompts for fair comparison
- **Runtime**: ~1.5 hours

### Vulnerability-Specific Analysis
- **`vulnerability_specific_analysis`**: Comprehensive vulnerability-specific detection evaluation
- **`vulnerability_prompt_comparison`**: Compare different vulnerability-specific prompt strategies  
- **`small_models_vulnerability_specific`**: Small models evaluation on vulnerability-specific detection
- **`large_models_vulnerability_specific`**: Large models evaluation on vulnerability-specific detection

## Configuration Structure

The VulBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulBench Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulBench benchmark",
    "version": "1.0",
    "dataset": "VulBench v1.0"
  },
  "dataset_configurations": {
    "binary_d2a": {
      "dataset_path": "datasets_processed/vulbench/vulbench_binary_d2a.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: D2A vulnerability detection"
    },
    "multiclass_big_vul": {
      "dataset_path": "datasets_processed/vulbench/vulbench_multiclass_big_vul.json",
      "task_type": "multiclass_vulnerability",
      "description": "Multi-class classification: Big-Vul vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "...",
      "user_prompt": "Analyze this C/C++ code for security vulnerabilities:\n\n{code}"
    },
    "multiclass_detailed": {
      "name": "Detailed Multiclass Vulnerability Analysis", 
      "system_prompt": "...",
      "user_prompt": "Analyze and classify the vulnerability type in this C/C++ code:\n\n```c\n{code}\n```"
    }
  },
  "model_configurations": {
    "qwen3-4b": {
      "model_name": "Qwen/Qwen3-4B",
      "model_type": "QWEN",
      "max_tokens": 512,
      "temperature": 0.1
      }
    }
  }
}
```

## Task Types

### Binary Vulnerability Detection
- **Purpose**: Determine if C/C++ code contains any security vulnerability
- **Output**: `VULNERABLE` or `SAFE`
- **Datasets**: `binary_d2a`, `binary_ctf`, `binary_magma`, `binary_big_vul`, `binary_devign`
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`

### Multiclass Vulnerability Classification  
- **Purpose**: Identify specific vulnerability type in C/C++ code
- **Output**: Specific vulnerability type (e.g., `Integer-Overflow`) or `SAFE`
- **Datasets**: `multiclass_d2a`, `multiclass_ctf`, `multiclass_magma`, `multiclass_big_vul`, `multiclass_devign`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`

## Key Features

### 1. Unified Configuration System
- **C/C++-Specific Prompts**: Tailored prompts for C/C++ vulnerability patterns
- **Task Separation**: Distinct prompts and experiments for binary vs. multiclass tasks
- **JSON Configuration**: All experiments defined in `vulbench_experiments.json`
- **Fixed Label Format**: Corrected binary labels (0/1) and multiclass response parsing
- **Experiment Plans**: Predefined experimental setups for different analysis needs

### 2. Enhanced VulBench Support
- **Multiple Datasets**: Support for all 5 VulBench sub-datasets
- **Vulnerability Types**: Proper handling of VulBench vulnerability type names
- **Response Parsing**: Updated parser to handle both CWE and VulBench formats
- **Label Consistency**: Fixed dataset loader to provide consistent integer labels for binary tasks

### 3. Comprehensive Evaluation
- **Standard Metrics**: AUC-ROC (primary), Accuracy, Precision, Recall, F1-score
- **Per-Dataset Analysis**: Individual performance analysis for each VulBench sub-dataset
- **Framework Integration**: Uses unified benchmark framework evaluation system
- **Error Analysis**: Detailed analysis of misclassifications

## Quick Start

### 1. Data Processing

First, process the raw VulBench data to create structured JSON datasets:

```bash
# Process VulBench data for all datasets
python src/scripts/process_vulbench_data.py

# Process specific dataset
python src/scripts/process_vulbench_data.py --dataset d2a

# Process both binary and multiclass variants
python src/scripts/process_vulbench_data.py --dataset big_vul --binary --multiclass
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vulbench --list-configs

# Using the direct VulBench runner
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs
```

### 3. Run Specific Experiments

```bash
# Run binary vulnerability detection on D2A dataset with GPT-4
python run_unified_benchmark.py vulbench \
  --plan quick_test \
  --model gpt-4-turbo \
  --dataset binary_d2a \
  --prompt vulnerability_detection_basic

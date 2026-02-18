# CVEFixes Benchmark - Multi-Language Vulnerability Detection

## Overview

The CVEFixes benchmark provides comprehensive vulnerability detection evaluation using the CVEFixes dataset containing real-world CVE fixes from open-source projects. The system supports **multi-language vulnerability detection** across C, Java, and Python codebases with both binary and multiclass classification tasks.

## Pre-requisites

1. Check [instruction](https://github.com/secureIT-project/CVEfixes/blob/main/INSTALL.md) how to download a db to your machine

2. Place database file to `datasets_processed\cvefixes\CVEfixes.db` project path

## Multi-Language Support ✅

### Supported Languages

- **C/C++**: Traditional memory safety vulnerabilities
- **Java**: Object-oriented and runtime vulnerabilities  
- **Python**: Scripting and dynamic language vulnerabilities

### Dataset Coverage

- **18 total datasets** across 3 languages and 2 granularities
- **Binary classification**: Vulnerable vs. Safe detection
- **Multiclass classification**: Specific CWE type identification
- **CWE-specific detection**: Focused analysis for specific vulnerability types

## Configuration Structure

The CVEFixes configuration follows a unified structure with comprehensive multi-language support:

```json
{
  "experiment_metadata": {
    "name": "CVEFixes Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on CVEFixes benchmark with multiple prompt strategies",
    "version": "1.0",
    "dataset": "CVEFixes-Benchmark v1.0",
    "created_date": "2025-06-12"
  },
  "dataset_configurations": {
    "binary_c_file": {
      "dataset_path": "datasets_processed/cvefixes/cvefixes_binary_c_file.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: C file-level vulnerability detection"
    },
    "multiclass_java_method": {
      "dataset_path": "datasets_processed/cvefixes/cvefixes_multiclass_java_method.json",
      "task_type": "multiclass_vulnerability", 
      "description": "Multi-class classification: Java method-level CWE identification"
    },
    "cwe_119": {
      "dataset_path": "datasets_processed/cvefixes/cvefixes_cwe_119.json",
      "task_type": "binary_cwe_specific",
      "cwe_type": "CWE-119",
      "description": "CWE-119: Improper Restriction of Operations within Bounds of Memory Buffer"
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "Multi-language security analysis prompt...",
      "user_prompt": "Analyze this code for security vulnerabilities:\n\n{code}"
    },
    "multiclass_detailed": {
      "name": "Detailed Multiclass Vulnerability Analysis", 
      "system_prompt": "Language-agnostic CWE classification prompt...",
      "user_prompt": "Analyze and classify the vulnerability type in this code:\n\n```\n{code}\n```"
    }
  }
}
```

## Available Datasets

### Binary Classification (12 datasets)

**File-level detection:**

- `binary_c_file` - C file-level vulnerability detection
- `binary_java_file` - Java file-level vulnerability detection  
- `binary_python_file` - Python file-level vulnerability detection

**Method-level detection:**

- `binary_c_method` - C method-level vulnerability detection
- `binary_java_method` - Java method-level vulnerability detection
- `binary_python_method` - Python method-level vulnerability detection

**CWE-specific detection:**

- `cwe_119` - CWE-119: Improper Restriction of Operations within Bounds of Memory Buffer
- `cwe_120` - CWE-120: Buffer Copy without Checking Size of Input (Classic Buffer Overflow)
- `cwe_125` - CWE-125: Out-of-bounds Read
- `cwe_190` - CWE-190: Integer Overflow or Wraparound
- `cwe_476` - CWE-476: NULL Pointer Dereference
- `cwe_787` - CWE-787: Out-of-bounds Write

### Multiclass Classification (6 datasets)

**File-level CWE identification:**

- `multiclass_c_file` - C file-level CWE identification
- `multiclass_java_file` - Java file-level CWE identification
- `multiclass_python_file` - Python file-level CWE identification

**Method-level CWE identification:**

- `multiclass_c_method` - C method-level CWE identification
- `multiclass_java_method` - Java method-level CWE identification
- `multiclass_python_method` - Python method-level CWE identification

## Prompt Strategies

### Binary Classification Prompts

- `basic_security` - Basic vulnerability detection across languages
- `detailed_analysis` - Comprehensive security analysis with language-specific patterns
- `cwe_focused` - Focused analysis for specific CWE types
- `context_aware` - Production context-aware security review
- `step_by_step` - Methodical systematic analysis

### Multiclass Classification Prompts  

- `multiclass_basic` - Basic CWE type classification
- `multiclass_detailed` - Detailed CWE analysis with language-specific patterns
- `multiclass_comprehensive` - Comprehensive production-level CWE classification

All prompts are **language-agnostic** and work effectively across C, Java, and Python code.

## Command Line Interface

### Core Entry Points

- **Main Runner**: `src/entrypoints/run_cvefixes_benchmark.py`
- **Dataset Preparation**: `src/entrypoints/run_setup_cvefixes_datasets.py`

#### Single Experiments

```bash
# Run specific model/dataset/prompt combination
python src/entrypoints/run_cvefixes_benchmark.py \
  --model qwen3-4b \
  --dataset binary_java_file \
  --prompt detailed_analysis

# Multi-language comparison
python src/entrypoints/run_cvefixes_benchmark.py \
  --model deepseek-coder-v2-lite-16b \
  --dataset multiclass_python_method \
  --prompt multiclass_detailed
```

#### Experiment Plans

```bash
# Run predefined experiment plan
python src/entrypoints/run_cvefixes_benchmark.py --plan quick_test

# Multi-language comparison plan
python src/entrypoints/run_cvefixes_benchmark.py --plan language_comparison_binary
```

#### Common Options

```bash
# List available configurations
python src/entrypoints/run_cvefixes_benchmark.py --list-configs

# Limit samples and set output directory
python src/entrypoints/run_cvefixes_benchmark.py \
  --plan multiclass_quick_test \
  --sample-limit 100 \
  --output-dir results/cvefixes_test
```

## Supported Models

### Current Model Support (11 models)

- **QWEN Series**: qwen3-4b, qwen3-30b, qwen3-30b-thinking
- **DeepSeek Series**: deepseek-coder-v2-lite-16b, deepseek-r1-distill-qwen2.5-7b, deepseek-r1-distill-qwen2.5-32b
- **Llama Series**: llama3.2-3B, llama4-scout-17b-16e
- **Gemma Series**: gemma3-1b, gemma3-27b
- **Wizard Series**: wizard-coder-34b

All models support quantization and are configured for vulnerability detection tasks.

## Available Task Types

- **binary_vulnerability**: Binary classification (Vulnerable vs. Safe)
- **multiclass_vulnerability**: Multi-class CWE type classification  
- **binary_cwe_specific**: CWE-specific binary classification

## Experiment Plans

### Quick Testing (2 plans)

- `quick_test` - Binary classification quick test (10 samples)
- `multiclass_quick_test` - Multiclass classification quick test (10 samples)

### Prompt Comparison (2 plans)  

- `prompt_comparison` - Compare binary classification prompts
- `multiclass_prompt_comparison` - Compare multiclass classification prompts

### Model Comparison (2 plans)

- `model_comparison` - Compare models on binary classification
- `multiclass_model_comparison` - Compare models on multiclass classification

### Comprehensive Evaluation (4 plans)

- `small_models_binary` - Small models on all binary tasks
- `small_models_multiclass` - Small models on all multiclass tasks  
- `large_models_binary` - Large models on all binary tasks
- `large_models_multiclass` - Large models on all multiclass tasks

### Cross-Language Analysis (3 plans)

- `language_comparison_binary` - Compare binary detection across C, Java, Python
- `language_comparison_multiclass` - Compare multiclass classification across languages
- `granularity_comparison` - Compare file-level vs method-level analysis

### CWE-Specific Analysis (1 plan)

- `cwe_specific_analysis` - Focused analysis on specific CWE types

## Key Features

### 1. Multi-Language Vulnerability Detection

- **Cross-Language Prompts**: Language-agnostic prompts work across C, Java, Python
- **Language-Specific Patterns**: Prompts consider language-specific vulnerability patterns
- **Comprehensive Coverage**: Both memory safety (C) and logic vulnerabilities (Java/Python)

### 2. Multiple Analysis Granularities

- **File-Level Analysis**: Broader context, inter-function vulnerabilities
- **Method-Level Analysis**: Focused analysis, function-specific vulnerabilities
- **Comparative Studies**: Built-in experiments to compare granularities

### 3. Real-World Vulnerability Data

- **Actual CVEs**: Real vulnerabilities from production code
- **Rich Metadata**: CVE IDs, CWE classifications, CVSS scores, commit information
- **Multi-Language Corpus**: Vulnerabilities across different programming paradigms

### 4. Comprehensive Task Coverage

- **Binary Classification**: Basic vulnerable/safe detection
- **Multiclass Classification**: Specific CWE type identification
- **CWE-Specific Detection**: Focused analysis for particular vulnerability types

### 5. Advanced Experiment Plans

- **Cross-Language Comparison**: Compare model performance across languages
- **Granularity Analysis**: File vs method-level detection effectiveness
- **Model Size Studies**: Small vs large model performance analysis

## Dataset Preparation

### Prerequisites

- CVEFixes SQLite database: `datasets_processed/cvefixes/CVEfixes.db`
- Available through CVEFixes dataset repository

### Generate Processed Datasets

```bash
# Generate all language datasets
python src/entrypoints/run_setup_cvefixes_datasets.py \
  --database-path datasets_processed/cvefixes/CVEfixes.db \
  --languages C Java Python

# Generate specific language datasets
python src/entrypoints/run_setup_cvefixes_datasets.py \
  --database-path datasets_processed/cvefixes/CVEfixes.db \
  --languages Java \
  --output-dir datasets_processed/cvefixes
```

## Usage Examples

### Quick Multi-Language Test

```bash
# Test binary classification across languages
python src/entrypoints/run_cvefixes_benchmark.py --plan quick_test

# Test multiclass classification
python src/entrypoints/run_cvefixes_benchmark.py --plan multiclass_quick_test
```

### Language Comparison Studies

```bash
# Compare binary detection across C, Java, Python
python src/entrypoints/run_cvefixes_benchmark.py --plan language_comparison_binary

# Compare multiclass classification across languages  
python src/entrypoints/run_cvefixes_benchmark.py --plan language_comparison_multiclass
```

### Custom Experiments

```bash
# Java-specific method-level analysis
python src/entrypoints/run_cvefixes_benchmark.py \
  --model deepseek-coder-v2-lite-16b \
  --dataset multiclass_java_method \
  --prompt multiclass_comprehensive \
  --sample-limit 500

# Python file-level binary classification
python src/entrypoints/run_cvefixes_benchmark.py \
  --model qwen3-30b \
  --dataset binary_python_file \
  --prompt context_aware
```

## Conclusion

The CVEFixes benchmark provides comprehensive **multi-language vulnerability detection** evaluation with:

- **18 datasets** across C, Java, Python with file/method granularity
- **8 language-agnostic prompts** for binary and multiclass classification
- **14 experiment plans** including cross-language and granularity comparisons
- **11 state-of-the-art models** optimized for code security analysis
- **Real-world CVE data** from production codebases

The system enables researchers to:

- Compare vulnerability detection across programming languages
- Evaluate model performance on different vulnerability types
- Study the impact of analysis granularity (file vs method level)
- Conduct comprehensive security-focused LLM evaluations

---

**Status**: ✅ MULTI-LANGUAGE READY  
**Last Updated**: June 2025  
**Dataset Coverage**: C, Java, Python  
**Total Configurations**: 18 datasets × 8 prompts × 11 models = 1,584 possible experiments

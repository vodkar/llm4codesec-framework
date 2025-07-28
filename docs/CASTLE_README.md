# CASTLE Benchmark Integration

This document describes the integration of the CASTLE (Code Analysis with Security Testing for Large-scale Evaluation) benchmark with the LLM code security evaluation framework.

## Overview

The CASTLE benchmark is a comprehensive dataset for evaluating static analysis tools on C/C++ code vulnerabilities. This integration provides:

- **Flexible Dataset Processing**: Convert CASTLE's C files into structured JSON datasets
- **Multiple Task Types**: Support for binary classification, CWE-specific detection, and multi-class classification
- **Prompt Experimentation**: Various prompt strategies for different analysis approaches
- **Batch Experiments**: Automated execution of multiple model/prompt/dataset combinations

## Quick Start

### 1. Setup CASTLE Dataset

First, ensure the CASTLE source is available and set up the processed datasets:

```powershell
# Setup processed datasets with all CWEs
python run_setup_castle_dataset.py --create-prompts --update-gitignore

# Or setup only specific CWEs
python run_setup_castle_dataset.py --cwes CWE-125 CWE-190 CWE-476 CWE-787
```

### 2. Run a Quick Test

```powershell
# Quick test for binary classification with 10 samples
python run_castle_benchmark.py --model qwen3-4b --dataset binary_all --prompt basic_security --sample-limit 10

# Quick test for multiclass classification with 10 samples
python run_castle_benchmark.py --model qwen3-4b --dataset multiclass_all --prompt multiclass_basic --sample-limit 10
```

### 3. Run Experiment Plans

```powershell
# List available experiment plans
python run_castle_experiments.py --list-plans

# Run binary prompt comparison experiment
python run_castle_experiments.py --plan prompt_comparison

# Run multiclass prompt comparison experiment
python run_castle_experiments.py --plan multiclass_prompt_comparison

# Run small models on binary tasks
python run_castle_experiments.py --plan small_models_binary

# Run small models on multiclass tasks
python run_castle_experiments.py --plan small_models_multiclass
```

## File Structure

```
├── castle_dataset_loader.py          # CASTLE dataset loader and processing
├── run_setup_castle_dataset.py           # Dataset setup and processing script
├── run_castle_benchmark.py           # Single experiment runner
├── run_castle_experiments.py         # Batch experiment runner
├── castle_experiments_config.json    # Experiment configurations
├── datasets_processed/               # Processed datasets (gitignored)
│   └── castle/
│       ├── castle_binary.json        # Binary classification dataset
│       ├── castle_multiclass.json    # Multi-class classification dataset
│       ├── castle_cwe_125.json       # CWE-125 specific dataset
│       └── ...                       # Other CWE-specific datasets
└── results/castle_experiments/       # Experiment results
```

## Dataset Types

### 1. Binary Classification (`binary_all`)
- **Task**: Determine if code is vulnerable or safe
- **Labels**: 0 (safe) or 1 (vulnerable)
- **Use Case**: General vulnerability detection

### 2. Multi-class Classification (`multiclass_all`)
- **Task**: Identify the specific vulnerability type
- **Labels**: "CWE-XXX" or "SAFE"
- **Use Case**: Vulnerability categorization

### 3. CWE-Specific Detection
- **Task**: Detect specific vulnerability types
- **Available CWEs**: CWE-125, CWE-134, CWE-190, CWE-22, CWE-253, CWE-327, etc.
- **Labels**: 0 (no target CWE) or 1 (target CWE present)
- **Use Case**: Focused vulnerability detection

## Prompt Strategies

### Binary Classification Prompts

### 1. Basic Security (`basic_security`)
Simple, direct security analysis prompt for binary vulnerability detection.

### 2. Detailed Analysis (`detailed_analysis`)
Comprehensive security analysis with specific guidelines for binary classification.

### 3. Context-Aware (`context_aware`)
Production environment focused analysis for binary classification.

### 4. Step-by-Step (`step_by_step`)
Systematic analysis approach with defined steps for binary classification.

### CWE-Specific Prompts

### 5. CWE-Focused (`cwe_focused`)
Specialized prompt for CWE-specific vulnerability detection in binary classification tasks.

### Multiclass Classification Prompts

### 6. Multiclass Basic (`multiclass_basic`)
Simple, direct vulnerability classification prompt that identifies specific CWE types or marks code as safe.

### 7. Multiclass Detailed (`multiclass_detailed`)
Comprehensive multiclass analysis with detailed CWE pattern descriptions and systematic vulnerability type identification.

### 8. Multiclass Comprehensive (`multiclass_comprehensive`)
Production-level context-aware multiclass analysis considering exploitation patterns and edge cases for precise CWE classification.

## Experiment Plans

### Quick Testing Plans

#### Quick Test (`quick_test`)
- **Purpose**: Fast validation with limited samples for binary classification
- **Configuration**: Multiple small models, binary prompts, 10 samples
- **Duration**: ~5-10 minutes

#### Multiclass Quick Test (`multiclass_quick_test`)
- **Purpose**: Fast validation with limited samples for multiclass classification
- **Configuration**: Multiple small models, multiclass prompts, 10 samples
- **Duration**: ~5-10 minutes

### Prompt Comparison Plans

#### Prompt Comparison (`prompt_comparison`)
- **Purpose**: Compare different binary classification prompt strategies
- **Configuration**: Single model, all binary prompts, binary classification
- **Duration**: ~15-30 minutes

#### Multiclass Prompt Comparison (`multiclass_prompt_comparison`)
- **Purpose**: Compare different multiclass classification prompt strategies
- **Configuration**: Single model, all multiclass prompts, multiclass classification
- **Duration**: ~15-30 minutes

### Model Comparison Plans

#### Model Comparison (`model_comparison`)
- **Purpose**: Compare different LLM models on binary classification
- **Configuration**: Multiple models, best binary prompt, binary classification
- **Duration**: ~30-60 minutes

#### Multiclass Model Comparison (`multiclass_model_comparison`)
- **Purpose**: Compare different LLM models on multiclass classification
- **Configuration**: Multiple models, best multiclass prompt, multiclass classification
- **Duration**: ~30-60 minutes

### Specialized Analysis Plans

#### CWE-Specific Analysis (`cwe_specific_analysis`)
- **Purpose**: Evaluate CWE-specific detection capabilities
- **Configuration**: Multiple models, CWE-focused prompt, all CWE datasets
- **Duration**: ~1-2 hours

### Comprehensive Evaluation Plans

#### Small Models Binary (`small_models_binary`)
- **Purpose**: Full evaluation of small models on binary classification tasks
- **Configuration**: Small models, binary prompts, binary + CWE datasets
- **Duration**: ~2-4 hours

#### Small Models Multiclass (`small_models_multiclass`)
- **Purpose**: Full evaluation of small models on multiclass classification
- **Configuration**: Small models, multiclass prompts, multiclass dataset
- **Duration**: ~1-2 hours

#### Large Models Binary (`large_models_binary`)
- **Purpose**: Full evaluation of large models on binary classification tasks
- **Configuration**: Large models, binary prompts, binary + CWE datasets
- **Duration**: ~4-8 hours

#### Large Models Multiclass (`large_models_multiclass`)
- **Purpose**: Full evaluation of large models on multiclass classification
- **Configuration**: Large models, multiclass prompts, multiclass dataset
- **Duration**: ~2-4 hours

## Usage Examples

### Basic Usage

```powershell
# Run binary classification with basic prompt
python run_castle_benchmark.py \
    --model qwen3-4b \
    --dataset binary_all \
    --prompt basic_security

# Run multiclass classification with detailed prompt
python run_castle_benchmark.py \
    --model qwen3-4b \
    --dataset multiclass_all \
    --prompt multiclass_detailed

# Run CWE-125 detection with focused prompt
python run_castle_benchmark.py \
    --model deepseek-r1-distill-qwen2.5-7b \
    --dataset cwe_125 \
    --prompt cwe_focused
```

### Batch Experiments

```powershell
# Run binary model comparison (test multiple models on binary classification)
python run_castle_experiments.py --plan model_comparison

# Run multiclass model comparison (test multiple models on multiclass classification)
python run_castle_experiments.py --plan multiclass_model_comparison

# Run comprehensive binary evaluation with sample limit for testing
python run_castle_experiments.py --plan small_models_binary --sample-limit 100

# Run comprehensive multiclass evaluation with sample limit for testing
python run_castle_experiments.py --plan small_models_multiclass --sample-limit 100
```

### Custom Configurations

Edit `castle_experiments.json` to:
- Add new models
- Create custom prompt strategies
- Define new experiment plans
- Modify evaluation settings

**Important**: Ensure prompt strategies match task types:
- Use binary prompts (`basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`) for binary classification
- Use multiclass prompts (`multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`) for multiclass classification
- Use CWE-focused prompts (`cwe_focused`) for CWE-specific detection

## Configuration Details

### Model Configurations
```json
{
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "model_type": "LLAMA",
  "max_tokens": 512,
  "temperature": 0.1,
  "batch_size": 1
}
```

### Dataset Configurations
```json
{
  "dataset_path": "datasets_processed/castle/castle_binary.json",
  "task_type": "binary_vulnerability",
  "description": "Binary classification: all vulnerability types"
}
```

### Binary Classification Prompt Strategies
```json
{
  "name": "Basic Security Analysis",
  "system_prompt": "You are an expert security analyst...",
  "user_prompt": "Analyze this code for security vulnerabilities:\n\n{code}"
}
```

### Multiclass Classification Prompt Strategies
```json
{
  "name": "Basic Multiclass Vulnerability Analysis",
  "system_prompt": "You are an expert security analyst specializing in vulnerability classification...",
  "user_prompt": "Classify the vulnerability type in this code:\n\n{code}"
}
```

## Results and Analysis

### Output Structure
```
results/castle_experiments/
├── plan_prompt_comparison_20250617_143022/
│   ├── experiment_plan_results.json
│   ├── qwen3-4b_binary_all_basic_security/
│   │   ├── benchmark_report_*.json
│   │   ├── metrics_summary_*.json
│   │   └── predictions_*.csv
│   └── ...
├── plan_multiclass_prompt_comparison_20250617_153022/
│   ├── experiment_plan_results.json
│   ├── qwen3-4b_multiclass_all_multiclass_basic/
│   │   ├── benchmark_report_*.json
│   │   ├── metrics_summary_*.json
│   │   └── predictions_*.csv
│   └── ...
```

### Key Metrics

#### Binary Classification
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Detection rate for vulnerabilities
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

#### Multiclass Classification
- **Accuracy**: Overall correctness across all classes
- **Per-class Precision/Recall/F1**: Metrics for each CWE type
- **Macro/Micro Averages**: Aggregated performance metrics
- **Confusion Matrix**: Detailed classification breakdown

#### CWE-Specific Detection
- **Binary metrics**: Applied to specific CWE vs. non-CWE classification
- **Per-CWE Analysis**: Individual performance for each vulnerability type

### Analysis Features
- Per-CWE performance breakdown
- Confusion matrices
- Error analysis with sample details
- Performance comparisons across experiments

## Advanced Usage

### Task Type Separation and Best Practices

The CASTLE integration now properly separates different task types to ensure optimal performance:

#### Binary Classification Tasks
- **Datasets**: `binary_all`, `cwe_*` (for CWE-specific binary detection)
- **Prompts**: `basic_security`, `detailed_analysis`, `context_aware`, `step_by_step`, `cwe_focused`
- **Output Format**: "VULNERABLE" or "SAFE"
- **Use Cases**: General vulnerability detection, specific CWE presence detection

#### Multiclass Classification Tasks
- **Datasets**: `multiclass_all`
- **Prompts**: `multiclass_basic`, `multiclass_detailed`, `multiclass_comprehensive`
- **Output Format**: "CWE-XXX" (e.g., "CWE-125", "CWE-190") or "SAFE"
- **Use Cases**: Vulnerability type identification, precise security categorization

#### Experiment Planning Guidelines
1. **Don't mix task types**: Use binary prompts for binary datasets and multiclass prompts for multiclass datasets
2. **Start small**: Use quick test plans to validate configurations before running comprehensive evaluations
3. **Resource planning**: Multiclass classification typically requires more computational resources and time
4. **Evaluation focus**: Binary classification for general detection capability, multiclass for precise categorization

### Custom Prompt Development

1. Edit `castle_experiments.json`
2. Add new prompt strategy in `prompt_strategies` section
3. Ensure prompt matches intended task type:
   - Binary prompts should output "VULNERABLE" or "SAFE"
   - Multiclass prompts should output specific CWE identifiers (e.g., "CWE-125") or "SAFE"
   - CWE-focused prompts should output "VULNERABLE" or "SAFE" for specific CWE detection
4. Test with single experiment:
   ```powershell
   # Test binary prompt
   python run_castle_benchmark.py --model qwen3-4b --dataset binary_all --prompt your_new_binary_prompt
   
   # Test multiclass prompt
   python run_castle_benchmark.py --model qwen3-4b --dataset multiclass_all --prompt your_new_multiclass_prompt
   ```

### Adding New Models

1. Update `model_configurations` in config file
2. Ensure model is supported by the framework
3. Test with quick experiment first


### Validation Commands

```powershell
# Validate all datasets exist
python run_castle_experiments.py --validate-datasets

# List available experiment plans
python run_castle_experiments.py --list-plans

# Test configuration with minimal samples
python run_castle_experiments.py --plan quick_test --sample-limit 5
python run_castle_experiments.py --plan multiclass_quick_test --sample-limit 5
```

## References

- [CASTLE Benchmark Works](https://github.com/CASTLE-Benchmark)
- [Original CASTLE Repository](https://github.com/CASTLE-Benchmark/CASTLE-Benchmark)
- [LLM Benchmark Framework Documentation](README.md)
# VulDetectBench Benchmark - Configuration-Based Runner

## Overview

The VulDetectBench benchmark provides comprehensive vulnerability detection evaluation using the VulDetectBench dataset containing 5 tasks of increasing difficulty for evaluating LLM vulnerability detection capabilities. The system has been implemented to use a unified configuration-based approach that matches the CASTLE, JitVul, CVEFixes, and VulBench benchmark patterns for consistency across all datasets.

## New Configuration-Based System

### Key Features ✅
- **Unified Configuration**: JSON-based experiment configuration following the established pattern
- **Consistent CLI**: Same command-line interface across all benchmarks (CASTLE, JitVul, CVEFixes, VulBench, VulDetectBench)
- **Five Task Types**: Support for all 5 VulDetectBench tasks with varying difficulty levels
- **Flexible Experiments**: Easy definition of model/dataset/prompt combinations
- **Single Entry Point**: All experiments configurable through JSON files
- **Model Synchronization**: Consistent model definitions across all datasets

### Core Components
- **Configuration File**: `src/configs/vuldetectbench_experiments.json`
- **Benchmark Runner**: `src/entrypoints/run_vuldetectbench_benchmark.py`
- **Unified Runner**: `src/entrypoints/run_unified_benchmark.py` (handles all datasets)
- **Dataset Loader**: `src/datasets/loaders/vuldetectbench_dataset_loader.py`
- **Data Processor**: `src/scripts/process_vuldetectbench_data.py`

## VulDetectBench Tasks

VulDetectBench contains 5 tasks of increasing difficulty, each designed to evaluate different aspects of LLM vulnerability detection capabilities:

| Task | Description | Type | Evaluation Metrics |
|------|-------------|------|-------------------|
| **Task 1** | Binary vulnerability existence detection | Binary Classification | Accuracy, F1-Score, Precision, Recall |
| **Task 2** | Multi-choice vulnerability type inference | Multi-class Classification | Moderate/Strict Evaluation Score |
| **Task 3** | Key objects and functions identification | Code Analysis | Token Recall (Macro/Micro) |
| **Task 4** | Root cause location identification | Code Analysis | Union Line Recall, Line Recall |
| **Task 5** | Trigger point location identification | Code Analysis | Union Line Recall, Line Recall |

### Task Details

#### Task 1: Vulnerability Existence Detection
- **Format**: Binary classification (YES/NO)
- **Samples**: 1000 (from original dataset)
- **Evaluation**: Standard binary classification metrics
- **Example**: "Is the code vulnerable? (YES/NO)"

#### Task 2: Vulnerability Type Inference  
- **Format**: Multi-choice question (A/B/C/D/E)
- **Samples**: 500 (from original dataset)
- **Evaluation**: Moderate (correct or sub-optimal) and Strict (only correct) evaluation
- **Example**: Multiple choice with vulnerability type options

#### Task 3: Key Objects & Functions Identification
- **Format**: Code snippet extraction
- **Samples**: 100 (from original dataset)
- **Evaluation**: Token recall between predicted and ground truth
- **Example**: Extract vulnerable functions and objects

#### Task 4: Root Cause Location
- **Format**: Code line identification  
- **Samples**: 100 (from original dataset)
- **Evaluation**: Line recall and union line recall
- **Example**: Identify the line causing the vulnerability

#### Task 5: Trigger Point Location
- **Format**: Code line identification
- **Samples**: 100 (from original dataset)  
- **Evaluation**: Line recall and union line recall
- **Example**: Identify the line that triggers the vulnerability

## Configuration Structure

The VulDetectBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulDetectBench LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulDetectBench with 5 tasks",
    "version": "1.0",
    "dataset": "VulDetectBench v1.0"
  },
  "dataset_configurations": {
    "task1_vulnerability": {
      "dataset_path": "datasets_processed/vuldetectbench/vuldetectbench_task1.json",
      "task_type": "binary_vulnerability",
      "description": "Task 1: Binary vulnerability existence detection"
    },
    "task2_multiclass": {
      "dataset_path": "datasets_processed/vuldetectbench/vuldetectbench_task2.json", 
      "task_type": "multiclass_vulnerability",
      "description": "Task 2: Multi-choice vulnerability type inference"
    }
    // ... additional task configurations
  },
  "prompt_strategies": {
    "basic_security": { /* Standard security analysis prompt */ },
    "task1_specific": { /* Task 1 optimized prompt */ },
    "task2_specific": { /* Task 2 optimized prompt */ }
    // ... task-specific prompts
  },
  "model_configurations": { /* Same as other benchmarks */ },
  "experiment_plans": { /* Predefined experiment combinations */ }
}
```

## Quick Start

### 1. Data Processing

First, process the raw VulDetectBench data to create structured JSON datasets:

```bash
# Process VulDetectBench data for all tasks
python src/scripts/process_vuldetectbench_data.py

# Process specific tasks
python src/scripts/process_vuldetectbench_data.py --tasks task1 task2

# Validate processed data
python src/scripts/process_vuldetectbench_data.py --validate-only
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vuldetectbench --list-configs

# Using the direct VulDetectBench runner
python src/entrypoints/run_vuldetectbench_benchmark.py --list-configs
```

### 3. Run Quick Test

```bash
# Quick test with limited samples
python run_unified_benchmark.py vuldetectbench --plan quick_test

# Or using direct runner
python src/entrypoints/run_vuldetectbench_benchmark.py --plan quick_test
```

### 4. Run Single Experiment

```bash
# Task 1 evaluation
python run_unified_benchmark.py vuldetectbench --model qwen3-4b --dataset task1_vulnerability --prompt task1_specific

# Task 2 evaluation  
python run_unified_benchmark.py vuldetectbench --model deepseek-coder-v2-lite-16b --dataset task2_multiclass --prompt task2_specific
```

## Available Experiment Plans

### Basic Plans
- **`quick_test`**: Fast validation with limited samples across basic tasks
- **`task1_evaluation`**: Comprehensive Task 1 (binary detection) evaluation
- **`task2_evaluation`**: Comprehensive Task 2 (multi-class) evaluation

### Advanced Plans
- **`advanced_tasks_evaluation`**: Tasks 3-5 (code analysis) evaluation
- **`model_comparison`**: Compare different model sizes on basic tasks
- **`prompt_comparison`**: Compare different prompt strategies
- **`comprehensive_evaluation`**: Complete evaluation across all tasks and models

### Example Plan Execution

```bash
# Evaluate Task 1 comprehensively
python run_unified_benchmark.py vuldetectbench --plan task1_evaluation

# Compare models on advanced tasks
python run_unified_benchmark.py vuldetectbench --plan advanced_tasks_evaluation

# Full comprehensive evaluation
python run_unified_benchmark.py vuldetectbench --plan comprehensive_evaluation
```

## Prompt Strategies

### Standard Prompts
- **`basic_security`**: General security analysis approach
- **`detailed_analysis`**: Thorough CWE-focused analysis
- **`step_by_step`**: Systematic analysis methodology

### Task-Specific Prompts
- **`task1_specific`**: Optimized for binary vulnerability detection
- **`task2_specific`**: Optimized for multi-choice vulnerability classification
- **`task3_specific`**: Optimized for object/function identification
- **`task4_specific`**: Optimized for root cause location
- **`task5_specific`**: Optimized for trigger point location

Each task-specific prompt is designed to match the original VulDetectBench evaluation format and expectations.

## Model Configurations

The system supports the same model configurations as other benchmarks:

- **Small Models**: `qwen3-4b`, `llama3.2-3B`, `gemma3-1b`, `deepseek-r1-distill-qwen2.5-1.5b`
- **Medium Models**: `deepseek-coder-v2-lite-16b`, `wizard-coder-34b`
- **Large Models**: `qwen3-30b`, `deepseek-r1-distill-qwen2.5-32b`, `gemma3-27b`

## Data Processing Details

### Raw VulDetectBench Structure
```
benchmarks/VulDetectBench/dataset/test/
├── task1_code.jsonl    # 1000 samples - binary detection
├── task2_code.jsonl    # 500 samples - multi-class
├── task3_code.jsonl    # 100 samples - object identification  
├── task4_code.jsonl    # 100 samples - root cause location
└── task5_code.jsonl    # 100 samples - trigger point location
```

### Processed Structure
```
datasets_processed/vuldetectbench/
├── vuldetectbench_task1.json
├── vuldetectbench_task2.json
├── vuldetectbench_task3.json
├── vuldetectbench_task4.json
├── vuldetectbench_task5.json
├── vuldetectbench_task1_stats.json
├── vuldetectbench_task2_stats.json
├── vuldetectbench_task3_stats.json
├── vuldetectbench_task4_stats.json
├── vuldetectbench_task5_stats.json
└── vuldetectbench_summary.json
```

### Data Processing Statistics

Based on the original VulDetectBench dataset:

| Task | Samples | Min Tokens | Max Tokens | Difficulty |
|------|---------|------------|------------|------------|
| Task 1 | 1000 | 50 | 3493 | Basic |
| Task 2 | 500 | 265 | 3372 | Intermediate |
| Task 3 | 100 | 1017 | 3269 | Advanced |
| Task 4 | 100 | 1010 | 3262 | Advanced |
| Task 5 | 100 | 1011 | 3363 | Advanced |

## Command Line Options

### VulDetectBench Runner Options

```bash
python src/entrypoints/run_vuldetectbench_benchmark.py [OPTIONS]

Options:
  --config PATH         Configuration file path
  --model MODEL         Model configuration key
  --dataset DATASET     Dataset configuration key  
  --prompt PROMPT       Prompt strategy key
  --plan PLAN          Experiment plan to execute
  --list-configs       List all available configurations
  --sample-limit N     Limit samples for testing
  --output-dir PATH    Output directory for results
  --verbose           Enable detailed logging
```

### Unified Runner Options

```bash  
python run_unified_benchmark.py vuldetectbench [OPTIONS]

Options:
  --plan PLAN          Experiment plan to execute
  --model MODEL        Model configuration
  --dataset DATASET    Dataset configuration
  --prompt PROMPT      Prompt strategy
  --sample-limit N     Limit samples for testing
  --list-configs      List available configurations
```

## Output and Results

### Result Structure

Each experiment generates:
- **Predictions**: Individual model responses and predictions
- **Metrics**: Task-specific evaluation metrics
- **Summary**: Overall performance statistics
- **Metadata**: Experiment configuration and timing

### Metrics Calculated

For each task type:
- **Task 1**: Accuracy, Precision, Recall, F1-Score
- **Task 2**: Moderate Evaluation Score, Strict Evaluation Score
- **Task 3**: Token Recall (Macro/Micro)
- **Task 4-5**: Line Recall, Union Line Recall

### Example Result Files

```
results/vuldetectbench_experiments/
└── qwen3-4b_task1_vulnerability_task1_specific_20250617_143022/
    ├── results.json
    ├── predictions.json
    ├── metrics.json
    └── config.json
```

## Integration with Framework

The VulDetectBench implementation maintains full compatibility with the benchmark framework:

- **Standard Interfaces**: Uses `BenchmarkSample`, `PredictionResult`, `BenchmarkConfig`
- **Consistent Patterns**: Matches CASTLE, JitVul, CVEFixes, and VulBench implementation patterns  
- **Framework Integration**: `VulDetectBenchDatasetLoaderFramework` for seamless integration
- **Unified Metrics**: Framework-standard evaluation metrics with task-specific extensions

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```bash
   # Ensure data is processed first
   python src/scripts/process_vuldetectbench_data.py
   ```

2. **Model Loading Errors**
   ```bash
   # Check model configurations match framework expectations
   python run_unified_benchmark.py vuldetectbench --list-configs
   ```

3. **Memory Issues with Large Models**
   ```bash
   # Use smaller models or reduce batch size
   python run_unified_benchmark.py vuldetectbench --plan quick_test --model qwen3-4b
   ```

4. **Task-Specific Evaluation Issues**
   - Tasks 3-5 use custom evaluation metrics (token recall, line recall)
   - Ensure proper response parsing for code extraction tasks
   - Verify prompt formatting matches VulDetectBench expectations

## Integration with Other Benchmarks

The VulDetectBench runner integrates seamlessly with the unified benchmark system:

```bash
# Compare across datasets
python run_unified_benchmark.py castle --plan quick_test
python run_unified_benchmark.py jitvul --plan quick_test  
python run_unified_benchmark.py cvefixes --plan quick_test
python run_unified_benchmark.py vulbench --plan quick_test
python run_unified_benchmark.py vuldetectbench --plan quick_test
```

## Next Steps

1. **Task-Specific Optimizations**: Develop advanced metrics for Tasks 3-5
2. **Cross-Task Analysis**: Compare model performance across different task types
3. **Result Visualization**: Create charts showing difficulty progression across tasks
4. **Advanced Prompting**: Experiment with few-shot and chain-of-thought prompting
5. **Performance Optimization**: Optimize for large-scale experiments across all 5 tasks

## Support

For issues specific to VulDetectBench implementation:
1. Check the logs in the output directory
2. Verify data processing completed successfully  
3. Ensure all required dependencies are installed
4. Review the experiment configuration for correct paths and task types

For general benchmark framework issues, refer to the main README.md and other benchmark documentation.

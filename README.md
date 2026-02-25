# LLM Code Security Benchmark Framework

A comprehensive framework for benchmarking Large Language Models on static code analysis and vulnerability detection tasks. This framework supports multiple LLM models, datasets, and evaluation metrics for binary and multi-class classification tasks.

## Features

- **Multiple Model Support**: Llama3.2 and Llama4, Qwen3, DeepSeek (R1, V2, Coder), Wizard Coder, Gemma.
- **Backend Choice**: Hugging Face transformers (default) or vLLM (Linux/CUDA only) or llama.cpp.
- **Flexible Task Types**:
  - Binary vulnerability detection
  - CWE-specific vulnerability detection  
  - Multi-class vulnerability classification
- **Dataset Agnostic**: Support for VulBench, JitVul, CASTLE, CVEFixes, VulDetectBench, VulnerabilityDetection
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Production Ready**: Full type annotations, logging, error handling, and result persistence
- **Easy Configuration**: Template-based configuration management
- **Local Execution**: Run models locally with optional quantization for efficiency

## Installation

### Prerequisites

- Linux is recommended (windows is not tested)
- Python 3.10 or higher
- CUDA-compatible GPU (recommended) or CPU
- 16GB+ vRAM (40GB+ recommended for larger models)

### Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd llm4codesec-framework
git submodule update --init --recursive
```

1. Setup your .env variables

```bash
cp .default.env .env
```

Put your hugging face token in `HF_TOKEN`. You can get your token [here](https://huggingface.co/settings/tokens)
Your `.env` should looks like:

```
PYTHONPATH=src/

LOG_LEVEL=INFO
LOG_FORMAT='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

HF_TOKEN='hf_blabla...'

PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

If you plan to use the vLLM backend, ensure vLLM is installed and you are running on Linux with CUDA support.

## ⚡️ Comprehensive experiment VERY QUICKSTART

Watch carefully for any warning/errors logs! You might get wrong results in case of incorrect configuration

### Option 1: Optimized Script (Recommended)

```shell
./build_docker.sh
./run_full_benchmark.sh
```

### Option 2: Manual Commands

```shell
./build_docker.sh
alias run_benchmark="docker-compose run --rm llm4codesec-benchmark python"

# CASTLE experiments
run_benchmark entrypoints/run_setup_castle_dataset.py
for plan in small_models_binary small_models_multiclass large_models_binary large_models_multiclass; do
  run_benchmark -m cli run-plan castle --plan $plan
done

# CVEFixes experiments
run_benchmark entrypoints/run_setup_cvefixes_datasets.py \
  --database-path datasets_processed/cvefixes/CVEfixes.db \
  --languages C Java Python
for plan in small_models_binary small_models_multiclass large_models_binary large_models_multiclass; do
  run_benchmark -m cli run-plan cvefixes --plan $plan
done
```

## Quick Start

### Using Docker

#### Setup nvidia container drivers

##### a

Check for actual instruction [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

##### OR b

Use docker cuda ready VPC image. For example in [selectel](https://docs.selectel.ru/en/cloud-servers/images/about-images/#default-images)

#### Build docker image

```bash
./build_docker.sh
```

### W/o Docker (not tested)

#### Install dependencies using uv

Install only the dependencies for your chosen backend:

```bash
# HuggingFace transformers backend
uv sync --extra hf

# vLLM backend (Linux/CUDA only)
uv sync --extra vllm

# llama.cpp backend
uv sync --extra llama-cpp

# Multiple backends at once
uv sync --extra hf --extra vllm
```

### How to run

#### Run quick experiment

```bash
PYTHONPATH=src python -m cli run-plan castle --plan quick_test

# Split config mode (shared models/prompts + explicit experiments/datasets)
PYTHONPATH=src python -m cli run-plan castle \
  --config-dir src/configs/shared \
  --experiments-config src/configs/castle/experiments.json \
  --datasets-config src/configs/castle/datasets.json \
  --plan quick_test
```

#### Run Specific Benchmarks

```bash
PYTHONPATH=src python -m cli run-plan cvefixes \
  --plan quick_test \
  --sample-limit 100 \
  --output-dir results/cvefixes_test

PYTHONPATH=src python -m cli run-plan jitvul \
  --plan quick_test \
  --sample-limit 100 \
  --output-dir results/jitvul_test

PYTHONPATH=src python -m cli run-plan vulbench \
  --plan quick_test \
  --output-dir results/vulbench_test
```

## Metrics

### Binary Classification

- Accuracy
- Precision
- Recall  
- F1-score
- Specificity
- Confusion Matrix (TP, TN, FP, FN)

### Multi-class Classification

- Accuracy
- Per-class Precision, Recall, F1-score
- Macro/Micro averages
- Confusion Matrix

### Example Results Analysis

```python
import json
import pandas as pd

# Load results
with open('./results/benchmark_report_20241203_143022.json', 'r') as f:
    results = json.load(f)

# Print summary
print(f"Model: {results['benchmark_info']['model_name']}")
print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"F1-Score: {results['metrics']['f1_score']:.4f}")

# Load predictions for detailed analysis
predictions_df = pd.read_csv('./results/predictions_20241203_143022.csv')
print(predictions_df.groupby(['true_label', 'predicted_label']).size())
```

## Documentation

Additional documentation can be found in [docs/](docs/) directory

### Datasets

- [CASTLE](docs/CASTLE_README.md)
- [CVEFixes](docs/CVEFIXES_README.md)

## Citation (NOT PUBLISHED YET ⚠️)

If you use this framework in your research, please cite:

```bibtex
@misc{llm4codesec-framework,
    title={...},
    author={Kirill Gladkikh},
    year={2025},
    url={https://github.com/vodkar/llm4codesec-framework}
}
```

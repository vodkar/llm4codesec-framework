# LLM Code Security Benchmark Framework

A comprehensive framework for benchmarking Large Language Models on static code analysis and vulnerability detection tasks. This framework supports multiple LLM models, datasets, and evaluation metrics for binary and multi-class classification tasks.

## Features

- **Multiple Model Support**: Llama3.2 and Llama4, Qwen3, DeepSeek (R1, V2, Coder), Wizard Coder, Gemma.
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
cd llm4codesec-llm-benchmark
git submodule update --init --recursive
```

2. Setup your .env variables

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
    run_benchmark entrypoints/run_castle_experiments.py --plan $plan
done

# CVEFixes experiments
run_benchmark entrypoints/prepare_cvefixes_datasets.py \
  --database-path datasets_processed/cvefixes/CVEfixes.db \
  --languages C Java Python
for plan in small_models_binary small_models_multiclass large_models_binary large_models_multiclass; do
    run_benchmark entrypoints/run_cvefixes_benchmark.py --plan $plan
done
```

## Quick Start

### Using Docker

#### Setup nvidia container drivers

##### a. 

Check for actual instruction [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

##### OR b. 

Use docker cuda ready VPC image. For example in [selectel](https://docs.selectel.ru/en/cloud-servers/images/about-images/#default-images)

#### Build docker image

```bash
./build_docker.sh
```

### W/o Docker (not tested)

#### Install dependencies using Poetry (recommended):

```bash
pip install poetry
poetry install
poetry env activate
```

### How to run

#### Run quick experiment

```bash
python src/entrypoints/run_castle_experiments.py --plan quick_test
```

#### Run Specific Benchmarks

```bash
python src/entrypoints/run_cvefixes_benchmark.py \
  --plan basic_evaluation \
  --sample-limit 100 \
  --output-dir results/cvefixes_test

python src/entrypoints/run_jitvul_benchmark.py \
  --plan basic_evaluation \
  --sample-limit 100 \
  --output-dir results/jitvul_test
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

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{llm4codesec-benchmark,
    title={LLM Code Security Benchmark Framework},
    author={Your Name},
    year={2024},
    url={https://github.com/your-repo}
}
```

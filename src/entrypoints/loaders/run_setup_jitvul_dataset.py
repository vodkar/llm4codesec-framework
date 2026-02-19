from pathlib import Path

from benchmark.enums import TaskType
from datasets.loaders.jitvul import JitVulDatasetLoader


def main() -> None:
    """Example usage and testing of the JitVul dataset loader."""
    dataset_path = Path("benchmarks/JitVul/data/final_benchmark.jsonl")
    loader = JitVulDatasetLoader(source_path=dataset_path)

    if dataset_path.exists():
        try:
            # Test different task types
            print("Testing JitVul Dataset Loader")
            print("=" * 40)

            # Binary task
            print("\n1. Binary Classification:")
            binary_samples = loader.load_dataset(
                task_type=TaskType.BINARY_VULNERABILITY, limit=10
            )
            print(f"   Loaded {len(binary_samples)} binary samples")

            # Multiclass task
            print("\n2. Multiclass Classification:")
            multiclass_samples = loader.load_dataset(
                task_type=TaskType.MULTICLASS_VULNERABILITY, limit=10
            )
            print(f"   Loaded {len(multiclass_samples)} multiclass samples")

            # CWE-specific task
            print("\n3. CWE-Specific Classification:")
            cwe_samples = loader.load_dataset(
                task_type=TaskType.BINARY_CWE_SPECIFIC,
                target_cwe="CWE-125",
                limit=10,
            )
            print(f"   Loaded {len(cwe_samples)} CWE-125 samples")

            # Get statistics
            print("\n4. Dataset Statistics:")
            stats = loader.get_dataset_stats(dataset_path)
            print(f"   Total items: {stats.get('total_items', 0)}")
            print(
                f"   CWE distribution: {list(stats.get('cwe_distribution', {}).keys())[:5]}"
            )
            print(f"   Severity distribution: {stats.get('severity_distribution', {})}")

        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Dataset file not found: {dataset_path}")
        print("Please ensure the JitVul dataset is available.")


if __name__ == "__main__":
    main()

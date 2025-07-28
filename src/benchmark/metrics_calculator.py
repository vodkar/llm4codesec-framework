import re
from abc import ABC, abstractmethod
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from benchmark.enums import TaskType
from benchmark.models import PredictionResult


class IMetricsCalculator(ABC):
    """
    Abstract interface for metrics calculation.

    This interface allows for different implementations of metrics calculation
    for various task types including binary classification, multiclass classification,
    and code analysis tasks.

    Example usage:
        # Binary classification
        calculator = BinaryMetricsCalculator()
        metrics = calculator.calculate(predictions)

        # Multiclass classification
        calculator = MulticlassMetricsCalculator()
        metrics = calculator.calculate(predictions)

        # Code analysis tasks
        calculator = CodeAnalysisMetricsCalculator("task3")
        metrics = calculator.calculate(predictions)
    """

    @abstractmethod
    def calculate(self, predictions: list[PredictionResult]) -> dict[str, Any]:
        """
        Calculate metrics for the given predictions.

        Args:
            predictions: List of prediction results from model evaluation

        Returns:
            Dictionary containing calculated metrics
        """
        pass


class BinaryMetricsCalculator(IMetricsCalculator):
    """Metrics calculator for binary classification tasks."""

    def calculate(self, predictions: list[PredictionResult]) -> dict[str, float]:
        """Calculate metrics for binary classification."""
        y_true = [pred.true_label for pred in predictions]
        y_pred = [pred.predicted_label for pred in predictions]

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Calculate metrics
        accuracy = float(accuracy_score(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }


class MulticlassMetricsCalculator(IMetricsCalculator):
    """Metrics calculator for multiclass classification tasks."""

    def calculate(self, predictions: list[PredictionResult]) -> dict[str, Any]:
        """Calculate metrics for multiclass classification."""
        y_true = [pred.true_label for pred in predictions]
        y_pred = [pred.predicted_label for pred in predictions]

        accuracy = float(accuracy_score(y_true, y_pred))
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        if not isinstance(report, dict):
            raise ValueError("Classification report is not a dictionary")

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }


class CodeAnalysisMetricsCalculator(IMetricsCalculator):
    """Metrics calculator for code analysis tasks (token/line recall)."""

    def __init__(self, task_type: str):
        """
        Initialize with specific task type.

        Args:
            task_type: Type of task ('task3', 'task4', 'task5', etc.)
        """
        self.task_type = task_type

    def calculate(self, predictions: list[PredictionResult]) -> dict[str, Any]:
        """Calculate metrics for code analysis tasks."""
        metrics = {"total_predictions": len(predictions), "task_type": self.task_type}

        if self.task_type == "task3":
            # Token recall for key objects identification
            token_recalls: list[float] = []
            for pred in predictions:
                recall = self._calculate_token_recall(
                    str(pred.predicted_label), str(pred.true_label)
                )
                token_recalls.append(recall)

            metrics["macro_token_recall"] = (
                sum(token_recalls) / len(token_recalls) if token_recalls else 0.0
            )
            metrics["micro_token_recall"] = metrics["macro_token_recall"]  # Simplified
            metrics["accuracy"] = metrics["macro_token_recall"]  # For consistency

        elif self.task_type in ["task4", "task5"]:
            # Line recall for root cause/trigger point location
            line_recalls: list[float] = []
            union_line_recalls: list[float] = []

            for pred in predictions:
                line_recall = self._calculate_line_recall(
                    str(pred.predicted_label), str(pred.true_label)
                )
                union_line_recall = self._calculate_union_line_recall(
                    str(pred.predicted_label), str(pred.true_label)
                )

                line_recalls.append(line_recall)
                union_line_recalls.append(union_line_recall)

            metrics["avg_line_recall"] = (
                sum(line_recalls) / len(line_recalls) if line_recalls else 0.0
            )
            metrics["avg_union_line_recall"] = (
                sum(union_line_recalls) / len(union_line_recalls)
                if union_line_recalls
                else 0.0
            )
            metrics["accuracy"] = metrics["avg_line_recall"]  # For consistency

        return metrics

    def _calculate_token_recall(self, predicted: str, true: str) -> float:
        """Calculate token recall for Task 3."""
        # Extract tokens from code snippets (simplified)
        pred_tokens = set(self._extract_code_tokens(predicted))
        true_tokens = set(self._extract_code_tokens(true))

        if not true_tokens:
            return 1.0 if not pred_tokens else 0.0

        intersection = pred_tokens.intersection(true_tokens)
        return len(intersection) / len(true_tokens)

    def _calculate_line_recall(self, predicted: str, true: str) -> float:
        """Calculate line recall for Task 4-5."""
        # Simplified line matching
        pred_lines = set(line.strip() for line in predicted.split("\n") if line.strip())
        true_lines = set(line.strip() for line in true.split("\n") if line.strip())

        if not true_lines:
            return 1.0 if not pred_lines else 0.0

        intersection = pred_lines.intersection(true_lines)
        return len(intersection) / len(true_lines)

    def _calculate_union_line_recall(self, predicted: str, true: str) -> float:
        """Calculate union line recall for Task 4-5."""
        # Simplified implementation - could be enhanced for actual union calculation
        return self._calculate_line_recall(predicted, true)

    def _extract_code_tokens(self, code_snippet: str) -> list[str]:
        """Extract code tokens from a snippet."""

        # Simple tokenization - extract identifiers and function names
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code_snippet)
        return tokens


class MetricsCalculatorFactory:
    """
    Legacy wrapper for backward compatibility and factory methods.

    This class provides both backward compatibility with existing code
    and factory methods for creating appropriate metrics calculators.
    """

    @staticmethod
    def create_calculator(
        task_type: TaskType, task_specific_type: str | None = None
    ) -> IMetricsCalculator:
        """
        Factory method to create appropriate metrics calculator.

        Args:
            task_type: The general task type (binary, multiclass, etc.)
            task_specific_type: Specific task type for code analysis (e.g., "task3", "task4")

        Returns:
            Appropriate metrics calculator instance

        Example:
            # Binary classification
            calculator = MetricsCalculator.create_calculator(TaskType.BINARY_VULNERABILITY)

            # Multiclass classification
            calculator = MetricsCalculator.create_calculator(TaskType.MULTICLASS_VULNERABILITY)

            # Code analysis task
            calculator = MetricsCalculator.create_calculator(TaskType.BINARY_VULNERABILITY, "task3")
        """
        if task_specific_type in ["task3", "task4", "task5"]:
            return CodeAnalysisMetricsCalculator(task_specific_type)
        elif task_type in [
            TaskType.BINARY_VULNERABILITY,
            TaskType.BINARY_CWE_SPECIFIC,
            TaskType.BINARY_VULNERABILITY_SPECIFIC,
        ]:
            return BinaryMetricsCalculator()
        else:
            return MulticlassMetricsCalculator()

    @staticmethod
    def calculate_binary_metrics(
        predictions: list[PredictionResult],
    ) -> dict[str, float]:
        """Calculate metrics for binary classification."""
        calculator = BinaryMetricsCalculator()
        return calculator.calculate(predictions)

    @staticmethod
    def calculate_multiclass_metrics(
        predictions: list[PredictionResult],
    ) -> dict[str, Any]:
        """Calculate metrics for multiclass classification."""
        calculator = MulticlassMetricsCalculator()
        return calculator.calculate(predictions)

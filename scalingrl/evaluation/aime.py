"""AIME 2025 evaluation."""

from typing import Any

from datasets import Dataset, load_dataset

from scalingrl.evaluation.evaluator import BaseEvaluator


class AIMEEvaluator(BaseEvaluator):
    """Evaluator for AIME 2025 dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize AIME evaluator."""
        super().__init__(*args, **kwargs)
        self.dataset_name = "AI-MO/aimo-validation-aime"  # AIME 2024 validation set

    def load_dataset(self) -> Dataset:
        """
        Load AIME 2025 dataset.

        Returns:
            Dataset with AIME problems
        """
        # Load dataset
        dataset = load_dataset(self.dataset_name, split="train")

        # Format for evaluation
        def format_example(example):
            """Format AIME example."""
            # AIME problems have specific format
            # Answers are integers from 000 to 999
            return {
                "query": example.get("problem", example.get("question", "")),
                "ground_truth": str(example.get("answer", "")),
            }

        formatted = dataset.map(
            format_example,
            remove_columns=dataset.column_names,
            desc="Formatting AIME dataset",
        )

        return formatted

    def evaluate(self, dataset: Dataset | None = None) -> dict[str, Any]:
        """
        Evaluate model on AIME 2025.

        Args:
            dataset: Optional pre-loaded dataset

        Returns:
            Evaluation results with accuracy
        """
        if dataset is None:
            dataset = self.load_dataset()

        print(f"Evaluating on AIME 2025 ({len(dataset)} problems)")

        # AIME answers are in \boxed{number} format
        results = self.evaluate_dataset(dataset, answer_format="boxed")

        print(f"AIME 2025 Accuracy: {results['accuracy']:.2%}")

        return results

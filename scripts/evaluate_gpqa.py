"""GPQA Diamond evaluation (separate script - requires dataset access).

GPQA is a gated dataset. Request access at:
https://huggingface.co/datasets/Idavidrein/gpqa

Usage:
    python scripts/evaluate_gpqa.py --checkpoint ./outputs/model_name/final
"""

import argparse
from typing import Any

from datasets import Dataset, load_dataset

from scalingrl.evaluation.evaluator import BaseEvaluator
from scalingrl.utils import set_seed


class GPQAEvaluator(BaseEvaluator):
    """Evaluator for GPQA Diamond dataset."""

    def __init__(self, *args, **kwargs):
        """Initialize GPQA evaluator."""
        super().__init__(*args, **kwargs)
        self.dataset_name = "Idavidrein/gpqa"  # GPQA benchmark
        self.subset = "gpqa_diamond"  # Diamond subset (hardest)

    def load_dataset(self) -> Dataset:
        """
        Load GPQA Diamond dataset.

        Returns:
            Dataset with GPQA problems
        """
        # Load dataset
        dataset = load_dataset(self.dataset_name, self.subset, split="train")

        # Format for evaluation
        def format_example(example):
            """Format GPQA example as multiple choice."""
            # GPQA has multiple choice questions
            question = example.get("Question", "")

            # Format choices
            choices = []
            for choice_key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
                if choice_key in example:
                    choices.append(example[choice_key])

            # Create formatted question with choices
            formatted_question = f"{question}\n\n"
            formatted_question += "Choose the correct answer:\n"
            for i, choice in enumerate(choices):
                letter = chr(65 + i)  # A, B, C, D
                formatted_question += f"{letter}. {choice}\n"

            # Ground truth is always "A" since we put correct answer first
            # (unless we shuffle - for now keep it simple)
            ground_truth = "A"

            return {
                "query": formatted_question,
                "ground_truth": ground_truth,
            }

        formatted = dataset.map(
            format_example,
            remove_columns=dataset.column_names,
            desc="Formatting GPQA dataset",
        )

        return formatted

    def evaluate(self, dataset: Dataset | None = None) -> dict[str, Any]:
        """
        Evaluate model on GPQA Diamond.

        Args:
            dataset: Optional pre-loaded dataset

        Returns:
            Evaluation results with accuracy
        """
        if dataset is None:
            dataset = self.load_dataset()

        print(f"Evaluating on GPQA Diamond ({len(dataset)} problems)")
        print("Note: Random baseline is 25% (4 choices)")

        # GPQA answers are multiple choice (A, B, C, D)
        results = self.evaluate_dataset(dataset, answer_format="choice")

        print(f"GPQA Diamond Accuracy: {results['accuracy']:.2%}")

        return results


def main():
    """Main evaluation function for GPQA."""
    parser = argparse.ArgumentParser(description="Evaluate model on GPQA Diamond")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load model (simplified - adapt from evaluate.py if needed)
    print(f"Loading checkpoint from: {args.checkpoint}")
    print("Note: You need to implement model loading from evaluate.py")
    print("This is a placeholder - integrate with main evaluate.py for full functionality")

    # For now, just test dataset loading
    try:
        evaluator = GPQAEvaluator(
            model=None,  # TODO: Load model
            tokenizer=None,  # TODO: Load tokenizer
            batch_size=args.batch_size,
        )
        dataset = evaluator.load_dataset()
        print(f"Successfully loaded GPQA dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading GPQA dataset: {e}")
        print("Make sure you have requested access at https://huggingface.co/datasets/Idavidrein/gpqa")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

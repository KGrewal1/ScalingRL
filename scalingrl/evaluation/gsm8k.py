"""GSM8K evaluation (pass@1 on test split)."""

from typing import Any

from datasets import Dataset, load_dataset

from scalingrl.evaluation.evaluator import BaseEvaluator
from scalingrl.evaluation.metrics import compute_pass_at_k_batch, extract_answer, normalize_answer


class GSM8KEvaluator(BaseEvaluator):
    """Evaluator for GSM8K test split."""

    def load_dataset(self) -> Dataset:
        """
        Load GSM8K test split.

        Returns:
            Dataset with GSM8K problems
        """
        import re

        dataset = load_dataset("openai/gsm8k", "main", split="test")

        def format_example(example):
            """Format GSM8K example for evaluation."""
            # Extract numeric answer from "#### <number>"
            match = re.search(r"####\s*(.+)", example["answer"])
            ground_truth = match.group(1).strip() if match else ""
            return {
                "query": example["question"],
                "ground_truth": ground_truth,
            }

        formatted = dataset.map(
            format_example,
            remove_columns=dataset.column_names,
            desc="Formatting GSM8K test set",
        )

        return formatted

    def evaluate(self, dataset: Dataset | None = None) -> dict[str, Any]:
        """
        Evaluate model on GSM8K test split with pass@1.

        Args:
            dataset: Optional pre-loaded dataset

        Returns:
            Evaluation results with pass@1 accuracy
        """
        if dataset is None:
            dataset = self.load_dataset()

        print(f"Evaluating on GSM8K test ({len(dataset)} problems)")
        print(f"  num_samples={self.num_samples}, computing pass@1")

        all_predictions_per_problem: list[list[str]] = []
        all_ground_truths: list[str] = []

        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        for batch_idx, i in enumerate(range(0, len(dataset), self.batch_size)):
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  Batch {batch_idx + 1}/{total_batches}")

            batch = dataset[i : i + self.batch_size]
            prompts = batch["query"] if isinstance(batch["query"], list) else [batch["query"]]
            ground_truths = (
                batch["ground_truth"] if isinstance(batch["ground_truth"], list) else [batch["ground_truth"]]
            )

            # Collect num_samples completions per problem
            batch_preds: list[list[str]] = [[] for _ in range(len(prompts))]
            temperature = 0.8 if self.num_samples > 1 else 0.0

            for _ in range(self.num_samples):
                completions = self.generate_batch(prompts, temperature=temperature)
                for j, comp in enumerate(completions):
                    pred = extract_answer(comp, "boxed")
                    batch_preds[j].append(pred)

            all_predictions_per_problem.extend(batch_preds)
            all_ground_truths.extend(ground_truths)

        # Compute pass@1
        pass_at_1 = compute_pass_at_k_batch(all_predictions_per_problem, all_ground_truths, k=1)

        # Also compute simple greedy accuracy (first sample per problem)
        first_preds = [preds[0] for preds in all_predictions_per_problem]
        correct = sum(
            1 for pred, gt in zip(first_preds, all_ground_truths) if normalize_answer(pred) == normalize_answer(gt)
        )
        greedy_accuracy = correct / len(all_ground_truths) if all_ground_truths else 0.0

        print(f"GSM8K pass@1: {pass_at_1:.2%}")
        print(f"GSM8K greedy accuracy: {greedy_accuracy:.2%}")

        return {
            "pass_at_1": pass_at_1,
            "greedy_accuracy": greedy_accuracy,
            "num_problems": len(all_ground_truths),
            "num_samples": self.num_samples,
        }

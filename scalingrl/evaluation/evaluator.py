"""Base evaluator class."""

from typing import Any

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from scalingrl.evaluation.metrics import compute_accuracy, extract_answer


class BaseEvaluator:
    """Base class for evaluation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        max_new_tokens: int = 512,
        num_samples: int = 1,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch_size: Batch size for generation
            max_new_tokens: Maximum tokens to generate
            num_samples: Number of samples per problem (for pass@k)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float = 0.0,
    ) -> list[str]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of prompts
            temperature: Sampling temperature (0.0 for greedy)

        Returns:
            List of generated completions
        """
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode outputs (skip the input prompt)
        completions = []
        for i, output in enumerate(outputs):
            # Remove input tokens
            input_length = inputs.input_ids[i].shape[0]
            generated_tokens = output[input_length:]
            completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            completions.append(completion)

        return completions

    def evaluate_dataset(
        self,
        dataset: Dataset,
        answer_format: str = "boxed",
    ) -> dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            dataset: Dataset with 'query' and 'ground_truth' fields
            answer_format: Answer format type ("boxed" or "choice")

        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_ground_truths = []

        # Process in batches
        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        for batch_idx, i in enumerate(range(0, len(dataset), self.batch_size)):
            print(f"Processing batch {batch_idx + 1}/{total_batches}...")
            batch = dataset[i : i + self.batch_size]

            # Get prompts and ground truths
            prompts = batch["query"] if isinstance(batch["query"], list) else [batch["query"]]
            ground_truths = (
                batch["ground_truth"] if isinstance(batch["ground_truth"], list) else [batch["ground_truth"]]
            )

            # Generate completions
            temperature = 0.8 if self.num_samples > 1 else 0.0

            for _ in range(self.num_samples):
                completions = self.generate_batch(prompts, temperature=temperature)

                # Extract answers
                predictions = [extract_answer(comp, answer_format) for comp in completions]

                all_predictions.extend(predictions)
                all_ground_truths.extend(ground_truths)

        # Compute accuracy
        accuracy = compute_accuracy(all_predictions, all_ground_truths)

        results = {
            "accuracy": accuracy,
            "num_samples": len(all_predictions),
            "predictions": all_predictions,
            "ground_truths": all_ground_truths,
        }

        return results

    def evaluate(self, dataset: Dataset) -> dict[str, Any]:
        """
        Evaluate model (to be implemented by subclasses).

        Args:
            dataset: Dataset to evaluate on

        Returns:
            Evaluation results
        """
        raise NotImplementedError("Subclasses must implement evaluate()")

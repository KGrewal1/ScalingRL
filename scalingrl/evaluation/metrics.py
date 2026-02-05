"""Evaluation metrics."""

import re

import numpy as np
from scipy.special import comb

from scalingrl.data import extract_boxed_answer, normalize_answer


def extract_answer(text: str, format_type: str = "boxed") -> str:
    """Extract answer from generated text.

    Handles nested braces for boxed format, and multiple-choice (A-D).
    """
    if format_type == "boxed":
        return extract_boxed_answer(text)

    elif format_type == "choice":
        patterns = [
            r"answer is ([A-D])",
            r"answer: ([A-D])",
            r"choose ([A-D])",
            r"option ([A-D])",
            r"\(([A-D])\)",
            r"^([A-D])\.",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()

        match = re.search(r"\b([A-D])\b", text)
        if match:
            return match.group(1).upper()

        return ""

    else:
        raise ValueError(f"Unknown format_type: {format_type}")


def compute_accuracy(predictions: list[str], ground_truths: list[str]) -> float:
    """Compute exact match accuracy."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if normalize_answer(pred) == normalize_answer(gt))
    return correct / len(predictions)


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k: probability that at least one of k samples is correct,
    given n total samples and c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - (comb(n - c, k) / comb(n, k))


def compute_pass_at_k_batch(
    predictions: list[list[str]],
    ground_truths: list[str],
    k: int = 1,
) -> float:
    """Compute pass@k averaged over a batch of multi-sample predictions."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    pass_at_k_scores = []
    for preds, gt in zip(predictions, ground_truths):
        n = len(preds)
        gt_norm = normalize_answer(gt)
        c = sum(1 for pred in preds if normalize_answer(pred) == gt_norm)
        pass_at_k_scores.append(compute_pass_at_k(n, c, k))

    return np.mean(pass_at_k_scores)

"""Data contamination evaluation via partial-prompt completion.

Implements the methodology from "Reasoning or Memorization?" (Wu et al. 2025):
truncate each problem to a prefix ratio (e.g. 60%), let the model complete the
rest greedily without a chat template, and measure how well it reconstructs the
held-out suffix. High completion rates signal memorization / data contamination.

Metrics:
  - ROUGE-L: longest-common-subsequence overlap between completion and suffix
  - EM: exact-match rate (completion == suffix after normalization)
  - Answer Accuracy: fraction of completions that contain the correct answer
"""

import re
from typing import Any

from datasets import Dataset, load_dataset

from scalingrl.evaluation.evaluator import BaseEvaluator

# ---------------------------------------------------------------------------
# ROUGE-L (F-measure) — simple, dependency-free implementation
# ---------------------------------------------------------------------------


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP (two rows)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 between a reference and hypothesis string."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


SUPPORTED_DATASETS = ("gsm8k", "math500")


class ContaminationEvaluator(BaseEvaluator):
    """Evaluate data contamination via partial-prompt completion.

    For each problem the evaluator:
      1. Truncates the question text to ``prefix_ratio`` of its characters.
      2. Feeds the prefix to the model (greedy, no chat template).
      3. Compares the generated continuation against the held-out suffix.
    """

    def __init__(self, *args, prefix_ratio: float = 0.6, dataset_name: str = "gsm8k", **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_ratio = prefix_ratio
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unknown dataset {dataset_name!r}, must be one of {SUPPORTED_DATASETS}")
        self.dataset_name = dataset_name

    # -- dataset loading ----------------------------------------------------

    def load_dataset(self) -> Dataset:
        """Load test split with full question text and ground truth."""
        if self.dataset_name == "gsm8k":
            return self._load_gsm8k()
        else:
            return self._load_math500()

    def _load_gsm8k(self) -> Dataset:
        dataset = load_dataset("openai/gsm8k", "main", split="test")

        def format_example(example):
            match = re.search(r"####\s*(.+)", example["answer"])
            ground_truth = match.group(1).strip() if match else ""
            return {
                "query": example["question"],
                "ground_truth": ground_truth,
            }

        return dataset.map(
            format_example,
            remove_columns=dataset.column_names,
            desc="Formatting GSM8K test set for contamination eval",
        )

    def _load_math500(self) -> Dataset:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

        def format_example(example):
            return {
                "query": example["problem"],
                "ground_truth": example["answer"],
            }

        return dataset.map(
            format_example,
            remove_columns=dataset.column_names,
            desc="Formatting MATH-500 test set for contamination eval",
        )

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, dataset: Dataset | None = None) -> dict[str, Any]:
        """Run partial-prompt completion evaluation.

        Returns dict with rouge_l, em, answer_accuracy, and per-problem details.
        """
        if dataset is None:
            dataset = self.load_dataset()

        ratio_pct = int(self.prefix_ratio * 100)
        print(f"Contamination eval on {self.dataset_name} test ({len(dataset)} problems)")
        print(f"  prefix_ratio={ratio_pct}%, greedy decoding, no chat template")

        rouge_l_scores: list[float] = []
        em_scores: list[float] = []
        answer_hits: list[float] = []

        total_batches = (len(dataset) + self.batch_size - 1) // self.batch_size

        for batch_idx, i in enumerate(range(0, len(dataset), self.batch_size)):
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  Batch {batch_idx + 1}/{total_batches}")

            batch = dataset[i : i + self.batch_size]
            questions = batch["query"] if isinstance(batch["query"], list) else [batch["query"]]
            ground_truths = (
                batch["ground_truth"] if isinstance(batch["ground_truth"], list) else [batch["ground_truth"]]
            )

            # Build truncated prefixes and reference suffixes.
            # Advance cutoff to next word boundary to avoid splitting mid-word,
            # matching the paper's truncation method.
            prefixes: list[str] = []
            suffixes: list[str] = []
            for q in questions:
                cutoff = int(len(q) * self.prefix_ratio)
                while cutoff < len(q) and q[cutoff] not in " \n,.!?)]}":
                    cutoff += 1
                prefixes.append(q[:cutoff].rstrip())
                suffixes.append(q[cutoff:])

            # Generate (greedy, no template)
            completions = self.generate_batch(prefixes, temperature=0.0)

            for comp, suffix, gt in zip(completions, suffixes, ground_truths):
                # ROUGE-L — truncate completion to suffix word count to
                # match the paper's methodology (avoids precision penalty
                # from the model generating beyond the suffix).
                suffix_words = suffix.split()
                comp_truncated = " ".join(comp.split()[: len(suffix_words)])
                rouge_l_scores.append(rouge_l_f1(suffix, comp_truncated))

                # Exact match (whitespace-normalized)
                em = 1.0 if comp.strip() == suffix.strip() else 0.0
                em_scores.append(em)

                # Answer accuracy — does the completion contain the correct
                # numeric answer anywhere (including inside \boxed{})?
                gt_clean = gt.replace(",", "").strip()
                comp_clean = comp.replace(",", "")
                hit = 1.0 if gt_clean and gt_clean in comp_clean else 0.0
                answer_hits.append(hit)

        n = len(rouge_l_scores)
        avg_rouge = sum(rouge_l_scores) / n if n else 0.0
        avg_em = sum(em_scores) / n if n else 0.0
        avg_ans = sum(answer_hits) / n if n else 0.0

        print(f"\n  Results (prefix={ratio_pct}%):")
        print(f"    ROUGE-L:          {avg_rouge:.4f}  ({avg_rouge:.2%})")
        print(f"    Exact Match:      {avg_em:.4f}  ({avg_em:.2%})")
        print(f"    Answer Accuracy:  {avg_ans:.4f}  ({avg_ans:.2%})")

        return {
            "prefix_ratio": self.prefix_ratio,
            "rouge_l": avg_rouge,
            "em": avg_em,
            "answer_accuracy": avg_ans,
            "num_problems": n,
        }

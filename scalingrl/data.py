"""Dataset loaders and reward functions for GRPO training."""

import re

from datasets import Dataset, load_dataset

# ---------------------------------------------------------------------------
# Answer extraction and normalization
# ---------------------------------------------------------------------------


def _extract_brace_content(text: str, start: int) -> str:
    """Extract content between balanced braces starting at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return ""


def extract_boxed_answer(text: str) -> str:
    """Extract answer from LaTeX \\boxed{answer} format.

    Handles nested braces (e.g. \\boxed{\\frac{1}{2}}).
    """
    for pattern in (r"\boxed{", "boxed{"):
        idx = text.find(pattern)
        if idx != -1:
            brace_start = idx + len(pattern) - 1
            return _extract_brace_content(text, brace_start).strip()
    return ""


def extract_gsm8k_ground_truth(answer_text: str) -> str:
    """Extract the numeric ground truth from GSM8K's '#### <number>' format."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip()
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    Handles numeric normalization: strips commas, whitespace,
    trailing zeros after decimal point, and unnecessary decimal points.
    """
    ans = answer.strip().lower()
    ans = ans.replace(",", "")
    ans = ans.replace(" ", "")
    if "." in ans:
        try:
            ans = str(float(ans))
            if "." in ans:
                ans = ans.rstrip("0").rstrip(".")
        except ValueError:
            pass
    return ans


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def math_accuracy_reward(
    prompts: list[str],
    completions: list[str],
    ground_truths: list[str],
    **kwargs,
) -> list[float]:
    """Math accuracy reward for GRPO. Returns 1.0 for correct, 0.0 for incorrect."""
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        predicted = extract_boxed_answer(completion)
        predicted_norm = normalize_answer(predicted)
        gt_norm = normalize_answer(gt)
        rewards.append(1.0 if predicted_norm == gt_norm else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_gsm8k_dataset(
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Dataset]:
    """Load GSM8K dataset (train + test splits) for GRPO training."""
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test")

    train_dataset = train_dataset.shuffle(seed=seed)
    test_dataset = test_dataset.shuffle(seed=seed)

    if max_samples is not None:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(max_samples // 5 or 1, len(test_dataset))))

    def format_example(example):
        ground_truth = extract_gsm8k_ground_truth(example["answer"])
        return {"prompt": example["question"], "ground_truth": ground_truth}

    train_formatted = train_dataset.map(
        format_example,
        remove_columns=train_dataset.column_names,
        desc="Formatting GSM8K train",
    )
    test_formatted = test_dataset.map(
        format_example,
        remove_columns=test_dataset.column_names,
        desc="Formatting GSM8K test",
    )

    return {"train": train_formatted, "test": test_formatted}


def load_dapo_math_dataset(
    dataset_name: str = "open-r1/DAPO-Math-17k-Processed",
    max_samples: int | None = None,
    train_split: str = "train",
    val_split: str | None = None,
    seed: int = 42,
) -> dict[str, Dataset]:
    """Load DAPO-Math-17k dataset for GRPO training."""
    dataset = load_dataset(dataset_name, split=train_split)
    dataset = dataset.shuffle(seed=seed)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def format_example(example):
        return {
            "query": example.get("prompt", example.get("problem", example.get("question", ""))),
            "ground_truth": example.get("solution", example.get("answer", "")),
            "original": example,
        }

    formatted_dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting dataset for GRPO",
    )

    result = {"train": formatted_dataset}

    if val_split is not None:
        val_dataset = load_dataset(dataset_name, split=val_split)
        val_dataset = val_dataset.shuffle(seed=seed)
        if max_samples is not None:
            val_dataset = val_dataset.select(range(min(max_samples // 5, len(val_dataset))))
        val_dataset = val_dataset.map(
            format_example,
            remove_columns=val_dataset.column_names,
            desc="Formatting validation dataset",
        )
        result["val"] = val_dataset

    return result

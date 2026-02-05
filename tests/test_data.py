"""Tests for data loading and reward functions."""

import pytest

from scalingrl.data import (
    extract_boxed_answer,
    extract_gsm8k_ground_truth,
    load_dapo_math_dataset,
    load_gsm8k_dataset,
    math_accuracy_reward,
    normalize_answer,
)


def test_extract_boxed_answer():
    """Test extracting answers from boxed format."""
    # Standard format
    text = "The answer is \\boxed{42}"
    assert extract_boxed_answer(text) == "42"

    # Without backslash
    text = "The answer is boxed{42}"
    assert extract_boxed_answer(text) == "42"

    # With complex expression
    text = "Therefore \\boxed{x + y = 10}"
    assert extract_boxed_answer(text) == "x + y = 10"

    # No boxed answer
    text = "The answer is 42"
    assert extract_boxed_answer(text) == ""


def test_normalize_answer():
    """Test answer normalization."""
    # Whitespace
    assert normalize_answer("  42  ") == "42"

    # Case
    assert normalize_answer("ABC") == "abc"

    # Commas
    assert normalize_answer("1,000") == "1000"

    # Trailing zeros
    assert normalize_answer("2.50") == "2.5"
    assert normalize_answer("1.0") == "1"

    # Integer with decimal
    assert normalize_answer("100.00") == "100"

    # Spaces in numbers
    assert normalize_answer("1, 234") == "1234"


def test_extract_gsm8k_ground_truth():
    """Test GSM8K ground truth extraction."""
    # Standard format
    assert extract_gsm8k_ground_truth("Some reasoning\n#### 42") == "42"

    # With commas
    assert extract_gsm8k_ground_truth("Steps...\n#### 1,000") == "1,000"

    # With spaces
    assert extract_gsm8k_ground_truth("#### 123") == "123"

    # No match
    assert extract_gsm8k_ground_truth("No answer here") == ""

    # Negative number
    assert extract_gsm8k_ground_truth("#### -5") == "-5"


def test_math_accuracy_reward():
    """Test math accuracy reward function."""
    prompts = ["Q1", "Q2", "Q3"]
    completions = [
        "Answer: \\boxed{42}",
        "Answer: \\boxed{100}",
        "Answer: \\boxed{7}",
    ]
    ground_truths = ["42", "50", "7"]

    rewards = math_accuracy_reward(prompts, completions, ground_truths)

    assert len(rewards) == 3
    assert rewards[0] == 1.0  # Correct
    assert rewards[1] == 0.0  # Incorrect
    assert rewards[2] == 1.0  # Correct


def test_math_accuracy_reward_numeric_normalization():
    """Test that numeric normalization works in reward computation."""
    prompts = ["Q1", "Q2"]
    completions = [
        "\\boxed{1000}",
        "\\boxed{2.5}",
    ]
    ground_truths = ["1,000", "2.50"]

    rewards = math_accuracy_reward(prompts, completions, ground_truths)
    assert rewards[0] == 1.0  # "1000" matches "1,000"
    assert rewards[1] == 1.0  # "2.5" matches "2.50"


def test_load_dapo_math_dataset():
    """Test loading DAPO-Math dataset."""
    # Load small sample
    datasets = load_dapo_math_dataset(max_samples=10, seed=42)

    assert "train" in datasets
    assert len(datasets["train"]) == 10

    # Check format
    example = datasets["train"][0]
    assert "query" in example
    assert "ground_truth" in example


def test_load_gsm8k_dataset():
    """Test loading GSM8K dataset."""
    datasets = load_gsm8k_dataset(max_samples=10, seed=42)

    assert "train" in datasets
    assert "test" in datasets
    assert len(datasets["train"]) == 10

    # Check format
    example = datasets["train"][0]
    assert "prompt" in example
    assert "ground_truth" in example
    assert example["ground_truth"] != ""  # Should have extracted a number


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

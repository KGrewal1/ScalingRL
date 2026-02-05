"""Tests for reward functions."""

import pytest

from scalingrl.data import extract_boxed_answer, extract_gsm8k_ground_truth, math_accuracy_reward, normalize_answer


def test_boxed_answer_formats():
    """Test various boxed answer formats."""
    test_cases = [
        ("\\boxed{123}", "123"),
        ("\\boxed{-5}", "-5"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("Some text \\boxed{answer} more text", "answer"),
        ("boxed{no_backslash}", "no_backslash"),
        ("no answer here", ""),
    ]

    for text, expected in test_cases:
        result = extract_boxed_answer(text)
        assert result == expected, f"Failed for: {text}"


def test_reward_function_edge_cases():
    """Test reward function edge cases."""
    # Empty completions
    rewards = math_accuracy_reward(["Q"], [""], ["42"])
    assert rewards[0] == 0.0

    # Case insensitive
    rewards = math_accuracy_reward(["Q"], ["\\boxed{ABC}"], ["abc"])
    assert rewards[0] == 1.0

    # Whitespace differences
    rewards = math_accuracy_reward(["Q"], ["\\boxed{ 42 }"], ["42"])
    assert rewards[0] == 1.0


def test_multiple_rewards():
    """Test batch reward computation."""
    prompts = ["Q1", "Q2", "Q3", "Q4"]
    completions = [
        "\\boxed{1}",
        "\\boxed{2}",
        "\\boxed{3}",
        "\\boxed{4}",
    ]
    ground_truths = ["1", "2", "5", "4"]

    rewards = math_accuracy_reward(prompts, completions, ground_truths)

    assert len(rewards) == 4
    assert rewards[0] == 1.0  # Correct
    assert rewards[1] == 1.0  # Correct
    assert rewards[2] == 0.0  # Incorrect
    assert rewards[3] == 1.0  # Correct


def test_extract_gsm8k_ground_truth():
    """Test GSM8K answer extraction."""
    assert extract_gsm8k_ground_truth("Step 1... Step 2...\n#### 42") == "42"
    assert extract_gsm8k_ground_truth("#### 1,200") == "1,200"
    assert extract_gsm8k_ground_truth("####  -7 ") == "-7"
    assert extract_gsm8k_ground_truth("no hash marks") == ""


def test_numeric_normalization():
    """Test numeric answer normalization."""
    # Commas
    assert normalize_answer("1,000") == "1000"
    assert normalize_answer("1,000,000") == "1000000"

    # Trailing zeros
    assert normalize_answer("2.50") == "2.5"
    assert normalize_answer("3.00") == "3"

    # Whitespace
    assert normalize_answer("  42  ") == "42"

    # Combination
    assert normalize_answer(" 1,000.00 ") == "1000"


def test_gsm8k_reward_with_commas():
    """Test that GSM8K-style comma numbers match model outputs."""
    # Model outputs "1000" via boxed, ground truth from GSM8K is "1,000"
    rewards = math_accuracy_reward(
        ["Q"],
        ["\\boxed{1000}"],
        ["1,000"],
    )
    assert rewards[0] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

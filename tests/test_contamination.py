"""Tests for contamination evaluation metrics."""

import pytest

from scalingrl.evaluation.contamination import rouge_l_f1


def test_rouge_l_identical():
    """Identical strings should score 1.0."""
    assert rouge_l_f1("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_rouge_l_no_overlap():
    """Completely different strings should score 0.0."""
    assert rouge_l_f1("the cat sat", "dogs run fast") == 0.0


def test_rouge_l_partial_overlap():
    """Partial overlap should give a score between 0 and 1."""
    ref = "the cat sat on the mat"
    hyp = "the cat on the mat today"
    score = rouge_l_f1(ref, hyp)
    assert 0.0 < score < 1.0
    # LCS = "the cat on the mat" (5 tokens)
    # precision = 5/6, recall = 5/6, F1 = 5/6
    assert abs(score - 5 / 6) < 1e-6


def test_rouge_l_empty():
    """Empty inputs should return 0."""
    assert rouge_l_f1("", "some text") == 0.0
    assert rouge_l_f1("some text", "") == 0.0
    assert rouge_l_f1("", "") == 0.0


def test_rouge_l_subsequence():
    """Hypothesis is a subsequence of reference."""
    ref = "a b c d e f"
    hyp = "a c e"
    score = rouge_l_f1(ref, hyp)
    # LCS = 3, precision = 3/3 = 1.0, recall = 3/6 = 0.5
    # F1 = 2 * 1.0 * 0.5 / 1.5 = 2/3
    assert abs(score - 2 / 3) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for model loading and configuration."""

import pytest

from scalingrl.models import create_lora_config


def test_create_lora_config():
    """Test LoRA configuration creation."""
    lora_config = create_lora_config(
        r=8,
        alpha=16,
        dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    assert lora_config.r == 8
    assert lora_config.lora_alpha == 16
    assert lora_config.lora_dropout == 0.05
    assert "q_proj" in lora_config.target_modules
    assert "v_proj" in lora_config.target_modules


def test_lora_ranks():
    """Test different LoRA ranks."""
    ranks = [1, 2, 4, 8, 16, 64]

    for rank in ranks:
        lora_config = create_lora_config(
            r=rank,
            alpha=rank * 2,
            dropout=0.05,
            target_modules=["q_proj"],
            bias="none",
        )
        assert lora_config.r == rank
        assert lora_config.lora_alpha == rank * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

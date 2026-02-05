"""Model loading, PEFT configuration, and parameter counting."""

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    use_flash_attention: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with specified configuration."""
    dtype_obj = getattr(torch, dtype)

    print(f"Loading model: {model_name}")
    print(f"  - dtype: {dtype_obj}")
    print(f"  - device_map: {device_map}")
    print(f"  - flash_attention: {use_flash_attention}")

    model_kwargs = {
        "torch_dtype": dtype_obj,
        "device_map": device_map,
        "trust_remote_code": True,
    }

    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model.config.use_cache = False  # Disable for gradient checkpointing

    print("Model loaded successfully")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, tokenizer


def create_lora_config(
    r: int,
    alpha: int,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
    bias: str = "none",
) -> LoraConfig:
    """Create LoRA configuration."""
    # Standard target modules across all supported architectures (Qwen, Mistral, OLMo, Gemma)
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    print("LoRA Configuration:")
    print(f"  - rank (r): {r}")
    print(f"  - alpha: {alpha}")
    print(f"  - dropout: {dropout}")
    print(f"  - target_modules: {target_modules}")

    return lora_config


def count_trainable_parameters(model) -> tuple[int, int]:
    """Count trainable parameters in model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable_params:,} / {all_params:,}")
    print(f"Trainable %: {100 * trainable_params / all_params:.2f}%")

    return trainable_params, all_params

"""Simple configuration using dataclasses."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "Qwen/Qwen2.5-7B"
    family: str = "qwen2.5"
    dtype: str = "bfloat16"
    device_map: str = "auto"


@dataclass
class LoRAConfig:
    """LoRA configuration."""

    adapter_type: str = "lora"  # "lora", "lora_xs", or "tiny_lora"
    r: int = 8
    alpha: int = 16
    # TinyLoRA-specific
    tiny_lora_u: int = 1  # projection dimension (trainable params per tying group)
    tiny_lora_n_tie: int | None = None  # weight tying factor (None = no tying)
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = "none"


@dataclass
class OptimizerConfig:
    """Optimizer configuration (AdamW only)."""

    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class DataConfig:
    """Data configuration."""

    dataset_name: str = "openai/gsm8k"
    max_samples: int | None = None
    train_split: str = "train"
    val_split: str | None = None
    max_length: int = 2048


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 10
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 2
    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class GRPOConfig:
    """GRPO specific configuration."""

    num_generations: int = 4
    max_completion_length: int = 4096
    temperature: float = 1.0
    beta: float = 0.0
    vllm_gpu_memory_utilization: float = 0.3


@dataclass
class LoggingConfig:
    """Logging configuration."""

    use_wandb: bool = True
    wandb_project: str = "scaling-rl-grpo"
    log_model: bool = False


@dataclass
class ProjectConfig:
    """Project configuration."""

    name: str = "scaling-rl-grpo"
    seed: int = 42
    output_dir: str = "./outputs"


@dataclass
class ExperimentConfig:
    """Full experiment configuration with defaults."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)

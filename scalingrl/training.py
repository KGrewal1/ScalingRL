"""GRPO trainer configuration and setup."""

from datasets import Dataset
from peft import LoraConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer

from scalingrl.data import math_accuracy_reward
from scalingrl.lora_xs import apply_lora_xs, apply_tiny_lora


def create_grpo_config(
    output_dir: str,
    run_name: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-5,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 2,
    num_generations: int = 4,
    max_completion_length: int = 512,
    temperature: float = 1.0,
    beta: float = 0.0,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    report_to: str = "wandb",
) -> GRPOConfig:
    """Create GRPOConfig."""
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=temperature,
        beta=beta,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        report_to=report_to,
        run_name=run_name,
    )

    print("GRPO Configuration:")
    print(f"  - output_dir: {output_dir}")
    print(f"  - run_name: {run_name}")
    print(f"  - num_generations: {num_generations}")
    print(f"  - max_completion_length: {max_completion_length}")
    print(f"  - beta (KL penalty): {beta}")

    return grpo_config


def create_grpo_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    grpo_config: GRPOConfig,
    peft_config: LoraConfig | None = None,
    adapter_type: str = "lora",
    tiny_lora_u: int = 1,
    tiny_lora_n_tie: int | None = None,
) -> GRPOTrainer:
    """Create GRPO trainer with built-in AdamW optimizer."""

    def reward_fn(prompts, completions, **kwargs):
        """Reward function wrapper."""
        indices = kwargs.get("indices", None)
        if indices is not None:
            ground_truths = [train_dataset[i]["ground_truth"] for i in indices]
        else:
            ground_truths = [ex["ground_truth"] for ex in train_dataset][: len(prompts)]

        return math_accuracy_reward(prompts, completions, ground_truths)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if adapter_type == "lora_xs" and peft_config is not None:
        apply_lora_xs(trainer.model, rank=peft_config.r)
    elif adapter_type == "tiny_lora" and peft_config is not None:
        apply_tiny_lora(trainer.model, rank=peft_config.r, u=tiny_lora_u, n_tie=tiny_lora_n_tie)

    print("GRPO Trainer created successfully")
    return trainer


def train_model(
    trainer: GRPOTrainer,
    resume_from_checkpoint: str | None = None,
) -> None:
    """Train model using GRPO trainer."""
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Training completed!")

    print(f"Saving final model to {trainer.args.output_dir}")
    trainer.save_model()

"""Main training script for GRPO experiments."""

import argparse
import os

from scalingrl.config import ExperimentConfig
from scalingrl.data import load_gsm8k_dataset
from scalingrl.models import count_trainable_parameters, create_lora_config, load_model_and_tokenizer
from scalingrl.training import create_grpo_config, create_grpo_trainer, train_model
from scalingrl.utils import finish_wandb, log_environment, set_seed, setup_wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train model with GRPO")

    # Model
    parser.add_argument("--model-name", type=str, help="Model name")
    parser.add_argument("--model-family", type=str, help="Model family label (e.g., qwen2.5, mistral)")

    # LoRA
    parser.add_argument("--lora-rank", type=int, help="LoRA rank")
    parser.add_argument("--adapter-type", type=str, choices=["lora", "lora_xs", "tiny_lora"], help="Adapter type")
    parser.add_argument("--tiny-lora-u", type=int, help="TinyLoRA projection dimension (trainable params per group)")
    parser.add_argument("--tiny-lora-n-tie", type=int, help="TinyLoRA weight tying factor")

    # Optimizer
    parser.add_argument("--lr", type=float, help="Learning rate")

    # Data
    parser.add_argument("--max-samples", type=int, help="Limit dataset size")

    # Training
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--grad-accum", type=int, help="Gradient accumulation steps")

    # GRPO
    parser.add_argument("--num-generations", type=int, help="GRPO num generations")

    # vLLM
    parser.add_argument("--vllm-gpu-memory", type=float, default=0.3, help="vLLM GPU memory utilization (default: 0.3)")

    # Other
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--wandb-group", type=str, help="Wandb group name (for separating sweep runs)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Start with defaults
    config = ExperimentConfig()

    # Override with command line args
    if args.model_name:
        config.model.name = args.model_name
    if args.model_family:
        config.model.family = args.model_family
    if args.adapter_type:
        config.lora.adapter_type = args.adapter_type
    if args.tiny_lora_u:
        config.lora.tiny_lora_u = args.tiny_lora_u
    if args.tiny_lora_n_tie:
        config.lora.tiny_lora_n_tie = args.tiny_lora_n_tie
    if args.lora_rank:
        config.lora.r = args.lora_rank
        config.lora.alpha = args.lora_rank * 2
    if args.lr:
        config.optimizer.lr = args.lr
    if args.max_samples:
        config.data.max_samples = args.max_samples
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.grad_accum:
        config.training.gradient_accumulation_steps = args.grad_accum
    if args.num_generations:
        config.grpo.num_generations = args.num_generations
    if args.no_wandb:
        config.logging.use_wandb = False
    if args.seed:
        config.project.seed = args.seed
    if args.output_dir:
        config.project.output_dir = args.output_dir
    config.grpo.vllm_gpu_memory_utilization = args.vllm_gpu_memory

    # Log configuration
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"Family: {config.model.family}")
    print(f"Adapter: {config.lora.adapter_type}")
    print(f"LoRA rank: {config.lora.r}")
    print(f"Optimizer: AdamW (lr={config.optimizer.lr})")
    print(f"Dataset: {config.data.dataset_name}")
    if config.data.max_samples:
        print(f"Max samples: {config.data.max_samples}")
    print(f"Epochs: {config.training.num_train_epochs}")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
    print(f"GRPO generations: {config.grpo.num_generations}")
    print(f"vLLM GPU memory: {config.grpo.vllm_gpu_memory_utilization}")
    print("=" * 60 + "\n")

    # Derive adapter label for run naming
    if config.lora.adapter_type == "lora_xs":
        adapter_label = "loraxs"
    elif config.lora.adapter_type == "tiny_lora":
        adapter_label = "tinylora"
    else:
        adapter_label = "lora"

    # Set random seed
    set_seed(config.project.seed)

    # Log environment
    log_environment()

    # Setup wandb
    if config.logging.use_wandb:
        run_name = f"{config.model.family}_{adapter_label}{config.lora.r}"
        wandb_group = args.wandb_group or f"family_{config.model.family}"
        setup_wandb(
            project=config.logging.wandb_project,
            run_name=run_name,
            config={
                "model": config.model.__dict__,
                "lora": config.lora.__dict__,
                "optimizer": config.optimizer.__dict__,
                "training": config.training.__dict__,
                "grpo": config.grpo.__dict__,
            },
            group=wandb_group,
            tags=[
                f"lora_r{config.lora.r}",
                config.model.family,
                "base",
            ],
        )

    print("\n" + "=" * 60)
    print("Step 1: Loading Data")
    print("=" * 60)

    # Load GSM8K dataset
    datasets = load_gsm8k_dataset(
        max_samples=config.data.max_samples,
        seed=config.project.seed,
    )
    train_dataset = datasets["train"]
    eval_dataset = datasets["test"]
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    print("\n" + "=" * 60)
    print("Step 2: Loading Model")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model.name,
        dtype=config.model.dtype,
        device_map=config.model.device_map,
        use_flash_attention=False,
    )

    print("\n" + "=" * 60)
    print("Step 3: Configuring LoRA")
    print("=" * 60)

    # Create LoRA config
    lora_config = create_lora_config(
        r=config.lora.r,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
    )

    print("\n" + "=" * 60)
    print("Step 4: Optimizer")
    print("=" * 60)
    print("Using GRPO's built-in AdamW optimizer")

    print("\n" + "=" * 60)
    print("Step 5: Creating GRPO Trainer")
    print("=" * 60)

    # Create GRPO config
    run_name = f"{config.model.family}_{adapter_label}{config.lora.r}"
    output_dir = os.path.join(config.project.output_dir, run_name)

    grpo_config = create_grpo_config(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.optimizer.lr,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        save_total_limit=config.training.save_total_limit,
        num_generations=config.grpo.num_generations,
        max_completion_length=config.grpo.max_completion_length,
        temperature=config.grpo.temperature,
        beta=config.grpo.beta,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        report_to="wandb" if config.logging.use_wandb else "none",
        vllm_gpu_memory_utilization=config.grpo.vllm_gpu_memory_utilization,
    )

    # Create trainer
    trainer = create_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        grpo_config=grpo_config,
        peft_config=lora_config,
        adapter_type=config.lora.adapter_type,
        tiny_lora_u=config.lora.tiny_lora_u,
        tiny_lora_n_tie=config.lora.tiny_lora_n_tie,
        eval_dataset=eval_dataset,
    )

    # Count trainable parameters
    print("\nTrainable parameters after LoRA:")
    count_trainable_parameters(trainer.model)

    print("\n" + "=" * 60)
    print("Step 6: Training")
    print("=" * 60)

    # Train
    train_model(trainer)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {trainer.args.output_dir}")

    # Save structured results
    import json

    result_entry = {
        "model": config.model.name,
        "family": config.model.family,
        "adapter_type": config.lora.adapter_type,
        "lora_rank": config.lora.r,
        "trainable_params": count_trainable_parameters(trainer.model)[0],
        "lr": config.optimizer.lr,
        "epochs": config.training.num_train_epochs,
        "num_generations": config.grpo.num_generations,
        "checkpoint": output_dir,
    }

    # Pull final metrics from trainer log history
    if trainer.state.log_history:
        last = trainer.state.log_history[-1]
        result_entry["train_reward"] = last.get("train/reward")
        result_entry["train_loss"] = last.get("train_loss", last.get("train/loss"))
        # Find last eval entry
        for entry in reversed(trainer.state.log_history):
            if "eval/reward" in entry:
                result_entry["eval_reward"] = entry["eval/reward"]
                break

    results_path = os.path.join(config.project.output_dir, "results.jsonl")
    with open(results_path, "a") as f:
        f.write(json.dumps(result_entry) + "\n")
    print(f"Results appended to {results_path}")

    # Finish wandb
    finish_wandb()


if __name__ == "__main__":
    main()

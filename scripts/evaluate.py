"""Evaluation script for trained models."""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scalingrl.evaluation.aime import AIMEEvaluator
from scalingrl.evaluation.gsm8k import GSM8KEvaluator
from scalingrl.utils import log_metrics, set_seed


def load_checkpoint(checkpoint_path: str):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Check if this is a PEFT checkpoint
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        print("Loading PEFT model...")
        # Load base model first
        base_model_name = None

        # Try to get base model from adapter config
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")

        if base_model_name is None:
            raise ValueError("Could not determine base model name")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = model.merge_and_unload()
    else:
        print("Loading full model...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    print("Model loaded successfully")

    return model, tokenizer


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["gsm8k"],
        help="Datasets to evaluate on (gsm8k, aime2025, gpqa_diamond)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per problem",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save evaluation results as JSON",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load model
    model, tokenizer = load_checkpoint(args.checkpoint)

    # Evaluate on each dataset
    results = {}

    for dataset_name in args.datasets:
        print("\n" + "=" * 60)
        print(f"Evaluating on {dataset_name}")
        print("=" * 60)

        if dataset_name == "gsm8k":
            evaluator = GSM8KEvaluator(
                model=model,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
            )
            dataset_results = evaluator.evaluate()
            results["gsm8k_pass_at_1"] = dataset_results["pass_at_1"]
            results["gsm8k_greedy_accuracy"] = dataset_results["greedy_accuracy"]

        elif dataset_name == "aime2025":
            evaluator = AIMEEvaluator(
                model=model,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
            )
            dataset_results = evaluator.evaluate()
            results[f"{dataset_name}_accuracy"] = dataset_results["accuracy"]

        else:
            print(f"Unknown dataset: {dataset_name}")
            continue

    # Log results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key, value in results.items():
        print(f"{key}: {value:.2%}")
    print("=" * 60)

    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "results": results}, f, indent=2)
        print(f"Results saved to: {args.output_json}")

    # Log to wandb if available
    log_metrics(results)


if __name__ == "__main__":
    main()

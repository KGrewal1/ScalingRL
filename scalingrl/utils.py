"""Logging, reproducibility, and environment utilities."""

import os
import random
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"Random seed set to {seed}")


def log_environment() -> None:
    """Log environment information."""
    print("\n" + "=" * 60)
    print("Environment Information")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print("CUDA available: True")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
    else:
        print("CUDA available: False")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------


def setup_wandb(
    project: str,
    run_name: str,
    config: dict[str, Any],
    group: str | None = None,
    tags: list | None = None,
) -> None:
    """Setup Weights & Biases logging."""
    from dotenv import load_dotenv

    load_dotenv()

    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Install with: pip install wandb")
        return

    api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)

    wandb.init(
        project=project,
        name=run_name,
        config=config,
        group=group,
        tags=tags or [],
    )

    print(f"Wandb initialized: {run_name}")


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to wandb."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def finish_wandb() -> None:
    """Finish wandb run."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

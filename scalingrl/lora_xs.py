"""LoRA-XS and TinyLoRA: extremely parameter-efficient adaptation.

Implements two methods as post-processing steps on standard PEFT LoRA models:

LoRA-XS (Balazy et al. 2025): replaces LoRA's W' = W + AB with W' = W + U Σ R V^T
where U, Σ, V come from truncated SVD of W and only R (r×r) is trainable.
Reduces per-module params from O(dr) to O(r²).

TinyLoRA (Morris et al. 2026): further reduces LoRA-XS by replacing R with a linear
combination of fixed random matrices: R = Σᵢ vᵢPᵢ, where P ∈ R^{u×r×r} are frozen
and v ∈ R^u is trainable. With weight tying across modules (n_tie), total trainable
params can go as low as u (e.g. 1 parameter).
"""

import torch
from peft.tuners.lora.layer import Linear as LoraLinear
from torch import nn


class _LoraAWithR(nn.Module):
    """Wraps frozen lora_A + trainable R into a single module.

    Exposes a `.weight` property pointing to lora_A's weight so that
    PEFT's forward (which accesses `lora_A.weight.dtype`) still works.
    """

    def __init__(self, lora_a: nn.Linear, r_matrix: nn.Module):
        super().__init__()
        self.lora_a = lora_a
        self.r_matrix = r_matrix

    @property
    def weight(self) -> torch.Tensor:
        return self.lora_a.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.r_matrix(self.lora_a(x))


def _compute_svd_factors(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute truncated SVD factors for LoRA-XS initialization.

    Given weight W of shape (d, k), computes truncated SVD and returns:
      - encoder: V_r^T of shape (r, k) — goes into lora_A
      - decoder: U_r @ diag(S_r) of shape (d, r) — goes into lora_B

    The forward path becomes: lora_B(R(lora_A(x))) = (U Σ) R (V^T x)
    """
    # Full SVD on CPU for numerical stability, then truncate
    W = weight.float().cpu()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    # Truncate to rank r
    U_r = U[:, :rank]  # (d, r)
    S_r = S[:rank]  # (r,)
    Vt_r = Vt[:rank, :]  # (r, k)

    # encoder = V_r^T (used as lora_A weight), shape (r, k)
    encoder = Vt_r
    # decoder = U_r @ diag(S_r) (used as lora_B weight), shape (d, r)
    decoder = U_r * S_r.unsqueeze(0)  # broadcast multiply

    return encoder.to(weight.dtype), decoder.to(weight.dtype)


def bake_r_into_a(model: nn.Module) -> None:
    """Temporarily replace frozen A weights with effective R @ A.

    This must be called before PEFT's merge_adapter() so that the merged delta
    is B @ (R @ A) * scaling instead of B @ A * scaling (which omits R).
    """
    adapter_name = "default"
    for module in model.modules():
        if not isinstance(module, LoraLinear):
            continue
        if adapter_name not in module.lora_A:
            continue
        wrapper = module.lora_A[adapter_name]
        if not isinstance(wrapper, _LoraAWithR):
            continue
        # Save original frozen A weight
        wrapper._saved_a_weight = wrapper.lora_a.weight.data.clone()
        # Compute effective weight: R @ A
        if isinstance(wrapper.r_matrix, _TinyLoRAMapping):
            R = torch.einsum("i,ijk->jk", wrapper.r_matrix.v, wrapper.r_matrix.P)
            effective = R @ wrapper.lora_a.weight.data
        else:
            # LoRA-XS: r_matrix is nn.Linear with weight (r, r)
            effective = wrapper.r_matrix.weight @ wrapper.lora_a.weight.data
        wrapper.lora_a.weight.data.copy_(effective)


def unbake_r_from_a(model: nn.Module) -> None:
    """Restore original frozen A weights after merge/unmerge cycle."""
    adapter_name = "default"
    for module in model.modules():
        if not isinstance(module, LoraLinear):
            continue
        if adapter_name not in module.lora_A:
            continue
        wrapper = module.lora_A[adapter_name]
        if not isinstance(wrapper, _LoraAWithR):
            continue
        if hasattr(wrapper, "_saved_a_weight"):
            wrapper.lora_a.weight.data.copy_(wrapper._saved_a_weight)
            del wrapper._saved_a_weight


def apply_lora_xs(model: nn.Module, rank: int) -> nn.Module:
    """Convert a PEFT LoRA model to LoRA-XS in-place.

    For each LoRA layer:
    1. Extract the base weight W and compute truncated SVD
    2. Replace lora_A weights with V_r^T (frozen)
    3. Replace lora_B weights with U_r @ diag(S_r) (frozen)
    4. Insert a trainable r×r linear layer R between A and B

    The R matrix is inserted by replacing lora_A with a wrapper module that
    chains lora_A → R, so the existing forward `lora_B(lora_A(x))` becomes
    `lora_B(R(lora_A(x)))` without needing to modify the forward method.
    """
    adapter_name = "default"

    # Collect modules first so we can show progress
    lora_modules: list[tuple[str, LoraLinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear) and adapter_name in module.lora_A:
            lora_modules.append((name, module))

    total = len(lora_modules)
    print(f"LoRA-XS: computing SVD for {total} modules (this may take a while)...")

    for idx, (name, module) in enumerate(lora_modules):
        lora_a = module.lora_A[adapter_name]  # nn.Linear(k, r, bias=False)
        lora_b = module.lora_B[adapter_name]  # nn.Linear(r, d, bias=False)

        # Get the base (pretrained) weight
        base_weight = module.get_base_layer().weight.data  # (d, k)

        # Compute SVD factors
        print(f"  SVD [{idx + 1}/{total}] {name} {tuple(base_weight.shape)}", flush=True)
        encoder, decoder = _compute_svd_factors(base_weight, rank)
        # encoder: (r, k), decoder: (d, r)

        device = lora_a.weight.device

        # Set lora_A weight to V_r^T and freeze
        with torch.no_grad():
            lora_a.weight.copy_(encoder.to(device))
        lora_a.weight.requires_grad = False

        # Set lora_B weight to U_r @ diag(S_r) and freeze
        with torch.no_grad():
            lora_b.weight.copy_(decoder.to(device))
        lora_b.weight.requires_grad = False

        # Create the trainable r×r mapping
        r_matrix = nn.Linear(rank, rank, bias=False, device=device, dtype=lora_a.weight.dtype)
        nn.init.normal_(r_matrix.weight, mean=0, std=1e-5)

        # Replace lora_A with a wrapper that chains lora_A → R, so the existing
        # forward `lora_B(lora_A(x))` becomes `lora_B(R(lora_A(x)))`.
        # The wrapper exposes .weight so PEFT's dtype casting still works.
        module.lora_A[adapter_name] = _LoraAWithR(lora_a, r_matrix)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA-XS: converted {total} modules (r={rank}, r²={rank**2})")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Expected: ~{total * rank**2:,} (from R matrices)")

    return model


# ---------------------------------------------------------------------------
# TinyLoRA
# ---------------------------------------------------------------------------


class _TinyLoRAMapping(nn.Module):
    """Replaces a trainable r×r matrix with R = Σᵢ vᵢPᵢ.

    P ∈ R^{u×r×r} are fixed random projection matrices, and v ∈ R^u is the
    only trainable parameter. The effective R matrix is the weighted sum of
    the u basis matrices.
    """

    def __init__(
        self,
        rank: int,
        u: int,
        v: nn.Parameter,
        device: torch.device,
        dtype: torch.dtype,
        seed: int = 0,
    ):
        super().__init__()
        # Fixed random projection bases — seeded for reproducibility
        gen = torch.Generator(device="cpu").manual_seed(seed)
        # Kaiming-style init scaled by 1/u so the sum has reasonable magnitude
        P = torch.randn(u, rank, rank, generator=gen, dtype=dtype) / (u * rank) ** 0.5
        self.register_buffer("P", P.to(device))
        # v is a shared parameter (passed in from outside for weight tying)
        self.v = v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # R = Σᵢ vᵢPᵢ, shape (r, r)
        R = torch.einsum("i,ijk->jk", self.v, self.P)
        return x @ R.T


def apply_tiny_lora(
    model: nn.Module,
    rank: int,
    u: int = 1,
    n_tie: int | None = None,
) -> nn.Module:
    """Convert a PEFT LoRA model to TinyLoRA in-place.

    Like LoRA-XS, initializes frozen A/B from truncated SVD. But instead of a
    trainable r×r matrix R, uses R = Σᵢ vᵢPᵢ where P are fixed random bases
    and v ∈ R^u is the only trainable vector.

    Args:
        model: A PEFT LoRA model (after get_peft_model).
        rank: SVD truncation rank (the paper recommends r=2).
        u: Projection dimension — number of random basis matrices per module.
            Each module has u trainable parameters.
        n_tie: Weight tying factor — number of modules sharing a single v.
            Default (None) means no tying (each module has its own v).
            Set to total number of LoRA modules for full tying (just u params).
    """
    adapter_name = "default"

    # Collect all LoRA modules
    lora_modules: list[tuple[str, LoraLinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear) and adapter_name in module.lora_A:
            lora_modules.append((name, module))

    if not lora_modules:
        raise ValueError("No LoRA modules found in model")

    total_modules = len(lora_modules)

    # Default: no tying (each module gets its own v)
    if n_tie is None:
        n_tie = 1

    # Determine device/dtype from first module
    first_a = lora_modules[0][1].lora_A[adapter_name]
    device = first_a.weight.device
    dtype = first_a.weight.dtype

    # Create shared v parameters — one per tying group
    n_groups = (total_modules + n_tie - 1) // n_tie
    shared_vs = [nn.Parameter(torch.zeros(u, device=device, dtype=dtype)) for _ in range(n_groups)]

    print(f"TinyLoRA: computing SVD for {total_modules} modules (this may take a while)...")

    for idx, (name, module) in enumerate(lora_modules):
        lora_a = module.lora_A[adapter_name]
        lora_b = module.lora_B[adapter_name]
        base_weight = module.get_base_layer().weight.data

        # SVD initialization (same as LoRA-XS)
        print(f"  SVD [{idx + 1}/{total_modules}] {name} {tuple(base_weight.shape)}", flush=True)
        encoder, decoder = _compute_svd_factors(base_weight, rank)

        with torch.no_grad():
            lora_a.weight.copy_(encoder.to(device))
            lora_b.weight.copy_(decoder.to(device))
        lora_a.weight.requires_grad = False
        lora_b.weight.requires_grad = False

        # Create TinyLoRA mapping with shared v
        group_idx = idx // n_tie
        v = shared_vs[group_idx]
        tiny_r = _TinyLoRAMapping(
            rank=rank,
            u=u,
            v=v,
            device=device,
            dtype=dtype,
            seed=idx,  # different random P per module
        )

        module.lora_A[adapter_name] = _LoraAWithR(lora_a, tiny_r)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TinyLoRA: converted {total_modules} modules (r={rank}, u={u}, n_tie={n_tie})")
    print(f"  Tying groups: {n_groups} (sharing v across {n_tie} modules each)")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Expected: {n_groups * u} ({n_groups} groups × {u} params)")

    return model

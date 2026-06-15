"""PyTorch reference for fused RMSNorm + RoPE — correctness ground truth."""

import torch
from evokernel.kernels.rmsnorm.reference import run as rmsnorm_ref
from evokernel.kernels.rope.reference import run as rope_ref


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_heads: int = 32,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfused reference: RMSNorm then RoPE separately."""
    seq_len, N = x.shape
    head_dim = N // (2 * n_heads)

    # RMSNorm over the full hidden dim
    x_norm = rmsnorm_ref(x, weight, eps)  # [seq_len, N]

    # Split into Q and K
    q = x_norm[:, :n_heads * head_dim].reshape(seq_len, n_heads, head_dim)
    k = x_norm[:, n_heads * head_dim:].reshape(seq_len, n_heads, head_dim)

    # Apply RoPE
    q_out, k_out = rope_ref(q, k, cos, sin)
    return q_out, k_out


def make_test_inputs(
    seq_len: int = 512,
    n_heads: int = 32,
    head_dim: int = 128,
    device: str = "cuda",
):
    torch.manual_seed(42)
    N = 2 * n_heads * head_dim
    x = torch.randn(seq_len, N, dtype=torch.float16, device=device)
    weight = torch.ones(N, dtype=torch.float16, device=device)

    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    cos_h = freqs.cos().half()
    sin_h = freqs.sin().half()
    cos = torch.cat([cos_h, cos_h], dim=-1)
    sin = torch.cat([sin_h, sin_h], dim=-1)

    return {"x": x, "weight": weight, "cos": cos, "sin": sin, "n_heads": n_heads}

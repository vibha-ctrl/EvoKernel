"""PyTorch reference implementation for RoPE — used as correctness ground truth."""

import torch


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch RoPE. Operates in float32 for precision."""
    q_f = q.float()
    k_f = k.float()
    # cos/sin: [seq_len, head_dim] -> [seq_len, 1, head_dim] for broadcast
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)

    half = q_f.shape[-1] // 2
    q0, q1 = q_f[..., :half], q_f[..., half:]
    k0, k1 = k_f[..., :half], k_f[..., half:]
    cos_h, sin_h = cos[..., :half], sin[..., :half]

    q_rot = torch.cat([q0 * cos_h - q1 * sin_h, q0 * sin_h + q1 * cos_h], dim=-1)
    k_rot = torch.cat([k0 * cos_h - k1 * sin_h, k0 * sin_h + k1 * cos_h], dim=-1)

    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def make_test_inputs(
    seq_len: int = 512,
    n_heads: int = 32,
    head_dim: int = 128,
    device: str = "cuda",
):
    torch.manual_seed(42)
    q = torch.randn(seq_len, n_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(seq_len, n_heads, head_dim, dtype=torch.float16, device=device)
    # Build sinusoidal freqs
    half = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().half()
    sin = freqs.sin().half()
    # Repeat to full head_dim
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return {"q": q, "k": k, "cos": cos, "sin": sin}

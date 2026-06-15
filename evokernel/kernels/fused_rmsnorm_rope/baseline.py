"""
Baseline fused RMSNorm + RoPE Triton kernel.

Fuses both operations in a single kernel pass:
  1. RMSNorm on input X
  2. Apply RoPE to the normalized Q/K head slices

Avoids writing intermediate normalized tensor to DRAM.
Interface contract: run(x, weight, cos, sin, n_heads, eps) -> (q_out, k_out)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rmsnorm_rope_kernel(
    X,
    W,
    Cos,
    Sin,
    Q_out,
    K_out,
    N,
    n_heads,
    head_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one row (one token).
    Row layout: [q_head_0 | q_head_1 | ... | k_head_0 | k_head_1 | ...]
    Total width N = 2 * n_heads * head_dim  (Q + K concatenated)
    """
    row = tl.program_id(0)
    X_row = X + row * N

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Step 1: load full row and compute RMSNorm
    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    x_norm = x * rstd * w  # normalized, still float32

    # Step 2: apply RoPE to Q portion (first n_heads * head_dim elements)
    q_size = n_heads * head_dim
    half_dim = head_dim // 2

    seq_idx = row
    cos_base = Cos + seq_idx * head_dim
    sin_base = Sin + seq_idx * head_dim

    # Process each head's Q
    for h in tl.static_range(0, 1):  # unrolled at generation time
        for hi in range(n_heads):
            h_offset = hi * head_dim
            h_cols = tl.arange(0, BLOCK_SIZE // (n_heads * 2))
            h_mask = h_cols < half_dim

            q0 = tl.load(X_row + h_offset + h_cols,
                          mask=h_mask, other=0.0).to(tl.float32)
            q1 = tl.load(X_row + h_offset + h_cols + half_dim,
                          mask=h_mask, other=0.0).to(tl.float32)
            rms_q0 = q0 * rstd * tl.load(W + h_offset + h_cols, mask=h_mask, other=1.0).to(tl.float32)
            rms_q1 = q1 * rstd * tl.load(W + h_offset + h_cols + half_dim, mask=h_mask, other=1.0).to(tl.float32)

            cos_v = tl.load(cos_base + h_cols, mask=h_mask, other=1.0).to(tl.float32)
            sin_v = tl.load(sin_base + h_cols, mask=h_mask, other=0.0).to(tl.float32)

            rq0 = rms_q0 * cos_v - rms_q1 * sin_v
            rq1 = rms_q0 * sin_v + rms_q1 * cos_v

            out_base = Q_out + row * q_size + h_offset
            tl.store(out_base + h_cols, rq0.to(tl.float16), mask=h_mask)
            tl.store(out_base + h_cols + half_dim, rq1.to(tl.float16), mask=h_mask)

    # Process each head's K (starts at q_size offset in X)
    for hi in range(n_heads):
        h_offset = q_size + hi * head_dim
        h_cols = tl.arange(0, BLOCK_SIZE // (n_heads * 2))
        h_mask = h_cols < half_dim

        k0 = tl.load(X_row + h_offset + h_cols,
                      mask=h_mask, other=0.0).to(tl.float32)
        k1 = tl.load(X_row + h_offset + h_cols + half_dim,
                      mask=h_mask, other=0.0).to(tl.float32)
        rms_k0 = k0 * rstd * tl.load(W + h_offset + h_cols, mask=h_mask, other=1.0).to(tl.float32)
        rms_k1 = k1 * rstd * tl.load(W + h_offset + h_cols + half_dim, mask=h_mask, other=1.0).to(tl.float32)

        cos_v = tl.load(cos_base + h_cols, mask=h_mask, other=1.0).to(tl.float32)
        sin_v = tl.load(sin_base + h_cols, mask=h_mask, other=0.0).to(tl.float32)

        rk0 = rms_k0 * cos_v - rms_k1 * sin_v
        rk1 = rms_k0 * sin_v + rms_k1 * cos_v

        out_base = K_out + row * q_size + hi * head_dim
        tl.store(out_base + h_cols, rk0.to(tl.float16), mask=h_mask)
        tl.store(out_base + h_cols + half_dim, rk1.to(tl.float16), mask=h_mask)


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_heads: int = 32,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x:       [seq_len, N]    float16  where N = 2 * n_heads * head_dim
        weight:  [N]             float16
        cos:     [seq_len, head_dim] float16
        sin:     [seq_len, head_dim] float16
        n_heads: number of attention heads

    Returns:
        (q_out, k_out): each [seq_len, n_heads, head_dim] float16
    """
    assert x.is_cuda
    seq_len, N = x.shape
    head_dim = N // (2 * n_heads)

    q_out = torch.empty(seq_len, n_heads, head_dim, dtype=torch.float16, device=x.device)
    k_out = torch.empty(seq_len, n_heads, head_dim, dtype=torch.float16, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(N)

    _fused_rmsnorm_rope_kernel[(seq_len,)](
        x, weight, cos, sin, q_out, k_out,
        N, n_heads, head_dim, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return q_out, k_out


KERNEL_TYPE = "fused_rmsnorm_rope"
DESCRIPTION = "Baseline fused RMSNorm+RoPE — single pass, no pipelining"

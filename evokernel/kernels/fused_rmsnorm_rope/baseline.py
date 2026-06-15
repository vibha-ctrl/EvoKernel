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
    HALF_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X + row * N

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    q_size = n_heads * head_dim

    cos_base = Cos + row * HALF_DIM
    sin_base = Sin + row * HALF_DIM

    h_cols = tl.arange(0, HALF_DIM)

    cos_v = tl.load(cos_base + h_cols, mask=h_cols < HALF_DIM, other=1.0).to(tl.float32)
    sin_v = tl.load(sin_base + h_cols, mask=h_cols < HALF_DIM, other=0.0).to(tl.float32)

    for hi in range(n_heads):
        h_offset = hi * head_dim
        h_mask = h_cols < HALF_DIM

        q0 = tl.load(X_row + h_offset + h_cols, mask=h_mask, other=0.0).to(tl.float32)
        q1 = tl.load(X_row + h_offset + h_cols + HALF_DIM, mask=h_mask, other=0.0).to(tl.float32)
        rms_q0 = q0 * rstd * tl.load(W + h_offset + h_cols, mask=h_mask, other=1.0).to(tl.float32)
        rms_q1 = q1 * rstd * tl.load(W + h_offset + h_cols + HALF_DIM, mask=h_mask, other=1.0).to(tl.float32)

        rq0 = rms_q0 * cos_v - rms_q1 * sin_v
        rq1 = rms_q0 * sin_v + rms_q1 * cos_v

        out_base = Q_out + row * q_size + h_offset
        tl.store(out_base + h_cols, rq0.to(tl.float16), mask=h_mask)
        tl.store(out_base + h_cols + HALF_DIM, rq1.to(tl.float16), mask=h_mask)

    for hi in range(n_heads):
        h_offset = hi * head_dim
        h_mask = h_cols < HALF_DIM

        k0 = tl.load(X_row + q_size + h_offset + h_cols, mask=h_mask, other=0.0).to(tl.float32)
        k1 = tl.load(X_row + q_size + h_offset + h_cols + HALF_DIM, mask=h_mask, other=0.0).to(tl.float32)
        rms_k0 = k0 * rstd * tl.load(W + q_size + h_offset + h_cols, mask=h_mask, other=1.0).to(tl.float32)
        rms_k1 = k1 * rstd * tl.load(W + q_size + h_offset + h_cols + HALF_DIM, mask=h_mask, other=1.0).to(tl.float32)

        rk0 = rms_k0 * cos_v - rms_k1 * sin_v
        rk1 = rms_k0 * sin_v + rms_k1 * cos_v

        out_base = K_out + row * q_size + hi * head_dim
        tl.store(out_base + h_cols, rk0.to(tl.float16), mask=h_mask)
        tl.store(out_base + h_cols + HALF_DIM, rk1.to(tl.float16), mask=h_mask)


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_heads: int = 32,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda
    seq_len, N = x.shape
    head_dim = N // (2 * n_heads)

    q_out = torch.empty(seq_len, n_heads, head_dim, dtype=torch.float16, device=x.device)
    k_out = torch.empty(seq_len, n_heads, head_dim, dtype=torch.float16, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(N)
    HALF_DIM = head_dim // 2

    _fused_rmsnorm_rope_kernel[(seq_len,)](
        x, weight, cos, sin, q_out, k_out,
        N, n_heads, head_dim, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HALF_DIM=HALF_DIM,
        num_warps=4,
        num_stages=1,
    )
    return q_out, k_out


KERNEL_TYPE = "fused_rmsnorm_rope"
DESCRIPTION = "Baseline fused RMSNorm+RoPE — single pass, no pipelining"

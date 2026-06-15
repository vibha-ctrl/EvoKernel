import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    Q,
    K,
    Cos,
    Sin,
    Q_out,
    K_out,
    seq_len,
    n_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_idx = pid // n_heads
    head_idx = pid % n_heads

    half_dim = head_dim // 2
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < half_dim

    base_q = Q + seq_idx * n_heads * head_dim + head_idx * head_dim
    base_k = K + seq_idx * n_heads * head_dim + head_idx * head_dim
    base_cos = Cos + seq_idx * head_dim
    base_sin = Sin + seq_idx * head_dim

    q0 = tl.load(base_q + cols, mask=mask, other=0.0).to(tl.float32)
    q1 = tl.load(base_q + cols + half_dim, mask=mask, other=0.0).to(tl.float32)
    k0 = tl.load(base_k + cols, mask=mask, other=0.0).to(tl.float32)
    k1 = tl.load(base_k + cols + half_dim, mask=mask, other=0.0).to(tl.float32)

    cos = tl.load(base_cos + cols, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(base_sin + cols, mask=mask, other=0.0).to(tl.float32)

    rq0 = q0 * cos - q1 * sin
    rq1 = q0 * sin + q1 * cos
    rk0 = k0 * cos - k1 * sin
    rk1 = k0 * sin + k1 * cos

    base_qo = Q_out + seq_idx * n_heads * head_dim + head_idx * head_dim
    base_ko = K_out + seq_idx * n_heads * head_dim + head_idx * head_dim

    tl.store(base_qo + cols, rq0.to(tl.float16), mask=mask)
    tl.store(base_qo + cols + half_dim, rq1.to(tl.float16), mask=mask)
    tl.store(base_ko + cols, rk0.to(tl.float16), mask=mask)
    tl.store(base_ko + cols + half_dim, rk1.to(tl.float16), mask=mask)


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_cuda
    seq_len, n_heads, head_dim = q.shape
    half_dim = head_dim // 2

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    grid = (seq_len * n_heads,)
    BLOCK_SIZE = triton.next_power_of_2(half_dim)

    _rope_kernel[grid](
        q, k, cos, sin, q_out, k_out,
        seq_len, n_heads, head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return q_out, k_out


KERNEL_TYPE = "rope"
DESCRIPTION = "Baseline RoPE — per-(seq, head) program, num_warps=4"

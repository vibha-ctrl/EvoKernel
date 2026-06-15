"""
Baseline RMSNorm Triton kernel.

This is the untuned starting point for the evolutionary search.
Interface contract: every candidate must expose run(x, weight, eps) -> Tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X,
    W,
    Y,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X + row * N
    Y_row = Y + row * N

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    y = (x * rstd * w).to(tl.float16)
    tl.store(Y_row + cols, y, mask=mask)


def run(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Args:
        x:      [M, N] float16
        weight: [N]    float16
        eps:    float

    Returns:
        y: [M, N] float16
    """
    assert x.is_cuda and weight.is_cuda
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _rmsnorm_kernel[(M,)](
        x, weight, y, N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return y


KERNEL_TYPE = "rmsnorm"
DESCRIPTION = "Baseline RMSNorm — single-pass, no pipelining, num_warps=4"

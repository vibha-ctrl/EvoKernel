# rmsnorm — Search Results
*Last updated: 2026-06-15 10:28 UTC*

## Progress

| Metric | Value |
|--------|-------|
| Baseline latency | 31.7 µs |
| Current best | **29.4 µs** |
| Current speedup | **1.08x** |
| Generations completed | 3 |
| Candidates evaluated | 35 (33 passed correctness) |

## Generation-by-Generation

```
Latency reduction over generations:

  Gen  0 | ██████████████████████████████████    31.4 µs  (+0.8%)
  Gen  1 | ████████████████████████████████      29.8 µs  (+6.1%)
  Gen  2 | ████████████████████████████████      29.4 µs  (+7.2%)
  Gen  3 | ████████████████████████████████      29.5 µs  (+6.8%)
```

| Generation | Best Latency (µs) | vs Baseline | Candidates | Passed |
|------------|-------------------|-------------|------------|--------|
| 0 | 31.4 | +0.8% | 5 | 3 |
| 1 | 29.8 | +6.1% | 10 | 10 |
| 2 | 29.4 | +7.2% | 10 | 10 |
| 3 | **29.5** | +6.8% | 10 | 10 |

## Current Best Kernel

**Candidate:** `gen2_4352896d`  
**Generation:** 2  
**Latency:** 29.4 µs  
**Throughput:** 1141 GB/s  
**BW utilization:** 57%  

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel_v4(
    X, W, Y, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row = X + row * N
    Y_row = Y + row * N
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    
    sq_sum = tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(sq_sum / N + eps)
    
    # Fuse multiply operations
    xw = x * w
    y = (xw * rstd).to(tl.float16)
    tl.store(Y_row + cols, y, mask=mask)

def run(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _rmsnorm_kernel_v4[(M,)](
        x, weight, y, N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return y
```
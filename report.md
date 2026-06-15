# EvoKernel Report — rmsnorm

## Summary

| Metric | Value |
|--------|-------|
| Kernel type | `rmsnorm` |
| Baseline latency | N/A |
| Best latency | **29.4 µs** |
| Speedup | N/A |
| Best candidate | `gen2_4352896d` |
| Generations run | 4 |
| Total candidates evaluated | 35 |

## Performance Progression

```
Latency (µs) by generation:

  Gen  0 | ████████████████████████████████████████ 31.4
  Gen  1 | █████████████████████████████████████    29.8
  Gen  2 | █████████████████████████████████████    29.4
  Gen  3 | █████████████████████████████████████    29.5
```

## Generation-by-Generation Results

| Generation | Best Latency (µs) | Candidates | Passed Verify |
|------------|-------------------|------------|----------------|
| 0 | 31.4 | 5 | 3 |
| 1 | 29.8 | 10 | 10 |
| 2 | 29.4 | 10 | 10 |
| 3 | 29.5 | 10 | 10 |

## Best Kernel Configuration

**Candidate:** `gen2_4352896d`  
**Latency:** 29.4 µs  
**Throughput:** 1141 GB/s  
**Bandwidth utilization:** 57%  

### Triton Parameters

| Parameter | Value |
|-----------|-------|
| `num_warps` | None |
| `num_stages` | None |
| `shared_mem_bytes` | None |
| `register_count` | None |
| `theoretical_occupancy` | None% |

### Nsight Compute Metrics

| Metric | Value |
|--------|-------|
| SM throughput | None% |
| DRAM utilization | None% |
| L1 hit rate | None% |
| Stall (memory dependency) | None% |
| Stall (long scoreboard) | None% |

## Best Kernel Source Code

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

## Optimization Journey

- **Generation 0**: 31.4 µs (baseline)
- **Generation 1**: 29.8 µs (+5.3%)
- **Generation 2**: 29.4 µs (+1.2%)
- **Generation 3**: 29.5 µs (-0.4%)
# EvoKernel Report — fused_rmsnorm_rope

## Summary

| Metric | Value |
|--------|-------|
| Kernel type | `fused_rmsnorm_rope` |
| Baseline latency | N/A |
| Best latency | **24.4 µs** |
| Speedup | N/A |
| Best candidate | `gen67_6fba567c` |
| Tool calls used | 27 |
| Total candidates evaluated | 29 |

## Performance Progression

```
Latency (µs) by generation:

  Call  0 | █████████████████████                    47.4
  Call  3 | █████████████████████                    47.5
  Call  4 | █████████████████████                    46.8
  Call  5 | ██████████████                           31.4
  Call 13 | █████████████████████                    47.3
  Call 14 | ██████████████                           32.1
  Call 15 | █████████████████████████████████        74.3
  Call 22 | ████████████████████████████████████████ 88.9
  Call 23 | ██████████████                           31.5
  Call 24 | ███████████████████                      42.6
  Call 32 | █████████████████████                    47.6
  Call 33 | ███████████████████                      44.3
  Call 34 | █████████████                            29.5
  Call 41 | ██████████████                           32.9
  Call 43 | ████████████                             28.6
  Call 49 | ███████████████                          33.7
  Call 50 | ██████████████                           32.4
  Call 51 | ████████████████████████████████         71.7
  Call 59 | ████████████                             28.0
  Call 60 | █████████████████                        38.6
  Call 65 | ████████████                             27.2
  Call 67 | ██████████                               24.4
  Call 75 | ███████████                              25.5
  Call 76 | ███████████████                          34.5
```

## Candidates by Tool Call

| Tool Call | Best Latency (µs) | Candidates | Passed Verify |
|-----------|-------------------|------------|----------------|
| 0 | 47.4 | 3 | 1 |
| 3 | 47.5 | 1 | 1 |
| 4 | 46.8 | 1 | 1 |
| 5 | 31.4 | 1 | 1 |
| 13 | 47.3 | 1 | 1 |
| 14 | 32.1 | 1 | 1 |
| 15 | 74.3 | 1 | 1 |
| 22 | 88.9 | 1 | 1 |
| 23 | 31.5 | 1 | 1 |
| 24 | 42.6 | 1 | 1 |
| 32 | 47.6 | 1 | 1 |
| 33 | 44.3 | 1 | 1 |
| 34 | 29.5 | 1 | 1 |
| 41 | 32.9 | 1 | 1 |
| 42 | — | 1 | 0 |
| 43 | 28.6 | 1 | 1 |
| 49 | 33.7 | 1 | 1 |
| 50 | 32.4 | 1 | 1 |
| 51 | 71.7 | 1 | 1 |
| 59 | 28.0 | 1 | 1 |
| 60 | 38.6 | 1 | 1 |
| 65 | 27.2 | 1 | 1 |
| 66 | — | 1 | 0 |
| 67 | 24.4 | 1 | 1 |
| 74 | — | 1 | 0 |
| 75 | 25.5 | 1 | 1 |
| 76 | 34.5 | 1 | 1 |

## Best Kernel Configuration

**Candidate:** `gen67_6fba567c`  
**Latency:** 24.4 µs  

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
def _fused_rmsnorm_rope_kernel_precompute_rstd(
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
    HEADS_PER_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head_block = tl.program_id(1)
    
    X_row = X + row * N

    # Compute RMSNorm statistics
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    # Pre-compute x * rstd for the entire row
    x_normalized = x * rstd

    q_size = n_heads * head_dim

    cos_base = Cos + row * head_dim
    sin_base = Sin + row * head_dim

    h_cols = tl.arange(0, HALF_DIM)
    h_mask = h_cols < HALF_DIM

    cos_v = tl.load(cos_base + h_cols, mask=h_mask, other=1.0).to(tl.float32)
    sin_v = tl.load(sin_base + h_cols, mask=h_mask, other=0.0).to(tl.float32)

    # Process multiple heads per block with static_range for compile-time unrolling
    start_head = head_block * HEADS_PER_BLOCK
    
    # Use tl.static_range for compile-time loop unrolling
    for hi_offset in tl.static_range(HEADS_PER_BLOCK):
        hi = start_head + hi_offset
        # Runtime check for valid head index
        if hi < n_heads:
            h_offset = hi * head_dim

            # Load pre-normalized Q data - first half and second half
            q0_norm = tl.load(X_row + h_offset + h_cols, mask=h_mask, other=0.0).to(tl.float32) * rstd
            q1_norm = tl.load(X_row + h_offset + h_cols + HALF_DIM, mask=h_mask, other=0.0).to(tl.float32) * rstd
            
            # Load pre-normalized K data (immediately after Q for better locality)
            k0_norm = tl.load(X_row + q_size + h_offset + h_cols, mask=h_mask, other=0.0).to(tl.float32) * rstd
            k1_norm = tl.load(X_row + q_size + h_offset + h_cols + HALF_DIM, mask=h_mask, other=0.0).to(tl.float32) * rstd
            
            # Load Q weights
            w_q0 = tl.load(W + h_offset + h_cols, mask=h_mask, other=1.0).to(tl.float32)
            w_q1 = tl.load(W + h_offset + h_cols + HALF_DIM, mask=h_mask, other=1.0).to(tl.float32)
            
            # Load K weights
            w_k0 = tl.load(W + q_size + h_offset + h_cols, mask=h_mask, other=1.0).to(tl.float32)
            w_k1 = tl.load(W + q_size + h_offset + h_cols + HALF_DIM, mask=h_mask, other=1.0).to(tl.float32)
            
            # Apply weights to pre-normalized values (rstd already applied)
            rms_q0 = q0_norm * w_q0
            rms_q1 = q1_norm * w_q1
            
            rms_k0 = k0_norm * w_k0
            rms_k1 = k1_norm * w_k1

            # Apply RoPE to Q
            rq0 = rms_q0 * cos_v - rms_q1 * sin_v
            rq1 = rms_q0 * sin_v + rms_q1 * cos_v
            
            # Apply RoPE to K
            rk0 = rms_k0 * cos_v - rms_k1 * sin_v
            rk1 = rms_k0 * sin_v + rms_k1 * cos_v

            # Store Q output
            q_out_base = Q_out + row * q_size + h_offset
            tl.store(q_out_base + h_cols, rq0.to(tl.float16), mask=h_mask)
            tl.store(q_out_base + h_cols + HALF_DIM, rq1.to(tl.float16), mask=h_mask)
            
            # Store K output
            k_out_base = K_out + row * q_size + h_offset
            tl.store(k_out_base + h_cols, rk0.to(tl.float16), mask=h_mask)
            tl.store(k_out_base + h_cols + HALF_DIM, rk1.to(tl.float16), mask=h_mask)


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
    HEADS_PER_BLOCK = 8  # Process 8 heads per thread block with static unrolling
    
    num_head_blocks = (n_heads + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK

    grid = (seq_len, num_head_blocks)
    
    _fused_rmsnorm_rope_kernel_precompute_rstd[grid](
        x, weight, cos, sin, q_out, k_out,
        N, n_heads, head_dim, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HALF_DIM=HALF_DIM,
        HEADS_PER_BLOCK=HEADS_PER_BLOCK,
        num_warps=2,
        num_stages=2,
    )
    return q_out, k_out


KERNEL_TYPE = "fused_rmsnorm_rope"
DESCRIPTION = "Pre-compute x * rstd once for the entire row before applying weights in the head loop"
```

## Optimization Journey

- **Tool call 0**: 47.4 µs (baseline)
- **Tool call 4**: 46.8 µs — new best (+1.3% vs baseline)
- **Tool call 5**: 31.4 µs — new best (+33.7% vs baseline)
- **Tool call 34**: 29.5 µs — new best (+37.8% vs baseline)
- **Tool call 43**: 28.6 µs — new best (+39.6% vs baseline)
- **Tool call 59**: 28.0 µs — new best (+41.0% vs baseline)
- **Tool call 65**: 27.2 µs — new best (+42.7% vs baseline)
- **Tool call 67**: 24.4 µs — new best (+48.5% vs baseline)
# EvoKernel Report — fused_rmsnorm_rope

## Performance Progression

```
Latency (µs) by generation:

  Call  0 | █████████████████████                    46.8
  Call  2 | ██████████████████████                   47.5
  Call  3 | ███████████████████                      43.0
  Call  4 | ███████████████████                      42.9
  Call  5 | ██████████████                           31.4
  Call 11 | █████████████████████                    47.1
  Call 12 | ███████████████████                      42.4
  Call 13 | █████████████████████                    46.7
  Call 14 | ██████████████                           32.1
  Call 15 | ██████████████████████████████████       74.3
  Call 19 | ██████████████████████                   48.8
  Call 21 | █████████████████████                    46.9
  Call 22 | ████████████████████                     43.2
  Call 23 | ██████████████                           31.5
  Call 24 | ███████████████████                      42.6
  Call 25 | ████████████                             26.9
  Call 28 | ███████████████                          33.8
  Call 31 | █████████████                            28.1
  Call 32 | ██████████████                           30.5
  Call 33 | ████████████                             26.4
  Call 34 | █████████████                            29.5
  Call 35 | ██████████                               23.5
  Call 37 | ██████████████████                       40.0
  Call 38 | ███████████████                          32.9
  Call 40 | ███████████                              25.6
  Call 41 | ████████████                             27.7
  Call 42 | ██████████                               23.5
  Call 43 | █████████████                            28.0
  Call 44 | ███████████████                          33.8
  Call 47 | ████████████████                         36.0
  Call 49 | ███████████████                          33.7
  Call 50 | ███████████████                          32.4
  Call 51 | ███████████                              25.7
  Call 52 | ███████████████████████████████████      75.9
  Call 53 | █████████████████████████████████        73.1
  Call 54 | ███████████████████████████████████████  85.6
  Call 55 | █████████                                20.6
  Call 58 | █████████████████████████████████████    81.6
  Call 59 | █████████████                            28.0
  Call 60 | █████████████                            28.1
  Call 61 | ███████████                              25.1
  Call 62 | ██████████                               23.6
  Call 64 | ██████████                               22.5
  Call 65 | ████████████                             27.2
  Call 67 | ███████████                              24.4
  Call 69 | ████████████████████████████████████████ 86.1
  Call 70 | ███████████████                          33.9
  Call 71 | █████████████████████████████████        72.1
  Call 75 | ███████████                              25.5
  Call 76 | ████████████████                         34.5
```

## Candidates by Tool Call

| Tool Call | Best Latency (µs) | Candidates | Passed Verify |
|-----------|-------------------|------------|----------------|
| 0 | 46.8 | 6 | 4 |
| 2 | 47.5 | 1 | 1 |
| 3 | 43.0 | 4 | 4 |
| 4 | 42.9 | 4 | 4 |
| 5 | 31.4 | 3 | 3 |
| 11 | 47.1 | 1 | 1 |
| 12 | 42.4 | 2 | 2 |
| 13 | 46.7 | 4 | 3 |
| 14 | 32.1 | 3 | 2 |
| 15 | 74.3 | 1 | 1 |
| 17 | — | 1 | 0 |
| 19 | 48.8 | 1 | 1 |
| 20 | — | 1 | 0 |
| 21 | 46.9 | 2 | 2 |
| 22 | 43.2 | 3 | 2 |
| 23 | 31.5 | 1 | 1 |
| 24 | 42.6 | 2 | 2 |
| 25 | 26.9 | 1 | 1 |
| 27 | — | 1 | 0 |
| 28 | 33.8 | 2 | 2 |
| 31 | 28.1 | 1 | 1 |
| 32 | 30.5 | 2 | 2 |
| 33 | 26.4 | 3 | 3 |
| 34 | 29.5 | 3 | 3 |
| 35 | 23.5 | 1 | 1 |
| 37 | 40.0 | 1 | 1 |
| 38 | 32.9 | 1 | 1 |
| 40 | 25.6 | 1 | 1 |
| 41 | 27.7 | 2 | 2 |
| 42 | 23.5 | 2 | 1 |
| 43 | 28.0 | 2 | 2 |
| 44 | 33.8 | 1 | 1 |
| 47 | 36.0 | 1 | 1 |
| 49 | 33.7 | 1 | 1 |
| 50 | 32.4 | 1 | 1 |
| 51 | 25.7 | 2 | 2 |
| 52 | 75.9 | 1 | 1 |
| 53 | 73.1 | 1 | 1 |
| 54 | 85.6 | 1 | 1 |
| 55 | 20.6 | 1 | 1 |
| 58 | 81.6 | 1 | 1 |
| 59 | 28.0 | 1 | 1 |
| 60 | 28.1 | 2 | 2 |
| 61 | 25.1 | 1 | 1 |
| 62 | 23.6 | 1 | 1 |
| 64 | 22.5 | 1 | 1 |
| 65 | 27.2 | 1 | 1 |
| 66 | — | 2 | 0 |
| 67 | 24.4 | 2 | 1 |
| 69 | 86.1 | 1 | 1 |
| 70 | 33.9 | 1 | 1 |
| 71 | 72.1 | 2 | 1 |
| 72 | — | 1 | 0 |
| 73 | — | 1 | 0 |
| 74 | — | 1 | 0 |
| 75 | 25.5 | 1 | 1 |
| 76 | 34.5 | 1 | 1 |
| 77 | — | 1 | 1 |
| 78 | — | 1 | 0 |
| 79 | — | 2 | 1 |
| 80 | — | 1 | 0 |
| 81 | — | 1 | 0 |

## Best Kernel Configuration

**Candidate:** `gen55_10f535a3`  
**Latency:** 20.6 µs  

### Triton Parameters

| Parameter | Value |
|-----------|-------|
| `num_warps` | 4 |
| `num_stages` | 2 |
| `shared_mem_bytes` | 16 |
| `register_count` | 407 |
| `theoretical_occupancy` | N/A |

### Nsight Systems Metrics

| Metric | Value |
|--------|-------|
| SM active cycles | 83.0% |
| DRAM utilization | 71.1% |

## Best Kernel Source Code

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rmsnorm_rope_kernel_matrix(
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
    NUM_HEADS: tl.constexpr,
):
    row = tl.program_id(0)
    
    X_row = X + row * N

    # Compute RMSNorm statistics
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    q_size = n_heads * head_dim

    cos_base = Cos + row * head_dim
    sin_base = Sin + row * head_dim

    h_cols = tl.arange(0, HALF_DIM)
    h_mask = h_cols < HALF_DIM

    cos_v = tl.load(cos_base + h_cols, mask=h_mask, other=1.0).to(tl.float32)
    sin_v = tl.load(sin_base + h_cols, mask=h_mask, other=0.0).to(tl.float32)

    # Process heads using 2D indexing for batch operations
    head_idx = tl.arange(0, NUM_HEADS)
    head_mask = head_idx < n_heads
    
    # Create 2D offsets: [NUM_HEADS, HALF_DIM]
    # h_offset for each head: head_idx * head_dim
    h_offsets = head_idx[:, None] * head_dim + h_cols[None, :]  # [NUM_HEADS, HALF_DIM]
    combined_mask = head_mask[:, None] & h_mask[None, :]  # [NUM_HEADS, HALF_DIM]
    
    # Load Q data for all heads - first half [NUM_HEADS, HALF_DIM]
    q0_raw = tl.load(X_row + h_offsets, mask=combined_mask, other=0.0).to(tl.float32)
    q1_raw = tl.load(X_row + h_offsets + HALF_DIM, mask=combined_mask, other=0.0).to(tl.float32)
    
    # Load K data for all heads [NUM_HEADS, HALF_DIM]
    k0_raw = tl.load(X_row + q_size + h_offsets, mask=combined_mask, other=0.0).to(tl.float32)
    k1_raw = tl.load(X_row + q_size + h_offsets + HALF_DIM, mask=combined_mask, other=0.0).to(tl.float32)
    
    # Apply RMSNorm scaling
    q0_norm = q0_raw * rstd
    q1_norm = q1_raw * rstd
    k0_norm = k0_raw * rstd
    k1_norm = k1_raw * rstd
    
    # Load weights for all heads [NUM_HEADS, HALF_DIM]
    w_q0 = tl.load(W + h_offsets, mask=combined_mask, other=1.0).to(tl.float32)
    w_q1 = tl.load(W + h_offsets + HALF_DIM, mask=combined_mask, other=1.0).to(tl.float32)
    w_k0 = tl.load(W + q_size + h_offsets, mask=combined_mask, other=1.0).to(tl.float32)
    w_k1 = tl.load(W + q_size + h_offsets + HALF_DIM, mask=combined_mask, other=1.0).to(tl.float32)
    
    # Apply weights
    rms_q0 = q0_norm * w_q0
    rms_q1 = q1_norm * w_q1
    rms_k0 = k0_norm * w_k0
    rms_k1 = k1_norm * w_k1
    
    # Broadcast cos and sin across heads: [1, HALF_DIM] -> broadcasts with [NUM_HEADS, HALF_DIM]
    cos_broadcast = cos_v[None, :]
    sin_broadcast = sin_v[None, :]
    
    # Apply RoPE to Q (batch across all heads)
    rq0 = rms_q0 * cos_broadcast - rms_q1 * sin_broadcast
    rq1 = rms_q0 * sin_broadcast + rms_q1 * cos_broadcast
    
    # Apply RoPE to K (batch across all heads)
    rk0 = rms_k0 * cos_broadcast - rms_k1 * sin_broadcast
    rk1 = rms_k0 * sin_broadcast + rms_k1 * cos_broadcast
    
    # Store Q output [NUM_HEADS, HALF_DIM]
    q_out_offsets = row * q_size + h_offsets
    tl.store(Q_out + q_out_offsets, rq0.to(tl.float16), mask=combined_mask)
    tl.store(Q_out + q_out_offsets + HALF_DIM, rq1.to(tl.float16), mask=combined_mask)
    
    # Store K output [NUM_HEADS, HALF_DIM]
    k_out_offsets = row * q_size + h_offsets
    tl.store(K_out + k_out_offsets, rk0.to(tl.float16), mask=combined_mask)
    tl.store(K_out + k_out_offsets + HALF_DIM, rk1.to(tl.float16), mask=combined_mask)


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
    NUM_HEADS = triton.next_power_of_2(n_heads)

    grid = (seq_len,)
    
    _fused_rmsnorm_rope_kernel_matrix[grid](
        x, weight, cos, sin, q_out, k_out,
        N, n_heads, head_dim, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HALF_DIM=HALF_DIM,
        NUM_HEADS=NUM_HEADS,
        num_warps=4,
        num_stages=2,
    )
    return q_out, k_out


KERNEL_TYPE = "fused_rmsnorm_rope"
DESCRIPTION = "Use matrix operations for batch processing multiple heads simultaneously with 2D tensor operations"
```

## Optimization Journey

- **Tool call 0**: 46.8 µs (baseline)
- **Tool call 3**: 43.0 µs — new best (+8.1% vs baseline)
- **Tool call 4**: 42.9 µs — new best (+8.3% vs baseline)
- **Tool call 5**: 31.4 µs — new best (+32.8% vs baseline)
- **Tool call 25**: 26.9 µs — new best (+42.5% vs baseline)
- **Tool call 33**: 26.4 µs — new best (+43.6% vs baseline)
- **Tool call 35**: 23.5 µs — new best (+49.9% vs baseline)
- **Tool call 55**: 20.6 µs — new best (+55.9% vs baseline)
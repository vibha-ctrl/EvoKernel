# EvoKernel Report — fused_rmsnorm_rope

## Performance Progression

```
Latency (µs) by generation:

  Call  0 | █████████████████████                    46.8
  Call  2 | ██████████████████████                   47.5
  Call  3 | ███████████████████                      43.0
  Call  4 | ███████████████████                      43.0
  Call  5 | ██████████████                           31.4
  Call 11 | █████████████████████                    47.1
  Call 12 | ███████████████████                      42.4
  Call 13 | █████████████████████                    47.3
  Call 14 | ██████████████                           32.1
  Call 15 | ██████████████████████████████████       74.3
  Call 19 | ██████████████████████                   48.8
  Call 21 | █████████████████████                    46.9
  Call 22 | ████████████████████                     43.2
  Call 23 | ██████████████                           31.5
  Call 24 | ███████████████████                      42.6
  Call 31 | █████████████                            28.1
  Call 32 | ██████████████                           30.5
  Call 33 | ████████████                             26.4
  Call 34 | █████████████                            29.5
  Call 40 | ███████████                              25.6
  Call 41 | ████████████                             27.7
  Call 42 | ██████████                               23.5
  Call 43 | █████████████                            28.6
  Call 49 | ███████████████                          33.7
  Call 50 | ███████████████                          32.4
  Call 51 | ███████████                              25.7
  Call 52 | ███████████████████████████████████      75.9
  Call 53 | █████████████████████████████████        73.1
  Call 59 | █████████████                            28.0
  Call 60 | █████████████                            28.1
  Call 61 | ███████████                              25.1
  Call 62 | ██████████                               23.6
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
| 0 | 46.8 | 4 | 2 |
| 2 | 47.5 | 1 | 1 |
| 3 | 43.0 | 2 | 2 |
| 4 | 43.0 | 2 | 2 |
| 5 | 31.4 | 1 | 1 |
| 11 | 47.1 | 1 | 1 |
| 12 | 42.4 | 1 | 1 |
| 13 | 47.3 | 2 | 1 |
| 14 | 32.1 | 1 | 1 |
| 15 | 74.3 | 1 | 1 |
| 19 | 48.8 | 1 | 1 |
| 21 | 46.9 | 1 | 1 |
| 22 | 43.2 | 2 | 2 |
| 23 | 31.5 | 1 | 1 |
| 24 | 42.6 | 1 | 1 |
| 31 | 28.1 | 1 | 1 |
| 32 | 30.5 | 2 | 2 |
| 33 | 26.4 | 2 | 2 |
| 34 | 29.5 | 1 | 1 |
| 40 | 25.6 | 1 | 1 |
| 41 | 27.7 | 2 | 2 |
| 42 | 23.5 | 2 | 1 |
| 43 | 28.6 | 1 | 1 |
| 49 | 33.7 | 1 | 1 |
| 50 | 32.4 | 1 | 1 |
| 51 | 25.7 | 2 | 2 |
| 52 | 75.9 | 1 | 1 |
| 53 | 73.1 | 1 | 1 |
| 59 | 28.0 | 1 | 1 |
| 60 | 28.1 | 2 | 2 |
| 61 | 25.1 | 1 | 1 |
| 62 | 23.6 | 1 | 1 |
| 65 | 27.2 | 1 | 1 |
| 66 | — | 1 | 0 |
| 67 | 24.4 | 1 | 1 |
| 69 | 86.1 | 1 | 1 |
| 70 | 33.9 | 1 | 1 |
| 71 | 72.1 | 1 | 1 |
| 74 | — | 1 | 0 |
| 75 | 25.5 | 1 | 1 |
| 76 | 34.5 | 1 | 1 |
| 79 | — | 1 | 0 |
| 80 | — | 1 | 0 |
| 81 | — | 1 | 0 |

## Best Kernel Configuration

**Candidate:** `gen42_85be03a9`  
**Latency:** 23.5 µs  

### Triton Parameters

| Parameter | Value |
|-----------|-------|
| `num_warps` | 2 |
| `num_stages` | 1 |
| `shared_mem_bytes` | 8 |
| `register_count` | 288 |
| `theoretical_occupancy` | N/A |

### Nsight Systems Metrics

| Metric | Value |
|--------|-------|
| SM active cycles | N/A |
| DRAM utilization | N/A |

## Best Kernel Source Code

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rmsnorm_rope_kernel_num_stages_1(
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
    
    _fused_rmsnorm_rope_kernel_num_stages_1[grid](
        x, weight, cos, sin, q_out, k_out,
        N, n_heads, head_dim, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        HALF_DIM=HALF_DIM,
        HEADS_PER_BLOCK=HEADS_PER_BLOCK,
        num_warps=2,
        num_stages=1,
    )
    return q_out, k_out


KERNEL_TYPE = "fused_rmsnorm_rope"
DESCRIPTION = "Use num_stages=1 instead of 2 to reduce register pressure for software pipelining"
```

## Optimization Journey

- **Tool call 0**: 46.8 µs (baseline)
- **Tool call 3**: 43.0 µs — new best (+8.1% vs baseline)
- **Tool call 5**: 31.4 µs — new best (+32.9% vs baseline)
- **Tool call 31**: 28.1 µs — new best (+39.9% vs baseline)
- **Tool call 33**: 26.4 µs — new best (+43.6% vs baseline)
- **Tool call 40**: 25.6 µs — new best (+45.3% vs baseline)
- **Tool call 42**: 23.5 µs — new best (+49.8% vs baseline)
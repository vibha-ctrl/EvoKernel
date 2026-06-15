# EvoKernel

Autonomous Triton kernel optimization via hardware-guided evolutionary search.

```
Generate → Verify → Benchmark → Profile → Select → Mutate → Repeat
```

## Architecture

```
Local machine (you)
  ├── run_search.py          evolutionary loop orchestration
  ├── evokernel/agents/      Claude claude-opus-4-5: generator + critic
  ├── evokernel/search/      search engine + SQLite candidate store
  ├── evokernel/mcp/         MCP tool server (optional interactive use)
  └── evokernel/reports/     Markdown report generator
        │
        │  HTTP (FastAPI)
        ▼
RunPod Pod (A100 / H100)
  └── gpu_server/server.py
        ├── POST /verify      correctness vs PyTorch reference
        ├── POST /benchmark   CUDA events + triton.testing.do_bench
        └── POST /profile     ncu + Triton compiler metadata
```

## Quick Start

### 1. Set up RunPod pod

Launch a pod with an A100/H100. In the pod terminal:

```bash
git clone <this-repo> /workspace/evokernel
pip install -r /workspace/evokernel/gpu_server/requirements.txt
cd /workspace/evokernel/gpu_server
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 2. Configure local environment

```bash
cp .env.example .env
# Edit .env:
#   RUNPOD_SERVER_URL=https://YOUR_POD_ID-8000.proxy.runpod.net
#   ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Install local dependencies

```bash
pip install -e .
```

### 4. Run a search

```bash
# Optimize RMSNorm
python run_search.py rmsnorm

# Optimize RoPE with custom settings
python run_search.py rope --generations 8 --candidates 10

# Optimize fused RMSNorm + RoPE (hardest)
python run_search.py fused_rmsnorm_rope

# Check progress mid-run
python run_search.py status rmsnorm

# Generate report from existing DB
python run_search.py report-only rmsnorm
```

## Kernels

| Kernel | Operation | Optimization Focus |
|--------|-----------|-------------------|
| `rmsnorm` | RMSNorm normalization | memory coalescing, vectorization |
| `rope` | Rotary positional encoding | indexing efficiency, memory access |
| `fused_rmsnorm_rope` | RMSNorm + RoPE in one pass | kernel fusion, launch overhead |

## How It Works

**Generation 0:** Baseline kernel benchmarked on GPU. Establishes starting latency.

**Each generation:**
1. Critic (Claude) reads profiler metrics → produces optimization hints
2. Generator (Claude) mutates top-k parents + hints → N new variants  
3. All variants verified against PyTorch reference (`torch.allclose`)
4. Passing variants benchmarked with CUDA events (100 reps, median)
5. Top-k survivors kept, profiled with ncu + Triton compiler metadata
6. Repeat until convergence

**Convergence:** Stops when best improvement < 2% per generation (configurable).

## MCP Server (interactive use)

```bash
fastmcp run evokernel/mcp/server.py
```

Tools exposed: `verify_kernel`, `benchmark_kernel`, `profile_kernel`,
`retrieve_history`, `generate_report`, `gpu_status`.

## Profiling on RunPod

RunPod pods run with Linux capabilities that allow `ncu` to access hardware
performance counters. The GPU server automatically detects `ncu` and falls
back to Triton compiler metadata + estimated occupancy if not available.

For full ncu access, install Nsight Compute on the pod:
```bash
apt-get install -y nsight-compute
```

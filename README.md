# EvoKernel

Autonomous Triton kernel optimization via hardware-guided evolutionary search.

> **Status:** Framework complete. GPU runs pending — results will appear in [`results/`](results/) after RunPod execution.

```
Generate → Verify → Benchmark → Profile → Select → Mutate → Repeat
```

---

## What It Does

Traditional GPU kernel optimization is a manual loop: write → benchmark → profile → modify → repeat. EvoKernel automates this entirely.

An LLM (Claude claude-opus-4-5) acts as the optimization engineer — generating kernel variants, reading real hardware profiler output, and proposing targeted improvements. The GPU is the source of truth. The fastest correct kernel wins each generation.

---

## Architecture

```
Local machine
  ├── run_search.py          orchestrates the evolutionary loop
  ├── evokernel/agents/      Claude claude-opus-4-5 generator + critic
  ├── evokernel/search/      search engine + SQLite candidate store
  ├── evokernel/mcp/         MCP tool server for interactive use
  └── evokernel/reports/     live Markdown results + final report
        │
        │  HTTP (FastAPI)
        ▼
RunPod Pod (A100 / H100)
  └── gpu_server/server.py
        ├── POST /verify      3-stage correctness gate vs PyTorch reference
        ├── POST /benchmark   CUDA events + triton.testing.do_bench (100 reps)
        └── POST /profile     real ncu (Nsight Compute) — required, no fallback
```

---

## Kernels Under Optimization

| Kernel | Operation | Optimization Target |
|--------|-----------|---------------------|
| `rmsnorm` | RMSNorm normalization | memory coalescing, vectorized loads, occupancy |
| `rope` | Rotary positional encoding | indexing efficiency, memory access patterns |
| `fused_rmsnorm_rope` | RMSNorm + RoPE in a single kernel pass | fusion, launch overhead, memory traffic |

All three are core LLM inference primitives used in LLaMA, Mistral, and similar architectures.

---

## How the Search Works

**Generation 0:** Baseline Triton kernel benchmarked on GPU — establishes starting latency.

**Each generation:**
1. **Critic** (Claude) reads ncu profiler output → diagnoses bottleneck → produces optimization hints
2. **Generator** (Claude) mutates top-k parents guided by hints → produces N new variants
3. All variants run through a **3-stage correctness gate**: syntax check → runtime check → `torch.allclose` vs PyTorch reference
4. Passing variants **benchmarked**: 25 warmup + 100 timed reps, median latency reported
5. **Top-k survivors** profiled with ncu (occupancy, DRAM utilization, warp stall reasons)
6. Repeat until improvement < 2% per generation

**Convergence** is automatic — typically 4–6 generations before diminishing returns.

---

## Profiling

ncu (Nsight Compute) is **required** — no fallback. The server refuses to start without it.

Metrics collected per candidate:
- Latency (µs), throughput (GB/s), bandwidth utilization %
- Register count, shared memory, `num_warps`, `num_stages`
- Theoretical occupancy
- DRAM utilization %, L1 cache hit rate %
- Warp stall reasons: memory dependency, long scoreboard

---

## Results

Search results are written to [`results/`](results/) after each generation and committed to this repo. Check there for live speedup numbers once GPU runs complete.

---

## Setup

### 1. RunPod pod

Launch an A100/H100 pod using the **RunPod PyTorch** template (1 GPU). In the pod web terminal:

```bash
# nsight-compute is a virtual package — must specify version
apt-get update -y && apt-get install -y nsight-compute-2024.3.2

# Verify ncu works
ncu --version

# Clone the repo
git clone https://github.com/vibha-ctrl/EvoKernel.git /workspace/EvoKernel

# Install GPU server dependencies
pip install -r /workspace/EvoKernel/gpu_server/requirements.txt

# Start the server — leave this terminal open
cd /workspace/EvoKernel/gpu_server
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 2. SSH tunnel (local machine — new terminal)

Port 8000 is not exposed via RunPod's HTTP proxy. Use SSH port forwarding instead.
Find your pod's SSH address in RunPod dashboard → Connect → SSH over exposed TCP.

```bash
# Keep this terminal open for the entire search session
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519 -L 8000:localhost:8000 -N
```

Verify connection:
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok","gpu":"NVIDIA A100-SXM4-80GB",...}
```

### 3. Local environment

```bash
cp .env.example .env
# Set in .env:
#   RUNPOD_SERVER_URL=http://localhost:8000
#   ANTHROPIC_API_KEY=sk-ant-...
pip install -e .
```

### 3. Run

```bash
python run_search.py rmsnorm
python run_search.py rope --generations 8
python run_search.py fused_rmsnorm_rope
python run_search.py status rmsnorm        # check progress mid-run
python run_search.py report-only rmsnorm   # regenerate report from DB
```

---

## MCP Server

```bash
fastmcp run evokernel/mcp/server.py
```

Exposes: `verify_kernel`, `benchmark_kernel`, `profile_kernel`, `retrieve_history`, `generate_report`, `gpu_status`

---

## Repository Structure

```
EvoKernel/
├── run_search.py                       main CLI (search / status / report-only)
├── gpu_server/
│   └── server.py                       FastAPI GPU server — runs on RunPod
├── evokernel/
│   ├── agents/
│   │   ├── generator.py                LLM kernel variant generator
│   │   └── critic.py                   LLM profiler output interpreter
│   ├── kernels/
│   │   ├── rmsnorm/                    baseline + PyTorch reference
│   │   ├── rope/                       baseline + PyTorch reference
│   │   └── fused_rmsnorm_rope/         baseline + PyTorch reference
│   ├── search/
│   │   ├── evolutionary.py             main search loop
│   │   └── candidate_store.py          SQLite persistence
│   ├── mcp/server.py                   MCP tool server
│   └── reports/
│       ├── report_generator.py         final Markdown report
│       └── results_tracker.py          live per-generation results writer
└── results/                            search results (updated each run)
```

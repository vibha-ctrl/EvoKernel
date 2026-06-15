# EvoKernel

Autonomous Triton kernel optimization via hardware-guided agentic search.

> **Status:** Framework complete. GPU runs pending — results will appear in [`results/`](results/) after RunPod execution.

```
Generate → Verify → Benchmark → Profile → Reason → Repeat
```

---

## What It Does

Traditional GPU kernel optimization is a manual loop: write → benchmark → profile → modify → repeat. EvoKernel automates this entirely.

Claude (claude-opus-4-5) acts as an autonomous optimization engineer. It decides its own strategy, generates kernel variants, reads real hardware profiler output, and drives the entire search loop via tool calls. The GPU is the source of truth. The fastest correct kernel wins.

---

## Architecture

```
Local machine
  ├── run_search.py          CLI entry point
  ├── evokernel/search/      agentic loop + SQLite candidate store
  └── evokernel/reports/     final report generator
        │
        │  HTTP (FastAPI)
        ▼
RunPod Pod (A100 / H100)
  └── gpu_server/server.py
        ├── POST /verify      3-stage correctness gate vs PyTorch reference
        ├── POST /benchmark   triton.testing.do_bench (25 warmup + 100 reps)
        └── POST /profile     nsys (Nsight Systems) + Triton compiler metadata
```

---

## Kernels Under Optimization

| Kernel | Operation | Optimization Target |
|--------|-----------|---------------------|
| `rmsnorm` | RMSNorm normalization | memory coalescing, vectorized loads, occupancy |
| `rope` | Rotary positional encoding | indexing efficiency, memory access patterns |
| `fused_rmsnorm_rope` | RMSNorm + RoPE in a single kernel pass | fusion, launch overhead, memory traffic |

All three are core LLM inference primitives used in LLaMA, Mistral, and similar architectures. Each has a hand-written baseline Triton kernel that is intentionally unoptimized — it establishes the starting latency that all variants are measured against.

---

## How the Search Works

Claude is given the baseline kernel code and access to 7 tools. It drives the entire loop autonomously:

1. **Generate** — writes a new variant mutated from a chosen parent, with a chosen strategy
2. **Verify** — 3-stage correctness gate: syntax → runtime → `torch.allclose` vs PyTorch reference
3. **Benchmark** — 25 warmup + 100 timed reps, median latency reported
4. **Profile** — Nsight Systems (`nsys`) for DRAM utilization and SM throughput; Triton compiler cache for `num_warps`, `num_stages`, occupancy
5. **Get history** — review past candidates and failures to avoid repeating mistakes
6. **Stop** — Claude decides when it has found a good speedup or exhausted strategies

Claude chooses which parent to build on, when to profile, and when to stop. The search runs for up to 80 tool calls.

---

## Results

Each run writes two files to [`results/`](results/), named with kernel type and timestamp so nothing is ever overwritten:

- `{kernel}_trace_{timestamp}.md` — Claude's full reasoning, every tool call in order, and every GPU result
- `{kernel}_report_{timestamp}.md` — performance summary: speedup, latency progression, best kernel code and hardware metrics

---

## Setup

### 1. RunPod pod

Launch an A100/H100 pod using the **RunPod PyTorch** template (1 GPU). In the pod web terminal:

```bash
# Install Nsight Systems
apt-get update -y && apt-get install -y nsight-systems

# Verify nsys works
nsys --version

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

### 4. Run

```bash
python run_search.py search rmsnorm
python run_search.py search rope
python run_search.py search fused_rmsnorm_rope
python run_search.py status rmsnorm        # check progress mid-run
```

---

## Repository Structure

```
EvoKernel/
├── run_search.py                       main CLI (search / status / report-only)
├── gpu_server/
│   └── server.py                       FastAPI GPU server — runs on RunPod
├── evokernel/
│   ├── kernels/
│   │   ├── rmsnorm/baseline.py         intentionally naive starting kernel
│   │   ├── rope/baseline.py
│   │   └── fused_rmsnorm_rope/baseline.py
│   ├── search/
│   │   ├── agent_loop.py               agentic loop — Claude drives via tool use
│   │   └── candidate_store.py          SQLite persistence for all candidates
│   └── reports/
│       └── report_generator.py         final Markdown report
└── results/                            per-run trace + report (timestamped)
```

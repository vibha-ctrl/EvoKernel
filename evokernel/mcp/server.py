"""
EvoKernel MCP Server — exposes all search capabilities as tools.

Tools:
  verify_kernel      — run correctness check on a kernel
  benchmark_kernel   — measure latency + throughput
  profile_kernel     — full profile (ncu + compiler metadata)
  retrieve_history   — query candidate history from DB
  generate_report    — generate final search report

Run with:
  python -m evokernel.mcp.server
  or: fastmcp run evokernel/mcp/server.py
"""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

from evokernel.search.candidate_store import CandidateStore
from evokernel.reports.report_generator import generate_report as _gen_report

load_dotenv()

mcp = FastMCP(
    name="EvoKernel",
    instructions=(
        "Tools for optimizing Triton GPU kernels via hardware-guided evolutionary search. "
        "Use verify_kernel before benchmarking. Use profile_kernel on top candidates only. "
        "Call retrieve_history to understand search progress. "
        "Call generate_report at the end of a search session."
    ),
)

_RUNPOD_URL = os.getenv("RUNPOD_SERVER_URL", "")
_DB_PATH = os.getenv("EVOKERNEL_DB", "evokernel.db")


def _gpu_client() -> httpx.Client:
    if not _RUNPOD_URL:
        raise RuntimeError(
            "RUNPOD_SERVER_URL not set. Set it in .env to point at your RunPod pod."
        )
    return httpx.Client(base_url=_RUNPOD_URL, timeout=120.0)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def verify_kernel(
    code: str,
    kernel_type: str,
    candidate_id: str = "manual",
) -> dict:
    """
    Verify a Triton kernel's correctness against the PyTorch reference implementation.

    Args:
        code:         Full Python source code of the kernel. Must define run().
        kernel_type:  One of: rmsnorm, rope, fused_rmsnorm_rope
        candidate_id: Optional label for tracking (e.g. "my_kernel_v1")

    Returns:
        {
          passed: bool,
          error_type: str | null,   # syntax_error | runtime_error | wrong_output | null
          error_msg:  str | null,
          max_error:  float | null  # max absolute diff from reference
        }
    """
    client = _gpu_client()
    resp = client.post("/verify", json={
        "code": code,
        "kernel_type": kernel_type,
        "candidate_id": candidate_id,
    })
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def benchmark_kernel(
    code: str,
    kernel_type: str,
    candidate_id: str = "manual",
) -> dict:
    """
    Benchmark a Triton kernel on the RunPod GPU.

    The kernel must pass verify_kernel first. Benchmarks are run with 25 warmup
    iterations and 100 timed repetitions. Returns median latency.

    Args:
        code:         Full Python source code. Must define run().
        kernel_type:  One of: rmsnorm, rope, fused_rmsnorm_rope
        candidate_id: Optional label

    Returns:
        {
          latency_us:               float,  # median latency in microseconds
          latency_p99_us:           float,  # 99th percentile latency
          throughput_gb_s:          float,  # memory throughput in GB/s
          peak_bandwidth_gb_s:      float,  # GPU peak memory bandwidth
          bandwidth_utilization_pct: float  # throughput / peak * 100
        }
    """
    client = _gpu_client()
    resp = client.post("/benchmark", json={
        "code": code,
        "kernel_type": kernel_type,
        "candidate_id": candidate_id,
    })
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def profile_kernel(
    code: str,
    kernel_type: str,
    candidate_id: str = "manual",
) -> dict:
    """
    Full performance profile of a Triton kernel using Nsight Compute + compiler metadata.

    This is expensive (~10-30s). Only use on top candidates.

    Args:
        code:         Full Python source code. Must define run().
        kernel_type:  One of: rmsnorm, rope, fused_rmsnorm_rope
        candidate_id: Optional label

    Returns:
        {
          num_warps, num_stages, shared_mem_bytes, register_count,
          theoretical_occupancy_pct,
          ncu_available: bool,      # false if ncu not accessible on pod
          dram_utilization_pct,
          l1_hit_rate_pct,
          stall_memory_dependency_pct,
          stall_long_scoreboard_pct,
          sm_active_cycles_pct,
          latency_us, throughput_gb_s
        }
    """
    client = _gpu_client()
    resp = client.post("/profile", json={
        "code": code,
        "kernel_type": kernel_type,
        "candidate_id": candidate_id,
    })
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def retrieve_history(
    kernel_type: str,
    query: str = "summary",
    generation: int | None = None,
    top_n: int = 10,
) -> dict:
    """
    Query the search history database.

    Args:
        kernel_type:  One of: rmsnorm, rope, fused_rmsnorm_rope
        query:        One of:
                        "summary"    — generation-by-generation best latency
                        "best"       — top_n fastest verified candidates
                        "generation" — all candidates from a specific generation (requires generation=N)
                        "failed"     — all failed candidates from a generation
        generation:   Required for query="generation" or "failed"
        top_n:        Number of results for query="best"

    Returns:
        List of candidate records or summary dicts.
    """
    store = CandidateStore(_DB_PATH)

    if query == "summary":
        return {"summary": store.generation_summary(kernel_type)}

    elif query == "best":
        candidates = store.get_best(kernel_type, n=top_n)
        return {"candidates": [_candidate_dict(c) for c in candidates]}

    elif query == "generation":
        if generation is None:
            return {"error": "generation parameter required for query='generation'"}
        candidates = store.get_generation(generation, kernel_type)
        return {"candidates": [_candidate_dict(c) for c in candidates]}

    elif query == "failed":
        if generation is None:
            return {"error": "generation parameter required for query='failed'"}
        candidates = store.get_failed(kernel_type, generation)
        return {
            "failed_candidates": [
                {
                    "id": c.id,
                    "error_type": c.verify_error_type,
                    "error_msg": c.verify_error_msg,
                    "code": c.code,
                }
                for c in candidates
            ]
        }

    else:
        return {"error": f"Unknown query type: {query}. Use: summary, best, generation, failed"}


@mcp.tool()
def generate_report(
    kernel_type: str,
    output_path: str = "report.md",
) -> dict:
    """
    Generate a final Markdown report summarizing the search results.

    Includes:
    - Search progression chart (ASCII)
    - Best kernel code
    - Performance comparison table
    - Winning optimizations

    Args:
        kernel_type:  One of: rmsnorm, rope, fused_rmsnorm_rope
        output_path:  Where to write the Markdown report

    Returns:
        { path: str, speedup: float, best_latency_us: float, baseline_latency_us: float }
    """
    store = CandidateStore(_DB_PATH)
    result = _gen_report(store, kernel_type, output_path)
    return result


@mcp.tool()
def gpu_status() -> dict:
    """
    Check the health and GPU specs of the RunPod pod.

    Returns GPU name, VRAM, CUDA version, Triton version.
    """
    client = _gpu_client()
    resp = client.get("/health")
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_dict(c) -> dict:
    return {
        "id": c.id,
        "label": c.label,
        "generation": c.generation,
        "latency_us": c.latency_us,
        "throughput_gb_s": c.throughput_gb_s,
        "bandwidth_utilization_pct": c.bandwidth_utilization_pct,
        "num_warps": c.num_warps,
        "num_stages": c.num_stages,
        "theoretical_occupancy_pct": c.theoretical_occupancy_pct,
        "parent_id": c.parent_id,
    }


if __name__ == "__main__":
    mcp.run()

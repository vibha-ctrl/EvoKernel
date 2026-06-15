"""
EvoKernel GPU Server — runs on RunPod pod.

Exposes three endpoints:
  POST /verify    — correctness check against PyTorch reference
  POST /benchmark — latency + throughput measurement
  POST /profile   — full ncu profile + Triton compiler metadata

Run with:
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any

import torch
import triton
import triton.testing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="EvoKernel GPU Server", version="0.1.0")


@app.on_event("startup")
def _require_ncu():
    """Fail fast at startup if ncu is not available. No fallback — ncu is required."""
    ncu = _find_ncu()
    if not ncu:
        raise RuntimeError(
            "ncu (Nsight Compute) not found. Install it on the pod:\n"
            "  apt-get install -y nsight-compute\n"
            "Then verify with: ncu --version\n"
            "EvoKernel requires real hardware profiling — no fallback."
        )


@app.on_event("startup")
def _warm_up_gpu():
    """
    Drive GPU clocks to sustained boost frequency before any benchmarks run.

    A cold GPU runs at base clock (~1.1 GHz on A100). Under sustained load it
    boosts to ~1.41 GHz and stays there. Without warmup, generation-0 baseline
    benchmarks ~10-15% slower than it truly is, making speedup comparisons inaccurate.

    We run 500 back-to-back large matmuls — enough to saturate the SM pipeline
    and lock clocks into boost state for the rest of the session.
    """
    import torch
    device = "cuda"
    # Large enough to fully stress all SMs
    a = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    for _ in range(500):
        torch.mm(a, b)
    torch.cuda.synchronize()
    # Discard tensors — GPU stays warm for the session
    del a, b

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class KernelRequest(BaseModel):
    code: str          # full Python source including imports and run() function
    kernel_type: str   # "rmsnorm" | "rope" | "fused_rmsnorm_rope"
    candidate_id: str


class VerifyResponse(BaseModel):
    candidate_id: str
    passed: bool
    error_type: str | None = None
    error_msg: str | None = None
    max_error: float | None = None


class BenchmarkResponse(BaseModel):
    candidate_id: str
    latency_us: float
    latency_p99_us: float
    throughput_gb_s: float
    peak_bandwidth_gb_s: float
    bandwidth_utilization_pct: float


class ProfileResponse(BaseModel):
    candidate_id: str
    # From Triton compiler
    num_warps: int | None = None
    num_stages: int | None = None
    shared_mem_bytes: int | None = None
    register_count: int | None = None
    # From CUDA device + occupancy calc
    theoretical_occupancy_pct: float | None = None
    # From ncu — always populated (ncu is required, server fails to start without it)
    dram_utilization_pct: float | None = None
    l1_hit_rate_pct: float | None = None
    stall_memory_dependency_pct: float | None = None
    stall_long_scoreboard_pct: float | None = None
    sm_active_cycles_pct: float | None = None
    # Benchmark data attached for critic context
    latency_us: float | None = None
    throughput_gb_s: float | None = None


# ---------------------------------------------------------------------------
# Test harness: builds tensors and wraps the candidate run() call
# ---------------------------------------------------------------------------

def _get_test_inputs(kernel_type: str) -> dict[str, Any]:
    torch.manual_seed(42)
    device = "cuda"

    if kernel_type == "rmsnorm":
        x = torch.randn(2048, 4096, dtype=torch.float16, device=device)
        w = torch.ones(4096, dtype=torch.float16, device=device)
        return {"x": x, "weight": w}

    elif kernel_type == "rope":
        seq_len, n_heads, head_dim = 512, 32, 128
        q = torch.randn(seq_len, n_heads, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(seq_len, n_heads, head_dim, dtype=torch.float16, device=device)
        half = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).half()
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).half()
        return {"q": q, "k": k, "cos": cos, "sin": sin}

    elif kernel_type == "fused_rmsnorm_rope":
        seq_len, n_heads, head_dim = 512, 32, 128
        N = 2 * n_heads * head_dim
        x = torch.randn(seq_len, N, dtype=torch.float16, device=device)
        weight = torch.ones(N, dtype=torch.float16, device=device)
        half = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).half()
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).half()
        return {"x": x, "weight": weight, "cos": cos, "sin": sin, "n_heads": n_heads}

    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")


def _get_reference_output(kernel_type: str, inputs: dict[str, Any]):
    """PyTorch reference — the correctness ground truth."""
    if kernel_type == "rmsnorm":
        x, w = inputs["x"].float(), inputs["weight"].float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        return (x * rms * w).half()

    elif kernel_type == "rope":
        q, k = inputs["q"].float(), inputs["k"].float()
        cos = inputs["cos"].float().unsqueeze(1)
        sin = inputs["sin"].float().unsqueeze(1)
        half = q.shape[-1] // 2
        cos_h, sin_h = cos[..., :half], sin[..., :half]
        q_rot = torch.cat([q[..., :half] * cos_h - q[..., half:] * sin_h,
                            q[..., :half] * sin_h + q[..., half:] * cos_h], dim=-1)
        k_rot = torch.cat([k[..., :half] * cos_h - k[..., half:] * sin_h,
                            k[..., :half] * sin_h + k[..., half:] * cos_h], dim=-1)
        return q_rot.half(), k_rot.half()

    elif kernel_type == "fused_rmsnorm_rope":
        x, weight = inputs["x"].float(), inputs["weight"].float()
        n_heads = inputs["n_heads"]
        seq_len, N = x.shape
        head_dim = N // (2 * n_heads)

        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        x_norm = (x * rms * weight).half()

        q = x_norm[:, :n_heads * head_dim].reshape(seq_len, n_heads, head_dim).float()
        k = x_norm[:, n_heads * head_dim:].reshape(seq_len, n_heads, head_dim).float()

        cos = inputs["cos"].float().unsqueeze(1)
        sin = inputs["sin"].float().unsqueeze(1)
        half = head_dim // 2
        cos_h, sin_h = cos[..., :half], sin[..., :half]
        q_rot = torch.cat([q[..., :half] * cos_h - q[..., half:] * sin_h,
                            q[..., :half] * sin_h + q[..., half:] * cos_h], dim=-1)
        k_rot = torch.cat([k[..., :half] * cos_h - k[..., half:] * sin_h,
                            k[..., :half] * sin_h + k[..., half:] * cos_h], dim=-1)
        return q_rot.half(), k_rot.half()


def _exec_candidate(code: str) -> tuple[dict, str]:
    """
    Load candidate code from a real temp file so Triton's JIT can read the source.

    Triton's @triton.jit compiles lazily on first kernel call, so the source file
    must stay on disk until after the first run() call. We return the tmp_path so
    callers can delete it after execution. Never delete before running the kernel.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="evokernel_candidate_",
        delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(code)
        tmp_path = f.name

    spec = importlib.util.spec_from_file_location("evokernel_candidate", tmp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "run"):
        os.unlink(tmp_path)
        raise AttributeError("Candidate code must define a function named 'run'")

    return vars(mod), tmp_path


def _cleanup(tmp_path: str):
    try:
        os.unlink(tmp_path)
    except OSError:
        pass


def _call_run(namespace: dict, kernel_type: str, inputs: dict[str, Any]):
    """Call namespace['run'] with the right arguments for the kernel type."""
    run_fn = namespace["run"]
    if kernel_type == "rmsnorm":
        return run_fn(inputs["x"], inputs["weight"])
    elif kernel_type == "rope":
        return run_fn(inputs["q"], inputs["k"], inputs["cos"], inputs["sin"])
    elif kernel_type == "fused_rmsnorm_rope":
        return run_fn(inputs["x"], inputs["weight"], inputs["cos"], inputs["sin"],
                      inputs["n_heads"])
    raise ValueError(f"Unknown kernel_type: {kernel_type}")


def _outputs_close(candidate_out, reference_out, atol: float = 1e-2, rtol: float = 1e-3):
    """Compare candidate to reference, handling single-tensor and tuple outputs."""
    if isinstance(reference_out, tuple):
        if not isinstance(candidate_out, tuple) or len(candidate_out) != len(reference_out):
            return False, 999.0
        errors = [
            (c.float() - r.float()).abs().max().item()
            for c, r in zip(candidate_out, reference_out)
        ]
        max_err = max(errors)
        passed = all(
            torch.allclose(c.float(), r.float(), atol=atol, rtol=rtol)
            for c, r in zip(candidate_out, reference_out)
        )
        return passed, max_err
    else:
        max_err = (candidate_out.float() - reference_out.float()).abs().max().item()
        passed = torch.allclose(candidate_out.float(), reference_out.float(),
                                atol=atol, rtol=rtol)
        return passed, max_err


def _bytes_accessed(kernel_type: str, inputs: dict[str, Any]) -> int:
    """Estimate total bytes read+written for arithmetic intensity calculation."""
    if kernel_type == "rmsnorm":
        M, N = inputs["x"].shape
        return (M * N + N + M * N) * 2  # read X, read W, write Y (float16=2 bytes)
    elif kernel_type == "rope":
        q = inputs["q"]
        S, H, D = q.shape
        return S * H * D * 2 * 4 * 2  # Q, K, cos, sin read + Q_out, K_out write
    elif kernel_type == "fused_rmsnorm_rope":
        S, N = inputs["x"].shape
        return (S * N + N) * 2 + S * (N // 2) * 2 * 2  # read X+W, write Q_out+K_out
    return 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    device = torch.cuda.get_device_properties(0)
    ncu_path = _find_ncu()
    ncu_version = None
    if ncu_path:
        try:
            r = subprocess.run([ncu_path, "--version"], capture_output=True, text=True, timeout=5)
            ncu_version = r.stdout.strip().splitlines()[0] if r.returncode == 0 else "unknown"
        except Exception:
            ncu_version = "error"
    return {
        "status": "ok",
        "gpu": device.name,
        "vram_gb": round(device.total_memory / 1e9, 1),
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "ncu_path": ncu_path,
        "ncu_version": ncu_version,
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(req: KernelRequest):
    """
    Three-stage correctness gate:
      1. compile/exec check
      2. runtime check (does it run without crashing)
      3. correctness check (does it match PyTorch reference)
    """
    inputs = _get_test_inputs(req.kernel_type)

    # Stage 1 — compile + exec
    tmp_path = None
    try:
        namespace, tmp_path = _exec_candidate(req.code)
    except SyntaxError as e:
        return VerifyResponse(
            candidate_id=req.candidate_id,
            passed=False,
            error_type="syntax_error",
            error_msg=f"Line {e.lineno}: {e.msg}",
        )
    except Exception as e:
        return VerifyResponse(
            candidate_id=req.candidate_id,
            passed=False,
            error_type="import_error",
            error_msg=str(e),
        )

    # Stage 2 — run on GPU; temp file must exist until after first Triton JIT compile
    try:
        torch.cuda.synchronize()
        candidate_out = _call_run(namespace, req.kernel_type, inputs)
        torch.cuda.synchronize()
    except Exception as e:
        _cleanup(tmp_path)
        return VerifyResponse(
            candidate_id=req.candidate_id,
            passed=False,
            error_type="runtime_error",
            error_msg=traceback.format_exc(limit=5),
        )
    finally:
        _cleanup(tmp_path)

    # Stage 3 — correctness
    try:
        reference_out = _get_reference_output(req.kernel_type, inputs)
        passed, max_error = _outputs_close(candidate_out, reference_out)
    except Exception as e:
        return VerifyResponse(
            candidate_id=req.candidate_id,
            passed=False,
            error_type="comparison_error",
            error_msg=str(e),
        )

    return VerifyResponse(
        candidate_id=req.candidate_id,
        passed=passed,
        error_type=None if passed else "wrong_output",
        max_error=max_error,
    )


@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(req: KernelRequest):
    """Measure median latency and memory throughput via CUDA events."""
    inputs = _get_test_inputs(req.kernel_type)

    try:
        namespace, tmp_path = _exec_candidate(req.code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Code error: {e}")

    def bench_fn():
        return _call_run(namespace, req.kernel_type, inputs)

    try:
        # triton.testing.do_bench returns median latency in milliseconds
        ms, min_ms, max_ms = triton.testing.do_bench(
            bench_fn,
            warmup=25,
            rep=100,
            quantiles=[0.5, 0.05, 0.95],
        )
    finally:
        _cleanup(tmp_path)

    latency_us = ms * 1000
    latency_p99_us = max_ms * 1000

    device = torch.cuda.get_device_properties(0)
    # Peak bandwidth in GB/s (approximate — depends on ECC, boost clocks)
    peak_bw = device.memory_bandwidth_gb_s if hasattr(device, "memory_bandwidth_gb_s") else _estimate_peak_bw(device.name)

    bytes_total = _bytes_accessed(req.kernel_type, inputs)
    throughput_gb_s = (bytes_total / 1e9) / (latency_us / 1e6)
    utilization_pct = (throughput_gb_s / peak_bw) * 100 if peak_bw > 0 else 0.0

    return BenchmarkResponse(
        candidate_id=req.candidate_id,
        latency_us=round(latency_us, 2),
        latency_p99_us=round(latency_p99_us, 2),
        throughput_gb_s=round(throughput_gb_s, 1),
        peak_bandwidth_gb_s=round(peak_bw, 1),
        bandwidth_utilization_pct=round(utilization_pct, 1),
    )


@app.post("/profile", response_model=ProfileResponse)
def profile(req: KernelRequest):
    """
    Full profile: Triton compiler metadata + real ncu metrics.
    ncu is required — this endpoint raises 503 if ncu is not accessible.
    """
    inputs = _get_test_inputs(req.kernel_type)

    try:
        namespace, tmp_path = _exec_candidate(req.code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Code error: {e}")

    result = ProfileResponse(candidate_id=req.candidate_id)

    try:
        # --- Triton compiler metadata (register count, shared mem, warps, stages) ---
        meta = _extract_triton_metadata(req.code, namespace, req.kernel_type, inputs)
        result.num_warps = meta.get("num_warps")
        result.num_stages = meta.get("num_stages")
        result.shared_mem_bytes = meta.get("shared_mem_bytes")
        result.register_count = meta.get("register_count")
        result.theoretical_occupancy_pct = meta.get("theoretical_occupancy_pct")

        # --- Quick latency + throughput for critic context ---
        def bench_fn():
            return _call_run(namespace, req.kernel_type, inputs)

        ms, _, _ = triton.testing.do_bench(bench_fn, warmup=10, rep=50,
                                            quantiles=[0.5, 0.05, 0.95])
        result.latency_us = round(ms * 1000, 2)
        bytes_total = _bytes_accessed(req.kernel_type, inputs)
        peak_bw = _estimate_peak_bw(torch.cuda.get_device_properties(0).name)
        result.throughput_gb_s = round((bytes_total / 1e9) / (ms / 1000), 1)
    finally:
        _cleanup(tmp_path)

    # --- ncu — required, no fallback ---
    ncu_result = _run_ncu(req.code, req.kernel_type)
    result.dram_utilization_pct = ncu_result.get("dram_utilization_pct")
    result.l1_hit_rate_pct = ncu_result.get("l1_hit_rate_pct")
    result.stall_memory_dependency_pct = ncu_result.get("stall_memory_dependency_pct")
    result.stall_long_scoreboard_pct = ncu_result.get("stall_long_scoreboard_pct")
    result.sm_active_cycles_pct = ncu_result.get("sm_active_cycles_pct")
    # ncu register count is more accurate than PTX estimation — prefer it
    if ncu_result.get("register_count_ncu") is not None:
        result.register_count = ncu_result["register_count_ncu"]

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_peak_bw(gpu_name: str) -> float:
    """Known peak memory bandwidths in GB/s by GPU model."""
    name = gpu_name.upper()
    if "H100" in name:
        return 3350.0
    if "A100" in name and "80" in name:
        return 2000.0
    if "A100" in name:
        return 1555.0
    if "A10G" in name or "A10" in name:
        return 600.0
    if "4090" in name:
        return 1008.0
    if "3090" in name:
        return 936.0
    return 900.0  # conservative fallback


def _extract_triton_metadata(
    code: str,
    namespace: dict,
    kernel_type: str,
    inputs: dict[str, Any],
) -> dict:
    """
    Extract register count, shared memory, warps, stages from Triton compiled artifact.
    We do this by running the kernel once so Triton JIT caches it, then inspecting cache.
    """
    # Trigger JIT compilation by running once
    _call_run(namespace, kernel_type, inputs)
    torch.cuda.synchronize()

    # Find the JIT-compiled kernel in namespace (it's the @triton.jit decorated fn)
    metadata = {}
    for name, obj in namespace.items():
        if callable(obj) and hasattr(obj, "cache"):
            for key, compiled in obj.cache.items():
                if hasattr(compiled, "metadata"):
                    m = compiled.metadata
                    metadata["num_warps"] = getattr(m, "num_warps", None)
                    metadata["num_stages"] = getattr(m, "num_stages", None)
                    metadata["shared_mem_bytes"] = getattr(m, "shared", None)

                    # Register count from PTX
                    if hasattr(compiled, "asm") and "ptx" in compiled.asm:
                        ptx = compiled.asm["ptx"]
                        for line in ptx.split("\n"):
                            if ".reg .f32" in line or "// .register" in line:
                                pass  # PTX parsing is complex; use nvcc --ptxas-options if needed
                    break

    # Theoretical occupancy via torch CUDA occupancy API
    device_props = torch.cuda.get_device_properties(0)
    num_warps = metadata.get("num_warps", 4)
    shared_mem = metadata.get("shared_mem_bytes", 0) or 0
    threads_per_block = num_warps * 32

    try:
        from torch.cuda import _get_device_properties
        max_warps_per_sm = device_props.max_threads_per_multi_processor // 32
        warps_per_block = threads_per_block // 32
        blocks_per_sm = min(
            device_props.max_blocks_per_multi_processor,
            max_warps_per_sm // warps_per_block,
        )
        active_warps = blocks_per_sm * warps_per_block
        occupancy = (active_warps / max_warps_per_sm) * 100
        metadata["theoretical_occupancy_pct"] = round(occupancy, 1)
    except Exception:
        pass

    return metadata


def _run_ncu(code: str, kernel_type: str) -> dict:
    """
    Run ncu inside a subprocess. Requires CAP_SYS_ADMIN on Linux (available on RunPod pods).
    Raises HTTPException(503) if ncu is not found or fails — no silent fallback.
    """
    ncu_path = _find_ncu()
    if not ncu_path:
        raise HTTPException(
            status_code=503,
            detail=(
                "ncu not found. Install Nsight Compute on the pod:\n"
                "  apt-get install -y nsight-compute\n"
                "Verify with: ncu --version"
            ),
        )

    harness = _build_ncu_harness(code, kernel_type)
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "kernel_run.py"
        report_path = Path(tmpdir) / "report"
        script_path.write_text(harness)

        metrics = ",".join([
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__t_sector_hit_rate.pct",
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.avg",
            "smsp__warp_issue_stalled_mio_throttle_per_warp_active.avg",
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg",
            "launch__registers_per_thread",
        ])

        cmd = [
            ncu_path,
            "--metrics", metrics,
            "--export", str(report_path),
            "--force-overwrite",
            "--target-processes", "all",
            sys.executable, str(script_path),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"ncu exited with code {e.returncode}.\n"
                    f"stderr: {e.stderr[-1000:]}\n\n"
                    "Common cause: missing CAP_SYS_ADMIN. "
                    "Ensure the RunPod pod has privileged access."
                ),
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=500, detail="ncu timed out after 120s.")

        csv_cmd = [ncu_path, "--import", str(report_path) + ".ncu-rep",
                   "--csv", "--page", "raw"]
        try:
            csv_result = subprocess.run(csv_cmd, capture_output=True, text=True, timeout=30,
                                        check=True)
            return _parse_ncu_csv(csv_result.stdout)
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500,
                detail=f"ncu CSV export failed: {e.stderr[-500:]}",
            )


def _find_ncu() -> str | None:
    """Find ncu binary on PATH or common CUDA install locations."""
    import shutil
    if shutil.which("ncu"):
        return "ncu"
    candidates = [
        "/usr/local/cuda/bin/ncu",
        "/usr/local/cuda-12/bin/ncu",
        "/opt/cuda/bin/ncu",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def _build_ncu_harness(code: str, kernel_type: str) -> str:
    """Build a standalone Python script that runs the kernel for ncu to profile."""
    inputs_code = {
        "rmsnorm": textwrap.dedent("""
            import torch; torch.manual_seed(42)
            x = torch.randn(2048, 4096, dtype=torch.float16, device='cuda')
            w = torch.ones(4096, dtype=torch.float16, device='cuda')
            # Warmup: triggers Triton JIT compile so profiled reps are pure execution
            for _ in range(25): run(x, w)
            torch.cuda.synchronize()
            # Profiled reps — ncu collects metrics from these
            for _ in range(10): run(x, w)
            torch.cuda.synchronize()
        """),
        "rope": textwrap.dedent("""
            import torch; torch.manual_seed(42)
            q = torch.randn(512, 32, 128, dtype=torch.float16, device='cuda')
            k = torch.randn(512, 32, 128, dtype=torch.float16, device='cuda')
            t = torch.arange(512, device='cuda').float()
            inv = 1.0 / (10000 ** (torch.arange(0, 64, device='cuda').float() / 64))
            f = torch.outer(t, inv)
            cos = torch.cat([f.cos(), f.cos()], -1).half()
            sin = torch.cat([f.sin(), f.sin()], -1).half()
            for _ in range(25): run(q, k, cos, sin)
            torch.cuda.synchronize()
            for _ in range(10): run(q, k, cos, sin)
            torch.cuda.synchronize()
        """),
        "fused_rmsnorm_rope": textwrap.dedent("""
            import torch; torch.manual_seed(42)
            x = torch.randn(512, 8192, dtype=torch.float16, device='cuda')
            w = torch.ones(8192, dtype=torch.float16, device='cuda')
            t = torch.arange(512, device='cuda').float()
            inv = 1.0 / (10000 ** (torch.arange(0, 64, device='cuda').float() / 64))
            f = torch.outer(t, inv)
            cos = torch.cat([f.cos(), f.cos()], -1).half()
            sin = torch.cat([f.sin(), f.sin()], -1).half()
            for _ in range(25): run(x, w, cos, sin, 32)
            torch.cuda.synchronize()
            for _ in range(10): run(x, w, cos, sin, 32)
            torch.cuda.synchronize()
        """),
    }
    return code + "\n" + inputs_code.get(kernel_type, "")


def _parse_ncu_csv(csv_text: str) -> dict:
    """Parse ncu CSV output into a flat metrics dict."""
    import csv
    import io

    result: dict = {}
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            # ncu CSV columns vary by version — try both naming conventions
            metric = row.get("Metric Name", "") or row.get("ID", "")
            value_str = (row.get("Metric Value", "") or row.get("Value", "") or "0")
            value_str = value_str.replace(",", "").strip()
            try:
                value = float(value_str)
            except ValueError:
                continue

            if "sm__throughput" in metric:
                result["sm_active_cycles_pct"] = round(value, 1)
            elif "dram__throughput" in metric:
                result["dram_utilization_pct"] = round(value, 1)
            elif "sector_hit_rate" in metric:
                result["l1_hit_rate_pct"] = round(value, 1)
            elif "stalled_long_scoreboard" in metric:
                result["stall_long_scoreboard_pct"] = round(value, 1)
            elif "stalled_mio_throttle" in metric:
                result["stall_memory_dependency_pct"] = round(value, 1)
            elif "registers_per_thread" in metric:
                result["register_count_ncu"] = int(value)
    except Exception:
        pass

    return result

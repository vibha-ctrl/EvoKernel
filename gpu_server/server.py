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
def _require_nsys():
    nsys = _find_nsys()
    if not nsys:
        raise RuntimeError(
            "nsys (Nsight Systems) not found.\n"
            "It should be pre-installed in the CUDA devel image. Check with: nsys --version\n"
            "If missing: apt-get install -y nsight-systems\n"
            "EvoKernel requires real hardware profiling — no fallback."
        )


@app.on_event("startup")
def _warm_up_gpu():
    import torch
    device = "cuda"
    a = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    b = torch.randn(4096, 4096, dtype=torch.float16, device=device)
    for _ in range(500):
        torch.mm(a, b)
    torch.cuda.synchronize()
    del a, b


class KernelRequest(BaseModel):
    code: str
    kernel_type: str
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
    num_warps: int | None = None
    num_stages: int | None = None
    shared_mem_bytes: int | None = None
    register_count: int | None = None
    theoretical_occupancy_pct: float | None = None
    dram_utilization_pct: float | None = None
    l1_hit_rate_pct: float | None = None
    stall_memory_dependency_pct: float | None = None
    stall_long_scoreboard_pct: float | None = None
    sm_active_cycles_pct: float | None = None
    latency_us: float | None = None
    throughput_gb_s: float | None = None


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
    import linecache
    import uuid as _uuid

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="evokernel_candidate_",
        delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(code)
        tmp_path = f.name

    # Populate linecache so inspect.getsource works for Triton's @triton.jit
    lines = code.splitlines(keepends=True)
    linecache.cache[tmp_path] = (len(code), None, lines, tmp_path)

    # Unique module name avoids stale sys.modules entries across calls
    module_name = f"evokernel_candidate_{_uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, tmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
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
    if kernel_type == "rmsnorm":
        M, N = inputs["x"].shape
        return (M * N + N + M * N) * 2
    elif kernel_type == "rope":
        q = inputs["q"]
        S, H, D = q.shape
        return S * H * D * 2 * 4 * 2
    elif kernel_type == "fused_rmsnorm_rope":
        S, N = inputs["x"].shape
        return (S * N + N) * 2 + S * (N // 2) * 2 * 2
    return 0


@app.get("/health")
def health():
    device = torch.cuda.get_device_properties(0)
    nsys_path = _find_nsys()
    nsys_version = None
    if nsys_path:
        try:
            r = subprocess.run([nsys_path, "--version"], capture_output=True, text=True, timeout=5)
            nsys_version = r.stdout.strip().splitlines()[0] if r.returncode == 0 else "unknown"
        except Exception:
            nsys_version = "error"
    return {
        "status": "ok",
        "gpu": device.name,
        "vram_gb": round(device.total_memory / 1e9, 1),
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "nsys_path": nsys_path,
        "nsys_version": nsys_version,
    }


@app.get("/debug/nsys")
def debug_nsys():
    """Run a minimal nsys profile and return raw CSV + any errors for debugging."""
    nsys_path = _find_nsys()
    if not nsys_path:
        return {"error": "nsys not found"}

    minimal_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def _debug_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N)
    tl.store(Y + offs, x * 2, mask=offs < N)

def run(x):
    y = torch.empty_like(x)
    _debug_kernel[(x.numel() // 128,)](x, y, x.numel(), BLOCK=128, num_warps=4)
    return y

import torch
x = torch.randn(4096, device='cuda', dtype=torch.float16)
for _ in range(10): run(x)
torch.cuda.synchronize()
"""
    import tempfile
    from pathlib import Path as _Path
    with tempfile.TemporaryDirectory() as tmpdir:
        script = _Path(tmpdir) / "debug_run.py"
        report = _Path(tmpdir) / "report"
        script.write_text(minimal_code)
        cmd = [nsys_path, "profile", "--trace=cuda", "--output", str(report),
               "--force-overwrite", "true", sys.executable, str(script)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        rep_file = _Path(str(report) + ".nsys-rep")

        if not rep_file.exists():
            return {
                "error": "no .nsys-rep produced",
                "returncode": proc.returncode,
                "stdout": proc.stdout[-800:],
                "stderr": proc.stderr[-800:],
            }

        report_name = "cuda_gpu_kern_sum"
        stats_cmd = [nsys_path, "stats", "--report", report_name,
                     "--format", "csv", "--output", "-", str(rep_file)]
        stats = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=30)
        if "could not be found" in stats.stderr or stats.returncode != 0:
            report_name = "gpukernsum"
            stats_cmd[3] = report_name
            stats = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=30)
        return {
            "nsys_path": nsys_path,
            "report_name_used": report_name,
            "profile_returncode": proc.returncode,
            "stats_returncode": stats.returncode,
            "csv_output": stats.stdout[:2000],
            "stats_stderr": stats.stderr[:500],
        }


@app.get("/debug/triton_cache")
def debug_triton_cache():
    """Compile a minimal Triton kernel via subprocess and dump cache structure."""
    probe_code = """
import torch, triton, triton.language as tl, json, sys

@triton.jit
def _probe(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N)
    tl.store(X + offs, x, mask=offs < N)

x = torch.randn(1024, device='cuda', dtype=torch.float32)
_probe[(8,)](x, 1024, BLOCK=128, num_warps=4, num_stages=2)
torch.cuda.synchronize()

cache = _probe.cache
info = {"cache_type": type(cache).__name__}
for k, v in cache.items():
    if isinstance(v, dict):
        info["structure"] = "nested"
        for k2, compiled in v.items():
            info["compiled_type"] = type(compiled).__name__
            info["has_metadata"] = hasattr(compiled, "metadata")
            if hasattr(compiled, "metadata"):
                m = compiled.metadata
                info["metadata_type"] = type(m).__name__
                info["metadata_attrs"] = [a for a in dir(m) if not a.startswith("_")]
                info["num_warps"] = getattr(m, "num_warps", "MISSING")
                info["num_stages"] = getattr(m, "num_stages", "MISSING")
                info["shared"] = getattr(m, "shared", "MISSING")
            break
    else:
        info["structure"] = "flat"
        info["compiled_type"] = type(v).__name__
        info["has_metadata"] = hasattr(v, "metadata")
        if hasattr(v, "metadata"):
            m = v.metadata
            info["num_warps"] = getattr(m, "num_warps", "MISSING")
            info["num_stages"] = getattr(m, "num_stages", "MISSING")
    break

print(json.dumps(info))
"""
    import tempfile, json as _json
    from pathlib import Path as _Path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(probe_code)
        tmp = f.name
    try:
        proc = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=30)
        last_line = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
        if last_line:
            return _json.loads(last_line[-1])
        return {"error": "no json output", "stdout": proc.stdout[-500:], "stderr": proc.stderr[-500:]}
    finally:
        os.unlink(tmp)


@app.post("/verify", response_model=VerifyResponse)
def verify(req: KernelRequest):
    inputs = _get_test_inputs(req.kernel_type)

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
    inputs = _get_test_inputs(req.kernel_type)

    try:
        namespace, tmp_path = _exec_candidate(req.code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Code error: {e}")

    def bench_fn():
        return _call_run(namespace, req.kernel_type, inputs)

    try:
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
    inputs = _get_test_inputs(req.kernel_type)

    try:
        namespace, tmp_path = _exec_candidate(req.code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Code error: {e}")

    result = ProfileResponse(candidate_id=req.candidate_id)

    try:
        meta = _extract_triton_metadata(req.code, namespace, req.kernel_type, inputs)
        result.num_warps = meta.get("num_warps")
        result.num_stages = meta.get("num_stages")
        result.shared_mem_bytes = meta.get("shared_mem_bytes")
        result.register_count = meta.get("register_count")
        result.theoretical_occupancy_pct = meta.get("theoretical_occupancy_pct")

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

    inputs_for_bytes = _get_test_inputs(req.kernel_type)
    bytes_total = _bytes_accessed(req.kernel_type, inputs_for_bytes)
    nsys_result = _run_nsys(req.code, req.kernel_type, bytes_total)
    result.dram_utilization_pct = nsys_result.get("dram_utilization_pct")
    result.sm_active_cycles_pct = nsys_result.get("sm_active_cycles_pct")

    return result


def _estimate_peak_bw(gpu_name: str) -> float:
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
    return 900.0


def _extract_triton_metadata(
    code: str,
    namespace: dict,
    kernel_type: str,
    inputs: dict[str, Any],
) -> dict:
    _call_run(namespace, kernel_type, inputs)
    torch.cuda.synchronize()

    metadata = {}
    for name, obj in namespace.items():
        if not (callable(obj) and hasattr(obj, "cache")):
            continue
        cache = obj.cache
        # Triton 3.x: cache is {device_id: {key: CompiledKernel}}
        # Triton 2.x: cache is {key: CompiledKernel}
        compiled = None
        for v in cache.values():
            if isinstance(v, dict):
                # Triton 3.x nested cache — pick the first compiled entry
                for inner in v.values():
                    if hasattr(inner, "metadata"):
                        compiled = inner
                        break
            elif hasattr(v, "metadata"):
                # Triton 2.x flat cache
                compiled = v
            if compiled:
                break

        if compiled and hasattr(compiled, "metadata"):
            m = compiled.metadata
            metadata["num_warps"] = getattr(m, "num_warps", None)
            metadata["num_stages"] = getattr(m, "num_stages", None)
            metadata["shared_mem_bytes"] = getattr(m, "shared", None)
            # register count from PTX
            if hasattr(compiled, "asm") and isinstance(compiled.asm, dict) and "ptx" in compiled.asm:
                for line in compiled.asm["ptx"].split("\n"):
                    if ".reg .b32" in line or ".reg .f32" in line:
                        import re
                        m2 = re.search(r"<(\d+)>", line)
                        if m2:
                            metadata["register_count"] = int(m2.group(1))
                            break
            break

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


def _find_nsys() -> str | None:
    import shutil
    if shutil.which("nsys"):
        return "nsys"
    candidates = [
        "/opt/nvidia/nsight-systems/2024.3.2/bin/nsys",
        "/opt/nvidia/nsight-systems/2024.1.1/bin/nsys",
        "/opt/nvidia/nsight-systems/2023.4.4/bin/nsys",
        "/usr/local/cuda/bin/nsys",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def _build_profile_harness(code: str, kernel_type: str) -> str:
    inputs_code = {
        "rmsnorm": textwrap.dedent("""
            import torch; torch.manual_seed(42)
            x = torch.randn(2048, 4096, dtype=torch.float16, device='cuda')
            w = torch.ones(4096, dtype=torch.float16, device='cuda')
            for _ in range(25): run(x, w)
            torch.cuda.synchronize()
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


def _run_nsys(code: str, kernel_type: str, bytes_accessed: int) -> dict:
    nsys_path = _find_nsys()
    if not nsys_path:
        raise HTTPException(
            status_code=503,
            detail=(
                "nsys (Nsight Systems) not found on the pod.\n"
                "It is pre-installed in CUDA devel images. Check with: nsys --version\n"
                "If missing: apt-get install -y nsight-systems"
            ),
        )

    harness = _build_profile_harness(code, kernel_type)
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "kernel_run.py"
        report_path = Path(tmpdir) / "report"
        script_path.write_text(harness)

        cmd = [
            nsys_path, "profile",
            "--trace=cuda",
            "--output", str(report_path),
            "--force-overwrite", "true",
            sys.executable, str(script_path),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            rep_file = Path(str(report_path) + ".nsys-rep")
            if not rep_file.exists():
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"nsys exited with code {proc.returncode}. No .nsys-rep produced.\n"
                        f"stdout: {proc.stdout[-600:]}\nstderr: {proc.stderr[-600:]}"
                    ),
                )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=500, detail="nsys timed out after 180s.")

        # nsys 2024+ uses "cuda_gpu_kern_sum"; older versions use "gpukernsum"
        report_name = "cuda_gpu_kern_sum"
        stats_cmd = [
            nsys_path, "stats",
            "--report", report_name,
            "--format", "csv",
            "--output", "-",
            str(rep_file),
        ]
        stats = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=30)
        if "could not be found" in stats.stderr or stats.returncode != 0:
            # Fall back to legacy report name
            report_name = "gpukernsum"
            stats_cmd[-3] = report_name
            stats = subprocess.run(stats_cmd, capture_output=True, text=True, timeout=30)

        try:
            return _parse_nsys_gpukernsum(stats.stdout, bytes_accessed)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"nsys stats failed: {e}")


def _parse_nsys_gpukernsum(csv_text: str, bytes_accessed: int) -> dict:
    import csv as csv_mod
    import io

    result: dict = {}
    try:
        # nsys stats prepends status lines before the CSV — skip to the actual header
        csv_lines = [l for l in csv_text.splitlines() if l.startswith("Time")]
        if not csv_lines:
            return result
        # Reconstruct CSV from the header line onward
        header_idx = csv_text.find(csv_lines[0])
        csv_text = csv_text[header_idx:]
        reader = csv_mod.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        if not rows:
            return result

        def total_time(r: dict) -> float:
            for key in ("Total Time (ns)", "Total Time(ns)", "TotalTime"):
                if key in r:
                    try:
                        return float(r[key].replace(",", ""))
                    except (ValueError, AttributeError):
                        pass
            return 0.0

        def avg_time(r: dict) -> float:
            for key in ("Avg (ns)", "Avg(ns)", "AvgTime"):
                if key in r:
                    try:
                        return float(r[key].replace(",", ""))
                    except (ValueError, AttributeError):
                        pass
            return 0.0

        def time_pct(r: dict) -> float:
            for key in ("Time (%)", "Time(%)", "Pct"):
                if key in r:
                    try:
                        return float(r[key].replace(",", ""))
                    except (ValueError, AttributeError):
                        pass
            return 0.0

        target = max(rows, key=total_time)
        avg_ns = avg_time(target)
        pct = time_pct(target)

        if avg_ns > 0 and bytes_accessed > 0:
            avg_s = avg_ns / 1e9
            throughput_gb_s = (bytes_accessed / 1e9) / avg_s
            peak_bw = _estimate_peak_bw(torch.cuda.get_device_properties(0).name)
            result["dram_utilization_pct"] = round(min(throughput_gb_s / peak_bw * 100, 100.0), 1)

        if pct > 0:
            result["sm_active_cycles_pct"] = round(min(pct, 100.0), 1)

    except Exception as e:
        import logging
        logging.warning(f"nsys CSV parse failed: {e}. CSV head: {csv_text[:300]!r}")

    return result

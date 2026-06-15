"""
Evolutionary Search Engine — the main loop.

Generate → Verify → Benchmark → Profile → Select → Mutate → Repeat

Orchestrates calls to:
  - RunPod GPU server (via HTTP)
  - Generator agent (LLM)
  - Critic agent (LLM)
  - CandidateStore (SQLite)
"""

import concurrent.futures
import time
from dataclasses import dataclass

import httpx
from rich.console import Console
from rich.table import Table

from evokernel.agents import generator as gen_agent
from evokernel.agents import critic as critic_agent
from evokernel.agents.generator import GeneratorContext, FailedCandidate
from evokernel.agents.critic import ProfileData
from evokernel.search.candidate_store import Candidate, CandidateStore
from evokernel.reports.results_tracker import write_generation_update, write_summary

console = Console()


@dataclass
class SearchConfig:
    kernel_type: str               # "rmsnorm" | "rope" | "fused_rmsnorm_rope"
    max_generations: int = 10
    candidates_per_generation: int = 10
    top_k: int = 3                 # survivors per generation
    convergence_threshold_pct: float = 2.0  # stop if improvement < this
    runpod_url: str = ""           # e.g. https://abc-8000.proxy.runpod.net
    anthropic_api_key: str = ""
    db_path: str = "evokernel.db"
    parallel_verify: bool = True
    parallel_benchmark: bool = True


def run_search(
    baseline_code: str,
    config: SearchConfig,
) -> Candidate:
    """
    Run the full evolutionary search.

    Args:
        baseline_code: Python source of the starting kernel (must define run())
        config: SearchConfig

    Returns:
        The best Candidate found.
    """
    store = CandidateStore(config.db_path)
    client = httpx.Client(base_url=config.runpod_url, timeout=120.0)

    console.rule("[bold blue]EvoKernel Search Starting")
    console.print(f"  Kernel type  : [cyan]{config.kernel_type}[/cyan]")
    console.print(f"  Max gens     : {config.max_generations}")
    console.print(f"  Candidates/gen: {config.candidates_per_generation}")
    console.print(f"  Top-k        : {config.top_k}")

    # ------------------------------------------------------------------ #
    # Generation 0: establish baseline                                     #
    # ------------------------------------------------------------------ #
    baseline = Candidate(
        code=baseline_code,
        kernel_type=config.kernel_type,
        generation=0,
    )
    store.save(baseline)

    console.print("\n[bold]Generation 0[/bold] — benchmarking baseline...")
    _verify_one(client, baseline, store)
    baseline = store.get(baseline.id)  # refresh — _verify_one updates DB, not the local object
    if not baseline.is_verified:
        raise RuntimeError(f"Baseline kernel failed verification: {baseline.verify_error_msg}")

    _benchmark_one(client, baseline, store)
    _profile_one(client, baseline, store)
    baseline = store.get(baseline.id)  # refresh

    console.print(f"  Baseline latency: [green]{baseline.latency_us:.1f} µs[/green]")

    # Write initial results file so gen-0 baseline is tracked
    write_generation_update(store, config.kernel_type, 0, baseline,
                            baseline.latency_us, critic_diagnosis="Baseline — search not started")
    write_summary(store, ["rmsnorm", "rope", "fused_rmsnorm_rope"])

    gpu_info = _get_gpu_info(client)
    gpu_name = gpu_info.get("gpu", "Unknown GPU")
    peak_bw = gpu_info.get("peak_bandwidth_gb_s", 2000.0)

    best_latency = baseline.latency_us
    parents = [baseline]

    # ------------------------------------------------------------------ #
    # Main evolutionary loop                                               #
    # ------------------------------------------------------------------ #
    for gen in range(1, config.max_generations + 1):
        gen_start = time.time()
        console.rule(f"[bold blue]Generation {gen}")

        # Pick the best parent to mutate from (lowest latency)
        primary_parent = min(parents, key=lambda c: c.latency_us or float("inf"))

        # Get critic hints from previous generation profiles
        critic_hints: list[str] = []
        critic_diagnosis = ""
        profiled_parents = [p for p in parents if p.is_profiled]
        if profiled_parents:
            profile_data = [_candidate_to_profile_data(p, peak_bw) for p in profiled_parents]
            critic_result = critic_agent.analyze_profiles(profile_data, gpu_name,
                                                          config.anthropic_api_key)
            critic_hints = critic_result.recommendations
            critic_diagnosis = critic_result.diagnosis
            console.print(f"  [yellow]Critic:[/yellow] {critic_result.diagnosis}")

        # Build generator context
        ctx = GeneratorContext(
            kernel_type=config.kernel_type,
            parent_code=primary_parent.code,
            parent_latency_us=primary_parent.latency_us,
            generation=gen,
            gpu_name=gpu_name,
            peak_bandwidth_gb_s=peak_bw,
            critic_hints=critic_hints,
            failed_candidates=[
                FailedCandidate(
                    candidate_id=f.id,
                    code=f.code,
                    error_type=f.verify_error_type or "unknown",
                    error_msg=f.verify_error_msg or "",
                )
                for f in store.get_failed(config.kernel_type, gen - 1)
            ],
            tried_configs=store.get_all_tried_configs(config.kernel_type),
            n_variants=config.candidates_per_generation,
        )

        # Generate candidates
        console.print(f"  Generating {config.candidates_per_generation} variants...")
        raw_variants = gen_agent.generate_variants(ctx, config.anthropic_api_key)
        console.print(f"  [dim]LLM returned {len(raw_variants)} code blocks[/dim]")

        candidates = [
            Candidate(code=code, kernel_type=config.kernel_type, generation=gen,
                      parent_id=primary_parent.id)
            for code in raw_variants
        ]
        for c in candidates:
            store.save(c)

        # Verify all candidates
        console.print(f"  Verifying {len(candidates)} candidates...")
        _verify_batch(client, candidates, store, parallel=config.parallel_verify)
        passed = [c for c in candidates if store.get(c.id).verify_passed]
        console.print(f"  [green]{len(passed)} passed[/green], "
                      f"[red]{len(candidates)-len(passed)} failed[/red]")

        if not passed:
            console.print("  [red]No candidates passed verification — skipping generation[/red]")
            continue

        # Benchmark all passing candidates
        console.print(f"  Benchmarking {len(passed)} valid candidates...")
        _benchmark_batch(client, passed, store, parallel=config.parallel_benchmark)

        # Refresh from DB (has updated latency)
        passed = [store.get(c.id) for c in passed]
        passed.sort(key=lambda c: c.latency_us or float("inf"))

        _print_generation_table(gen, passed)

        # Profile top-k
        top = passed[:config.top_k]
        console.print(f"  Profiling top {len(top)} candidates...")
        for c in top:
            _profile_one(client, c, store)

        # Check convergence
        gen_best = passed[0].latency_us
        improvement_pct = (best_latency - gen_best) / best_latency * 100
        console.print(
            f"  Generation best: [green]{gen_best:.1f} µs[/green]  "
            f"(improvement: {improvement_pct:+.1f}%)"
        )

        if gen_best < best_latency:
            best_latency = gen_best

        # Write live results after every generation
        write_generation_update(
            store, config.kernel_type, gen, passed[0],
            baseline.latency_us, critic_diagnosis,
        )
        write_summary(store, ["rmsnorm", "rope", "fused_rmsnorm_rope"])

        if gen > 2 and improvement_pct < config.convergence_threshold_pct:
            console.print(
                f"  [yellow]Converged[/yellow] — improvement {improvement_pct:.1f}% "
                f"< threshold {config.convergence_threshold_pct}%"
            )
            break

        parents = [store.get(c.id) for c in top]
        console.print(f"  Elapsed: {time.time()-gen_start:.1f}s")

    # ------------------------------------------------------------------ #
    # Return overall best                                                  #
    # ------------------------------------------------------------------ #
    bests = store.get_best(config.kernel_type, n=1)
    best = bests[0] if bests else baseline
    console.rule("[bold green]Search Complete")
    console.print(f"  Baseline : {baseline.latency_us:.1f} µs")
    console.print(f"  Best     : [bold green]{best.latency_us:.1f} µs[/bold green]")
    speedup = baseline.latency_us / best.latency_us
    console.print(f"  Speedup  : [bold green]{speedup:.2f}x[/bold green]")
    return best


# ---------------------------------------------------------------------------
# RunPod client helpers
# ---------------------------------------------------------------------------

def _get_gpu_info(client: httpx.Client) -> dict:
    try:
        resp = client.get("/health")
        resp.raise_for_status()
        data = resp.json()
        gpu_name = data.get("gpu", "")
        peak_bw = {
            "H100": 3350.0, "A100": 2000.0, "A10G": 600.0, "4090": 1008.0,
        }
        for key, bw in peak_bw.items():
            if key in gpu_name:
                data["peak_bandwidth_gb_s"] = bw
                break
        data.setdefault("peak_bandwidth_gb_s", 900.0)
        return data
    except Exception:
        return {"gpu": "Unknown", "peak_bandwidth_gb_s": 900.0}


def _verify_one(client: httpx.Client, c: Candidate, store: CandidateStore):
    try:
        resp = client.post("/verify", json={
            "code": c.code,
            "kernel_type": c.kernel_type,
            "candidate_id": c.id,
        })
        resp.raise_for_status()
        data = resp.json()
        store.update_verify(c.id, data["passed"], data.get("error_type"),
                            data.get("error_msg"), data.get("max_error"))
    except Exception as e:
        store.update_verify(c.id, False, "http_error", str(e), None)


def _benchmark_one(client: httpx.Client, c: Candidate, store: CandidateStore):
    try:
        resp = client.post("/benchmark", json={
            "code": c.code,
            "kernel_type": c.kernel_type,
            "candidate_id": c.id,
        })
        resp.raise_for_status()
        data = resp.json()
        store.update_benchmark(
            c.id,
            data["latency_us"],
            data["latency_p99_us"],
            data["throughput_gb_s"],
            data["bandwidth_utilization_pct"],
        )
    except Exception as e:
        console.print(f"  [red]Benchmark failed for {c.id}: {e}[/red]")


def _profile_one(client: httpx.Client, c: Candidate, store: CandidateStore):
    try:
        resp = client.post("/profile", json={
            "code": c.code,
            "kernel_type": c.kernel_type,
            "candidate_id": c.id,
        })
        resp.raise_for_status()
        store.update_profile(c.id, resp.json())
    except Exception as e:
        console.print(f"  [yellow]Profile failed for {c.id}: {e}[/yellow]")


def _verify_batch(client: httpx.Client, candidates: list[Candidate],
                  store: CandidateStore, parallel: bool = True):
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(lambda c: _verify_one(client, c, store), candidates))
    else:
        for c in candidates:
            _verify_one(client, c, store)


def _benchmark_batch(client: httpx.Client, candidates: list[Candidate],
                     store: CandidateStore, parallel: bool = False):
    # Benchmarks are ALWAYS sequential — parallel GPU execution causes memory bandwidth
    # contention between candidates, inflating all latency numbers and making comparisons
    # meaningless. parallel flag kept for API compatibility but ignored here.
    for c in candidates:
        _benchmark_one(client, c, store)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_generation_table(gen: int, candidates: list[Candidate]):
    table = Table(title=f"Generation {gen} Results", show_lines=False)
    table.add_column("Rank", style="dim", width=4)
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Latency (µs)", justify="right")
    table.add_column("Throughput (GB/s)", justify="right")
    table.add_column("BW Util %", justify="right")

    for i, c in enumerate(candidates[:10]):
        style = "bold green" if i == 0 else ""
        table.add_row(
            str(i + 1),
            c.label,
            f"{c.latency_us:.1f}" if c.latency_us else "—",
            f"{c.throughput_gb_s:.0f}" if c.throughput_gb_s else "—",
            f"{c.bandwidth_utilization_pct:.0f}%" if c.bandwidth_utilization_pct else "—",
            style=style,
        )
    console.print(table)


def _candidate_to_profile_data(c: Candidate, peak_bw: float) -> ProfileData:
    return ProfileData(
        candidate_id=c.label,
        kernel_type=c.kernel_type,
        latency_us=c.latency_us or 0.0,
        throughput_gb_s=c.throughput_gb_s or 0.0,
        peak_bandwidth_gb_s=peak_bw,
        num_warps=c.num_warps,
        num_stages=c.num_stages,
        shared_mem_bytes=c.shared_mem_bytes,
        register_count=c.register_count,
        theoretical_occupancy_pct=c.theoretical_occupancy_pct,
        dram_utilization_pct=c.dram_utilization_pct,
        l1_hit_rate_pct=c.l1_hit_rate_pct,
        stall_memory_dependency_pct=c.stall_memory_dependency_pct,
        stall_long_scoreboard_pct=c.stall_long_scoreboard_pct,
        sm_active_cycles_pct=c.sm_active_cycles_pct,
    )

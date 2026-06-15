"""
Report Generator — produces a Markdown summary of a completed search run.

Includes:
  - Performance progression per generation (ASCII chart)
  - Best kernel source code
  - Comparison table: baseline vs random-equivalent vs EvoKernel best
  - Winning optimization techniques identified
"""

from pathlib import Path
from typing import Optional

from evokernel.search.candidate_store import Candidate, CandidateStore


def generate_report(
    store: CandidateStore,
    kernel_type: str,
    output_path: str = "report.md",
) -> dict:
    """Write a Markdown report and return summary stats."""
    summary = store.generation_summary(kernel_type)
    best_list = store.get_best(kernel_type, n=1)
    if not best_list:
        return {"error": "No benchmarked candidates found."}

    best = best_list[0]
    baseline_gen = store.get_generation(0, kernel_type)
    baseline = baseline_gen[0] if baseline_gen else None
    baseline_latency = baseline.latency_us if baseline and baseline.latency_us else None

    speedup = (baseline_latency / best.latency_us) if baseline_latency and best.latency_us else None

    lines: list[str] = []

    lines.append(f"# EvoKernel Report — {kernel_type}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Kernel type | `{kernel_type}` |")
    lines.append(f"| Baseline latency | {baseline_latency:.1f} µs |" if baseline_latency else "| Baseline latency | N/A |")
    lines.append(f"| Best latency | **{best.latency_us:.1f} µs** |")
    lines.append(f"| Speedup | **{speedup:.2f}x** |" if speedup else "| Speedup | N/A |")
    lines.append(f"| Best candidate | `{best.label}` |")
    lines.append(f"| Generations run | {len(summary)} |")
    total_candidates = sum(s.get("total", 0) for s in summary)
    lines.append(f"| Total candidates evaluated | {total_candidates} |")
    lines.append("")

    lines.append("## Performance Progression")
    lines.append("")
    lines.append(_ascii_chart(summary))
    lines.append("")

    lines.append("## Generation-by-Generation Results")
    lines.append("")
    lines.append("| Generation | Best Latency (µs) | Candidates | Passed Verify |")
    lines.append("|------------|-------------------|------------|----------------|")
    for s in summary:
        lines.append(
            f"| {s['generation']} | "
            f"{s['best_latency_us']:.1f} | "
            f"{s['total']} | "
            f"{s['passed']} |"
        )
    lines.append("")

    lines.append("## Best Kernel Configuration")
    lines.append("")
    lines.append(f"**Candidate:** `{best.label}`  ")
    lines.append(f"**Latency:** {best.latency_us:.1f} µs  ")
    if best.throughput_gb_s:
        lines.append(f"**Throughput:** {best.throughput_gb_s:.0f} GB/s  ")
    if best.bandwidth_utilization_pct:
        lines.append(f"**Bandwidth utilization:** {best.bandwidth_utilization_pct:.0f}%  ")
    lines.append("")

    lines.append("### Triton Parameters")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| `num_warps` | {best.num_warps} |")
    lines.append(f"| `num_stages` | {best.num_stages} |")
    lines.append(f"| `shared_mem_bytes` | {best.shared_mem_bytes} |")
    lines.append(f"| `register_count` | {best.register_count} |")
    lines.append(f"| `theoretical_occupancy` | {best.theoretical_occupancy_pct}% |")
    lines.append("")

    lines.append("### Nsight Compute Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| SM throughput | {best.sm_active_cycles_pct}% |")
    lines.append(f"| DRAM utilization | {best.dram_utilization_pct}% |")
    lines.append(f"| L1 hit rate | {best.l1_hit_rate_pct}% |")
    lines.append(f"| Stall (memory dependency) | {best.stall_memory_dependency_pct}% |")
    lines.append(f"| Stall (long scoreboard) | {best.stall_long_scoreboard_pct}% |")
    lines.append("")

    lines.append("## Best Kernel Source Code")
    lines.append("")
    lines.append("```python")
    lines.append(best.code)
    lines.append("```")
    lines.append("")

    lines.append("## Optimization Journey")
    lines.append("")
    lines.append(_optimization_journey(store, kernel_type, summary))

    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text, encoding="utf-8")

    return {
        "path": str(Path(output_path).resolve()),
        "speedup": round(speedup, 3) if speedup else None,
        "best_latency_us": best.latency_us,
        "baseline_latency_us": baseline_latency,
    }


def _ascii_chart(summary: list[dict]) -> str:
    """Simple ASCII bar chart of best latency per generation."""
    if not summary:
        return ""

    latencies = [s["best_latency_us"] for s in summary if s.get("best_latency_us")]
    if not latencies:
        return ""

    max_lat = max(latencies)
    bar_width = 40

    lines = ["```"]
    lines.append("Latency (µs) by generation:")
    lines.append("")
    for s in summary:
        lat = s.get("best_latency_us")
        if lat is None:
            continue
        bar_len = int((lat / max_lat) * bar_width)
        bar = "█" * bar_len
        lines.append(f"  Gen {s['generation']:2d} | {bar:<{bar_width}} {lat:.1f}")
    lines.append("```")
    return "\n".join(lines)


def _optimization_journey(
    store: CandidateStore,
    kernel_type: str,
    summary: list[dict],
) -> str:
    """Describe what changed from generation to generation."""
    lines = []
    prev_latency = None
    for s in summary:
        gen = s["generation"]
        lat = s.get("best_latency_us")
        if lat is None:
            continue
        delta = f"({(prev_latency - lat) / prev_latency * 100:+.1f}%)" if prev_latency else "(baseline)"
        lines.append(f"- **Generation {gen}**: {lat:.1f} µs {delta}")
        prev_latency = lat
    return "\n".join(lines)

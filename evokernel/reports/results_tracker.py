"""
Live results tracker — writes human-readable Markdown to results/ after every generation.

These files are meant to be committed to git so the search progress
is visible on GitHub without needing to open the SQLite database.
"""

from datetime import datetime
from pathlib import Path

from evokernel.search.candidate_store import Candidate, CandidateStore

RESULTS_DIR = Path("results")


def write_generation_update(
    store: CandidateStore,
    kernel_type: str,
    generation: int,
    best_this_gen: Candidate,
    baseline_latency_us: float,
    critic_diagnosis: str = "",
):
    """
    Called after every generation completes.
    Updates results/<kernel_type>.md with the latest generation result.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{kernel_type}.md"

    summary = store.generation_summary(kernel_type)
    best_overall = store.get_best(kernel_type, n=1)
    best = best_overall[0] if best_overall else best_this_gen

    speedup = baseline_latency_us / best.latency_us if best.latency_us else None

    lines = []
    lines.append(f"# {kernel_type} — Search Results")
    lines.append(f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")

    lines.append("## Progress")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Baseline latency | {baseline_latency_us:.1f} µs |")
    lines.append(f"| Current best | **{best.latency_us:.1f} µs** |")
    if speedup:
        lines.append(f"| Current speedup | **{speedup:.2f}x** |")
    lines.append(f"| Generations completed | {generation} |")
    total = sum(s.get("total", 0) for s in summary)
    passed = sum(s.get("passed", 0) for s in summary)
    lines.append(f"| Candidates evaluated | {total} ({passed} passed correctness) |")
    lines.append("")

    lines.append("## Generation-by-Generation")
    lines.append("")
    lines.append(_ascii_progress_chart(summary, baseline_latency_us))
    lines.append("")
    lines.append("| Generation | Best Latency (µs) | vs Baseline | Candidates | Passed |")
    lines.append("|------------|-------------------|-------------|------------|--------|")
    for s in summary:
        lat = s.get("best_latency_us")
        delta = ""
        if lat and baseline_latency_us:
            pct = (baseline_latency_us - lat) / baseline_latency_us * 100
            delta = f"{pct:+.1f}%"
        lat_str = f"{lat:.1f}" if lat else "—"
        bold = "**" if s['generation'] == generation else ""
        lines.append(
            f"| {s['generation']} | "
            f"{bold}{lat_str}{bold} | "
            f"{delta} | "
            f"{s['total']} | "
            f"{s['passed']} |"
        )
    lines.append("")

    if critic_diagnosis:
        lines.append(f"## Latest Critic Diagnosis (Generation {generation})")
        lines.append("")
        lines.append(f"> {critic_diagnosis}")
        lines.append("")

    lines.append("## Current Best Kernel")
    lines.append("")
    lines.append(f"**Candidate:** `{best.label}`  ")
    lines.append(f"**Generation:** {best.generation}  ")
    lines.append(f"**Latency:** {best.latency_us:.1f} µs  ")
    if best.throughput_gb_s:
        lines.append(f"**Throughput:** {best.throughput_gb_s:.0f} GB/s  ")
    if best.bandwidth_utilization_pct:
        lines.append(f"**BW utilization:** {best.bandwidth_utilization_pct:.0f}%  ")
    lines.append("")

    if best.num_warps is not None:
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| `num_warps` | {best.num_warps} |")
        lines.append(f"| `num_stages` | {best.num_stages} |")
        lines.append(f"| `register_count` | {best.register_count} |")
        lines.append(f"| `theoretical_occupancy` | {best.theoretical_occupancy_pct}% |")
        if best.dram_utilization_pct is not None:
            lines.append(f"| `dram_utilization` | {best.dram_utilization_pct}% |")
        if best.stall_memory_dependency_pct is not None:
            lines.append(f"| `stall_memory_dep` | {best.stall_memory_dependency_pct}% |")
        lines.append("")

    lines.append("```python")
    lines.append(best.code)
    lines.append("```")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(store: CandidateStore, kernel_types: list[str]):
    """
    Write results/summary.md comparing all kernels side by side.
    Called after each generation for any kernel.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "summary.md"

    lines = []
    lines.append("# EvoKernel — Summary")
    lines.append(f"*Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append("")
    lines.append("| Kernel | Baseline (µs) | Best (µs) | Speedup | Generations |")
    lines.append("|--------|---------------|-----------|---------|-------------|")

    for kt in kernel_types:
        summary = store.generation_summary(kt)
        if not summary:
            lines.append(f"| `{kt}` | — | — | — | 0 |")
            continue
        baseline = next((s["best_latency_us"] for s in summary if s["generation"] == 0), None)
        best_list = store.get_best(kt, n=1)
        best_lat = best_list[0].latency_us if best_list else None
        speedup = f"{baseline / best_lat:.2f}x" if baseline and best_lat else "—"
        lines.append(
            f"| `{kt}` | "
            f"{baseline:.1f if baseline else '—'} | "
            f"**{best_lat:.1f if best_lat else '—'}** | "
            f"**{speedup}** | "
            f"{len(summary) - 1} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _ascii_progress_chart(summary: list[dict], baseline: float) -> str:
    """ASCII bar chart of latency reduction over generations."""
    if not summary:
        return ""

    max_lat = baseline
    bar_width = 35

    lines = ["```"]
    lines.append("Latency reduction over generations:")
    lines.append("")
    for s in summary:
        lat = s.get("best_latency_us")
        if lat is None:
            continue
        bar_len = int((lat / max_lat) * bar_width)
        bar = "█" * bar_len
        pct = (max_lat - lat) / max_lat * 100
        label = f"Gen {s['generation']:2d}"
        lines.append(f"  {label} | {bar:<{bar_width}} {lat:6.1f} µs  ({pct:+.1f}%)")
    lines.append("```")
    return "\n".join(lines)

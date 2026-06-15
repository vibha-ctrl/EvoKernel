from pathlib import Path
from typing import Optional

from evokernel.search.candidate_store import Candidate, CandidateStore


def generate_report(
    store: CandidateStore,
    kernel_type: str,
    output_path: str = "report.md",
) -> dict:
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
    lines.append("## Performance Progression")
    lines.append("")
    lines.append(_ascii_chart(summary))
    lines.append("")

    lines.append("## Candidates by Tool Call")
    lines.append("")
    lines.append("| Tool Call | Best Latency (µs) | Candidates | Passed Verify |")
    lines.append("|-----------|-------------------|------------|----------------|")
    for s in summary:
        lat = s['best_latency_us']
        lat_str = f"{lat:.1f}" if lat is not None else "—"
        lines.append(
            f"| {s['tool_call']} | "
            f"{lat_str} | "
            f"{s['total']} | "
            f"{s['passed']} |"
        )
    lines.append("")

    lines.append("## Best Kernel Configuration")
    lines.append("")
    lines.append(f"**Candidate:** `{best.label}`  ")
    lines.append(f"**Latency:** {best.latency_us:.1f} µs  ")
    lines.append("")

    def _fmt(val, suffix="", fmt=None):
        if val is None:
            return "N/A"
        return f"{val:{fmt}}{suffix}" if fmt else f"{val}{suffix}"

    lines.append("### Triton Parameters")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| `num_warps` | {_fmt(best.num_warps)} |")
    lines.append(f"| `num_stages` | {_fmt(best.num_stages)} |")
    lines.append(f"| `shared_mem_bytes` | {_fmt(best.shared_mem_bytes)} |")
    lines.append(f"| `register_count` | {_fmt(best.register_count)} |")
    lines.append(f"| `theoretical_occupancy` | {_fmt(best.theoretical_occupancy_pct, suffix='%', fmt='.1f')} |")
    lines.append("")

    lines.append("### Nsight Systems Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| SM active cycles | {_fmt(best.sm_active_cycles_pct, suffix='%', fmt='.1f')} |")
    lines.append(f"| DRAM utilization | {_fmt(best.dram_utilization_pct, suffix='%', fmt='.1f')} |")
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
        lines.append(f"  Call {s['tool_call']:2d} | {bar:<{bar_width}} {lat:.1f}")
    lines.append("```")
    return "\n".join(lines)


def _optimization_journey(
    store: CandidateStore,
    kernel_type: str,
    summary: list[dict],
) -> str:
    baseline_lat = None
    best_lat = None
    lines = []
    for s in summary:
        lat = s.get("best_latency_us")
        if lat is None:
            continue
        if baseline_lat is None:
            baseline_lat = lat
            lines.append(f"- **Tool call {s['tool_call']}**: {lat:.1f} µs (baseline)")
            best_lat = lat
            continue
        if lat < best_lat:
            delta = (baseline_lat - lat) / baseline_lat * 100
            lines.append(f"- **Tool call {s['tool_call']}**: {lat:.1f} µs — new best ({delta:+.1f}% vs baseline)")
            best_lat = lat
    return "\n".join(lines)

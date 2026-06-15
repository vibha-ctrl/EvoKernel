"""
Critic Agent — uses Claude claude-opus-4-5 to interpret profiler output and
produce actionable optimization hints for the generator.

The critic does NOT write code. It only diagnoses bottlenecks
and recommends strategies.
"""

from dataclasses import dataclass

import anthropic

MODEL = "claude-opus-4-5"

SYSTEM_PROMPT = """\
You are a GPU performance engineer who specializes in diagnosing Triton kernel bottlenecks.

You will be given profiler metrics for one or more kernel candidates.
Your job is to:
1. Identify the primary performance bottleneck (memory-bound, compute-bound, launch-overhead, etc.)
2. Explain WHY each metric indicates a problem
3. Provide SPECIFIC, ACTIONABLE optimization recommendations for a Triton kernel engineer

Output format — respond with a structured list:

DIAGNOSIS: <one sentence summary of the bottleneck>

OBSERVATIONS:
- <metric name>: <value> — <what this means for performance>
- ...

RECOMMENDATIONS:
- <specific Triton optimization technique> — <why this will help>
- ...

Be precise. Reference specific Triton API features (num_warps, num_stages, eviction_policy,
tl.multiple_of, tl.load cache_modifier, BLOCK_SIZE, etc.) when relevant.
"""


@dataclass
class ProfileData:
    candidate_id: str
    kernel_type: str
    latency_us: float
    throughput_gb_s: float
    peak_bandwidth_gb_s: float
    # Triton metadata
    num_warps: int | None = None
    num_stages: int | None = None
    shared_mem_bytes: int | None = None
    register_count: int | None = None
    theoretical_occupancy_pct: float | None = None
    # ncu metrics — always populated (ncu is required)
    dram_utilization_pct: float | None = None
    l1_hit_rate_pct: float | None = None
    stall_memory_dependency_pct: float | None = None
    stall_long_scoreboard_pct: float | None = None
    sm_active_cycles_pct: float | None = None


@dataclass
class CriticResult:
    diagnosis: str
    observations: list[str]
    recommendations: list[str]
    raw_response: str


def analyze_profiles(
    profiles: list[ProfileData],
    gpu_name: str,
    api_key: str,
) -> CriticResult:
    """
    Send profiler data for top candidates to Claude and get optimization hints back.
    """
    prompt = _build_prompt(profiles, gpu_name)
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    return _parse_response(raw)


def _build_prompt(profiles: list[ProfileData], gpu_name: str) -> str:
    parts = [f"## GPU: {gpu_name}\n"]

    for p in profiles:
        bw_util = (p.throughput_gb_s / p.peak_bandwidth_gb_s * 100) if p.peak_bandwidth_gb_s else 0
        parts.append(f"### Candidate: {p.candidate_id}  ({p.kernel_type})")
        parts.append(f"Latency: {p.latency_us:.1f} µs")
        parts.append(f"Throughput: {p.throughput_gb_s:.0f} GB/s  "
                     f"({bw_util:.0f}% of {p.peak_bandwidth_gb_s:.0f} GB/s peak)")

        parts.append("\n**Triton Compiler Metadata:**")
        parts.append(f"- num_warps: {p.num_warps}")
        parts.append(f"- num_stages: {p.num_stages}")
        parts.append(f"- shared_mem_bytes: {p.shared_mem_bytes}")
        parts.append(f"- register_count: {p.register_count}")
        parts.append(f"- theoretical_occupancy: {p.theoretical_occupancy_pct}%")

        parts.append("\n**Nsight Compute Metrics:**")
        parts.append(f"- SM throughput: {p.sm_active_cycles_pct}%")
        parts.append(f"- DRAM utilization: {p.dram_utilization_pct}%")
        parts.append(f"- L1 cache hit rate: {p.l1_hit_rate_pct}%")
        parts.append(f"- Stall (memory dependency): {p.stall_memory_dependency_pct}%")
        parts.append(f"- Stall (long scoreboard): {p.stall_long_scoreboard_pct}%")

        parts.append("")

    parts.append("Analyze these kernels and provide optimization recommendations "
                 "for the Triton kernel generator to use in the next generation.")

    return "\n".join(parts)


def _parse_response(raw: str) -> CriticResult:
    """Parse structured critic response into CriticResult."""
    diagnosis = ""
    observations: list[str] = []
    recommendations: list[str] = []

    current_section = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("DIAGNOSIS:"):
            diagnosis = line[len("DIAGNOSIS:"):].strip()
        elif line.startswith("OBSERVATIONS:"):
            current_section = "obs"
        elif line.startswith("RECOMMENDATIONS:"):
            current_section = "rec"
        elif line.startswith("- ") and current_section == "obs":
            observations.append(line[2:])
        elif line.startswith("- ") and current_section == "rec":
            recommendations.append(line[2:])

    # Fallback if parsing fails
    if not diagnosis:
        diagnosis = raw[:200]
    if not recommendations:
        recommendations = [raw]

    return CriticResult(
        diagnosis=diagnosis,
        observations=observations,
        recommendations=recommendations,
        raw_response=raw,
    )

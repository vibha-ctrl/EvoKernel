"""
Generator Agent — uses Claude claude-opus-4-5 to produce Triton kernel variants.

Given a parent kernel, profiler hints, and a history of failed attempts,
the generator produces N syntactically correct and semantically sound
Triton kernel candidates for the next generation.
"""

import re
from dataclasses import dataclass, field

import anthropic

MODEL = "claude-opus-4-5"

SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in Triton, CUDA, and GPU performance optimization.

Your job is to generate Triton kernel variants that are FASTER than the parent kernel on NVIDIA GPUs.

STRICT RULES:
1. Every variant MUST define a function named exactly `run` with the correct signature for the kernel type.
2. Every variant MUST include all necessary imports (torch, triton, triton.language as tl).
3. Never change the mathematical correctness of the operation — only change performance parameters.
4. Only mutate: BLOCK_SIZE, num_warps, num_stages, eviction_policy, data type casting strategy,
   memory access patterns, loop structure, tl.multiple_of hints, constexpr usage.
5. Output ONLY code blocks, no explanation text between variants.
6. Each variant must be wrapped in a markdown code block tagged exactly: ```variant_N where N is 1,2,3...

KERNEL TYPE SIGNATURES (must match exactly):
- rmsnorm:            run(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor
- rope:               run(q, k, cos, sin) -> tuple[torch.Tensor, torch.Tensor]
- fused_rmsnorm_rope: run(x, weight, cos, sin, n_heads: int = 32, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]
"""


@dataclass
class FailedCandidate:
    candidate_id: str
    code: str
    error_type: str
    error_msg: str


@dataclass
class BenchmarkedCandidate:
    candidate_id: str
    code: str
    latency_us: float


@dataclass
class GeneratorContext:
    kernel_type: str
    parent_code: str
    parent_latency_us: float
    generation: int
    gpu_name: str
    peak_bandwidth_gb_s: float
    critic_hints: list[str] = field(default_factory=list)
    failed_candidates: list[FailedCandidate] = field(default_factory=list)
    tried_configs: list[dict] = field(default_factory=list)
    n_variants: int = 5


def generate_variants(ctx: GeneratorContext, api_key: str) -> list[str]:
    """
    Call Claude claude-opus-4-5 with full context and return a list of code strings,
    one per variant. Returns up to ctx.n_variants items.
    """
    prompt = _build_prompt(ctx)
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text
    return _parse_variants(raw)


def _build_prompt(ctx: GeneratorContext) -> str:
    parts = []

    parts.append(f"## Task\nGenerate {ctx.n_variants} Triton kernel variants for: **{ctx.kernel_type}**")
    parts.append(f"Generation: {ctx.generation}")
    parts.append(f"Target GPU: {ctx.gpu_name} (peak bandwidth: {ctx.peak_bandwidth_gb_s} GB/s)")

    parts.append("\n## Parent Kernel (mutate from this)")
    parts.append(f"Current latency: {ctx.parent_latency_us:.1f} µs")
    parts.append("```python")
    parts.append(ctx.parent_code)
    parts.append("```")

    if ctx.critic_hints:
        parts.append("\n## Profiler Analysis & Optimization Hints")
        for hint in ctx.critic_hints:
            parts.append(f"- {hint}")

    if ctx.tried_configs:
        parts.append("\n## Previously Tried Configs (do NOT repeat these exactly)")
        for cfg in ctx.tried_configs[-20:]:  # last 20 to stay in context
            parts.append(
                f"- BLOCK_SIZE={cfg.get('BLOCK_SIZE','?')}, "
                f"num_warps={cfg.get('num_warps','?')}, "
                f"num_stages={cfg.get('num_stages','?')} "
                f"→ {cfg.get('latency_us','?')} µs"
            )

    if ctx.failed_candidates:
        parts.append("\n## Failed Candidates (understand WHY they failed, avoid same mistakes)")
        for fc in ctx.failed_candidates[-5:]:  # last 5 failures
            parts.append(f"\n### {fc.candidate_id} — {fc.error_type}")
            parts.append(f"Error: {fc.error_msg}")
            parts.append("Code that failed:")
            parts.append("```python")
            parts.append(fc.code)
            parts.append("```")

    parts.append(f"\n## Your Task")
    parts.append(
        f"Generate exactly {ctx.n_variants} distinct variants. "
        f"Each must attempt a DIFFERENT optimization strategy. "
        f"Focus on reducing latency below {ctx.parent_latency_us:.1f} µs. "
        f"Wrap each in a code block tagged ```variant_1 through ```variant_{ctx.n_variants}."
    )

    return "\n".join(parts)


def _parse_variants(response_text: str) -> list[str]:
    """Extract code blocks tagged ```variant_N from LLM response."""
    pattern = r"```variant_\d+\s*\n(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    return [m.strip() for m in matches]

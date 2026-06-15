import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

import anthropic
import httpx
from rich.console import Console

from evokernel.search.candidate_store import Candidate, CandidateStore

console = Console()

MODEL = "claude-opus-4-5"
MAX_TOOL_CALLS = 80


TOOLS = [
    {
        "name": "generate_kernel_variant",
        "description": (
            "Generate a single new Triton kernel variant based on a parent candidate. "
            "You choose the optimization strategy. Returns a candidate_id you can pass "
            "to verify_kernel, benchmark_kernel, or profile_kernel."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_candidate_id": {
                    "type": "string",
                    "description": "ID of the parent candidate to mutate from. Use 'baseline' for the original kernel.",
                },
                "strategy": {
                    "type": "string",
                    "description": (
                        "Natural language description of what optimization to try. "
                        "Examples: 'increase num_warps to 8 for better parallelism', "
                        "'try num_stages=3 for software pipelining', "
                        "'use larger BLOCK_SIZE to increase arithmetic intensity'."
                    ),
                },
            },
            "required": ["parent_candidate_id", "strategy"],
        },
    },
    {
        "name": "verify_kernel",
        "description": (
            "Check correctness of a candidate kernel against the PyTorch reference. "
            "Must pass before benchmarking. Returns passed/failed with error details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "description": "Candidate ID to verify."},
            },
            "required": ["candidate_id"],
        },
    },
    {
        "name": "benchmark_kernel",
        "description": (
            "Measure latency and memory throughput of a verified kernel on the A100. "
            "Returns latency_us, throughput_gb_s, bandwidth_utilization_pct. "
            "Only call this after verify_kernel has passed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "description": "Candidate ID to benchmark."},
            },
            "required": ["candidate_id"],
        },
    },
    {
        "name": "profile_kernel",
        "description": (
            "Run nsys profiler on a benchmarked kernel to get hardware metrics: "
            "dram_utilization_pct, sm_active_cycles_pct, num_warps, num_stages, "
            "shared_mem_bytes, theoretical_occupancy_pct. "
            "Use this when you want to understand WHY a kernel is fast or slow."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_id": {"type": "string", "description": "Candidate ID to profile."},
            },
            "required": ["candidate_id"],
        },
    },
    {
        "name": "get_best_candidates",
        "description": (
            "Retrieve the N fastest verified+benchmarked candidates so far. "
            "Use this to pick a good parent for the next generation or to assess progress."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of best candidates to retrieve (1-10).",
                    "default": 3,
                },
            },
        },
    },
    {
        "name": "get_failed_candidates",
        "description": (
            "Retrieve the N most recent candidates that failed verification. "
            "Use this to understand what mistakes to avoid."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of recent failures to retrieve (1-10).",
                    "default": 5,
                },
            },
        },
    },
    {
        "name": "stop_search",
        "description": (
            "End the optimization search. Call this when you've found a satisfactory speedup, "
            "when further improvement seems unlikely, or when you've exhausted good strategies. "
            "Provide a clear reason and the ID of the best candidate found."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why you are stopping the search.",
                },
                "best_candidate_id": {
                    "type": "string",
                    "description": "ID of the best candidate found.",
                },
            },
            "required": ["reason", "best_candidate_id"],
        },
    },
]


AGENT_SYSTEM_PROMPT = """\
You are an autonomous GPU kernel optimizer. Your goal is to find the fastest correct \
implementation of a Triton kernel on an NVIDIA A100 80GB GPU.

You have access to tools: generate, verify, benchmark, profile, get history, and stop.

WORKFLOW GUIDELINES:
- Always verify a new variant before benchmarking it.
- Profile candidates strategically — profiling takes ~60s, so only profile when you need \
  to understand why something is fast or slow.
- Use profiling data to guide your next generation: if DRAM utilization is low, try \
  different memory access patterns; if SM utilization is low, try more warps.
- If a generation of similar strategies all fail verification, step back and diagnose \
  the pattern before generating more.
- If latency improvement stalls below 1% for 3 consecutive strategies, consider a \
  fundamentally different approach or stop.
- Track which configs you've tried. Don't repeat the same (BLOCK_SIZE, num_warps, \
  num_stages) combination.
- Stop when you've achieved a good speedup (>5%) or when further improvement is unlikely.

KERNEL TYPE SIGNATURES (candidates MUST define run() with these exact signatures):
- rmsnorm:            run(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor
- rope:               run(q, k, cos, sin) -> tuple[torch.Tensor, torch.Tensor]
- fused_rmsnorm_rope: run(x, weight, cos, sin, n_heads: int = 32, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]

Every generated kernel MUST include all imports (torch, triton, triton.language as tl).
"""


@dataclass
class AgentState:
    kernel_type: str
    baseline_id: str
    baseline_latency_us: float
    store: CandidateStore
    http_client: httpx.Client
    api_key: str
    gpu_name: str
    peak_bw: float
    tool_calls: int = 0
    stop_reason: str = ""
    best_candidate_id: str = ""

    _pending_codes: dict = None

    def __post_init__(self):
        self._pending_codes = {}


def _execute_tool(name: str, inputs: dict, state: AgentState) -> str:
    if name == "generate_kernel_variant":
        return _tool_generate(inputs, state)
    elif name == "verify_kernel":
        return _tool_verify(inputs, state)
    elif name == "benchmark_kernel":
        return _tool_benchmark(inputs, state)
    elif name == "profile_kernel":
        return _tool_profile(inputs, state)
    elif name == "get_best_candidates":
        return _tool_get_best(inputs, state)
    elif name == "get_failed_candidates":
        return _tool_get_failed(inputs, state)
    elif name == "stop_search":
        state.stop_reason = inputs.get("reason", "")
        state.best_candidate_id = inputs.get("best_candidate_id", "")
        return f"Search stopped. Reason: {state.stop_reason}"
    else:
        return f"Unknown tool: {name}"


def _tool_generate(inputs: dict, state: AgentState) -> str:
    parent_id = inputs["parent_candidate_id"]
    strategy = inputs["strategy"]

    if parent_id == "baseline":
        parent = state.store.get(state.baseline_id)
    else:
        parent = state.store.get(parent_id)

    if not parent:
        return f"ERROR: parent candidate '{parent_id}' not found."

    import anthropic as _anthropic
    client = _anthropic.Anthropic(api_key=state.api_key)

    generation_prompt = f"""\
Parent kernel ({state.kernel_type}, latency={parent.latency_us:.1f if parent.latency_us else '?'} µs):

```python
{parent.code}
```

Generate ONE Triton kernel variant that implements this strategy:
{strategy}

Rules:
- Keep the same mathematical operation (correctness must pass torch.allclose)
- Include all imports (torch, triton, triton.language as tl)
- Define run() with the exact signature for {state.kernel_type}
- Output ONLY the Python code, no explanation

```python"""

    msg = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": generation_prompt}],
    )

    raw = msg.content[0].text
    code = raw.strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    candidate = Candidate(
        code=code,
        kernel_type=state.kernel_type,
        generation=state.tool_calls,
        parent_id=parent.id,
    )
    state.store.save(candidate)
    state._pending_codes[candidate.id] = code

    console.print(f"  [dim]Generated candidate {candidate.id} — strategy: {strategy[:60]}[/dim]")
    return (
        f"Generated candidate_id={candidate.id}\n"
        f"Strategy applied: {strategy}\n"
        f"Parent: {parent_id} (latency={parent.latency_us:.1f if parent.latency_us else '?'} µs)\n"
        f"Next step: call verify_kernel(candidate_id='{candidate.id}')"
    )


def _tool_verify(inputs: dict, state: AgentState) -> str:
    candidate_id = inputs["candidate_id"]
    c = state.store.get(candidate_id)
    if not c:
        return f"ERROR: candidate '{candidate_id}' not found."

    try:
        resp = state.http_client.post("/verify", json={
            "code": c.code,
            "kernel_type": c.kernel_type,
            "candidate_id": c.id,
        })
        resp.raise_for_status()
        data = resp.json()
        state.store.update_verify(c.id, data["passed"], data.get("error_type"),
                                   data.get("error_msg"), data.get("max_error"))

        if data["passed"]:
            console.print(f"  [green]✓[/green] Verified {candidate_id}")
            return (
                f"PASSED verification\n"
                f"candidate_id={candidate_id}\n"
                f"max_error={data.get('max_error', 'N/A')}\n"
                f"Next step: call benchmark_kernel(candidate_id='{candidate_id}')"
            )
        else:
            console.print(f"  [red]✗[/red] Failed {candidate_id}: {data.get('error_type')}")
            return (
                f"FAILED verification\n"
                f"error_type={data.get('error_type')}\n"
                f"error_msg={data.get('error_msg', '')[:300]}\n"
                f"This candidate is unusable. Generate a new variant fixing this issue."
            )
    except Exception as e:
        state.store.update_verify(c.id, False, "http_error", str(e), None)
        return f"ERROR connecting to GPU server: {e}"


def _tool_benchmark(inputs: dict, state: AgentState) -> str:
    candidate_id = inputs["candidate_id"]
    c = state.store.get(candidate_id)
    if not c:
        return f"ERROR: candidate '{candidate_id}' not found."
    if not c.verify_passed:
        return f"ERROR: candidate '{candidate_id}' has not passed verification. Call verify_kernel first."

    try:
        resp = state.http_client.post("/benchmark", json={
            "code": c.code,
            "kernel_type": c.kernel_type,
            "candidate_id": c.id,
        })
        resp.raise_for_status()
        data = resp.json()
        state.store.update_benchmark(c.id, data["latency_us"], data["latency_p99_us"],
                                      data["throughput_gb_s"], data["bandwidth_utilization_pct"])

        speedup = state.baseline_latency_us / data["latency_us"]
        delta_pct = (state.baseline_latency_us - data["latency_us"]) / state.baseline_latency_us * 100
        console.print(
            f"  [cyan]⏱[/cyan] {candidate_id}: {data['latency_us']:.1f} µs  "
            f"({delta_pct:+.1f}% vs baseline, {speedup:.3f}x)"
        )
        return (
            f"BENCHMARKED candidate_id={candidate_id}\n"
            f"latency_us={data['latency_us']:.2f}\n"
            f"throughput_gb_s={data['throughput_gb_s']:.1f}\n"
            f"bandwidth_utilization_pct={data['bandwidth_utilization_pct']:.1f}\n"
            f"vs_baseline: {delta_pct:+.1f}% ({speedup:.3f}x speedup)\n"
            f"baseline_latency_us={state.baseline_latency_us:.2f}"
        )
    except Exception as e:
        return f"ERROR benchmarking: {e}"


def _tool_profile(inputs: dict, state: AgentState) -> str:
    candidate_id = inputs["candidate_id"]
    c = state.store.get(candidate_id)
    if not c:
        return f"ERROR: candidate '{candidate_id}' not found."

    try:
        resp = state.http_client.post("/profile", json={
            "code": c.code,
            "kernel_type": c.kernel_type,
            "candidate_id": c.id,
        })
        resp.raise_for_status()
        data = resp.json()
        state.store.update_profile(c.id, data)
        console.print(f"  [magenta]📊[/magenta] Profiled {candidate_id}")

        lines = [f"PROFILED candidate_id={candidate_id}"]
        if data.get("num_warps"):      lines.append(f"num_warps={data['num_warps']}")
        if data.get("num_stages"):     lines.append(f"num_stages={data['num_stages']}")
        if data.get("shared_mem_bytes"): lines.append(f"shared_mem_bytes={data['shared_mem_bytes']}")
        if data.get("register_count"): lines.append(f"register_count={data['register_count']}")
        if data.get("theoretical_occupancy_pct"): lines.append(f"theoretical_occupancy_pct={data['theoretical_occupancy_pct']:.1f}")
        if data.get("dram_utilization_pct"):       lines.append(f"dram_utilization_pct={data['dram_utilization_pct']:.1f}")
        if data.get("sm_active_cycles_pct"):       lines.append(f"sm_active_cycles_pct={data['sm_active_cycles_pct']:.1f}")

        return "\n".join(lines)
    except Exception as e:
        return f"ERROR profiling: {e}"


def _tool_get_best(inputs: dict, state: AgentState) -> str:
    n = min(inputs.get("n", 3), 10)
    bests = state.store.get_best(state.kernel_type, n=n)
    if not bests:
        return "No benchmarked candidates yet."

    lines = [f"TOP {len(bests)} CANDIDATES:"]
    for i, c in enumerate(bests):
        speedup = state.baseline_latency_us / c.latency_us if c.latency_us else 0
        delta = (state.baseline_latency_us - c.latency_us) / state.baseline_latency_us * 100 if c.latency_us else 0
        lines.append(
            f"{i+1}. id={c.id} latency={c.latency_us:.1f}µs "
            f"({delta:+.1f}%, {speedup:.3f}x) "
            f"profiled={'yes' if c.is_profiled else 'no'}"
        )
    return "\n".join(lines)


def _tool_get_failed(inputs: dict, state: AgentState) -> str:
    n = min(inputs.get("n", 5), 10)
    failed = state.store.get_failed(state.kernel_type, generation=None)[-n:]
    if not failed:
        return "No failed candidates yet."

    lines = [f"RECENT {len(failed)} FAILURES:"]
    for f in failed:
        lines.append(
            f"- id={f.id} error_type={f.verify_error_type} "
            f"msg={str(f.verify_error_msg)[:100]}"
        )
    return "\n".join(lines)


def run_agentic_search(
    baseline_code: str,
    kernel_type: str,
    runpod_url: str,
    anthropic_api_key: str,
    db_path: str = "evokernel.db",
) -> Candidate:
    store = CandidateStore(db_path)
    http_client = httpx.Client(base_url=runpod_url, timeout=300.0)

    console.rule("[bold blue]EvoKernel Agentic Search")
    console.print(f"  Kernel type : [cyan]{kernel_type}[/cyan]")
    console.print(f"  Mode        : [bold magenta]AGENTIC[/bold magenta] — Claude drives the loop")
    console.print(f"  Max calls   : {MAX_TOOL_CALLS}")

    baseline = Candidate(code=baseline_code, kernel_type=kernel_type, generation=0)
    store.save(baseline)

    console.print("\n[bold]Establishing baseline...[/bold]")
    try:
        resp = http_client.post("/verify", json={
            "code": baseline_code, "kernel_type": kernel_type, "candidate_id": baseline.id
        })
        data = resp.json()
        store.update_verify(baseline.id, data["passed"], data.get("error_type"),
                             data.get("error_msg"), data.get("max_error"))
        if not data["passed"]:
            raise RuntimeError(f"Baseline failed verification: {data.get('error_msg')}")

        resp = http_client.post("/benchmark", json={
            "code": baseline_code, "kernel_type": kernel_type, "candidate_id": baseline.id
        })
        bdata = resp.json()
        store.update_benchmark(baseline.id, bdata["latency_us"], bdata["latency_p99_us"],
                                bdata["throughput_gb_s"], bdata["bandwidth_utilization_pct"])
    except Exception as e:
        raise RuntimeError(f"Baseline setup failed: {e}")

    baseline = store.get(baseline.id)
    console.print(f"  Baseline: [green]{baseline.latency_us:.1f} µs[/green]")

    gpu_name, peak_bw = "A100", 2000.0
    try:
        hdata = http_client.get("/health").json()
        gpu_name = hdata.get("gpu", "A100")
        for key, bw in {"H100": 3350.0, "A100": 2000.0, "A10G": 600.0}.items():
            if key in gpu_name:
                peak_bw = bw
                break
    except Exception:
        pass

    state = AgentState(
        kernel_type=kernel_type,
        baseline_id=baseline.id,
        baseline_latency_us=baseline.latency_us,
        store=store,
        http_client=http_client,
        api_key=anthropic_api_key,
        gpu_name=gpu_name,
        peak_bw=peak_bw,
        best_candidate_id=baseline.id,
    )

    initial_message = f"""\
Optimize this {kernel_type} Triton kernel to run as fast as possible on {gpu_name} \
(peak memory bandwidth: {peak_bw:.0f} GB/s).

Baseline kernel (latency: {baseline.latency_us:.1f} µs, candidate_id='{baseline.id}'):

```python
{baseline_code}
```

Use the available tools to generate, verify, benchmark, and profile variants. \
Decide your own strategy. Stop when you've found a good speedup or exhausted approaches.
"""

    messages = [{"role": "user", "content": initial_message}]
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    console.print("\n[bold magenta]Handing control to Claude...[/bold magenta]\n")

    while state.tool_calls < MAX_TOOL_CALLS:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=AGENT_SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            console.print("[dim]Claude finished without calling stop_search.[/dim]")
            break

        if response.stop_reason != "tool_use":
            console.print(f"[dim]Unexpected stop_reason: {response.stop_reason}[/dim]")
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                state.tool_calls += 1
                tool_name = block.name
                tool_inputs = block.input

                console.print(f"\n[bold]Tool call {state.tool_calls}[/bold]: [cyan]{tool_name}[/cyan]")

                result = _execute_tool(tool_name, tool_inputs, state)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

                if tool_name == "stop_search":
                    messages.append({"role": "user", "content": tool_results})
                    break

        messages.append({"role": "user", "content": tool_results})

        if state.stop_reason:
            break

    best_candidates = store.get_best(kernel_type, n=1)
    best = best_candidates[0] if best_candidates else baseline
    speedup = baseline.latency_us / best.latency_us if best.latency_us else 1.0

    _save_trace(messages, kernel_type, state, baseline, best, speedup)

    console.rule("[bold green]Agentic Search Complete")
    console.print(f"  Baseline     : {baseline.latency_us:.1f} µs")
    console.print(f"  Best found   : [bold green]{best.latency_us:.1f} µs[/bold green]")
    console.print(f"  Speedup      : [bold green]{speedup:.2f}x[/bold green]")
    console.print(f"  Tool calls   : {state.tool_calls}")
    if state.stop_reason:
        console.print(f"  Claude said  : [italic]{state.stop_reason}[/italic]")

    return best


def _save_trace(
    messages: list,
    kernel_type: str,
    state: AgentState,
    baseline: Candidate,
    best: Candidate,
    speedup: float,
):
    import os
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results/{kernel_type}_trace_{timestamp}.md"

    lines = []
    lines.append(f"# {kernel_type} — Agent Trace")
    lines.append(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')}  |  GPU: {state.gpu_name}*")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Baseline latency | {baseline.latency_us:.1f} µs |")
    lines.append(f"| Best latency | **{best.latency_us:.1f} µs** |")
    lines.append(f"| Speedup | **{speedup:.2f}x** |")
    lines.append(f"| Tool calls used | {state.tool_calls} |")
    if state.stop_reason:
        lines.append(f"| Claude's reason for stopping | *{state.stop_reason}* |")
    lines.append("")
    lines.append("## Reasoning & Tool Call Trace")
    lines.append("")

    call_num = 0
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant":
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text" and block.text.strip():
                            lines.append(f"**Claude:** {block.text.strip()}")
                            lines.append("")
                        elif block.type == "tool_use":
                            call_num += 1
                            inputs_str = json.dumps(block.input, indent=2)
                            lines.append(f"**Tool call {call_num}: `{block.name}`**")
                            lines.append(f"```json")
                            lines.append(inputs_str)
                            lines.append("```")
                            lines.append("")

        elif role == "user":
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        lines.append(f"**Result:**")
                        lines.append(f"> {str(block.get('content', '')).strip()}")
                        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"  Trace saved  : [dim]{path}[/dim]")

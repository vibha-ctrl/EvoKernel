"""
EvoKernel — Main Entry Point

Usage:
  python run_search.py rmsnorm
  python run_search.py rope --generations 8 --candidates 10
  python run_search.py fused_rmsnorm_rope --top-k 3

Before running:
  1. Start a RunPod pod with an A100 / H100 / A10G
  2. SSH in and start the GPU server:
       pip install -r gpu_server/requirements.txt
       cd gpu_server && uvicorn server:app --host 0.0.0.0 --port 8000
  3. Set RUNPOD_SERVER_URL and ANTHROPIC_API_KEY in .env
"""

import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from evokernel.search.evolutionary import SearchConfig, run_search
from evokernel.search.agent_loop import run_agentic_search
from evokernel.reports.report_generator import generate_report
from evokernel.search.candidate_store import CandidateStore

load_dotenv()
app = typer.Typer(help="EvoKernel — autonomous Triton kernel optimizer")
console = Console()

KERNEL_BASELINES = {
    "rmsnorm": "evokernel/kernels/rmsnorm/baseline.py",
    "rope": "evokernel/kernels/rope/baseline.py",
    "fused_rmsnorm_rope": "evokernel/kernels/fused_rmsnorm_rope/baseline.py",
}


@app.command()
def search(
    kernel_type: str = typer.Argument(
        ...,
        help="Kernel to optimize: rmsnorm | rope | fused_rmsnorm_rope",
    ),
    generations: int = typer.Option(
        int(os.getenv("MAX_GENERATIONS", "10")),
        "--generations", "-g",
        help="Maximum number of evolutionary generations",
    ),
    candidates: int = typer.Option(
        int(os.getenv("CANDIDATES_PER_GENERATION", "10")),
        "--candidates", "-c",
        help="Candidate variants to generate per generation",
    ),
    top_k: int = typer.Option(
        int(os.getenv("TOP_K", "3")),
        "--top-k", "-k",
        help="Top candidates to keep per generation",
    ),
    convergence: float = typer.Option(
        float(os.getenv("CONVERGENCE_THRESHOLD_PCT", "2.0")),
        "--convergence",
        help="Stop if improvement < this percentage",
    ),
    db: str = typer.Option("evokernel.db", "--db", help="SQLite database path"),
    report: str = typer.Option("report.md", "--report", help="Output report path"),
    sequential: bool = typer.Option(
        False, "--sequential",
        help="Run verify/benchmark sequentially (slower, less GPU pressure)",
    ),
    agentic: bool = typer.Option(
        False, "--agentic",
        help="Use agentic mode: Claude drives the loop via tool use instead of a fixed evolutionary schedule",
    ),
):
    """Run the evolutionary kernel optimization search."""

    # Validate kernel type
    if kernel_type not in KERNEL_BASELINES:
        console.print(f"[red]Unknown kernel type: {kernel_type}[/red]")
        console.print(f"Choose from: {', '.join(KERNEL_BASELINES)}")
        raise typer.Exit(1)

    # Check required env vars
    runpod_url = os.getenv("RUNPOD_SERVER_URL", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not runpod_url:
        console.print("[red]RUNPOD_SERVER_URL not set.[/red]")
        console.print("Set it in .env: RUNPOD_SERVER_URL=https://YOUR_POD-8000.proxy.runpod.net")
        raise typer.Exit(1)

    if not anthropic_key:
        console.print("[red]ANTHROPIC_API_KEY not set.[/red]")
        raise typer.Exit(1)

    # Load baseline
    baseline_path = Path(KERNEL_BASELINES[kernel_type])
    if not baseline_path.exists():
        console.print(f"[red]Baseline not found: {baseline_path}[/red]")
        raise typer.Exit(1)

    baseline_code = baseline_path.read_text()

    # Run search
    if agentic:
        best = run_agentic_search(
            baseline_code=baseline_code,
            kernel_type=kernel_type,
            runpod_url=runpod_url,
            anthropic_api_key=anthropic_key,
            db_path=db,
        )
    else:
        config = SearchConfig(
            kernel_type=kernel_type,
            max_generations=generations,
            candidates_per_generation=candidates,
            top_k=top_k,
            convergence_threshold_pct=convergence,
            runpod_url=runpod_url,
            anthropic_api_key=anthropic_key,
            db_path=db,
            parallel_verify=not sequential,
            parallel_benchmark=not sequential,
        )
        best = run_search(baseline_code, config)

    # Generate report
    store = CandidateStore(db)
    result = generate_report(store, kernel_type, report)
    console.print(f"\n[bold green]Report written:[/bold green] {result['path']}")
    if result.get("speedup"):
        console.print(f"Final speedup: [bold green]{result['speedup']:.2f}x[/bold green]")


@app.command()
def report_only(
    kernel_type: str = typer.Argument(..., help="Kernel type to report on"),
    db: str = typer.Option("evokernel.db", "--db"),
    output: str = typer.Option("report.md", "--output"),
):
    """Generate a report from an existing database without running a new search."""
    store = CandidateStore(db)
    result = generate_report(store, kernel_type, output)
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
    else:
        console.print(f"[green]Report written to {result['path']}[/green]")
        console.print(f"Speedup: {result.get('speedup', 'N/A')}x")


@app.command()
def status(
    kernel_type: str = typer.Argument(...),
    db: str = typer.Option("evokernel.db", "--db"),
):
    """Show current search progress from the database."""
    from rich.table import Table

    store = CandidateStore(db)
    summary = store.generation_summary(kernel_type)

    if not summary:
        console.print(f"[yellow]No data found for {kernel_type}[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Search Progress — {kernel_type}")
    table.add_column("Generation")
    table.add_column("Best Latency (µs)", justify="right")
    table.add_column("Candidates", justify="right")
    table.add_column("Passed", justify="right")

    baseline_latency = None
    for s in summary:
        lat = s.get("best_latency_us")
        if s["generation"] == 0 and lat:
            baseline_latency = lat
        delta = ""
        if baseline_latency and lat and s["generation"] > 0:
            delta = f"  ({(baseline_latency - lat) / baseline_latency * 100:+.1f}%)"
        table.add_row(
            str(s["generation"]),
            f"{lat:.1f}{delta}" if lat else "—",
            str(s["total"]),
            str(s["passed"]),
        )

    console.print(table)

    bests = store.get_best(kernel_type, n=1)
    if bests and baseline_latency:
        speedup = baseline_latency / bests[0].latency_us
        console.print(f"\nCurrent best speedup: [bold green]{speedup:.2f}x[/bold green]")


if __name__ == "__main__":
    app()

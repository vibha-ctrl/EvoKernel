import os
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

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
    db: str = typer.Option("evokernel.db", "--db", help="SQLite database path"),
):
    if kernel_type not in KERNEL_BASELINES:
        console.print(f"[red]Unknown kernel type: {kernel_type}[/red]")
        console.print(f"Choose from: {', '.join(KERNEL_BASELINES)}")
        raise typer.Exit(1)

    runpod_url = os.getenv("RUNPOD_SERVER_URL", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not runpod_url:
        console.print("[red]RUNPOD_SERVER_URL not set.[/red]")
        console.print("Set it in .env: RUNPOD_SERVER_URL=https://YOUR_POD-8000.proxy.runpod.net")
        raise typer.Exit(1)

    if not anthropic_key:
        console.print("[red]ANTHROPIC_API_KEY not set.[/red]")
        raise typer.Exit(1)

    baseline_path = Path(KERNEL_BASELINES[kernel_type])
    if not baseline_path.exists():
        console.print(f"[red]Baseline not found: {baseline_path}[/red]")
        raise typer.Exit(1)

    baseline_code = baseline_path.read_text()

    best = run_agentic_search(
        baseline_code=baseline_code,
        kernel_type=kernel_type,
        runpod_url=runpod_url,
        anthropic_api_key=anthropic_key,
        db_path=db,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    report_path = f"results/{kernel_type}_report_{timestamp}.md"

    store = CandidateStore(db)
    result = generate_report(store, kernel_type, report_path)
    console.print(f"\n[bold green]Report written:[/bold green] {result['path']}")
    if result.get("speedup"):
        console.print(f"Final speedup: [bold green]{result['speedup']:.2f}x[/bold green]")



@app.command()
def status(
    kernel_type: str = typer.Argument(...),
    db: str = typer.Option("evokernel.db", "--db"),
):
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

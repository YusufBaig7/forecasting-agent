"""Script to run experiments across multiple configurations."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from forecasting.config import get_settings
from forecasting.eval.backtest import run_backtest
from forecasting.feeds.stub import StubFeed
from forecasting.forecast.baseline_market import MarketBaselineForecaster
from forecasting.forecast.calibration import CalibratedForecaster, PlattCalibrator
from forecasting.forecast.llm_forecaster import LLMForecaster
from forecasting.forecast.multi_agent import MultiAgentForecaster
from forecasting.llm.mock import MockLLM
from forecasting.retrieval.stub import StubRetriever
from forecasting.storage.forecast_store import FileForecastStore
from forecasting.storage.snapshot_store import FileSnapshotStore

app = typer.Typer(add_completion=False)
console = Console()


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    forecaster_type: str
    n_agents: int = 1
    use_supervisor: bool = False
    use_calibration: bool = False
    calibrator_path: Optional[str] = None


@dataclass
class ExperimentResult:
    """Results from a single experiment."""

    config: ExperimentConfig
    brier: float
    ece: float
    return_pct: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    num_bets: Optional[int] = None
    latency_ms: Optional[float] = None
    coverage: float = 0.0
    n_forecasts: int = 0
    n_possible: int = 0
    error: Optional[str] = None


def _create_forecaster(config: ExperimentConfig, llm, retriever):
    """Create forecaster based on config."""
    if config.forecaster_type == "market_baseline":
        forecaster = MarketBaselineForecaster()
    elif config.forecaster_type == "llm":
        forecaster = LLMForecaster(llm=llm, retriever=retriever)
    elif config.forecaster_type == "multi_agent":
        forecaster = MultiAgentForecaster(
            llm=llm,
            retriever=retriever,
            n_agents=config.n_agents,
            use_supervisor=config.use_supervisor,
        )
    else:
        raise ValueError(f"Unknown forecaster type: {config.forecaster_type}")

    # Apply calibration if requested
    if config.use_calibration:
        calibrator = None
        if config.calibrator_path:
            calibrator = PlattCalibrator.load(Path(config.calibrator_path))
        else:
            # Try to find default calibrator
            settings = get_settings()
            default_path = settings.data_dir / "calibration" / "default.json"
            if default_path.exists():
                calibrator = PlattCalibrator.load(default_path)
        
        if calibrator is None:
            console.print(f"[yellow]Warning: No calibrator found, skipping calibration for {config.name}[/yellow]")
        else:
            forecaster = CalibratedForecaster(base_forecaster=forecaster, calibrator=calibrator)

    return forecaster


def _run_experiment(
    config: ExperimentConfig,
    feed,
    as_of_times: list[datetime],
    snapshot_store: Optional[FileSnapshotStore],
    llm,
    retriever,
) -> ExperimentResult:
    """Run a single experiment and collect metrics."""
    try:
        # Create forecaster
        forecaster = _create_forecaster(config, llm, retriever)

        # Measure latency
        start_time = time.time()

        # Run backtest
        result = run_backtest(
            feed=feed,
            forecaster=forecaster,
            as_of_times=as_of_times,
            snapshot_store=snapshot_store,
        )

        elapsed_time = time.time() - start_time

        # Calculate metrics
        n_forecasts = len(result.predictions)
        n_possible = len(as_of_times) * len(feed.list_events(as_of_times[0]))
        coverage = n_forecasts / n_possible if n_possible > 0 else 0.0

        # Latency per forecast
        latency_ms = (elapsed_time / n_forecasts * 1000) if n_forecasts > 0 else None

        # Trading metrics
        return_pct = None
        sharpe = None
        max_drawdown = None
        num_bets = None
        if result.trading is not None:
            return_pct = result.trading.total_return * 100
            sharpe = result.trading.sharpe
            max_drawdown = result.trading.max_drawdown * 100
            num_bets = result.trading.num_bets

        return ExperimentResult(
            config=config,
            brier=result.brier,
            ece=result.ece,
            return_pct=return_pct,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            num_bets=num_bets,
            latency_ms=latency_ms,
            coverage=coverage,
            n_forecasts=n_forecasts,
            n_possible=n_possible,
        )

    except Exception as e:
        return ExperimentResult(
            config=config,
            brier=0.0,
            ece=0.0,
            error=str(e),
        )


def _generate_markdown_report(results: list[ExperimentResult], run_id: str) -> str:
    """Generate Markdown report from results."""
    lines = []
    lines.append(f"# Experiment Report: {run_id}")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append("## Summary\n")

    # Main metrics table
    lines.append("| Config | Brier | ECE | Return % | Sharpe | Max DD % | Latency (ms) | Coverage |")
    lines.append("|--------|-------|-----|----------|--------|----------|--------------|----------|")

    for r in results:
        if r.error:
            lines.append(
                f"| {r.config.name} | ERROR | ERROR | - | - | - | - | - |"
            )
            continue

        return_str = f"{r.return_pct:.2f}" if r.return_pct is not None else "-"
        sharpe_str = f"{r.sharpe:.3f}" if r.sharpe is not None else "-"
        dd_str = f"{r.max_drawdown:.2f}" if r.max_drawdown is not None else "-"
        latency_str = f"{r.latency_ms:.1f}" if r.latency_ms is not None else "-"

        lines.append(
            f"| {r.config.name} | {r.brier:.6f} | {r.ece:.6f} | "
            f"{return_str} | {sharpe_str} | {dd_str} | {latency_str} | {r.coverage:.2%} |"
        )

    lines.append("\n## Details\n")

    for r in results:
        lines.append(f"### {r.config.name}\n")
        if r.error:
            lines.append(f"**Error:** {r.error}\n")
            continue

        lines.append(f"- **Forecaster:** {r.config.forecaster_type}")
        if r.config.forecaster_type == "multi_agent":
            lines.append(f"  - Agents: {r.config.n_agents}")
            lines.append(f"  - Supervisor: {r.config.use_supervisor}")
        if r.config.use_calibration:
            lines.append(f"  - Calibrated: Yes")
        lines.append(f"- **Forecasts:** {r.n_forecasts} / {r.n_possible} ({r.coverage:.2%})")
        lines.append(f"- **Brier Score:** {r.brier:.6f} (lower is better)")
        lines.append(f"- **ECE:** {r.ece:.6f} (lower is better)")
        if r.return_pct is not None:
            lines.append(f"- **Return:** {r.return_pct:.2f}%")
            lines.append(f"- **Sharpe:** {r.sharpe:.3f}")
            lines.append(f"- **Max Drawdown:** {r.max_drawdown:.2f}%")
            lines.append(f"- **Bets:** {r.num_bets}")
        if r.latency_ms is not None:
            lines.append(f"- **Latency:** {r.latency_ms:.1f} ms per forecast")
        lines.append("")

    return "\n".join(lines)


@app.command()
def main(
    n_events: int = typer.Option(20, help="Number of stub events"),
    n_asofs: int = typer.Option(4, help="Number of as_of timestamps"),
    hours_step: int = typer.Option(12, help="Hours between as_of timestamps"),
    run_id: Optional[str] = typer.Option(None, help="Run ID (auto-generated if not provided)"),
    calibrator_path: Optional[str] = typer.Option(None, help="Path to calibrator JSON file"),
):
    """Run experiments across multiple configurations and generate report."""
    settings = get_settings()
    
    # Generate run ID
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    
    console.print(f"[bold]Experiment Run ID:[/bold] {run_id}\n")

    # Setup feed and timestamps
    feed = StubFeed(n_events=n_events)
    now = datetime.now(timezone.utc)
    as_of_times = [now - timedelta(hours=hours_step * i) for i in range(n_asofs)][::-1]

    # Setup stores
    snapshot_store = FileSnapshotStore(settings.snapshots_dir)

    # Setup LLM and retriever (using mock for reproducibility)
    llm = MockLLM()
    retriever = StubRetriever()

    # Define experiment configurations
    configs = [
        ExperimentConfig(
            name="Baseline (Market)",
            forecaster_type="market_baseline",
        ),
        ExperimentConfig(
            name="LLM (Mock)",
            forecaster_type="llm",
        ),
        ExperimentConfig(
            name="Multi-Agent (3 agents)",
            forecaster_type="multi_agent",
            n_agents=3,
            use_supervisor=False,
        ),
        ExperimentConfig(
            name="Multi-Agent + Supervisor",
            forecaster_type="multi_agent",
            n_agents=3,
            use_supervisor=True,
        ),
    ]

    # Add calibrated version if calibrator exists
    if calibrator_path or (settings.data_dir / "calibration" / "default.json").exists():
        configs.append(
            ExperimentConfig(
                name="Baseline + Calibration",
                forecaster_type="market_baseline",
                use_calibration=True,
                calibrator_path=calibrator_path,
            )
        )

    # Run experiments
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for config in configs:
            task = progress.add_task(f"Running {config.name}...", total=None)
            result = _run_experiment(
                config, feed, as_of_times, snapshot_store, llm, retriever
            )
            results.append(result)
            progress.update(task, completed=True)

    # Generate reports
    reports_dir = settings.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = reports_dir / f"{run_id}.json"
    json_data = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_events": n_events,
            "n_asofs": n_asofs,
            "hours_step": hours_step,
        },
        "results": [asdict(r) for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)

    # Markdown report
    md_path = reports_dir / f"{run_id}.md"
    md_content = _generate_markdown_report(results, run_id)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    console.print(f"\n[green]Reports saved:[/green]")
    console.print(f"  JSON: {json_path}")
    console.print(f"  Markdown: {md_path}\n")

    # Print summary table
    from rich.table import Table
    table = Table(title="Experiment Results")
    table.add_column("Config", style="cyan")
    table.add_column("Brier", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Coverage", justify="right")

    for r in results:
        if r.error:
            table.add_row(r.config.name, "[red]ERROR[/red]", "[red]ERROR[/red]", "-", "-", "-")
        else:
            return_str = f"{r.return_pct:.2f}" if r.return_pct is not None else "-"
            sharpe_str = f"{r.sharpe:.3f}" if r.sharpe is not None else "-"
            table.add_row(
                r.config.name,
                f"{r.brier:.6f}",
                f"{r.ece:.6f}",
                return_str,
                sharpe_str,
                f"{r.coverage:.2%}",
            )

    console.print(table)


if __name__ == "__main__":
    app()


"""Script to fit calibration models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console

from forecasting.config import get_settings
from forecasting.eval.backtest import run_backtest
from forecasting.eval.metrics import brier_score, expected_calibration_error
from forecasting.feeds.stub import StubFeed
from forecasting.forecast.baseline_market import MarketBaselineForecaster
from forecasting.forecast.calibration import PlattCalibrator
from forecasting.storage.forecast_store import FileForecastStore
from forecasting.storage.snapshot_store import FileSnapshotStore

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    n_events: int = typer.Option(50, help="Number of stub events"),
    n_asofs: int = typer.Option(10, help="Number of as_of timestamps"),
    hours_step: int = typer.Option(12, help="Hours between as_of timestamps"),
    train_split: float = typer.Option(0.7, help="Fraction of data for training (earliest)"),
    name: str = typer.Option("default", help="Name for saved calibrator"),
    cache_snapshots: bool = typer.Option(True, help="Enable snapshot caching"),
):
    """Fit a calibration model on backtest predictions."""
    settings = get_settings()
    
    feed = StubFeed(n_events=n_events)
    forecaster = MarketBaselineForecaster()

    now = datetime.now(timezone.utc)
    as_ofs = [now - timedelta(hours=hours_step * i) for i in range(n_asofs)][::-1]

    snapshot_store = None
    if cache_snapshots:
        snapshot_store = FileSnapshotStore(settings.snapshots_dir)

    # Run backtest to get predictions
    console.print("[bold]Running backtest to collect predictions...[/bold]")
    result = run_backtest(
        feed=feed,
        forecaster=forecaster,
        as_of_times=as_ofs,
        snapshot_store=snapshot_store,
    )

    # Filter to rows with outcomes
    df = result.predictions.dropna(subset=["y"]).copy()
    
    if len(df) == 0:
        console.print("[red]No predictions with outcomes available.[/red]")
        raise typer.Exit(1)

    # Sort by as_of (chronological)
    df = df.sort_values("as_of")
    
    # Split by time: earliest train_split% for training, latest (1-train_split)% for validation
    n_train = int(len(df) * train_split)
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:].copy()

    console.print(f"[dim]Train: {len(df_train)} samples, Val: {len(df_val)} samples[/dim]")

    if len(df_train) == 0:
        console.print("[red]No training data available.[/red]")
        raise typer.Exit(1)

    if len(df_val) == 0:
        console.print("[red]No validation data available.[/red]")
        raise typer.Exit(1)

    # Extract predictions and outcomes
    p_train = df_train["p_yes"].to_numpy(dtype=float)
    y_train = df_train["y"].to_numpy(dtype=float)
    
    p_val = df_val["p_yes"].to_numpy(dtype=float)
    y_val = df_val["y"].to_numpy(dtype=float)

    # Fit calibrator
    console.print("[bold]Fitting calibrator...[/bold]")
    calibrator = PlattCalibrator()
    calibrator.fit(p_train, y_train)
    
    console.print(f"[dim]Calibrator parameters: a={calibrator.a:.4f}, b={calibrator.b:.4f}[/dim]")

    # Evaluate on validation set
    p_val_cal = calibrator.predict(p_val)
    
    brier_uncal = brier_score(y_val, p_val)
    brier_cal = brier_score(y_val, p_val_cal)
    ece_uncal = expected_calibration_error(y_val, p_val)
    ece_cal = expected_calibration_error(y_val, p_val_cal)

    console.print("\n[bold]Validation Results:[/bold]")
    console.print(f"Uncalibrated Brier: {brier_uncal:.6f}")
    console.print(f"Calibrated Brier:   {brier_cal:.6f} ({brier_cal - brier_uncal:+.6f})")
    console.print(f"Uncalibrated ECE:   {ece_uncal:.6f}")
    console.print(f"Calibrated ECE:     {ece_cal:.6f} ({ece_cal - ece_uncal:+.6f})")

    # Save calibrator
    calibration_dir = settings.data_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibrator_path = calibration_dir / f"{name}.json"
    
    calibrator.save(calibrator_path)
    console.print(f"\n[green]Calibrator saved to: {calibrator_path}[/green]")


if __name__ == "__main__":
    app()


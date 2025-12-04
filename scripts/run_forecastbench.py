"""Script to run ForecastBench evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from forecasting.config import get_settings
from forecasting.eval.backtest import run_backtest
from forecasting.eval.metrics import brier_score, expected_calibration_error
from forecasting.feeds.forecastbench_feed import ForecastBenchFeed
from forecasting.feeds.forecastbench_humans import aggregate_crowd, load_individual_forecasts
from forecasting.forecast.baseline_market import MarketBaselineForecaster
from forecasting.forecast.calibration import CalibratedForecaster, PlattCalibrator
from forecasting.forecast.llm_forecaster import LLMForecaster
from forecasting.forecast.multi_agent import MultiAgentForecaster
from forecasting.llm.base import LLM
from forecasting.llm.mock import MockLLM
from forecasting.retrieval.stub import StubRetriever

app = typer.Typer(add_completion=False)
console = Console()


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Compute log loss (negative log likelihood)."""
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _get_llm() -> LLM:
    """Get LLM instance (auto-selects if available, else mock)."""
    try:
        from forecasting.llm.gemini_client import get_llm_if_available
        llm = get_llm_if_available()
        if llm is not None:
            return llm
    except ImportError:
        pass
    return MockLLM()


@app.command()
def main(
    question_set: Path = typer.Option(
        Path("data/forecastbench/question_sets/2024-07-21-human.json"),
        help="Path to ForecastBench question set JSON (can be question set with 'questions' array or human forecast file with 'forecasts' array)",
    ),
    resolution_set: Path = typer.Option(
        Path("data/forecastbench/resolution_sets/2024-07-21_resolution_set.json"),
        help="Path to ForecastBench resolution set JSON",
    ),
    public_forecasts: Optional[Path] = typer.Option(
        Path("data/forecastbench/question_sets/2024-07-21.ForecastBench.human_public_individual.json"),
        help="Path to human public individual forecasts JSON",
    ),
    super_forecasts: Optional[Path] = typer.Option(
        Path("data/forecastbench/question_sets/2024-07-21.ForecastBench.human_super_individual.json"),
        help="Path to human superforecaster individual forecasts JSON",
    ),
    split: str = typer.Option("fb_market", help="Split: 'fb_market' (prediction markets only) or 'all'"),
    use_llm: bool = typer.Option(False, help="Include single LLM forecaster"),
    use_multi_agent: bool = typer.Option(False, help="Include multi-agent forecaster"),
    calibrator_path: Optional[Path] = typer.Option(None, help="Path to saved PlattCalibrator JSON"),
):
    """
    Run ForecastBench evaluation comparing multiple forecasters and human baselines.
    """
    settings = get_settings()
    
    question_set = Path(question_set)
    resolution_set = Path(resolution_set)
    
    if not question_set.exists():
        raise typer.BadParameter(f"Question set not found: {question_set}")
    if not resolution_set.exists():
        raise typer.BadParameter(f"Resolution set not found: {resolution_set}")
    
    console.print(f"[bold]ForecastBench Evaluation[/bold]")
    console.print(f"Question set: {question_set}")
    console.print(f"Resolution set: {resolution_set}")
    console.print(f"Split: {split}\n")
    
    # Create feed
    feed = ForecastBenchFeed(
        question_set_path=question_set,
        resolution_set_path=resolution_set,
        split=split,
    )
    
    # Single as_of horizon
    as_of_times = [feed.forecast_due_date]
    
    # Build forecasters dictionary
    forecasters = {}
    
    # Always include market baseline
    market_baseline = MarketBaselineForecaster()
    forecasters["Market Baseline"] = market_baseline
    
    # Add LLM if requested
    if use_llm:
        llm = _get_llm()
        retriever = StubRetriever()
        forecasters["LLM"] = LLMForecaster(llm=llm, retriever=retriever)
    
    # Add multi-agent if requested
    if use_multi_agent:
        llm = _get_llm()
        retriever = StubRetriever()
        forecasters["Multi-Agent"] = MultiAgentForecaster(
            llm=llm,
            retriever=retriever,
            n_agents=3,
            use_supervisor=True,
        )
    
    # Add calibrated variants if calibrator provided
    if calibrator_path and calibrator_path.exists():
        calibrator = PlattCalibrator.load(calibrator_path)
        forecasters["Market Baseline + Calibration"] = CalibratedForecaster(
            base_forecaster=market_baseline,
            calibrator=calibrator,
        )
        if use_llm:
            llm_forecaster = forecasters["LLM"]
            forecasters["LLM + Calibration"] = CalibratedForecaster(
                base_forecaster=llm_forecaster,
                calibrator=calibrator,
            )
    
    # Run backtests for each forecaster
    results = {}
    for name, forecaster in forecasters.items():
        console.print(f"[dim]Running {name}...[/dim]")
        try:
            result = run_backtest(
                feed=feed,
                forecaster=forecaster,
                as_of_times=as_of_times,
                snapshot_store=None,
                forecast_store=None,
            )
            
            # Compute coverage
            n_forecasts = len(result.predictions)
            n_events = len(feed.list_events(as_of_times[0]))
            coverage = n_forecasts / n_events if n_events > 0 else 0.0
            
            # Compute NLL if we have outcomes
            df_scored = result.predictions.dropna(subset=["y", "p_yes"])
            nll = float("nan")
            if len(df_scored) > 0:
                y_true = df_scored["y"].to_numpy(dtype=float)
                y_prob = df_scored["p_yes"].to_numpy(dtype=float)
                nll = log_loss(y_true, y_prob)
            
            results[name] = {
                "brier": result.brier,
                "ece": result.ece,
                "nll": nll,
                "coverage": coverage,
                "n_forecasts": n_forecasts,
                "n_events": n_events,
            }
        except Exception as e:
            console.print(f"[red]Error running {name}: {e}[/red]")
            results[name] = {
                "brier": float("nan"),
                "ece": float("nan"),
                "nll": float("nan"),
                "coverage": 0.0,
                "error": str(e),
            }
    
    # Load human baselines
    console.print("\n[dim]Loading human baselines...[/dim]")
    
    # Load outcomes from resolution set
    import json
    with resolution_set.open("r") as f:
        res_data = json.load(f)
    
    outcomes_rows = []
    for r in res_data.get("resolutions", []):
        if not r.get("resolved", False):
            continue
        qid = r.get("id")
        # Handle case where id might be a list (take first element)
        if isinstance(qid, list):
            if not qid:
                continue
            qid = qid[0]
        if not isinstance(qid, str):
            continue
        outcomes_rows.append({
            "question_id": qid,
            "y": int(float(r["resolved_to"])),
        })
    outcomes_df = pd.DataFrame(outcomes_rows)
    
    # Public crowd
    if public_forecasts and Path(public_forecasts).exists():
        try:
            public_df = load_individual_forecasts(Path(public_forecasts))
            public_crowd = aggregate_crowd(public_df, agg="median")
            public_merged = public_crowd.merge(outcomes_df, on="question_id", how="inner")
            public_merged = public_merged.dropna(subset=["p_crowd", "y"])
            
            if len(public_merged) > 0:
                y_true = public_merged["y"].to_numpy(dtype=float)
                y_prob = public_merged["p_crowd"].to_numpy(dtype=float)
                results["Public Crowd (ForecastBench)"] = {
                    "brier": brier_score(y_true, y_prob),
                    "ece": expected_calibration_error(y_true, y_prob, n_bins=15),
                    "nll": log_loss(y_true, y_prob),
                    "coverage": len(public_merged) / len(outcomes_df) if len(outcomes_df) > 0 else 0.0,
                    "n_forecasts": len(public_merged),
                    "n_events": len(outcomes_df),
                }
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load public forecasts: {e}[/yellow]")
    
    # Superforecasters
    if super_forecasts and Path(super_forecasts).exists():
        try:
            super_df = load_individual_forecasts(Path(super_forecasts))
            super_crowd = aggregate_crowd(super_df, agg="median")
            super_merged = super_crowd.merge(outcomes_df, on="question_id", how="inner")
            super_merged = super_merged.dropna(subset=["p_crowd", "y"])
            
            if len(super_merged) > 0:
                y_true = super_merged["y"].to_numpy(dtype=float)
                y_prob = super_merged["p_crowd"].to_numpy(dtype=float)
                results["Superforecasters (ForecastBench)"] = {
                    "brier": brier_score(y_true, y_prob),
                    "ece": expected_calibration_error(y_true, y_prob, n_bins=15),
                    "nll": log_loss(y_true, y_prob),
                    "coverage": len(super_merged) / len(outcomes_df) if len(outcomes_df) > 0 else 0.0,
                    "n_forecasts": len(super_merged),
                    "n_events": len(outcomes_df),
                }
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load superforecaster forecasts: {e}[/yellow]")
    
    # Display results table
    table = Table(title="ForecastBench Results")
    table.add_column("Forecaster", style="cyan")
    table.add_column("Brier", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("NLL", justify="right")
    table.add_column("Coverage", justify="right")
    
    for name, res in results.items():
        if "error" in res:
            table.add_row(name, "[red]ERROR[/red]", "[red]ERROR[/red]", "[red]ERROR[/red]", "-")
        else:
            brier_str = f"{res['brier']:.6f}" if not np.isnan(res['brier']) else "N/A"
            ece_str = f"{res['ece']:.6f}" if not np.isnan(res['ece']) else "N/A"
            nll_str = f"{res['nll']:.6f}" if not np.isnan(res['nll']) else "N/A"
            coverage_str = f"{res['coverage']:.2%}"
            table.add_row(name, brier_str, ece_str, nll_str, coverage_str)
    
    console.print("\n")
    console.print(table)
    
    # Summary
    console.print(f"\n[dim]Total events: {len(feed.list_events(as_of_times[0]))}[/dim]")
    console.print(f"[dim]Resolved outcomes: {len(outcomes_df)}[/dim]")


if __name__ == "__main__":
    app()


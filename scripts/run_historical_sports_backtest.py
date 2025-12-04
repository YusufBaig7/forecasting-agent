from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional


import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from forecasting.config import get_settings
from forecasting.eval.metrics import brier_score, expected_calibration_error
from forecasting.feeds.sports_odds import SportsOddsFeed
from forecasting.storage.snapshot_store import FileSnapshotStore
from forecasting.storage.forecast_store import FileForecastStore

# Import your forecasters the same way run_backtest.py does
from forecasting.forecast.baseline_market import MarketBaselineForecaster
from forecasting.forecast.llm_forecaster import LLMForecaster  # only if you need it


app = typer.Typer(add_completion=False)
console = Console()


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def make_as_of_times(kickoff: datetime) -> list[datetime]:
    """Opening / day-of / 1h pregame (heuristics)."""
    kickoff = _utc(kickoff)

    opening = kickoff - timedelta(days=2)

    # "day-of" = noon UTC on game date, but ensure it's < kickoff
    day_of = datetime.combine(kickoff.date(), time(12, 0), tzinfo=timezone.utc)
    if day_of >= kickoff:
        day_of = kickoff - timedelta(hours=6)

    one_hour = kickoff - timedelta(hours=1)

    # unique + sorted + strictly pregame
    out = sorted({_utc(t) for t in [opening, day_of, one_hour] if _utc(t) < kickoff})
    return out


def build_forecaster(name: str):
    if name == "market_baseline":
        return MarketBaselineForecaster()
    if name == "llm":
        # Use whatever constructor your LLMForecaster expects in your repo
        raise RuntimeError("LLM forecaster wiring not added in this script yet—use market_baseline for now.")
    raise typer.BadParameter(f"Unknown forecaster: {name}")


def make_as_of_times(kickoff: datetime) -> list[tuple[str, datetime]]:
    kickoff = _utc(kickoff)
    return [
        ("open", kickoff - timedelta(hours=24)),   # proxy for “opening”
        ("day_of", kickoff - timedelta(hours=6)),  # proxy for “day-of”
        ("1h", kickoff - timedelta(hours=1)),      # 1 hour pre-game
    ]

@app.command()
def main(
    league: str = typer.Option("nba"),
    forecaster: str = typer.Option("market_baseline"),
    start_date: str = typer.Option(..., help="YYYY-MM-DD (inclusive)"),
    end_date: str = typer.Option(..., help="YYYY-MM-DD (inclusive)"),
    limit_events: int = typer.Option(300),
    cache_snapshots: bool = typer.Option(True),
    log_forecasts: bool = typer.Option(True),
):
    """
    Historical sports backtest:
      - iterates games in [start_date, end_date]
      - forecasts at opening/day-of/1h-pre
      - scores Brier/ECE/log-loss on resolved outcomes
    """
    settings = get_settings()

    try:
        start_d = date.fromisoformat(start_date)
        end_d = date.fromisoformat(end_date)
    except ValueError as e:
        raise typer.BadParameter("Dates must be YYYY-MM-DD") from e

    forecaster_instance = build_forecaster(forecaster)

    feed = SportsOddsFeed(
        settings.sportsradar_api_key,
        settings.sportsradar_base_url,
        settings.opticodds_api_key,
        settings.opticodds_base_url,
        league=league,
    )

    snapshot_store = FileSnapshotStore(settings.snapshots_dir) if cache_snapshots else None
    forecast_store = FileForecastStore(settings.forecasts_dir) if log_forecasts else None
    if forecast_store is not None:
        forecast_store.start_run(forecaster_instance.model_name)

    console.print(f"[dim]Snapshots: {settings.snapshots_dir}[/dim]")
    if forecast_store is not None:
        console.print(f"[dim]Forecasts:  {settings.forecasts_dir}[/dim]")
        console.print(f"[dim]Run ID:     {forecast_store.run_id}[/dim]")

    events = feed.list_events_between(start_d, end_d)
    console.print(f"Loaded {len(events)} events from {start_d} to {end_d}")


    rows = []
    outcome_cache: dict[str, Optional[int]] = {}

    for ev in events:
        kickoff = _utc(ev.close_time)
        asofs = make_as_of_times(kickoff)

        # outcome only depends on event, cache it
        if ev.event_id not in outcome_cache:
            out = feed.get_outcome(ev.event_id)
            outcome_cache[ev.event_id] = (out.outcome if out is not None else None)

        y = outcome_cache[ev.event_id]

    rows = []

    for ev in events:
        kickoff = _utc(ev.close_time)

        for asof_kind, as_of in make_as_of_times(kickoff):
            # don’t time-travel past kickoff
            if as_of >= kickoff:
                continue

            # ---- snapshot: cache -> fetch -> cache ----
            snap = None
            if snapshot_store is not None:
                snap = snapshot_store.get(ev.event_id, as_of)

            if snap is None:
                snap = feed.get_snapshot(ev.event_id, as_of=as_of)
                if snap is not None and snapshot_store is not None:
                    snapshot_store.put(snap)

            if snap is None:
                continue

            # ---- forecast ----
            fc = forecaster_instance.predict(ev, snap, as_of=as_of)
            if forecast_store is not None:
                forecast_store.log_forecast(fc)

            # ---- outcome ----
            out = feed.get_outcome(ev.event_id)
            y = out.outcome if out is not None else None

            q = getattr(snap, "market_prob", None)

            rows.append(
                {
                    "event_id": ev.event_id,
                    "kickoff": kickoff,
                    "as_of": as_of,
                    "asof_kind": asof_kind,
                    "p_yes": fc.p_yes,
                    "model": fc.model,
                    "q_market": q,
                    "y": y,
                }
            )


    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No predictions produced. (No snapshots matched / no odds returned.)")

    df_scored = df.dropna(subset=["y", "p_yes"]).copy()
    if df_scored.empty:
        raise RuntimeError("No resolved outcomes in this date range (all y are NA). Pick older dates.")

    # ---- overall ----
    y_true = df_scored["y"].to_numpy(dtype=float)
    y_prob = df_scored["p_yes"].to_numpy(dtype=float)

    brier = brier_score(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)
    nll = log_loss(y_true, y_prob)  # make sure you imported/sklearn-installed this

    console.print(f"\nBrier: {brier:.6f}")
    console.print(f"ECE:   {ece:.6f}")
    console.print(f"NLL:   {nll:.6f}")

    # ---- coverage (super useful) ----
    console.print(
        f"Coverage: rows={len(df)} scored={len(df_scored)} "
        f"({len(df_scored)/max(1,len(df))*100:.1f}%)"
    )

    # ---- per as-of bucket ----
    if "asof_kind" in df_scored.columns:
        console.print("\n[bold]Metrics by asof_kind[/bold]")
        for kind, g in df_scored.groupby("asof_kind"):
            yt = g["y"].to_numpy(dtype=float)
            yp = g["p_yes"].to_numpy(dtype=float)

            console.print(
                f"{kind:>6}  n={len(g):4d}  "
                f"brier={brier_score(yt, yp):.6f}  "
                f"ece={expected_calibration_error(yt, yp, n_bins=15):.6f}  "
                f"nll={log_loss(yt, yp):.6f}"
            )
    else:
        console.print("[yellow]No asof_kind column found; add it to rows to get per-timestamp metrics.[/yellow]")


    # show sample
    sample = df.sort_values(["as_of", "event_id"]).head(12)

    table = Table(title="Sample predictions (first 12 rows)")
    for col in ["as_of", "event_id", "p_yes", "y", "q_market"]:
        table.add_column(col)
    for _, r in sample.iterrows():
        table.add_row(
            str(r["as_of"]),
            str(r["event_id"]),
            f"{float(r['p_yes']):.3f}",
            "NA" if pd.isna(r["y"]) else str(int(r["y"])),
            "NA" if pd.isna(r["q_market"]) else f"{float(r['q_market']):.3f}",
        )
    console.print(table)

    # persist results
    out_path = Path(settings.forecasts_dir) / f"historical_{league}_{forecaster_instance.model_name.replace('/','_')}.csv"
    df.to_csv(out_path, index=False)
    console.print(f"[dim]Wrote rows={len(df)} to {out_path}[/dim]")


if __name__ == "__main__":
    app()

"""Backtesting implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd
import numpy as np

from forecasting.feeds.base import Feed
from forecasting.eval.metrics import brier_score, expected_calibration_error
from forecasting.eval.trading import TradingResult, simulate_trading
from forecasting.storage.snapshot_store import FileSnapshotStore
from forecasting.storage.forecast_store import FileForecastStore


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame
    brier: float
    ece: float
    trading: Optional[TradingResult] = None


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def run_backtest(
    feed: Feed,
    forecaster,  # Any object with predict(event, snapshot, as_of) and model_name
    as_of_times: Iterable[datetime],
    limit_events: Optional[int] = None,
    snapshot_store: Optional[FileSnapshotStore] = None,
    forecast_store: Optional[FileForecastStore] = None,
) -> BacktestResult:
    """
    Run a backtest with optional snapshot caching and forecast logging.

    Args:
        feed: Data feed for events, snapshots, and outcomes
        forecaster: Forecaster to generate predictions
        as_of_times: Timestamps to evaluate at (must be in chronological order for caching)
        limit_events: Optional limit on number of events to process
        snapshot_store: Optional store for caching snapshots (checks cache before fetching)
        forecast_store: Optional store for logging forecasts (must call start_run() before use)

    Returns:
        BacktestResult with predictions dataframe and metrics
    """
    as_of_times = [_utc(t) for t in as_of_times]
    if not as_of_times:
        raise ValueError("as_of_times must be non-empty")

    # Start forecast logging if store is provided
    if forecast_store is not None:
        forecast_store.start_run(forecaster.model_name)

    events = feed.list_events(as_of_times[0])
    if limit_events is not None:
        events = events[:limit_events]

    rows = []
    for as_of in as_of_times:
        for ev in events:
            # Try cache first, then fetch from feed
            snap = None
            if snapshot_store is not None:
                snap = snapshot_store.get(ev.event_id, as_of)
            if snap is None:
                snap = feed.get_snapshot(ev.event_id, as_of=as_of)
                # Cache the snapshot if we have a store
                if snap is not None and snapshot_store is not None:
                    snapshot_store.put(snap)

            if snap is None:
                continue

            # Generate forecast
            fc = forecaster.predict(ev, snap, as_of=as_of)
            
            # Always log the forecast if store is provided
            if forecast_store is not None:
                forecast_store.log_forecast(fc)

            outcome = feed.get_outcome(ev.event_id)
            y = outcome.outcome if outcome is not None else None
            
            # Get market probability from snapshot
            q = snap.best_market_prob()
            rows.append(
                {
                    "event_id": ev.event_id,
                    "as_of": as_of,
                    "p_yes": fc.p_yes,
                    "model": fc.model,
                    "q_market": q,
                    "y": y,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No predictions produced (empty dataset).")

    df_known = df.dropna(subset=["y"]).copy()
    if df_known.empty:
        return BacktestResult(
            predictions=df,
            brier=float("nan"),
            ece=float("nan"),
            trading=None,
        )


    y_true = df_known["y"].to_numpy(dtype=float)
    y_prob = df_known["p_yes"].to_numpy(dtype=float)

    # Compute trading metrics if market probabilities are available
    trading_result = None
    df_with_market = df_known.dropna(subset=["q_market"]).copy()
    if len(df_with_market) > 0:
        trading_result = simulate_trading(
            df_with_market,
            forecast_col="p_yes",
            market_col="q_market",
            outcome_col="y",
        )

    return BacktestResult(
        predictions=df,
        brier=brier_score(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob, n_bins=15),
        trading=trading_result,
    )

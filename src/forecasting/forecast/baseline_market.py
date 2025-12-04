"""Baseline market forecast implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from forecasting.models import Event, MarketSnapshot, Forecast


@dataclass(frozen=True)
class MarketBaselineForecaster:
    model_name: str = "market_baseline/v1"

    def predict(self, event: Event, snapshot: MarketSnapshot, as_of: datetime) -> Forecast:
        p = snapshot.best_market_prob()
        if p is None:
            p = 0.5
        return Forecast(
            event_id=event.event_id,
            as_of=as_of,
            p_yes=float(p),
            model=self.model_name,
            rationale="Baseline: uses market implied probability (or 0.5 if unavailable).",
            metadata={"source": event.source, "liquidity": snapshot.liquidity},
        )

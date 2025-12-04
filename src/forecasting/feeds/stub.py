"""Stub data feed implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import math
from typing import Optional

from forecasting.models import Event, MarketSnapshot, ResolvedOutcome


def _hash_to_unit_float(s: str) -> float:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    # Take 52 bits -> float in [0,1)
    x = int(h[:13], 16)  # 13 hex chars ~ 52 bits
    return x / float(1 << 52)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class StubFeed:
    """
    Deterministic fake data for development.
    - event list is stable given (n_events).
    - snapshot probabilities drift smoothly over time.
    - outcomes are stable per event.
    """
    n_events: int = 20
    seed: str = "stub-v1"

    def list_events(self, as_of: datetime) -> list[Event]:
        # Normalize as_of to UTC
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        else:
            as_of = as_of.astimezone(timezone.utc)

        out: list[Event] = []
        base_close = as_of + timedelta(days=7)
        for i in range(self.n_events):
            event_id = f"stub:{i:04d}"
            title = f"Stub Event {i:04d}"
            question = f"Will stub event {i:04d} resolve YES?"
            close_time = base_close + timedelta(hours=i)
            out.append(
                Event(
                    event_id=event_id,
                    title=title,
                    question=question,
                    close_time=close_time,
                    resolution_time=close_time + timedelta(days=3),
                    resolution_criteria="Synthetic: outcome determined by stable hash.",
                    source="stub",
                )
            )
        return out

    def get_snapshot(self, event_id: str, as_of: datetime) -> Optional[MarketSnapshot]:
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        else:
            as_of = as_of.astimezone(timezone.utc)

        # Drift: base logit per event + sinusoidal time component
        base = (_hash_to_unit_float(self.seed + "|" + event_id) - 0.5) * 3.0  # ~[-1.5, 1.5]
        t = as_of.timestamp() / 86400.0  # days
        drift = math.sin(t * 0.5) * 0.7  # smooth nonstationary drift
        p = _sigmoid(base + drift)
        liquidity = 1000.0 * (0.2 + _hash_to_unit_float("liq|" + event_id))
        return MarketSnapshot(
            event_id=event_id,
            as_of=as_of,
            market_prob=float(p),
            liquidity=float(liquidity),
            raw={"provider": "stub", "version": "v1"},
        )

    def get_outcome(self, event_id: str) -> Optional[ResolvedOutcome]:
        # Stable binary label
        u = _hash_to_unit_float(self.seed + "|outcome|" + event_id)
        y = 1 if u >= 0.5 else 0
        return ResolvedOutcome(event_id=event_id, outcome=y)

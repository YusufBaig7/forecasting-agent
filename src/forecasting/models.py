"""Data models for the forecasting agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        # Treat naive datetimes as UTC to avoid ambiguity.
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class Event(BaseModel):
    event_id: str = Field(..., description="Canonical event identifier (your internal ID)")
    title: str
    question: str
    close_time: datetime = Field(..., description="When trading/forecasting closes (UTC)")
    resolution_time: Optional[datetime] = Field(None, description="When it resolves (UTC), if known")
    resolution_criteria: Optional[str] = None
    source: Optional[str] = Field(None, description="Provider/source name")

    @field_validator("close_time", "resolution_time")
    @classmethod
    def _dt_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        return _ensure_utc(v) if v is not None else None


class MarketSnapshot(BaseModel):
    event_id: str
    as_of: datetime = Field(..., description="Snapshot timestamp (UTC)")

    # Preferred: store a probability directly if the provider exposes it.
    market_prob: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Implied market probability (0..1)"
    )

    # Optional: store odds if that's the raw source.
    # (For binary markets, you usually need both sides to remove vig; this is kept minimal.)
    odds_format: Optional[Literal["decimal", "american"]] = None
    odds: Optional[float] = None

    liquidity: Optional[float] = Field(None, ge=0.0, description="Any liquidity proxy, if available")
    raw: dict[str, Any] = Field(default_factory=dict, description="Provider-specific raw payload")

    @field_validator("as_of")
    @classmethod
    def _as_of_utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)

    def implied_prob_from_odds(self) -> Optional[float]:
        if self.odds is None or self.odds_format is None:
            return None
        if self.odds_format == "decimal":
            if self.odds <= 1.0:
                return None
            return 1.0 / self.odds
        if self.odds_format == "american":
            a = self.odds
            if a == 0:
                return None
            # American odds: +150 means win 150 on 100 stake; -150 means stake 150 to win 100.
            if a > 0:
                return 100.0 / (a + 100.0)
            return (-a) / ((-a) + 100.0)
        return None

    def best_market_prob(self) -> Optional[float]:
        return self.market_prob if self.market_prob is not None else self.implied_prob_from_odds()


class Forecast(BaseModel):
    event_id: str
    as_of: datetime
    p_yes: float = Field(..., ge=0.0, le=1.0, description="Forecast probability of YES/outcome=1")
    model: str = Field(..., description="Model identifier/version")
    rationale: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("as_of")
    @classmethod
    def _as_of_utc(cls, v: datetime) -> datetime:
        return _ensure_utc(v)


class ResolvedOutcome(BaseModel):
    event_id: str
    outcome: int = Field(..., description="Binary outcome: 1 for YES, 0 for NO")

    @field_validator("outcome")
    @classmethod
    def _binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("outcome must be 0 or 1")
        return v

"""Base class for data feeds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from forecasting.models import Event, MarketSnapshot, ResolvedOutcome


class Feed(ABC):
    @abstractmethod
    def list_events(self, as_of: datetime) -> list[Event]:
        """Return all active/known events as of timestamp."""
        raise NotImplementedError

    @abstractmethod
    def get_snapshot(self, event_id: str, as_of: datetime) -> Optional[MarketSnapshot]:
        """Return a market snapshot for (event_id, as_of) or None."""
        raise NotImplementedError

    @abstractmethod
    def get_outcome(self, event_id: str) -> Optional[ResolvedOutcome]:
        """Return resolved outcome if known, else None."""
        raise NotImplementedError

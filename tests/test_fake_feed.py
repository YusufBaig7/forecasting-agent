"""Fake feed for testing that counts calls."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Optional

from forecasting.feeds.base import Feed
from forecasting.models import Event, MarketSnapshot, ResolvedOutcome


class FakeFeed(Feed):
    """
    A test feed that counts method calls and returns configurable data.
    
    Useful for testing caching behavior - you can verify that get_snapshot
    is called fewer times when caching is enabled.
    """

    def __init__(self, events: list[Event], snapshots: dict[tuple[str, datetime], MarketSnapshot]):
        """
        Initialize with events and snapshots.
        
        Args:
            events: List of events to return
            snapshots: Dict mapping (event_id, as_of) -> MarketSnapshot
        """
        self.events = events
        self.snapshots = snapshots
        self.call_counts = defaultdict(int)

    def list_events(self, as_of: datetime) -> list[Event]:
        """Return all events."""
        self.call_counts["list_events"] += 1
        return self.events

    def get_snapshot(self, event_id: str, as_of: datetime) -> Optional[MarketSnapshot]:
        """Return snapshot if available, else None."""
        self.call_counts["get_snapshot"] += 1
        key = (event_id, as_of)
        return self.snapshots.get(key)

    def get_outcome(self, event_id: str) -> Optional[ResolvedOutcome]:
        """Return outcome if available, else None."""
        self.call_counts["get_outcome"] += 1
        # Default: return None (unresolved)
        return None

    def get_call_count(self, method: str) -> int:
        """Get the number of times a method was called."""
        return self.call_counts.get(method, 0)

    def reset_counts(self) -> None:
        """Reset all call counts."""
        self.call_counts.clear()


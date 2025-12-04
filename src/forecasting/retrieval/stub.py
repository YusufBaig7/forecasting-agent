"""Stub retriever with deterministic fake items."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from forecasting.models import Event
from forecasting.retrieval.base import ContextBundle, RetrievedItem, Retriever


def _hash_to_unit_float(s: str) -> float:
    """Convert string to deterministic float in [0, 1)."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    x = int(h[:13], 16)  # 13 hex chars ~ 52 bits
    return x / float(1 << 52)


class StubRetriever:
    """
    Deterministic stub retriever that generates fake items based on event_id.
    
    Useful for testing and development without external dependencies.
    """

    def __init__(self, n_items: int = 5, seed: str = "stub-retriever-v1"):
        """
        Initialize stub retriever.
        
        Args:
            n_items: Number of fake items to return
            seed: Seed for deterministic generation
        """
        self.n_items = n_items
        self.seed = seed

    def get_context(self, event: Event, as_of: datetime) -> ContextBundle:
        """
        Generate deterministic fake context for an event.
        
        Args:
            event: Event to retrieve context for
            as_of: Timestamp for retrieval
            
        Returns:
            ContextBundle with fake items
        """
        # Normalize as_of to UTC
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        else:
            as_of = as_of.astimezone(timezone.utc)

        items = []
        for i in range(self.n_items):
            # Deterministic hash based on event_id, as_of, and item index
            key = f"{self.seed}|{event.event_id}|{as_of.isoformat()}|{i}"
            h = _hash_to_unit_float(key)
            
            # Generate deterministic fake data
            title = f"Article {i+1} about {event.title[:30]}"
            url = f"https://example.com/article/{event.event_id}/{i}"
            snippet = (
                f"This is a synthetic article snippet about {event.question[:50]}. "
                f"Generated deterministically from hash {h:.6f}."
            )
            # Published at some time before as_of
            days_ago = int(h * 30) + 1  # 1-30 days ago
            published_at = as_of - timedelta(days=days_ago)
            
            source = f"stub_source_{i % 3}"  # 3 different sources
            
            items.append(
                RetrievedItem(
                    title=title,
                    url=url,
                    snippet=snippet,
                    published_at=published_at,
                    source=source,
                )
            )

        summary_text = (
            f"Retrieved {len(items)} items for event {event.event_id} "
            f"as of {as_of.isoformat()}. All items are synthetic."
        )

        return ContextBundle(
            as_of=as_of,
            items=items,
            summary_text=summary_text,
        )


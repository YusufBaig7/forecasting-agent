"""Base classes for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from forecasting.models import Event


@dataclass
class RetrievedItem:
    """A single retrieved item (article, document, etc.)."""

    title: str
    url: str
    snippet: str
    published_at: datetime
    source: str


@dataclass
class ContextBundle:
    """Bundle of retrieved context for an event."""

    as_of: datetime
    items: list[RetrievedItem]
    summary_text: str = ""


class Retriever(Protocol):
    """Protocol for retrievers that fetch context for events."""

    def get_context(self, event: Event, as_of: datetime) -> ContextBundle:
        """
        Retrieve context for an event at a given timestamp.
        
        Args:
            event: Event to retrieve context for
            as_of: Timestamp for retrieval (must be time-bounded)
            
        Returns:
            ContextBundle with retrieved items and summary
        """
        ...


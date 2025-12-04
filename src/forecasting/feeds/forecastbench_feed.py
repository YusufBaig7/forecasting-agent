"""ForecastBench feed implementation."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from dateutil import parser as dateutil_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False

from forecasting.feeds.base import Feed
from forecasting.models import Event, MarketSnapshot, ResolvedOutcome


MARKET_SOURCES = {
    "manifold",
    "polymarket",
    "kalshi",
    "metaculus",
    # add others as you see in the question_set file
}


def _parse_dt(s: str) -> datetime:
    """Parse ISO datetime string to UTC datetime."""
    if isinstance(s, str):
        if HAS_DATEUTIL:
            dt = dateutil_parser.isoparse(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        else:
            # Fallback to fromisoformat
            if "T" in s or " " in s:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            else:
                # Date-only string, assume midnight UTC
                dt = datetime.fromisoformat(s + "T00:00:00+00:00")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    raise ValueError(f"Invalid datetime string: {s}")


@dataclass
class _FBQuestion:
    raw: dict
    event: Event
    freeze_datetime: datetime
    freeze_value: float


class ForecastBenchFeed(Feed):
    """
    Feed over a static ForecastBench question set.
    
    Supports both standard question set files (with 'questions' array) and
    human forecast files (with 'forecasts' array) as a fallback.
    """

    def __init__(
        self,
        question_set_path: Path,
        resolution_set_path: Optional[Path] = None,
        split: str = "fb_market",
    ) -> None:
        self.question_set_path = Path(question_set_path)
        self.resolution_set_path = Path(resolution_set_path) if resolution_set_path else None
        self.split = split

        with self.question_set_path.open("r", encoding="utf-8") as f:
            qs = json.load(f)

        self.forecast_due_date: datetime = _parse_dt(qs["forecast_due_date"])
        
        # Check if this is a human forecast file (has "forecasts" array) or question set (has "questions" array)
        if "forecasts" in qs:
            # Human forecast file - extract questions from forecasts
            forecasts = qs["forecasts"]
            
            # Group forecasts by question ID
            forecasts_by_qid = defaultdict(list)
            for f in forecasts:
                qid = f.get("id")
                if qid:
                    forecasts_by_qid[qid].append(f)
            
            # Create question objects from forecasts
            questions_raw = []
            for qid, q_forecasts in forecasts_by_qid.items():
                # Use first forecast as template for question metadata
                first_f = q_forecasts[0]
                
                # Calculate median forecast as market probability
                forecast_values = [float(f.get("forecast", 0.5)) for f in q_forecasts if f.get("forecast") is not None]
                median_forecast = float(np.median(forecast_values)) if forecast_values else 0.5
                
                # Filter by source if needed
                source = first_f.get("source", "")
                if split == "fb_market" and source not in MARKET_SOURCES:
                    continue
                
                questions_raw.append({
                    "id": qid,
                    "source": source,
                    "question": f"Question {qid}",
                    "freeze_datetime": qs["forecast_due_date"],
                    "freeze_datetime_value": median_forecast,
                    "market_info_close_datetime": qs["forecast_due_date"],
                })
        else:
            # Standard question set file
            questions_raw = qs.get("questions", [])
            
            # Filter to FB-Market-ish subset: only prediction-market sources
            if split == "fb_market":
                questions_raw = [
                    q for q in questions_raw
                    if q.get("source") in MARKET_SOURCES
                ]

        self._questions: Dict[str, _FBQuestion] = {}
        for q in questions_raw:
            event = self._to_event(q)
            fbq = _FBQuestion(
                raw=q,
                event=event,
                freeze_datetime=_parse_dt(q.get("freeze_datetime", qs["forecast_due_date"])),
                freeze_value=float(q.get("freeze_datetime_value", 0.5)),
            )
            self._questions[event.event_id] = fbq

        # Outcomes (may be partially unresolved)
        self._outcomes: Dict[str, ResolvedOutcome] = {}
        if self.resolution_set_path and self.resolution_set_path.exists():
            self._load_resolutions()

    # -------- Feed interface implementation --------

    def list_events(self, as_of: datetime) -> List[Event]:
        """Return all events when as_of >= forecast_due_date (no early peeking)."""
        if as_of < self.forecast_due_date:
            return []
        return [fbq.event for fbq in self._questions.values()]

    def list_events_between(
        self,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None,
    ) -> Iterable[Event]:
        """List events between start and end dates (delegates to list_events for static dataset)."""
        return self.list_events(as_of or self.forecast_due_date)

    def get_snapshot(self, event_id: str, as_of: datetime) -> Optional[MarketSnapshot]:
        fbq = self._questions.get(event_id)
        if fbq is None:
            return None

        # Allow snapshot if as_of is at or after freeze_datetime, or if as_of matches forecast_due_date
        # (forecast_due_date is the evaluation horizon, so we allow it to access freeze_datetime snapshots)
        time_diff = abs((as_of - fbq.freeze_datetime).total_seconds())
        forecast_due_diff = abs((as_of - self.forecast_due_date).total_seconds())
        
        # Allow if: (1) as_of matches freeze_datetime closely, OR (2) as_of matches forecast_due_date closely
        if time_diff > 1 and forecast_due_diff > 1:
            return None

        return MarketSnapshot(
            event_id=event_id,
            as_of=as_of,
            market_prob=fbq.freeze_value,
            liquidity=None,
            raw=fbq.raw,
        )

    def get_outcome(self, event_id: str) -> Optional[ResolvedOutcome]:
        return self._outcomes.get(event_id)

    # -------- Helpers --------

    def _to_event(self, q: dict) -> Event:
        """Convert ForecastBench question dict to Event."""
        return Event(
            event_id=q["id"],
            title=q.get("question", q.get("title", "")),
            question=q.get("question", ""),
            close_time=_parse_dt(
                q.get("market_info_close_datetime") 
                or q.get("close_time") 
                or q.get("freeze_datetime")
            ),
            resolution_criteria=q.get("resolution_criteria"),
            source=q.get("source"),
        )

    def _load_resolutions(self) -> None:
        """Load resolved outcomes from resolution set JSON."""
        if not self.resolution_set_path or not self.resolution_set_path.exists():
            return
            
        with self.resolution_set_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for r in data.get("resolutions", []):
            if not r.get("resolved", False):
                continue
                
            qid = r.get("id")
            # Handle case where id might be a list (take first element)
            if isinstance(qid, list):
                if not qid:
                    continue
                qid = qid[0]
            
            if not isinstance(qid, str) or qid not in self._questions:
                continue

            raw_value = r.get("resolved_to")
            if raw_value is None:
                continue

            self._outcomes[qid] = ResolvedOutcome(
                event_id=qid,
                outcome=int(float(raw_value)),
            )

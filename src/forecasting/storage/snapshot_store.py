"""Snapshot storage implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from forecasting.models import MarketSnapshot


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_event_dir(event_id: str) -> str:
    # Minimal sanitization for filesystem paths
    return event_id.replace("/", "_").replace(":", "_")


class FileSnapshotStore:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, event_id: str, as_of: datetime) -> Path:
        as_of = _utc(as_of)
        event_dir = self.root_dir / _safe_event_dir(event_id)
        event_dir.mkdir(parents=True, exist_ok=True)
        ts = as_of.strftime("%Y%m%dT%H%M%SZ")
        return event_dir / f"{ts}.json"

    def put(self, snap: MarketSnapshot) -> Path:
        p = self._path(snap.event_id, snap.as_of)
        p.write_text(snap.model_dump_json(indent=2), encoding="utf-8")
        return p

    def get(self, event_id: str, as_of: datetime) -> Optional[MarketSnapshot]:
        p = self._path(event_id, as_of)
        if not p.exists():
            return None
        return MarketSnapshot.model_validate_json(p.read_text(encoding="utf-8"))

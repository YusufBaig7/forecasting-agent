"""
Sports odds feed using SportsRadar and OpticOdds APIs.

Binary market definition:
  YES = home team wins (moneyline)
  NO  = home team does not win

Supports:
- list_events(as_of) for upcoming games (next 7 days)
- list_events_between(start_date, end_date) for historical iteration
- get_snapshot(event_id, as_of) for:
    * upcoming games (current odds)
    * historical games (historical odds proxy, within OpticOdds retention window)
- get_outcome(event_id) from SportsRadar summaries

Notes:
- OpticOdds /api/v3/fixtures/odds/historical is retained on a rolling ~2-month basis. Older games
  require your own stored snapshots (or a higher-tier data plan).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Any
from urllib.parse import urljoin
import os
import re

import httpx

from forecasting.feeds.base import Feed
from forecasting.models import Event, MarketSnapshot, ResolvedOutcome


# ------------------ utils ------------------

def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso_z(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _norm_team(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _vig_free_probability(p_home: float, p_away: float) -> float:
    total = p_home + p_away
    if total <= 0:
        return 0.5
    return p_home / total


def _american_to_prob(price: float) -> float:
    # price is American odds, e.g. -150 or +120
    p = float(price)
    if p == 0:
        return 0.5
    if p < 0:
        return (-p) / ((-p) + 100.0)
    return 100.0 / (p + 100.0)


def _to_date(x: Any) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return _utc(x).date()
    if isinstance(x, str):
        return date.fromisoformat(x)
    raise TypeError(f"Unsupported date type: {type(x)}")


LEAGUE_ENDPOINTS = {
    "nba": {
        "schedule": "/nba/trial/v8/en/games/{year}/{month:02d}/{day:02d}/schedule.json",
        "game_summary": "/nba/trial/v8/en/games/{game_id}/summary.json",
        "optic_sport": "basketball",
        "optic_league": "nba",
    },
    # extend later: nfl, nhl, mlb...
}


@dataclass(frozen=True)
class _EventMeta:
    home: str
    away: str
    start: datetime  # UTC


@dataclass(frozen=True)
class _FixtureMatch:
    fixture_id: str
    fixture_home: str
    fixture_away: str
    swapped: bool


class SportsOddsFeed(Feed):
    def __init__(
        self,
        sportsradar_api_key: str,
        sportsradar_base_url: str = "https://api.sportradar.com",
        opticodds_api_key: Optional[str] = None,
        opticodds_base_url: str = "https://api.opticodds.com",
        league: str = "nba",
        timeout: float = 15.0,
    ):
        if league not in LEAGUE_ENDPOINTS:
            raise ValueError(f"Unsupported league: {league}. Supported: {list(LEAGUE_ENDPOINTS.keys())}")

        self.sportsradar_api_key = sportsradar_api_key
        self.sportsradar_base_url = sportsradar_base_url.rstrip("/")
        self.opticodds_api_key = opticodds_api_key
        self.opticodds_base_url = opticodds_base_url.rstrip("/")
        self.league = league
        self.timeout = timeout
        self.endpoints = LEAGUE_ENDPOINTS[league]

        self.client = httpx.Client(timeout=timeout)

        # ---- caches ----
        self._event_meta: dict[str, _EventMeta] = {}                 # event_id -> home/away/start
        self._fixture_match_cache: dict[str, _FixtureMatch] = {}     # event_id -> matched Optic fixture
        self._fixtures_cache: dict[str, tuple[datetime, list[dict]]] = {}  # cache_key -> (fetched_at, fixtures)
        self._sportsbooks_cache: Optional[tuple[datetime, list[str]]] = None  # (fetched_at, sorted book_ids)

    # ------------------ API helpers ------------------

    def _fetch_sportsradar(self, path: str, params: Optional[dict] = None) -> dict:
        url = urljoin(self.sportsradar_base_url + "/", path.lstrip("/"))
        params = dict(params or {})
        params["api_key"] = self.sportsradar_api_key
        r = self.client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def _fetch_opticodds(self, path: str, params: Optional[dict] = None) -> dict:
        if not self.opticodds_api_key:
            raise RuntimeError("OpticOdds API key not provided")
        url = urljoin(self.opticodds_base_url + "/", path.lstrip("/"))
        params = dict(params or {})
        r = self.client.get(url, params=params, headers={"X-Api-Key": self.opticodds_api_key})
        r.raise_for_status()
        return r.json()

    def _get_sportsbooks_cached(self, now: datetime, ttl_seconds: int = 3600) -> list[str]:
        now = _utc(now)
        if self._sportsbooks_cache is not None:
            fetched_at, books = self._sportsbooks_cache
            if (now - fetched_at).total_seconds() < ttl_seconds:
                return books

        payload = self._fetch_opticodds(
            "/api/v3/sportsbooks/active",
            params={"league": self.endpoints["optic_league"]},
        )
        raw = payload.get("data", []) or []
        # Prefer onshore books first; then fill.
        onshore = [b for b in raw if b.get("is_active") and b.get("is_onshore")]
        offshore = [b for b in raw if b.get("is_active") and not b.get("is_onshore")]
        book_ids = [b["id"] for b in (onshore + offshore) if b.get("id")]
        self._sportsbooks_cache = (now, book_ids)
        return book_ids

    def _fixtures_window_cached(
        self,
        start: datetime,
        window_hours: int = 18,
        ttl_seconds: int = 3600,
    ) -> list[dict]:
        """
        Fetch OpticOdds fixtures near a given start time (cached).
        Uses /api/v3/fixtures (works for historical lookups).
        """
        start = _utc(start)
        now = _utc(datetime.now(timezone.utc))

        start_after = (start - timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        start_before = (start + timedelta(hours=window_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        cache_key = f"{self.endpoints['optic_sport']}|{self.endpoints['optic_league']}|{start_after}|{start_before}"

        if cache_key in self._fixtures_cache:
            fetched_at, fixtures = self._fixtures_cache[cache_key]
            if (now - fetched_at).total_seconds() < ttl_seconds:
                return fixtures

        payload = self._fetch_opticodds(
            "/api/v3/fixtures",
            params={
                "sport": self.endpoints["optic_sport"],
                "league": self.endpoints["optic_league"],
                "start_date_after": start_after,
                "start_date_before": start_before,
            },
        )
        fixtures = payload.get("data", []) or []
        self._fixtures_cache[cache_key] = (now, fixtures)
        return fixtures

    def _match_fixture_for_event(self, event_id: str) -> Optional[_FixtureMatch]:
        if event_id in self._fixture_match_cache:
            return self._fixture_match_cache[event_id]

        meta = self._event_meta.get(event_id)
        if meta is None:
            return None

        nh, na = _norm_team(meta.home), _norm_team(meta.away)
        start = _utc(meta.start)

        fixtures = self._fixtures_window_cached(start)

        best: Optional[_FixtureMatch] = None
        for fx in fixtures:
            fx_home = fx.get("home_team_display", "") or ""
            fx_away = fx.get("away_team_display", "") or ""
            fx_start_s = fx.get("start_date")
            if not fx_start_s:
                continue

            try:
                fx_start = _parse_iso_z(fx_start_s)
            except Exception:
                continue

            # Require time proximity.
            if abs((fx_start - start).total_seconds()) > 4 * 3600:
                continue

            nfxh, nfxa = _norm_team(fx_home), _norm_team(fx_away)

            # Direct match
            if nfxh == nh and nfxa == na:
                best = _FixtureMatch(
                    fixture_id=str(fx["id"]),
                    fixture_home=fx_home,
                    fixture_away=fx_away,
                    swapped=False,
                )
                break

            # Swapped match (data-source home/away mismatch)
            if nfxh == na and nfxa == nh:
                best = _FixtureMatch(
                    fixture_id=str(fx["id"]),
                    fixture_home=fx_home,
                    fixture_away=fx_away,
                    swapped=True,
                )
                break

        if best is not None:
            self._fixture_match_cache[event_id] = best
        return best

    # ------------------ Feed interface ------------------

    def list_events(self, as_of: datetime) -> list[Event]:
        """Upcoming games for the next 7 days."""
        as_of = _utc(as_of)
        return self.list_events_between(as_of.date(), as_of.date() + timedelta(days=6), as_of=as_of)

    def list_events_for_date(self, d: date | str, as_of: Optional[datetime] = None) -> list[Event]:
        d0 = _to_date(d)
        return self.list_events_between(d0, d0, as_of=as_of)

    def list_events_between(
        self,
        start_date: date | str,
        end_date: date | str,
        as_of: Optional[datetime] = None,
    ) -> list[Event]:
        """
        Historical iterator: inclusive [start_date, end_date].
        If as_of is provided, filters to games strictly after as_of (useful for “upcoming only”).
        """
        start_d = _to_date(start_date)
        end_d = _to_date(end_date)
        if end_d < start_d:
            return []

        as_of_dt = _utc(as_of) if as_of else None

        out: list[Event] = []
        cur = start_d
        while cur <= end_d:
            schedule_path = self.endpoints["schedule"].format(year=cur.year, month=cur.month, day=cur.day)
            try:
                data = self._fetch_sportsradar(schedule_path)
            except Exception:
                cur = cur + timedelta(days=1)
                continue

            games = data.get("games") or data.get("schedule", {}).get("games") or []
            for game in games:
                game_id = game.get("id") or game.get("game_id")
                if not game_id:
                    continue

                home = (game.get("home") or {}).get("name") or (game.get("home_team") or {}).get("name")
                away = (game.get("away") or {}).get("name") or (game.get("away_team") or {}).get("name")
                if not home or not away:
                    continue

                scheduled_str = game.get("scheduled") or game.get("scheduled_time")
                if isinstance(scheduled_str, str) and scheduled_str:
                    try:
                        scheduled = datetime.fromisoformat(scheduled_str.replace("Z", "+00:00"))
                    except ValueError:
                        scheduled = datetime.combine(cur, datetime.min.time(), tzinfo=timezone.utc).replace(hour=12)
                else:
                    scheduled = datetime.combine(cur, datetime.min.time(), tzinfo=timezone.utc).replace(hour=12)

                scheduled = _utc(scheduled)

                # Optional filter to “upcoming only”
                if as_of_dt is not None and scheduled <= as_of_dt:
                    continue

                event_id = f"sports:{self.league}:{game_id}"
                self._event_meta[event_id] = _EventMeta(home=home, away=away, start=scheduled)

                out.append(
                    Event(
                        event_id=event_id,
                        title=f"{away} @ {home} (home moneyline)",
                        question=f"Will {home} win vs {away}?",
                        close_time=scheduled,
                        resolution_criteria="YES if home team wins (moneyline).",
                        source="sportsradar",
                    )
                )

            cur = cur + timedelta(days=1)

        return out

    # ------------------ snapshots ------------------

    def _resolve_book_list(self, now: datetime) -> list[str]:
        # If user pins a book, try it first.
        pinned = os.getenv("OPTICODDS_SPORTSBOOK")
        if pinned:
            # allow comma-separated
            pinned_list = [p.strip() for p in pinned.split(",") if p.strip()]
        else:
            pinned_list = []

        # then fall back to active list (prefer onshore first)
        fallback = self._get_sportsbooks_cached(now)
        # keep order, remove duplicates
        seen = set()
        out = []
        for b in pinned_list + fallback:
            if b and b not in seen:
                out.append(b)
                seen.add(b)
        return out

    def _snapshot_from_current_odds(
        self,
        event_id: str,
        as_of: datetime,
        fixture: _FixtureMatch,
        meta: _EventMeta,
    ) -> Optional[MarketSnapshot]:
        """
        Uses /api/v3/fixtures/odds (best for upcoming games).
        """
        nh, na = _norm_team(meta.home), _norm_team(meta.away)
        start = _utc(meta.start)

        books = self._resolve_book_list(as_of)

        for sportsbook in books[:25]:
            payload = self._fetch_opticodds(
                "/api/v3/fixtures/odds",
                params={
                    "fixture_id": fixture.fixture_id,
                    "sportsbook": sportsbook,
                    "market": "moneyline",
                    "odds_format": "PROBABILITY",
                },
            )
            data = payload.get("data", []) or []
            odds = (data[0].get("odds") if data else None) or []
            if not odds:
                continue

            p_home = None
            p_away = None

            for o in odds:
                mid = (o.get("market_id") or o.get("market") or "").lower()
                if mid and "moneyline" not in mid:
                    continue

                name = o.get("name") or o.get("selection") or o.get("competitor") or ""
                n = _norm_team(str(name))

                if "probability" in o and o.get("probability") is not None:
                    prob = float(o["probability"])
                elif "price" in o and o.get("price") is not None:
                    # if odds_format isn't honored, fall back to American odds -> prob
                    prob = _american_to_prob(float(o["price"]))
                else:
                    continue

                if n == nh:
                    p_home = prob
                elif n == na:
                    p_away = prob

            if p_home is None or p_away is None:
                continue

            q_home = _vig_free_probability(p_home, p_away)
            return MarketSnapshot(
                event_id=event_id,
                as_of=as_of,
                market_prob=q_home,
                raw={
                    "provider": "opticodds",
                    "mode": "current",
                    "fixture_id": fixture.fixture_id,
                    "sportsbook": sportsbook,
                    "event_home": meta.home,
                    "event_away": meta.away,
                    "fixture_home": fixture.fixture_home,
                    "fixture_away": fixture.fixture_away,
                    "swapped": fixture.swapped,
                    "start": start.isoformat(),
                    "p_home": p_home,
                    "p_away": p_away,
                },
            )

        return None

    def _pick_historical_line_prob(
        self,
        as_of: datetime,
        start: datetime,
        odds_obj: dict,
    ) -> Optional[float]:
        """
        Turn one historical odds object into an implied probability at as_of.
        Handles:
        - entries[] timeseries (if present)
        - otherwise uses olv/clv and simple blend by time-to-start
        """
        # 1) timeseries
        entries = odds_obj.get("entries") or []
        if entries:
            best_price = None
            best_ts = None
            for e in entries:
                ts = e.get("timestamp")
                price = e.get("price")
                if not ts or price is None:
                    continue
                try:
                    dt = _parse_iso_z(ts) if isinstance(ts, str) else None
                except Exception:
                    continue
                if dt is None or dt > as_of:
                    continue
                if best_ts is None or dt > best_ts:
                    best_ts = dt
                    best_price = price
            if best_price is not None:
                return _american_to_prob(float(best_price))

        # 2) olv/clv blend
        olv = odds_obj.get("olv") or {}
        clv = odds_obj.get("clv") or {}
        olv_price = olv.get("price")
        clv_price = clv.get("price")
        if olv_price is None and clv_price is None:
            return None

        # If only one exists, use it.
        if olv_price is None:
            return _american_to_prob(float(clv_price))
        if clv_price is None:
            return _american_to_prob(float(olv_price))

        p_open = _american_to_prob(float(olv_price))
        p_close = _american_to_prob(float(clv_price))

        # Blend: far out -> open, close to kickoff -> close, middle -> interpolate
        secs_to_start = (start - as_of).total_seconds()
        if secs_to_start >= 12 * 3600:
            return p_open
        if secs_to_start <= 2 * 3600:
            return p_close

        # linear time blend between 12h and 2h before start
        t = (12 * 3600 - secs_to_start) / (10 * 3600)  # 0..1
        t = max(0.0, min(1.0, t))
        return (1.0 - t) * p_open + t * p_close

    def _snapshot_from_historical_odds(
        self,
        event_id: str,
        as_of: datetime,
        fixture: _FixtureMatch,
        meta: _EventMeta,
    ) -> Optional[MarketSnapshot]:
        """
        Uses /api/v3/fixtures/odds/historical (best for backtests).
        """
        nh, na = _norm_team(meta.home), _norm_team(meta.away)
        start = _utc(meta.start)

        books = self._resolve_book_list(as_of)

        for sportsbook in books[:10]:
            payload = self._fetch_opticodds(
                "/api/v3/fixtures/odds/historical",
                params={
                    "fixture_id": fixture.fixture_id,
                    "sportsbook": sportsbook,
                    "market": "moneyline",
                    "is_main": "true",
                },
            )
            data = payload.get("data", []) or []
            if not data:
                continue

            row = data[0]
            odds = row.get("odds") or []
            if not odds:
                continue

            # Extract selection probs for event-home and event-away.
            p_home = None
            p_away = None

            # Prefer matching by text name; fallback to team_id if present.
            # (Moneyline typically has one entry per team.)
            for o in odds:
                mid = (o.get("market_id") or o.get("market") or "").lower()
                if mid and "moneyline" not in mid:
                    continue

                name = o.get("name") or o.get("selection") or o.get("normalized_selection") or ""
                n = _norm_team(str(name))

                prob = self._pick_historical_line_prob(as_of, start, o)
                if prob is None:
                    continue

                if n == nh:
                    p_home = prob
                elif n == na:
                    p_away = prob

            if p_home is None or p_away is None:
                # Some responses might not carry team names in odds entries; try rough matching
                # using fixture_home/fixture_away names.
                fxh = _norm_team(fixture.fixture_home)
                fxa = _norm_team(fixture.fixture_away)

                for o in odds:
                    mid = (o.get("market_id") or o.get("market") or "").lower()
                    if mid and "moneyline" not in mid:
                        continue
                    name = o.get("name") or o.get("selection") or o.get("normalized_selection") or ""
                    n = _norm_team(str(name))
                    prob = self._pick_historical_line_prob(as_of, start, o)
                    if prob is None:
                        continue

                    if n == fxh:
                        # fixture home prob
                        if fixture.swapped:
                            p_away = prob  # event away == fixture home
                        else:
                            p_home = prob
                    elif n == fxa:
                        if fixture.swapped:
                            p_home = prob
                        else:
                            p_away = prob

            if p_home is None or p_away is None:
                continue

            q_home = _vig_free_probability(p_home, p_away)

            return MarketSnapshot(
                event_id=event_id,
                as_of=as_of,
                market_prob=q_home,
                raw={
                    "provider": "opticodds",
                    "mode": "historical",
                    "fixture_id": fixture.fixture_id,
                    "sportsbook": sportsbook,
                    "event_home": meta.home,
                    "event_away": meta.away,
                    "fixture_home": fixture.fixture_home,
                    "fixture_away": fixture.fixture_away,
                    "swapped": fixture.swapped,
                    "start": start.isoformat(),
                    "p_home": p_home,
                    "p_away": p_away,
                },
            )

        return None

    def get_snapshot(self, event_id: str, as_of: datetime) -> Optional[MarketSnapshot]:
        as_of = _utc(as_of)

        meta = self._event_meta.get(event_id)
        if meta is None:
            return None

        start = _utc(meta.start)

        # No live/in-play snapshots for this binary pregame market.
        if start <= as_of:
            return None

        fixture = self._match_fixture_for_event(event_id)
        if fixture is None:
            return None

        now = _utc(datetime.now(timezone.utc))
        game_already_started_in_real_life = start <= now

        # If the game is already in the past "today", we must use historical odds.
        if game_already_started_in_real_life:
            return self._snapshot_from_historical_odds(event_id, as_of, fixture, meta)

        # Upcoming: use current odds endpoint.
        return self._snapshot_from_current_odds(event_id, as_of, fixture, meta)

    # ------------------ outcomes ------------------

    def get_outcome(self, event_id: str) -> Optional[ResolvedOutcome]:
        if not event_id.startswith(f"sports:{self.league}:"):
            return None

        game_id = event_id.split(":", 2)[2]
        game_summary_path = self.endpoints["game_summary"].format(game_id=game_id)

        try:
            data = self._fetch_sportsradar(game_summary_path)
        except Exception:
            return None

        status = data.get("status") or data.get("game", {}).get("status")
        if not status:
            return None
        if status.lower() not in {"closed", "complete", "final", "finished"}:
            return None

        home_score = away_score = None
        if "home" in data and "away" in data:
            home_score = data["home"].get("points") or data["home"].get("score")
            away_score = data["away"].get("points") or data["away"].get("score")

        if home_score is None or away_score is None:
            return None

        home_score = int(home_score)
        away_score = int(away_score)
        if home_score == away_score:
            return None  # push/tie

        outcome = 1 if home_score > away_score else 0
        return ResolvedOutcome(event_id=event_id, outcome=outcome)

    def __del__(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

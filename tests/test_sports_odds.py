"""Tests for sports odds feed with mocked HTTP responses."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from forecasting.feeds.sports_odds import SportsOddsFeed, _vig_free_probability
from forecasting.models import Event, MarketSnapshot, ResolvedOutcome


class TestVigFreeProbability:
    """Tests for vig-free probability calculation."""

    def test_basic_calculation(self):
        """Test basic vig-free calculation."""
        # Home odds: 2.0 (50% implied), Away odds: 2.0 (50% implied)
        # Vig-free: 50% / (50% + 50%) = 50%
        prob = _vig_free_probability(0.5, 0.5)
        assert abs(prob - 0.5) < 0.01

    def test_favorite_underdog(self):
        """Test with favorite and underdog."""
        # Home odds: 1.5 (66.7% implied), Away odds: 3.0 (33.3% implied)
        # Total: 100%, so no vig adjustment needed, but test the function
        implied_home = 1.0 / 1.5
        implied_away = 1.0 / 3.0
        prob = _vig_free_probability(implied_home, implied_away)
        assert prob > 0.5  # Home is favorite
        assert abs(prob - (implied_home / (implied_home + implied_away))) < 0.01

    def test_with_vig(self):
        """Test with vig (total implied > 1.0)."""
        # Home: 1.8 (55.6%), Away: 2.2 (45.5%) -> Total: 101.1% (1.1% vig)
        implied_home = 1.0 / 1.8
        implied_away = 1.0 / 2.2
        prob = _vig_free_probability(implied_home, implied_away)
        # Should normalize to sum to 1.0
        assert 0 < prob < 1


class TestSportsOddsFeed:
    """Tests for SportsOddsFeed with mocked HTTP responses."""

    @pytest.fixture
    def mock_feed(self):
        """Create a SportsOddsFeed instance for testing."""
        return SportsOddsFeed(
            sportsradar_api_key="test_key",
            opticodds_api_key="test_odds_key",
            league="nba",
        )

    def test_list_events_success(self, mock_feed):
        """Test successful event listing from mocked schedule."""
        as_of = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        # Mock schedule response
        mock_schedule = {
            "games": [
                {
                    "id": "abc123",
                    "home": {"name": "Lakers"},
                    "away": {"name": "Warriors"},
                    "scheduled": "2024-01-16T19:00:00Z",
                },
                {
                    "id": "def456",
                    "home": {"name": "Celtics"},
                    "away": {"name": "Heat"},
                    "scheduled": "2024-01-17T20:00:00Z",
                },
            ]
        }
        
        with patch.object(mock_feed.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_schedule
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            events = mock_feed.list_events(as_of)
            
            assert len(events) == 2
            assert events[0].event_id == "sports:nba:abc123"
            assert "Lakers" in events[0].title
            assert "Warriors" in events[0].title
            assert events[0].question == "Will Lakers win vs Warriors?"
            assert events[0].close_time.tzinfo == timezone.utc

    def test_get_snapshot_with_odds(self, mock_feed):
        """Test getting snapshot with moneyline odds."""
        event_id = "sports:nba:abc123"
        as_of = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        # Mock OpticOdds response
        mock_odds = {
            "markets": [
                {
                    "type": "moneyline",
                    "outcomes": [
                        {"side": "home", "decimal": 1.8},
                        {"side": "away", "decimal": 2.2},
                    ],
                }
            ]
        }
        
        with patch.object(mock_feed.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_odds
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            snapshot = mock_feed.get_snapshot(event_id, as_of)
            
            assert snapshot is not None
            assert snapshot.event_id == event_id
            assert snapshot.as_of == as_of
            assert 0 < snapshot.market_prob < 1
            # Home odds 1.8 -> implied 0.556, Away 2.2 -> implied 0.455
            # Vig-free: 0.556 / (0.556 + 0.455) ≈ 0.55
            assert snapshot.market_prob > 0.5  # Home is favorite
            assert "opticodds" in snapshot.raw["provider"]

    def test_get_snapshot_no_odds(self, mock_feed):
        """Test getting snapshot when odds are unavailable."""
        event_id = "sports:nba:abc123"
        as_of = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        # Mock empty response
        mock_odds = {"markets": []}
        
        with patch.object(mock_feed.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_odds
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            snapshot = mock_feed.get_snapshot(event_id, as_of)
            
            assert snapshot is None

    def test_get_outcome_final_home_win(self, mock_feed):
        """Test getting outcome when home team wins."""
        event_id = "sports:nba:abc123"
        
        # Mock game summary with home win
        mock_summary = {
            "status": "closed",
            "home": {"points": 110},
            "away": {"points": 95},
        }
        
        with patch.object(mock_feed.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_summary
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            outcome = mock_feed.get_outcome(event_id)
            
            assert outcome is not None
            assert outcome.event_id == event_id
            assert outcome.outcome == 1  # Home wins

    def test_get_outcome_final_away_win(self, mock_feed):
        """Test getting outcome when away team wins."""
        event_id = "sports:nba:abc123"
        
        # Mock game summary with away win
        mock_summary = {
            "status": "closed",
            "home": {"points": 95},
            "away": {"points": 110},
        }
        
        with patch.object(mock_feed.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_summary
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            outcome = mock_feed.get_outcome(event_id)
            
            assert outcome is not None
            assert outcome.outcome == 0  # Away wins (home loses)

    def test_get_outcome_tie(self, mock_feed):
        """Test that tie returns None (push)."""
        event_id = "sports:nba:abc123"
        
        # Mock game summary with tie
        mock_summary = {
            "status": "closed",
            "home": {"points": 100},
            "away": {"points": 100},
        }
        
        with patch.object(mock_feed.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_summary
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            outcome = mock_feed.get_outcome(event_id)
            
            assert outcome is None  # Tie/push

    def test_get_outcome_not_final(self, mock_feed):
        """Test that non-final games return None."""
        event_id = "sports:nba:abc123"
        
        # Mock game summary with in-progress status
        mock_summary = {
            "status": "inprogress",
            "home": {"points": 50},
            "away": {"points": 45},
        }
        
        with patch.object(mock_feed.client, "get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_summary
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            outcome = mock_feed.get_outcome(event_id)
            
            assert outcome is None  # Not final

    def test_invalid_event_id(self, mock_feed):
        """Test that invalid event IDs return None."""
        as_of = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        # Invalid event ID format
        snapshot = mock_feed.get_snapshot("invalid:format", as_of)
        assert snapshot is None
        
        outcome = mock_feed.get_outcome("invalid:format")
        assert outcome is None

    def test_wrong_league(self, mock_feed):
        """Test that event IDs from wrong league return None."""
        as_of = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        # Event ID for different league
        snapshot = mock_feed.get_snapshot("sports:nfl:abc123", as_of)
        assert snapshot is None

    def test_http_error_handling(self, mock_feed):
        """Test that HTTP errors are handled gracefully."""
        as_of = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        with patch.object(mock_feed.client, "get") as mock_get:
            import httpx
            mock_get.side_effect = httpx.HTTPError("Network error")
            
            # Should return None or raise RuntimeError
            with pytest.raises(RuntimeError, match="SportsRadar API error"):
                mock_feed.list_events(as_of)


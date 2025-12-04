"""Tests for backtest caching integration."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from forecasting.eval.backtest import run_backtest
from forecasting.feeds.stub import StubFeed
from forecasting.forecast.baseline_market import MarketBaselineForecaster
from forecasting.storage.forecast_store import FileForecastStore
from forecasting.storage.snapshot_store import FileSnapshotStore
from tests.test_fake_feed import FakeFeed
from forecasting.models import Event, MarketSnapshot


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBacktestCaching:
    """Tests for snapshot caching and forecast logging in backtests."""

    def test_snapshot_caching(self, temp_dir):
        """Test that snapshots are cached during backtest."""
        feed = StubFeed(n_events=5)
        forecaster = MarketBaselineForecaster()
        snapshot_store = FileSnapshotStore(temp_dir / "snapshots")

        now = datetime.now(timezone.utc)
        as_ofs = [now - timedelta(hours=12 * i) for i in range(2)][::-1]

        # Run backtest with caching
        result1 = run_backtest(
            feed=feed,
            forecaster=forecaster,
            as_of_times=as_ofs,
            snapshot_store=snapshot_store,
        )

        # Verify snapshots were cached
        events = feed.list_events(as_ofs[0])
        for ev in events[:2]:  # Check first 2 events
            for as_of in as_ofs:
                cached = snapshot_store.get(ev.event_id, as_of)
                assert cached is not None, f"Snapshot not cached for {ev.event_id} at {as_of}"

    def test_forecast_logging(self, temp_dir):
        """Test that forecasts are logged during backtest."""
        feed = StubFeed(n_events=5)
        forecaster = MarketBaselineForecaster()
        forecast_store = FileForecastStore(temp_dir / "forecasts")

        now = datetime.now(timezone.utc)
        as_ofs = [now - timedelta(hours=12 * i) for i in range(2)][::-1]

        # Run backtest with logging
        result = run_backtest(
            feed=feed,
            forecaster=forecaster,
            as_of_times=as_ofs,
            forecast_store=forecast_store,
        )

        # Verify forecasts were logged to JSONL file
        assert forecast_store._file_path is not None
        assert forecast_store._file_path.exists()
        
        metadata, forecasts = forecast_store.read_run(forecast_store._file_path)
        assert metadata["run_id"] == forecast_store.run_id
        assert metadata["model_name"] == forecaster.model_name
        assert len(forecasts) > 0
        
        # Verify forecast content
        for fc in forecasts:
            assert fc.p_yes >= 0.0
            assert fc.p_yes <= 1.0
            assert fc.model == forecaster.model_name

    def test_cache_hit_optimization(self, temp_dir):
        """Test that cached snapshots are used instead of fetching from feed."""
        feed = StubFeed(n_events=3)
        forecaster = MarketBaselineForecaster()
        snapshot_store = FileSnapshotStore(temp_dir / "snapshots")

        now = datetime.now(timezone.utc)
        as_ofs = [now - timedelta(hours=12 * i) for i in range(2)][::-1]

        # Pre-populate cache
        events = feed.list_events(as_ofs[0])
        for ev in events[:2]:
            for as_of in as_ofs:
                snap = feed.get_snapshot(ev.event_id, as_of)
                if snap:
                    snapshot_store.put(snap)

        # Run backtest - should use cached snapshots
        result = run_backtest(
            feed=feed,
            forecaster=forecaster,
            as_of_times=as_ofs,
            snapshot_store=snapshot_store,
            limit_events=2,
        )

        # Should still produce valid results
        assert len(result.predictions) > 0
        assert result.brier >= 0.0
        assert result.brier <= 1.0

    def test_snapshot_caching_reduces_feed_calls(self, temp_dir):
        """Test that snapshot caching reduces calls to feed.get_snapshot."""
        # Create events and snapshots
        events = [
            Event(
                event_id=f"test:event:{i}",
                title=f"Event {i}",
                question=f"Will event {i} happen?",
                close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
                source="test",
            )
            for i in range(3)
        ]
        
        as_of1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        as_of2 = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        
        snapshots = {}
        for ev in events:
            for as_of in [as_of1, as_of2]:
                snapshots[(ev.event_id, as_of)] = MarketSnapshot(
                    event_id=ev.event_id,
                    as_of=as_of,
                    market_prob=0.5,
                    liquidity=1000.0,
                )
        
        fake_feed = FakeFeed(events, snapshots)
        forecaster = MarketBaselineForecaster()
        snapshot_store = FileSnapshotStore(temp_dir / "snapshots")
        
        # First run: should call feed.get_snapshot for all
        result1 = run_backtest(
            feed=fake_feed,
            forecaster=forecaster,
            as_of_times=[as_of1, as_of2],
            snapshot_store=snapshot_store,
        )
        
        calls_first_run = fake_feed.get_call_count("get_snapshot")
        assert calls_first_run == 6  # 3 events * 2 timestamps
        
        # Reset counts
        fake_feed.reset_counts()
        
        # Second run: should use cache, fewer calls
        result2 = run_backtest(
            feed=fake_feed,
            forecaster=forecaster,
            as_of_times=[as_of1, as_of2],
            snapshot_store=snapshot_store,
        )
        
        calls_second_run = fake_feed.get_call_count("get_snapshot")
        # Should be 0 if all snapshots are cached
        assert calls_second_run == 0, "Expected 0 calls when all snapshots are cached"
        
        # Results should be the same
        assert len(result1.predictions) == len(result2.predictions)

    def test_backtest_without_stores(self, temp_dir):
        """Test that backtest works without stores (backward compatibility)."""
        feed = StubFeed(n_events=5)
        forecaster = MarketBaselineForecaster()

        now = datetime.now(timezone.utc)
        as_ofs = [now - timedelta(hours=12 * i) for i in range(2)][::-1]

        # Run backtest without stores
        result = run_backtest(
            feed=feed,
            forecaster=forecaster,
            as_of_times=as_ofs,
        )

        assert len(result.predictions) > 0
        assert result.brier >= 0.0
        assert result.brier <= 1.0
        assert result.ece >= 0.0

    def test_forecast_logging_always_writes(self, temp_dir):
        """Test that forecasts are always logged when store is provided."""
        feed = StubFeed(n_events=3)
        forecaster = MarketBaselineForecaster()
        forecast_store = FileForecastStore(temp_dir / "forecasts")

        now = datetime.now(timezone.utc)
        as_ofs = [now - timedelta(hours=12 * i) for i in range(2)][::-1]

        # Run backtest
        result = run_backtest(
            feed=feed,
            forecaster=forecaster,
            as_of_times=as_ofs,
            forecast_store=forecast_store,
        )

        # Verify all forecasts were logged
        metadata, forecasts = forecast_store.read_run(forecast_store._file_path)
        assert len(forecasts) == len(result.predictions)
        
        # Each forecast should have been logged
        for fc in forecasts:
            assert fc.model == forecaster.model_name
            assert 0.0 <= fc.p_yes <= 1.0


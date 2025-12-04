"""Tests for storage modules."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from forecasting.models import Forecast, MarketSnapshot
from forecasting.storage.forecast_store import FileForecastStore
from forecasting.storage.snapshot_store import FileSnapshotStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestFileSnapshotStore:
    """Tests for FileSnapshotStore."""

    def test_put_and_get(self, temp_dir):
        """Test storing and retrieving snapshots."""
        store = FileSnapshotStore(temp_dir / "snapshots")
        event_id = "test:event:001"
        as_of = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        snapshot = MarketSnapshot(
            event_id=event_id,
            as_of=as_of,
            market_prob=0.75,
            liquidity=1000.0,
            raw={"test": "data"},
        )

        # Store
        path = store.put(snapshot)
        assert path.exists()

        # Retrieve
        retrieved = store.get(event_id, as_of)
        assert retrieved is not None
        assert retrieved.event_id == event_id
        assert retrieved.market_prob == 0.75
        assert retrieved.liquidity == 1000.0
        assert retrieved.raw == {"test": "data"}

    def test_get_nonexistent(self, temp_dir):
        """Test retrieving non-existent snapshot returns None."""
        store = FileSnapshotStore(temp_dir / "snapshots")
        event_id = "test:event:999"
        as_of = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        result = store.get(event_id, as_of)
        assert result is None

    def test_utc_normalization(self, temp_dir):
        """Test that naive datetimes are normalized to UTC."""
        store = FileSnapshotStore(temp_dir / "snapshots")
        event_id = "test:event:002"
        # Naive datetime
        as_of = datetime(2024, 1, 1, 12, 0, 0)

        snapshot = MarketSnapshot(
            event_id=event_id,
            as_of=as_of,
            market_prob=0.5,
        )

        store.put(snapshot)
        retrieved = store.get(event_id, as_of.replace(tzinfo=timezone.utc))
        assert retrieved is not None
        assert retrieved.as_of.tzinfo == timezone.utc


class TestFileForecastStore:
    """Tests for FileForecastStore with JSONL format."""

    def test_start_run_and_log_forecast(self, temp_dir):
        """Test starting a run and logging forecasts."""
        store = FileForecastStore(temp_dir / "forecasts", run_id="test-run-123")
        model = "test_model/v1"
        
        # Start run
        file_path = store.start_run(model)
        assert file_path.exists()
        assert file_path.suffix == ".jsonl"
        
        # Log forecasts
        forecasts = [
            Forecast(
                event_id=f"test:event:{i}",
                as_of=datetime(2024, 1, 1, 12, i, 0, tzinfo=timezone.utc),
                p_yes=0.5 + i * 0.1,
                model=model,
                rationale=f"Forecast {i}",
            )
            for i in range(3)
        ]
        
        for fc in forecasts:
            store.log_forecast(fc)
        
        # Read back the run
        metadata, read_forecasts = store.read_run(file_path)
        
        assert metadata["run_id"] == "test-run-123"
        assert metadata["model_name"] == model
        assert "timestamp" in metadata
        assert len(read_forecasts) == 3
        assert read_forecasts[0].event_id == "test:event:0"
        assert read_forecasts[0].p_yes == 0.5

    def test_jsonl_parseable(self, temp_dir):
        """Test that JSONL file is properly formatted and parseable."""
        store = FileForecastStore(temp_dir / "forecasts")
        model = "test_model/v1"
        
        file_path = store.start_run(model)
        
        forecast = Forecast(
            event_id="test:event:001",
            as_of=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            p_yes=0.75,
            model=model,
            metadata={"key": "value"},
        )
        store.log_forecast(forecast)
        
        # Read file manually to verify JSONL format
        lines = file_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2  # metadata + 1 forecast
        
        import json
        metadata = json.loads(lines[0])
        forecast_dict = json.loads(lines[1])
        
        assert metadata["run_id"] == store.run_id
        assert metadata["model_name"] == model
        assert forecast_dict["event_id"] == "test:event:001"
        assert forecast_dict["p_yes"] == 0.75

    def test_run_id_and_git_hash(self, temp_dir):
        """Test that run_id and git_hash are included in metadata."""
        custom_run_id = "custom-run-456"
        store = FileForecastStore(temp_dir / "forecasts", run_id=custom_run_id)
        model = "test_model/v1"
        
        file_path = store.start_run(model)
        metadata, _ = store.read_run(file_path)
        
        assert metadata["run_id"] == custom_run_id
        # git_hash may be None if not in git repo, that's OK
        assert "git_hash" in metadata

    def test_list_runs(self, temp_dir):
        """Test listing forecast run files."""
        store1 = FileForecastStore(temp_dir / "forecasts", run_id="run1")
        store2 = FileForecastStore(temp_dir / "forecasts", run_id="run2")
        
        file_path1 = store1.start_run("model_a")
        file_path2 = store2.start_run("model_b")
        
        runs = store1.list_runs()
        assert len(runs) == 2
        
        # Filter by model
        runs_model_a = store1.list_runs(model_name="model_a")
        assert len(runs_model_a) == 1
        assert runs_model_a[0] == file_path1

    def test_model_name_sanitization(self, temp_dir):
        """Test that model names with special chars are sanitized in filename."""
        store = FileForecastStore(temp_dir / "forecasts")
        model = "model/with/slashes:v1"
        
        file_path = store.start_run(model)
        
        # Filename should have sanitized model name
        assert "model_with_slashes_v1" in file_path.name
        assert file_path.exists()
        
        # Should still work
        forecast = Forecast(
            event_id="test:event:001",
            as_of=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            p_yes=0.5,
            model=model,
        )
        store.log_forecast(forecast)
        
        metadata, forecasts = store.read_run(file_path)
        assert metadata["model_name"] == model  # Original name preserved in metadata
        assert forecasts[0].model == model

    def test_double_start_run_error(self, temp_dir):
        """Test that starting a run twice raises an error."""
        store = FileForecastStore(temp_dir / "forecasts")
        store.start_run("model")
        
        with pytest.raises(RuntimeError, match="already started"):
            store.start_run("model")

    def test_log_before_start_error(self, temp_dir):
        """Test that logging before starting raises an error."""
        store = FileForecastStore(temp_dir / "forecasts")
        forecast = Forecast(
            event_id="test:event:001",
            as_of=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            p_yes=0.5,
            model="test_model/v1",
        )
        
        with pytest.raises(RuntimeError, match="Must call start_run"):
            store.log_forecast(forecast)

    def test_model_mismatch_error(self, temp_dir):
        """Test that logging forecast with wrong model raises an error."""
        store = FileForecastStore(temp_dir / "forecasts")
        store.start_run("model_a")
        
        forecast = Forecast(
            event_id="test:event:001",
            as_of=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            p_yes=0.5,
            model="model_b",  # Different model
        )
        
        with pytest.raises(ValueError, match="doesn't match"):
            store.log_forecast(forecast)


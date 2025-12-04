"""Tests for LLM forecaster."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from forecasting.forecast.llm_forecaster import LLMForecaster
from forecasting.llm.mock import MockLLM
from forecasting.models import Event, MarketSnapshot
from forecasting.retrieval.stub import StubRetriever


class TestLLMForecaster:
    """Tests for LLMForecaster."""

    def test_basic_forecast(self):
        """Test basic forecast generation with mock LLM."""
        llm = MockLLM()
        forecaster = LLMForecaster(llm=llm)
        
        event = Event(
            event_id="test:001",
            title="Test Event",
            question="Will this happen?",
            close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        )
        
        as_of = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            event_id="test:001",
            as_of=as_of,
            market_prob=0.6,
        )
        
        forecast = forecaster.predict(event, snapshot, as_of)
        
        assert forecast.event_id == event.event_id
        assert forecast.as_of == as_of
        assert 0.0 <= forecast.p_yes <= 1.0
        assert forecast.model == "llm/v1"
        assert len(forecast.rationale) > 0

    def test_with_retriever(self):
        """Test forecast with retriever."""
        llm = MockLLM()
        retriever = StubRetriever(n_items=3)
        forecaster = LLMForecaster(llm=llm, retriever=retriever)
        
        event = Event(
            event_id="test:002",
            title="Test Event 2",
            question="Will this happen?",
            close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        )
        
        as_of = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            event_id="test:002",
            as_of=as_of,
            market_prob=0.5,
        )
        
        forecast = forecaster.predict(event, snapshot, as_of)
        
        assert "context_items_count" in forecast.metadata
        assert forecast.metadata["context_items_count"] == 3
        assert "context_summary" in forecast.metadata

    def test_as_of_propagation(self):
        """Test that as_of is properly propagated to retriever."""
        llm = MockLLM()
        
        # Create a retriever that tracks the as_of it was called with
        class TrackingRetriever:
            def __init__(self):
                self.last_as_of = None
                self.call_count = 0
            
            def get_context(self, event, as_of):
                self.last_as_of = as_of
                self.call_count += 1
                return StubRetriever().get_context(event, as_of)
        
        tracking_retriever = TrackingRetriever()
        forecaster = LLMForecaster(llm=llm, retriever=tracking_retriever)
        
        event = Event(
            event_id="test:003",
            title="Test Event 3",
            question="Will this happen?",
            close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        )
        
        as_of = datetime(2024, 1, 5, 15, 30, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            event_id="test:003",
            as_of=as_of,
            market_prob=0.7,
        )
        
        forecast = forecaster.predict(event, snapshot, as_of)
        
        # Verify retriever was called with correct as_of
        assert tracking_retriever.call_count == 1
        assert tracking_retriever.last_as_of == as_of
        assert forecast.as_of == as_of

    def test_as_of_mismatch_error(self):
        """Test that as_of mismatch raises error."""
        llm = MockLLM()
        forecaster = LLMForecaster(llm=llm)
        
        event = Event(
            event_id="test:004",
            title="Test Event 4",
            question="Will this happen?",
            close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        )
        
        as_of = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        snapshot_as_of = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            event_id="test:004",
            as_of=snapshot_as_of,
            market_prob=0.6,
        )
        
        with pytest.raises(ValueError, match="must match"):
            forecaster.predict(event, snapshot, as_of)

    def test_probability_clamping(self):
        """Test that probabilities are clamped to [0, 1]."""
        # Create a mock LLM that returns out-of-range values
        class BadMockLLM:
            def complete_json(self, system_prompt, user_prompt, schema):
                return {
                    "p_yes": 1.5,  # Out of range
                    "rationale": "Test",
                }
        
        llm = BadMockLLM()
        forecaster = LLMForecaster(llm=llm)
        
        event = Event(
            event_id="test:005",
            title="Test Event 5",
            question="Will this happen?",
            close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        )
        
        as_of = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            event_id="test:005",
            as_of=as_of,
            market_prob=0.5,
        )
        
        forecast = forecaster.predict(event, snapshot, as_of)
        
        # Should be clamped to 1.0
        assert forecast.p_yes == 1.0

    def test_llm_error_fallback(self):
        """Test that LLM errors fall back to market probability."""
        # Create a mock LLM that raises errors
        class ErrorMockLLM:
            def complete_json(self, system_prompt, user_prompt, schema):
                raise RuntimeError("LLM API error")
        
        llm = ErrorMockLLM()
        forecaster = LLMForecaster(llm=llm)
        
        event = Event(
            event_id="test:006",
            title="Test Event 6",
            question="Will this happen?",
            close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        )
        
        as_of = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            event_id="test:006",
            as_of=as_of,
            market_prob=0.75,
        )
        
        forecast = forecaster.predict(event, snapshot, as_of)
        
        # Should fall back to market probability
        assert forecast.p_yes == 0.75
        assert "llm_error" in forecast.metadata
        assert "LLM error" in forecast.rationale

    def test_deterministic_output(self):
        """Test that mock LLM produces deterministic outputs."""
        llm1 = MockLLM(seed="test-seed")
        llm2 = MockLLM(seed="test-seed")
        
        forecaster1 = LLMForecaster(llm=llm1)
        forecaster2 = LLMForecaster(llm=llm2)
        
        event = Event(
            event_id="test:007",
            title="Test Event 7",
            question="Will this happen?",
            close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        )
        
        as_of = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            event_id="test:007",
            as_of=as_of,
            market_prob=0.5,
        )
        
        forecast1 = forecaster1.predict(event, snapshot, as_of)
        forecast2 = forecaster2.predict(event, snapshot, as_of)
        
        # Should produce same results with same seed
        assert abs(forecast1.p_yes - forecast2.p_yes) < 1e-6


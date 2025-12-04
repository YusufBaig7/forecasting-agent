"""Tests for multi-agent forecaster."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import pytest

from forecasting.forecast.agents import AgentOutput, MarketAgent, NewsAgent, SkepticAgent
from forecasting.forecast.multi_agent import MultiAgentForecaster
from forecasting.forecast.supervisor import Supervisor
from forecasting.llm.mock import MockLLM
from forecasting.models import Event, MarketSnapshot
from forecasting.retrieval.base import ContextBundle, Retriever


class TrackingRetriever:
    """Retriever that tracks call counts."""

    def __init__(self):
        self.call_count = 0
        self.calls = []

    def get_context(self, event, as_of):
        self.call_count += 1
        self.calls.append((event.event_id, as_of))
        # Return a simple context bundle
        from forecasting.retrieval.stub import StubRetriever
        stub = StubRetriever()
        return stub.get_context(event, as_of)


class TestMultiAgentForecaster:
    """Tests for MultiAgentForecaster."""

    def test_basic_forecast(self):
        """Test basic forecast with 3 agents."""
        llm = MockLLM()
        forecaster = MultiAgentForecaster(llm=llm, n_agents=3, use_supervisor=True)

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
        assert forecast.model == "multi_agent/v1"
        assert "n_agents" in forecast.metadata
        assert forecast.metadata["n_agents"] == 3

    def test_single_agent(self):
        """Test with single agent."""
        llm = MockLLM()
        forecaster = MultiAgentForecaster(llm=llm, n_agents=1, use_supervisor=True)

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

        assert forecast.metadata["n_agents"] == 1
        assert len(forecast.metadata["agent_outputs"]) == 1

    def test_disagreement_triggers_follow_up(self):
        """Test that disagreement triggers follow-up retrieval."""
        llm = MockLLM()
        tracking_retriever = TrackingRetriever()

        # Create a supervisor with low threshold to trigger disagreement
        supervisor = Supervisor(retriever=tracking_retriever, disagreement_threshold=0.1)

        # Create agent outputs with high disagreement
        agent_outputs = [
            AgentOutput(p_yes=0.2, rationale="Low", confidence=0.8, citations=[]),
            AgentOutput(p_yes=0.5, rationale="Medium", confidence=0.7, citations=[]),
            AgentOutput(p_yes=0.9, rationale="High", confidence=0.9, citations=[]),
        ]
        agent_roles = ["market", "news", "skeptic"]

        event = Event(
            event_id="test:003",
            title="Test Event 3",
            question="Will this happen?",
            close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        )

        as_of = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            event_id="test:003",
            as_of=as_of,
            market_prob=0.5,
        )

        # Initial call count
        initial_count = tracking_retriever.call_count

        # Combine with disagreement
        output = supervisor.combine(
            agent_outputs, agent_roles, event, snapshot, as_of
        )

        # Should have triggered follow-up retrieval
        assert output.used_follow_up
        assert tracking_retriever.call_count == initial_count + 1
        assert len(tracking_retriever.calls) == 1
        assert tracking_retriever.calls[0][0] == event.event_id
        assert tracking_retriever.calls[0][1] == as_of

    def test_no_disagreement_no_follow_up(self):
        """Test that agreement doesn't trigger follow-up."""
        llm = MockLLM()
        tracking_retriever = TrackingRetriever()

        supervisor = Supervisor(retriever=tracking_retriever, disagreement_threshold=0.25)

        # Create agent outputs with low disagreement
        agent_outputs = [
            AgentOutput(p_yes=0.45, rationale="Low", confidence=0.8, citations=[]),
            AgentOutput(p_yes=0.50, rationale="Medium", confidence=0.7, citations=[]),
            AgentOutput(p_yes=0.55, rationale="High", confidence=0.9, citations=[]),
        ]
        agent_roles = ["market", "news", "skeptic"]

        event = Event(
            event_id="test:004",
            title="Test Event 4",
            question="Will this happen?",
            close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        )

        as_of = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            event_id="test:004",
            as_of=as_of,
            market_prob=0.5,
        )

        initial_count = tracking_retriever.call_count

        output = supervisor.combine(
            agent_outputs, agent_roles, event, snapshot, as_of
        )

        # Should NOT have triggered follow-up (disagreement = 0.1 < 0.25)
        assert not output.used_follow_up
        assert tracking_retriever.call_count == initial_count

    def test_deterministic_output(self):
        """Test that mock LLM produces deterministic outputs."""
        llm1 = MockLLM(seed="test-seed")
        llm2 = MockLLM(seed="test-seed")

        forecaster1 = MultiAgentForecaster(llm=llm1, n_agents=3, use_supervisor=True)
        forecaster2 = MultiAgentForecaster(llm=llm2, n_agents=3, use_supervisor=True)

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

        forecast1 = forecaster1.predict(event, snapshot, as_of)
        forecast2 = forecaster2.predict(event, snapshot, as_of)

        # Should produce same results with same seed
        assert abs(forecast1.p_yes - forecast2.p_yes) < 1e-6

    def test_no_supervisor_simple_average(self):
        """Test that without supervisor, uses simple average."""
        llm = MockLLM()
        forecaster = MultiAgentForecaster(llm=llm, n_agents=3, use_supervisor=False)

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
            market_prob=0.5,
        )

        forecast = forecaster.predict(event, snapshot, as_of)

        # Should have used simple average
        assert "Average of" in forecast.rationale
        assert forecast.metadata["n_agents"] == 3

    def test_weighted_average_with_confidence(self):
        """Test that supervisor uses weighted average based on confidence."""
        llm = MockLLM()
        supervisor = Supervisor(retriever=None, disagreement_threshold=0.25)

        # Create outputs with different confidences
        agent_outputs = [
            AgentOutput(p_yes=0.3, rationale="Low", confidence=0.9, citations=[]),
            AgentOutput(p_yes=0.5, rationale="Medium", confidence=0.5, citations=[]),
            AgentOutput(p_yes=0.7, rationale="High", confidence=0.3, citations=[]),
        ]
        agent_roles = ["market", "news", "skeptic"]

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

        output = supervisor.combine(
            agent_outputs, agent_roles, event, snapshot, as_of
        )

        # Weighted average: (0.3*0.9 + 0.5*0.5 + 0.7*0.3) / (0.9+0.5+0.3)
        # = (0.27 + 0.25 + 0.21) / 1.7 = 0.73 / 1.7 ≈ 0.429
        expected = (0.3 * 0.9 + 0.5 * 0.5 + 0.7 * 0.3) / (0.9 + 0.5 + 0.3)
        assert abs(output.p_yes - expected) < 0.01

    def test_as_of_propagation(self):
        """Test that as_of is properly propagated to retriever."""
        llm = MockLLM()
        tracking_retriever = TrackingRetriever()
        forecaster = MultiAgentForecaster(
            llm=llm, retriever=tracking_retriever, n_agents=3, use_supervisor=True
        )

        event = Event(
            event_id="test:008",
            title="Test Event 8",
            question="Will this happen?",
            close_time=datetime(2024, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        )

        as_of = datetime(2024, 1, 5, 15, 30, 0, tzinfo=timezone.utc)
        snapshot = MarketSnapshot(
            event_id="test:008",
            as_of=as_of,
            market_prob=0.6,
        )

        forecast = forecaster.predict(event, snapshot, as_of)

        # Retriever should have been called with correct as_of
        # (at least once for initial context, possibly more for follow-up)
        assert tracking_retriever.call_count >= 1
        assert all(call[1] == as_of for call in tracking_retriever.calls)
        assert forecast.as_of == as_of


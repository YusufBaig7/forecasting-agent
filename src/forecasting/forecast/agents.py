"""Agent-based forecasters with different perspectives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from forecasting.llm.base import LLM
from forecasting.models import Event, MarketSnapshot
from forecasting.retrieval.base import ContextBundle, Retriever


@dataclass
class AgentOutput:
    """Output from a single agent."""

    p_yes: float
    rationale: str
    confidence: float  # 0-1, how confident the agent is
    citations: list[str]  # List of source identifiers used


class BaseAgent:
    """Base class for agents."""

    def __init__(
        self,
        llm: LLM,
        retriever: Optional[Retriever] = None,
        role_name: str = "agent",
    ):
        """
        Initialize agent.
        
        Args:
            llm: LLM provider
            retriever: Optional retriever for context
            role_name: Name of the agent role
        """
        self.llm = llm
        self.retriever = retriever
        self.role_name = role_name

    def get_system_prompt(self) -> str:
        """Get system prompt for this agent."""
        raise NotImplementedError

    def get_context_slice(
        self, context: Optional[ContextBundle]
    ) -> Optional[ContextBundle]:
        """
        Get relevant context slice for this agent.
        
        Args:
            context: Full context bundle
            
        Returns:
            Filtered context bundle or None
        """
        return context  # Default: use all context

    def predict(
        self,
        event: Event,
        snapshot: MarketSnapshot,
        as_of: datetime,
        context: Optional[ContextBundle] = None,
    ) -> AgentOutput:
        """
        Generate agent prediction.
        
        Args:
            event: Event to forecast
            snapshot: Market snapshot
            as_of: Timestamp for forecast
            context: Optional context bundle
            
        Returns:
            AgentOutput with prediction and metadata
        """
        # Get context slice for this agent
        agent_context = self.get_context_slice(context)

        # Build prompts
        system_prompt = self.get_system_prompt()
        user_prompt = self._build_user_prompt(event, snapshot, agent_context)

        # Define expected JSON schema
        schema = {
            "type": "object",
            "properties": {
                "p_yes": {
                    "type": "number",
                    "description": "Probability that the event resolves YES (0-1)",
                },
                "rationale": {
                    "type": "string",
                    "description": "Explanation of the forecast",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence in the forecast (0-1)",
                },
                "citations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source identifiers used in the analysis",
                },
            },
            "required": ["p_yes", "rationale", "confidence"],
        }

        # Get LLM response
        try:
            response = self.llm.complete_json(system_prompt, user_prompt, schema)
        except Exception as e:
            # Fallback to market probability
            market_prob = snapshot.best_market_prob() or 0.5
            return AgentOutput(
                p_yes=float(market_prob),
                rationale=f"LLM error: {e}. Using market probability as fallback.",
                confidence=0.3,
                citations=[],
            )

        # Extract and validate response
        p_yes = response.get("p_yes", snapshot.best_market_prob() or 0.5)
        p_yes = max(0.0, min(1.0, float(p_yes)))

        confidence = response.get("confidence", 0.5)
        confidence = max(0.0, min(1.0, float(confidence)))

        rationale = response.get("rationale", "No rationale provided.")
        citations = response.get("citations", [])

        return AgentOutput(
            p_yes=p_yes,
            rationale=rationale,
            confidence=confidence,
            citations=citations if isinstance(citations, list) else [],
        )

    def _build_user_prompt(
        self,
        event: Event,
        snapshot: MarketSnapshot,
        context: Optional[ContextBundle],
    ) -> str:
        """Build user prompt with event, market, and context information."""
        parts = []

        parts.append(f"Event: {event.title}")
        parts.append(f"Question: {event.question}")
        parts.append(f"Close time: {event.close_time.isoformat()}")
        if event.resolution_criteria:
            parts.append(f"Resolution criteria: {event.resolution_criteria}")

        # Market information
        market_prob = snapshot.best_market_prob()
        if market_prob is not None:
            parts.append(f"\nMarket probability: {market_prob:.3f}")
        if snapshot.liquidity is not None:
            parts.append(f"Market liquidity: {snapshot.liquidity:.0f}")

        # Context information
        if context is not None and len(context.items) > 0:
            parts.append(f"\nRetrieved context ({len(context.items)} items):")
            for i, item in enumerate(context.items[:5], 1):  # Limit to 5 items
                parts.append(f"\n{i}. {item.title}")
                parts.append(f"   Source: {item.source}")
                parts.append(f"   Published: {item.published_at.date()}")
                parts.append(f"   {item.snippet[:200]}...")

        parts.append(
            "\nProvide your forecast as JSON with p_yes (0-1), rationale, confidence (0-1), and citations."
        )

        return "\n".join(parts)


class MarketAgent(BaseAgent):
    """Agent focused on market signals and pricing."""

    def __init__(self, llm: LLM, retriever: Optional[Retriever] = None):
        super().__init__(llm, retriever, role_name="market")

    def get_system_prompt(self) -> str:
        return """You are a market analysis expert specializing in prediction markets.
Focus on market signals, pricing efficiency, liquidity, and crowd wisdom.
Your forecasts should heavily weight current market probabilities and trading patterns.
Be pragmatic and trust market efficiency when liquidity is high."""


class NewsAgent(BaseAgent):
    """Agent focused on news and information sources."""

    def __init__(self, llm: LLM, retriever: Optional[Retriever] = None):
        super().__init__(llm, retriever, role_name="news")

    def get_system_prompt(self) -> str:
        return """You are a news and information analysis expert.
Focus on recent developments, news articles, expert opinions, and factual information.
Weight recent information more heavily than older sources.
Be thorough in citing your sources and explaining how information influences your forecast."""

    def get_context_slice(
        self, context: Optional[ContextBundle]
    ) -> Optional[ContextBundle]:
        """News agent uses all context (default behavior)."""
        return context


class SkepticAgent(BaseAgent):
    """Agent that takes a skeptical, contrarian view."""

    def __init__(self, llm: LLM, retriever: Optional[Retriever] = None):
        super().__init__(llm, retriever, role_name="skeptic")

    def get_system_prompt(self) -> str:
        return """You are a skeptical analyst who questions assumptions and looks for contrarian signals.
Challenge consensus views, consider alternative scenarios, and identify potential biases.
Be cautious about overconfidence and consider what could go wrong.
Your forecasts should reflect healthy skepticism and uncertainty."""


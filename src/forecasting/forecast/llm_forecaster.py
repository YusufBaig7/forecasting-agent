"""LLM-based forecaster."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from forecasting.llm.base import LLM
from forecasting.models import Event, Forecast, MarketSnapshot
from forecasting.retrieval.base import Retriever


class LLMForecaster:
    """
    Forecaster that uses LLM to generate predictions from events, market data, and context.
    """

    def __init__(
        self,
        llm: LLM,
        retriever: Optional[Retriever] = None,
        model_name: str = "llm/v1",
    ):
        """
        Initialize LLM forecaster.
        
        Args:
            llm: LLM provider
            retriever: Optional retriever for context
            model_name: Model identifier
        """
        self.llm = llm
        self.retriever = retriever
        self.model_name = model_name

    def predict(
        self, event: Event, snapshot: MarketSnapshot, as_of: datetime
    ) -> Forecast:
        """
        Generate forecast using LLM.
        
        Args:
            event: Event to forecast
            snapshot: Market snapshot
            as_of: Timestamp for forecast (must match snapshot.as_of)
            
        Returns:
            Forecast with LLM-generated probability and rationale
        """
        # Verify as_of matches snapshot
        if snapshot.as_of != as_of:
            raise ValueError(
                f"as_of ({as_of}) must match snapshot.as_of ({snapshot.as_of})"
            )

        # Retrieve context if retriever is available
        context = None
        if self.retriever is not None:
            context = self.retriever.get_context(event, as_of)

        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(event, snapshot, context)

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
                "key_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key factors influencing the forecast",
                },
                "sources_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sources referenced in the analysis",
                },
            },
            "required": ["p_yes", "rationale"],
        }

        # Get LLM response
        try:
            response = self.llm.complete_json(system_prompt, user_prompt, schema)
        except Exception as e:
            # Fallback to market probability if LLM fails
            market_prob = snapshot.best_market_prob() or 0.5
            return Forecast(
                event_id=event.event_id,
                as_of=as_of,
                p_yes=float(market_prob),
                model=self.model_name,
                rationale=f"LLM error: {e}. Using market probability as fallback.",
                metadata={"llm_error": str(e)},
            )

        # Validate and extract response
        p_yes = response.get("p_yes", snapshot.best_market_prob() or 0.5)
        p_yes = float(p_yes)
        # Clamp to [0, 1]
        p_yes = max(0.0, min(1.0, p_yes))

        rationale = response.get("rationale", "No rationale provided.")
        key_factors = response.get("key_factors", [])
        sources_used = response.get("sources_used", [])

        # Build metadata
        metadata = {
            "key_factors": key_factors,
            "sources_used": sources_used,
        }
        if context is not None:
            metadata["context_items_count"] = len(context.items)
            metadata["context_summary"] = context.summary_text

        return Forecast(
            event_id=event.event_id,
            as_of=as_of,
            p_yes=p_yes,
            model=self.model_name,
            rationale=rationale,
            metadata=metadata,
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM."""
        return """You are a forecasting expert analyzing binary prediction markets.
Your task is to provide well-calibrated probability forecasts (0-1) for events.
Consider all available information including market probabilities, context, and your reasoning.
Be honest about uncertainty and provide clear rationales for your predictions."""

    def _build_user_prompt(
        self,
        event: Event,
        snapshot: MarketSnapshot,
        context: Optional[object],
    ) -> str:
        """Build user prompt with event, market, and context information."""
        parts = []

        # Event information
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
        if context is not None:
            parts.append(f"\nRetrieved context ({len(context.items)} items):")
            for i, item in enumerate(context.items[:5], 1):  # Limit to 5 items
                parts.append(f"\n{i}. {item.title}")
                parts.append(f"   Source: {item.source}")
                parts.append(f"   Published: {item.published_at.date()}")
                parts.append(f"   {item.snippet[:200]}...")  # Truncate long snippets

        parts.append(
            "\nProvide your forecast as JSON with p_yes (0-1), rationale, key_factors, and sources_used."
        )

        return "\n".join(parts)


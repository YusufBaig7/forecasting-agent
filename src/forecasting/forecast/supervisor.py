"""Supervisor that combines agent outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from forecasting.forecast.agents import AgentOutput
from forecasting.models import Event, MarketSnapshot
from forecasting.retrieval.base import ContextBundle, Retriever


@dataclass
class SupervisorOutput:
    """Output from supervisor."""

    p_yes: float
    rationale: str
    used_follow_up: bool  # Whether follow-up retrieval was triggered


class Supervisor:
    """
    Supervisor that combines agent outputs and handles disagreement.
    
    If agents disagree significantly, requests focused follow-up retrieval.
    Otherwise combines via weighted average using confidence.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        disagreement_threshold: float = 0.25,
    ):
        """
        Initialize supervisor.
        
        Args:
            retriever: Optional retriever for follow-up queries
            disagreement_threshold: Max-min difference that triggers follow-up (default: 0.25)
        """
        self.retriever = retriever
        self.disagreement_threshold = disagreement_threshold

    def combine(
        self,
        agent_outputs: list[AgentOutput],
        agent_roles: list[str],
        event: Event,
        snapshot: MarketSnapshot,
        as_of: datetime,
        initial_context: Optional[ContextBundle] = None,
    ) -> SupervisorOutput:
        """
        Combine agent outputs into final forecast.
        
        Args:
            agent_outputs: List of agent outputs
            event: Event being forecasted
            snapshot: Market snapshot
            as_of: Timestamp for forecast
            initial_context: Initial context bundle (if any)
            
        Returns:
            SupervisorOutput with final forecast
        """
        if not agent_outputs:
            # Fallback to market probability
            market_prob = snapshot.best_market_prob() or 0.5
            return SupervisorOutput(
                p_yes=float(market_prob),
                rationale="No agent outputs available. Using market probability.",
                used_follow_up=False,
            )

        if len(agent_outputs) == 1:
            # Single agent: use directly
            agent = agent_outputs[0]
            role = agent_roles[0] if agent_roles else "agent"
            return SupervisorOutput(
                p_yes=agent.p_yes,
                rationale=f"{agent.rationale}\n\n[Single agent: {role}]",
                used_follow_up=False,
            )

        # Check for disagreement
        probabilities = [a.p_yes for a in agent_outputs]
        max_prob = max(probabilities)
        min_prob = min(probabilities)
        disagreement = max_prob - min_prob

        used_follow_up = False
        follow_up_context = None

        # If disagreement exceeds threshold, request follow-up
        if disagreement > self.disagreement_threshold and self.retriever is not None:
            used_follow_up = True
            # Request focused follow-up retrieval
            # In a real implementation, this might use a different query or filter
            follow_up_context = self.retriever.get_context(event, as_of)

        # Combine via weighted average using confidence
        total_weight = sum(a.confidence for a in agent_outputs)
        if total_weight > 0:
            weighted_sum = sum(a.p_yes * a.confidence for a in agent_outputs)
            p_yes = weighted_sum / total_weight
        else:
            # Fallback to simple average if no confidence
            p_yes = sum(probabilities) / len(probabilities)

        # Clamp to [0, 1]
        p_yes = max(0.0, min(1.0, p_yes))

        # Build rationale
        rationale_parts = []
        rationale_parts.append(f"Combined forecast from {len(agent_outputs)} agents:")
        rationale_parts.append(f"  Disagreement range: {min_prob:.3f} - {max_prob:.3f} (span: {disagreement:.3f})")
        
        for i, (agent, role) in enumerate(zip(agent_outputs, agent_roles), 1):
            role_name = role if role else f'agent_{i}'
            rationale_parts.append(f"  {role_name.capitalize()}: {agent.p_yes:.3f} (confidence: {agent.confidence:.2f})")
            if agent.citations:
                rationale_parts.append(f"    Citations: {', '.join(agent.citations[:3])}")

        if used_follow_up:
            rationale_parts.append(f"\nFollow-up retrieval triggered due to disagreement > {self.disagreement_threshold}.")
            if follow_up_context:
                rationale_parts.append(f"Retrieved {len(follow_up_context.items)} additional items.")

        rationale_parts.append(f"\nFinal weighted average: {p_yes:.3f}")

        rationale = "\n".join(rationale_parts)

        return SupervisorOutput(
            p_yes=p_yes,
            rationale=rationale,
            used_follow_up=used_follow_up,
        )


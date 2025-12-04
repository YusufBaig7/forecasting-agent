"""Multi-agent forecaster with supervisor."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from forecasting.forecast.agents import AgentOutput, BaseAgent, MarketAgent, NewsAgent, SkepticAgent
from forecasting.forecast.supervisor import Supervisor, SupervisorOutput
from forecasting.llm.base import LLM
from forecasting.models import Event, Forecast, MarketSnapshot
from forecasting.retrieval.base import Retriever


class MultiAgentForecaster:
    """
    Multi-agent forecaster that combines multiple agent perspectives with a supervisor.
    """

    def __init__(
        self,
        llm: LLM,
        retriever: Optional[Retriever] = None,
        n_agents: int = 3,
        use_supervisor: bool = True,
        model_name: str = "multi_agent/v1",
    ):
        """
        Initialize multi-agent forecaster.
        
        Args:
            llm: LLM provider for agents
            retriever: Optional retriever for context
            n_agents: Number of agents to use (1 or 3)
            use_supervisor: Whether to use supervisor for combination
            model_name: Model identifier
        """
        self.llm = llm
        self.retriever = retriever
        self.n_agents = n_agents
        self.use_supervisor = use_supervisor
        self.model_name = model_name

        # Create agents
        if n_agents == 1:
            self.agents = [MarketAgent(llm, retriever)]
        elif n_agents == 3:
            self.agents = [
                MarketAgent(llm, retriever),
                NewsAgent(llm, retriever),
                SkepticAgent(llm, retriever),
            ]
        else:
            raise ValueError(f"n_agents must be 1 or 3, got {n_agents}")

        # Create supervisor if enabled
        self.supervisor = Supervisor(retriever=retriever) if use_supervisor else None

    def predict(
        self, event: Event, snapshot: MarketSnapshot, as_of: datetime
    ) -> Forecast:
        """
        Generate forecast using multi-agent system.
        
        Args:
            event: Event to forecast
            snapshot: Market snapshot
            as_of: Timestamp for forecast (must match snapshot.as_of)
            
        Returns:
            Forecast with combined agent predictions
        """
        # Verify as_of matches snapshot
        if snapshot.as_of != as_of:
            raise ValueError(
                f"as_of ({as_of}) must match snapshot.as_of ({snapshot.as_of})"
            )

        # Get initial context if retriever is available
        initial_context = None
        if self.retriever is not None:
            initial_context = self.retriever.get_context(event, as_of)

        # Get predictions from all agents
        agent_outputs: list[AgentOutput] = []
        agent_roles: list[str] = []
        for agent in self.agents:
            output = agent.predict(event, snapshot, as_of, context=initial_context)
            agent_outputs.append(output)
            agent_roles.append(agent.role_name)

        # Combine via supervisor or simple average
        if self.use_supervisor and self.supervisor is not None:
            supervisor_output = self.supervisor.combine(
                agent_outputs, agent_roles, event, snapshot, as_of, initial_context
            )
            p_yes = supervisor_output.p_yes
            rationale = supervisor_output.rationale
            used_follow_up = supervisor_output.used_follow_up
        else:
            # Simple average if no supervisor
            p_yes = sum(a.p_yes for a in agent_outputs) / len(agent_outputs)
            rationale = f"Average of {len(agent_outputs)} agents: " + " | ".join(
                f"{role}: {a.p_yes:.3f}" for role, a in zip(agent_roles, agent_outputs)
            )
            used_follow_up = False

        # Build metadata
        metadata = {
            "n_agents": len(agent_outputs),
            "agent_outputs": [
                {
                    "role": role,
                    "p_yes": float(a.p_yes),
                    "confidence": float(a.confidence),
                    "citations": a.citations,
                }
                for role, a in zip(agent_roles, agent_outputs)
            ],
        }
        if used_follow_up:
            metadata["used_follow_up"] = True

        return Forecast(
            event_id=event.event_id,
            as_of=as_of,
            p_yes=float(p_yes),
            model=self.model_name,
            rationale=rationale,
            metadata=metadata,
        )


"""Forecast modules."""

from .agents import AgentOutput, BaseAgent, MarketAgent, NewsAgent, SkepticAgent
from .baseline_market import MarketBaselineForecaster
from .calibration import (
    CalibratedForecaster,
    Extremizer,
    PlattCalibrator,
    logit,
    sigmoid,
)
from .llm_forecaster import LLMForecaster
from .multi_agent import MultiAgentForecaster
from .supervisor import Supervisor, SupervisorOutput

__all__ = [
    "MarketBaselineForecaster",
    "PlattCalibrator",
    "Extremizer",
    "CalibratedForecaster",
    "logit",
    "sigmoid",
    "LLMForecaster",
    "MultiAgentForecaster",
    "AgentOutput",
    "BaseAgent",
    "MarketAgent",
    "NewsAgent",
    "SkepticAgent",
    "Supervisor",
    "SupervisorOutput",
]

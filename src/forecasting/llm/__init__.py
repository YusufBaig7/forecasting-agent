"""LLM interface modules."""

from .base import LLM
from .mock import MockLLM

__all__ = ["LLM", "MockLLM"]

# Conditionally export Gemini client if available
try:
    from .gemini_client import GeminiClientLLM
    __all__.append("GeminiClientLLM")
except ImportError:
    pass


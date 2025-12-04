"""Mock LLM that returns deterministic outputs."""

from __future__ import annotations

import hashlib
import json

from forecasting.llm.base import LLM


def _hash_to_unit_float(s: str) -> float:
    """Convert string to deterministic float in [0, 1)."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    x = int(h[:13], 16)  # 13 hex chars ~ 52 bits
    return x / float(1 << 52)


class MockLLM:
    """
    Mock LLM that returns deterministic outputs based on prompt hash.
    
    Useful for testing and development without API keys.
    """

    def __init__(self, seed: str = "mock-llm-v1"):
        """
        Initialize mock LLM.
        
        Args:
            seed: Seed for deterministic generation
        """
        self.seed = seed

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
    ) -> dict:
        """
        Generate deterministic mock JSON response.
        
        Args:
            system_prompt: System prompt (used for hash)
            user_prompt: User prompt (used for hash)
            schema: Expected schema (used to determine structure)
            
        Returns:
            Dictionary matching schema with deterministic values
        """
        # Create deterministic hash from prompts
        combined = f"{self.seed}|{system_prompt}|{user_prompt}"
        h = _hash_to_unit_float(combined)
        
        # Generate deterministic probability
        p_yes = h  # Will be in [0, 1)
        
        # Build response based on schema
        response = {}
        
        # Extract expected fields from schema (if available)
        if "properties" in schema:
            props = schema["properties"]
            
            # Handle p_yes
            if "p_yes" in props:
                response["p_yes"] = float(p_yes)
            
            # Handle rationale
            if "rationale" in props:
                response["rationale"] = (
                    f"Mock LLM analysis based on hash {h:.6f}. "
                    f"This is a deterministic synthetic response."
                )
            
            # Handle key_factors
            if "key_factors" in props:
                n_factors = int(h * 5) + 1  # 1-5 factors
                response["key_factors"] = [
                    f"Factor {i+1} (hash: {_hash_to_unit_float(f'{combined}|factor{i}'):.6f})"
                    for i in range(n_factors)
                ]
            
            # Handle sources_used
            if "sources_used" in props:
                n_sources = int(h * 3) + 1  # 1-3 sources
                response["sources_used"] = [
                    f"source_{i}" for i in range(n_sources)
                ]
        else:
            # Fallback: return basic structure
            response = {
                "p_yes": float(p_yes),
                "rationale": "Mock LLM response",
                "key_factors": [],
                "sources_used": [],
            }
        
        return response


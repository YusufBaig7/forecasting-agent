"""Base LLM interface."""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    """Protocol for LLM providers."""

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
    ) -> dict:
        """
        Complete a JSON response based on prompts and schema.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User query/prompt
            schema: JSON schema describing expected output structure
            
        Returns:
            Dictionary matching the schema
        """
        ...


"""OpenAI client LLM implementation."""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from forecasting.llm.base import LLM


class OpenAIClientLLM:
    """
    OpenAI client LLM implementation.
    
    Only works if OPENAI_API_KEY environment variable is set.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model name (default: gpt-4o-mini)
            api_key: API key (defaults to OPENAI_API_KEY env var)
            
        Raises:
            RuntimeError: If OpenAI is not available or API key is missing
        """
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it or use MockLLM for testing."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
    ) -> dict:
        """
        Complete JSON response using OpenAI API.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User query/prompt
            schema: JSON schema describing expected output
            
        Returns:
            Dictionary matching the schema
        """
        # Build prompt that requests JSON output
        full_prompt = f"{user_prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI")
            
            # Parse JSON response
            result = json.loads(content)
            return result
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e


def get_llm_if_available() -> Optional[LLM]:
    """
    Get OpenAI LLM if API key is available, else None.
    
    Returns:
        OpenAIClientLLM if key exists, else None
    """
    if not OPENAI_AVAILABLE:
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    try:
        return OpenAIClientLLM()
    except Exception:
        return None


"""Google Gemini client LLM implementation."""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from forecasting.llm.base import LLM


class GeminiClientLLM:
    """
    Google Gemini client LLM implementation.
    
    Only works if GEMINI_API_KEY environment variable is set.
    """

    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            model: Model name (default: gemini-1.5-flash)
            api_key: API key (defaults to GEMINI_API_KEY env var)
            
        Raises:
            RuntimeError: If Gemini is not available or API key is missing
        """
        if not GEMINI_AVAILABLE:
            raise RuntimeError(
                "Google Generative AI package not installed. Install with: pip install google-generativeai"
            )
        
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Set it or use MockLLM for testing."
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
    ) -> dict:
        """
        Complete JSON response using Gemini API.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User query/prompt
            schema: JSON schema describing expected output
            
        Returns:
            Dictionary matching the schema
        """
        # Build prompt that requests JSON output
        full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
        
        try:
            # Generate content with JSON mode
            # Use dict for generation_config (Gemini SDK accepts both dict and GenerationConfig)
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                response_mime_type="application/json",
            )
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            
            content = response.text
            if not content:
                raise ValueError("Empty response from Gemini")
            
            # Parse JSON response
            result = json.loads(content)
            return result
            
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e


def get_llm_if_available() -> Optional[LLM]:
    """
    Get Gemini LLM if API key is available, else None.
    
    Returns:
        GeminiClientLLM if key exists, else None
    """
    if not GEMINI_AVAILABLE:
        return None
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    try:
        return GeminiClientLLM()
    except Exception:
        return None


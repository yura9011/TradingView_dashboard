"""
Base Agent Class - Common functionality for all specialist agents.
"""

import os
import yaml
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

# Default prompts directory
PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"


@dataclass
class AgentResponse:
    """Structured response from an agent."""
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all specialist agents."""
    
    def __init__(
        self,
        prompt_file: str,
        api_key: Optional[str] = None,
        model: str = "models/gemini-flash-latest",
    ):
        """Initialize base agent.
        
        Args:
            prompt_file: Name of prompt YAML file (without path)
            api_key: Gemini API key
            model: Model to use
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.system_prompt = self._load_prompt(prompt_file)
        
        # Configure Gemini
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                },
                system_instruction=self.system_prompt,
            )
        else:
            raise ValueError("GEMINI_API_KEY is required")
    
    def _load_prompt(self, prompt_file: str) -> str:
        """Load system prompt from YAML file."""
        prompt_path = PROMPTS_DIR / prompt_file
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        return config.get("system_prompt", "")
    
    def _load_image(self, image_path: str) -> Dict[str, Any]:
        """Load image for Gemini API."""
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        mime_type = "image/png" if image_path.endswith(".png") else "image/jpeg"
        
        return {
            "mime_type": mime_type,
            "data": base64.b64encode(image_data).decode("utf-8"),
        }
    
    def analyze(self, image_path: str, additional_context: str = "") -> AgentResponse:
        """Analyze chart image.
        
        Args:
            image_path: Path to chart image
            additional_context: Optional additional context
            
        Returns:
            AgentResponse with parsed results
        """
        try:
            image_data = self._load_image(image_path)
            
            # Build content
            content = [
                {"inline_data": image_data},
                f"Analyze this chart. {additional_context}".strip(),
            ]
            
            # Generate response
            response = self.model.generate_content(content)
            raw_text = response.text
            
            logger.info(f"{self.__class__.__name__} analysis complete")
            logger.info(f"Raw response (first 500 chars): {raw_text[:500]}")
            
            # Parse response
            parsed = self._parse_response(raw_text)
            
            return AgentResponse(
                raw_text=raw_text,
                parsed=parsed,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"{self.__class__.__name__} failed: {e}")
            return AgentResponse(
                raw_text="",
                parsed={},
                success=False,
                error=str(e),
            )
    
    @abstractmethod
    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse agent-specific response format.
        
        Args:
            raw_text: Raw text from model
            
        Returns:
            Parsed dictionary
        """
        pass

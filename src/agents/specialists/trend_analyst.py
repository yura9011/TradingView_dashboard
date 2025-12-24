"""
Trend Analyst Agent - Determines trend direction and market phase.
"""

import logging
from typing import Dict, Any, Optional

from src.agents.specialists.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class TrendAnalystAgent(BaseAgent):
    """Specialist agent for analyzing trend direction."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            prompt_file="trend_analyst.yaml",
            api_key=api_key,
        )
    
    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse trend analysis response.
        
        Expected format (v2.0):
        TREND: [up/down/sideways]
        STRENGTH: [strong/moderate/weak]
        PHASE: [accumulation/markup/distribution/markdown/unclear]
        WYCKOFF_EVENT: [specific event or "none"]
        WAVE: [current wave or "unclear"]
        WAVE_COUNT: [brief wave count or "unclear"]
        DESCRIPTION: [text]
        """
        result = {
            "trend": "unknown",
            "strength": "unknown",
            "phase": "unclear",
            "wyckoff_event": None,
            "wave": None,
            "wave_count": None,
            "description": "",
        }
        
        lines = raw_text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("TREND:"):
                trend = line.replace("TREND:", "").strip().lower()
                if trend in ["up", "uptrend", "bullish"]:
                    result["trend"] = "up"
                elif trend in ["down", "downtrend", "bearish"]:
                    result["trend"] = "down"
                else:
                    result["trend"] = "sideways"
                    
            elif line.startswith("STRENGTH:"):
                strength = line.replace("STRENGTH:", "").strip().lower()
                if strength in ["strong", "moderate", "weak"]:
                    result["strength"] = strength
                    
            elif line.startswith("PHASE:"):
                phase = line.replace("PHASE:", "").strip().lower()
                valid_phases = ["accumulation", "markup", "distribution", "markdown", "unclear"]
                if phase in valid_phases:
                    result["phase"] = phase
            
            elif line.startswith("WYCKOFF_EVENT:"):
                event = line.replace("WYCKOFF_EVENT:", "").strip()
                if event.lower() not in ["none", "n/a", "-", ""]:
                    result["wyckoff_event"] = event
                    
            elif line.startswith("WAVE:"):
                wave = line.replace("WAVE:", "").strip()
                if wave.lower() not in ["unclear", "none", "n/a", "-", ""]:
                    result["wave"] = wave
            
            elif line.startswith("WAVE_COUNT:"):
                wave_count = line.replace("WAVE_COUNT:", "").strip()
                if wave_count.lower() not in ["unclear", "none", "n/a", "-", ""]:
                    result["wave_count"] = wave_count
                    
            elif line.startswith("DESCRIPTION:"):
                result["description"] = line.replace("DESCRIPTION:", "").strip()
        
        # If description spans multiple lines, capture the rest
        if "DESCRIPTION:" in raw_text:
            desc_start = raw_text.find("DESCRIPTION:") + 12
            result["description"] = raw_text[desc_start:].strip()
        
        logger.info(f"Trend: {result['trend']} ({result['strength']}), Phase: {result['phase']}, Event: {result['wyckoff_event']}")
        
        return result

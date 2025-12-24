"""
Trend Analyst Agent (Local) - Uses Phi-3.5-vision-instruct.
"""

import logging
from typing import Dict, Any, Optional

from src.agents.specialists.base_agent_local import BaseAgentLocal, AgentResponse

logger = logging.getLogger(__name__)


class TrendAnalystAgentLocal(BaseAgentLocal):
    """Specialist agent for analyzing trend direction using local model."""
    
    def __init__(self, model_name: str = None):
        super().__init__(
            prompt_file="trend_analyst.yaml",
            model_name=model_name or "microsoft/Phi-3.5-vision-instruct",
        )
    
    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse trend analysis response."""
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
        
        if "DESCRIPTION:" in raw_text:
            desc_start = raw_text.find("DESCRIPTION:") + 12
            result["description"] = raw_text[desc_start:].strip()
        
        logger.info(f"Trend: {result['trend']} ({result['strength']}), Phase: {result['phase']}")
        
        return result

"""
Trend Analyst Agent (Local) - Uses Qwen2-VL-7B-Instruct.
"""

import logging
from typing import Dict, Any, Optional

from src.agents.specialists.base_agent_local import BaseAgentLocal

logger = logging.getLogger(__name__)


class TrendAnalystAgentLocal(BaseAgentLocal):
    """Specialist agent for analyzing trend direction using local model."""
    
    def __init__(self, model_name: str = None):
        super().__init__(
            prompt_file="trend_analyst.yaml",
            model_name=model_name or "Qwen/Qwen2-VL-7B-Instruct",
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
                result["strength"] = strength
                    
            elif line.startswith("MARKET_PHASE:") or line.startswith("PHASE:"):
                phase = line.replace("MARKET_PHASE:", "").replace("PHASE:", "").strip().lower()
                result["phase"] = phase
            
            elif line.startswith("VSA_SIGNAL:"):
                signal = line.replace("VSA_SIGNAL:", "").strip()
                if signal.lower() not in ["none", "n/a", "-", ""]:
                    result["vsa_signal"] = signal
            
            elif line.startswith("VOLUME_ACTION:"):
                action = line.replace("VOLUME_ACTION:", "").strip()
                result["volume_action"] = action
                    
            elif line.startswith("WYCKOFF_EVENT:"):
                event = line.replace("WYCKOFF_EVENT:", "").strip()
                result["wyckoff_event"] = event if event.lower() not in ["none", ""] else None

            elif line.startswith("DESCRIPTION:"):
                result["description"] = line.replace("DESCRIPTION:", "").strip()
        
        if "DESCRIPTION:" in raw_text:
            desc_start = raw_text.find("DESCRIPTION:") + 12
            result["description"] = raw_text[desc_start:].strip()
        
        logger.info(f"Trend: {result['trend']} ({result['strength']}), Phase: {result['phase']}, Signal: {result.get('vsa_signal', 'none')}")
        logger.debug(f"Trend raw response: {raw_text[:500]}")
        
        return result

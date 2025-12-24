"""
Levels Calculator Agent (Local) - Uses Phi-3.5-vision-instruct.
"""

import re
import logging
from typing import Dict, Any, Optional

from src.agents.specialists.base_agent_local import BaseAgentLocal

logger = logging.getLogger(__name__)


class LevelsCalculatorAgentLocal(BaseAgentLocal):
    """Specialist agent for calculating technical levels using local model."""
    
    def __init__(self, model_name: str = None):
        super().__init__(
            prompt_file="levels_calculator.yaml",
            model_name=model_name or "microsoft/Phi-3.5-vision-instruct",
        )
    
    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse levels calculation response."""
        result = {
            "support": None,
            "support_reason": None,
            "resistance": None,
            "resistance_reason": None,
            "support_secondary": None,
            "resistance_secondary": None,
            "fibonacci": None,
            "fibonacci_confluence": None,
            "key_level": None,
            "key_level_reason": None,
            "description": "",
        }
        
        lines = raw_text.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("SUPPORT_SECONDARY:"):
                result["support_secondary"] = self._parse_price(
                    line.replace("SUPPORT_SECONDARY:", "").strip()
                )
            elif line.startswith("SUPPORT_REASON:"):
                result["support_reason"] = line.replace("SUPPORT_REASON:", "").strip()
            elif line.startswith("SUPPORT:"):
                result["support"] = self._parse_price(
                    line.replace("SUPPORT:", "").strip()
                )
                
            elif line.startswith("RESISTANCE_SECONDARY:"):
                result["resistance_secondary"] = self._parse_price(
                    line.replace("RESISTANCE_SECONDARY:", "").strip()
                )
            elif line.startswith("RESISTANCE_REASON:"):
                result["resistance_reason"] = line.replace("RESISTANCE_REASON:", "").strip()
            elif line.startswith("RESISTANCE:"):
                result["resistance"] = self._parse_price(
                    line.replace("RESISTANCE:", "").strip()
                )
            
            elif line.startswith("FIBONACCI_CONFLUENCE:"):
                result["fibonacci_confluence"] = line.replace("FIBONACCI_CONFLUENCE:", "").strip()
            elif line.startswith("FIBONACCI:"):
                result["fibonacci"] = line.replace("FIBONACCI:", "").strip()
            
            elif line.startswith("KEY_LEVEL_REASON:"):
                result["key_level_reason"] = line.replace("KEY_LEVEL_REASON:", "").strip()
            elif line.startswith("KEY_LEVEL:"):
                result["key_level"] = self._parse_price(
                    line.replace("KEY_LEVEL:", "").strip()
                )
                
            elif line.startswith("DESCRIPTION:"):
                result["description"] = line.replace("DESCRIPTION:", "").strip()
        
        if "DESCRIPTION:" in raw_text:
            desc_start = raw_text.find("DESCRIPTION:") + 12
            result["description"] = raw_text[desc_start:].strip()
        
        logger.info(f"Levels: S={result['support']}, R={result['resistance']}, Key={result['key_level']}")
        
        return result
    
    def _parse_price(self, val: str) -> Optional[float]:
        """Parse price value from various formats."""
        if not val or val.lower() in ["unknown", "n/a", "none", "-"]:
            return None
        
        try:
            cleaned = val.replace("$", "").replace(",", "").strip()
            match = re.search(r'[\d.]+', cleaned)
            if match:
                return float(match.group())
        except (ValueError, AttributeError):
            pass
        
        return None

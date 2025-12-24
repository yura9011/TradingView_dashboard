"""
Levels Calculator Agent - Identifies support, resistance, and Fibonacci levels.
"""

import re
import logging
from typing import Dict, Any, Optional

from src.agents.specialists.base_agent import BaseAgent, AgentResponse

logger = logging.getLogger(__name__)


class LevelsCalculatorAgent(BaseAgent):
    """Specialist agent for calculating technical levels."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            prompt_file="levels_calculator.yaml",
            api_key=api_key,
        )
    
    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse levels calculation response.
        
        Expected format (v2.0):
        SUPPORT: [$price]
        SUPPORT_REASON: [justification]
        RESISTANCE: [$price]
        RESISTANCE_REASON: [justification]
        SUPPORT_SECONDARY: [$price or "none"]
        RESISTANCE_SECONDARY: [$price or "none"]
        FIBONACCI: [level% at $price (measured from $low to $high)]
        FIBONACCI_CONFLUENCE: [confluence description]
        KEY_LEVEL: [$price]
        KEY_LEVEL_REASON: [justification]
        DESCRIPTION: [text]
        """
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
            
            # Parse in specific order to avoid partial matches
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
        
        # If description spans multiple lines, capture the rest
        if "DESCRIPTION:" in raw_text:
            desc_start = raw_text.find("DESCRIPTION:") + 12
            result["description"] = raw_text[desc_start:].strip()
        
        logger.info(f"Levels: S={result['support']} ({result['support_reason'][:30] if result['support_reason'] else 'N/A'}...), R={result['resistance']}, Key={result['key_level']}")
        
        return result
    
    def _parse_price(self, val: str) -> Optional[float]:
        """Parse price value from various formats."""
        if not val or val.lower() in ["unknown", "n/a", "none", "-"]:
            return None
        
        try:
            # Remove currency symbols, commas, and extra text
            cleaned = val.replace("$", "").replace(",", "").strip()
            # Extract first number found
            match = re.search(r'[\d.]+', cleaned)
            if match:
                return float(match.group())
        except (ValueError, AttributeError):
            pass
        
        return None

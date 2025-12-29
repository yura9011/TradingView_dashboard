"""
Levels Calculator Agent (Local) - Uses Qwen2-VL-7B-Instruct.
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
            model_name=model_name or "Qwen/Qwen2-VL-7B-Instruct",
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
        
        logger.info(f"Levels parsed: S={result['support']}, R={result['resistance']}, Key={result['key_level']}")
        logger.debug(f"Levels raw response: {raw_text[:500]}")
        
        return result
    
    def analyze(self, image_path: str, market_context: str = "") -> Any:
        # Run base analysis
        response = super().analyze(image_path, market_context)
        
        # Sanity check against current price if available in context
        try:
             price_match = re.search(r"Current Price: \$([\d,]+\.?\d*)", market_context)
             if price_match and response.success:
                 current_price = float(price_match.group(1).replace(",", ""))
                 parsed = response.parsed
                 
                 support = parsed.get("support")
                 resistance = parsed.get("resistance")
                 
                 # 1. Hallucination Check - Copying old examples
                 if support == 15.50 and resistance == 18.00:
                      logger.warning("🚨 Hallucination detected: Model copied example values!")
                      parsed["support"] = None
                      parsed["resistance"] = None
                      parsed["description"] += " [Wait: Model copied example values, ignoring]"
                      return response
                 
                 # 2. Logic Check - Support should be < Current Price (mostly)
                 # If Support is > Current Price by >2%, it's likely wrong (or it's resistance)
                 if support and isinstance(support, (int, float)):
                     if support > current_price * 1.02:
                         logger.warning(f"⚠️ Logical Swing: Support ${support} is > Current ${current_price}")
                         # If resistance is unavailable or less than support, maybe they are swapped?
                         if resistance and resistance < support:
                             logger.info("Swapping Support/Resistance due to logical inversion")
                             parsed["support"], parsed["resistance"] = resistance, support
                         else:
                             # Just invalid support
                             parsed["support_reason"] = f"(Invalid: > Spot ${current_price:.2f}) " + (parsed.get("support_reason") or "")
                             parsed["support"] = None

                 # 3. Logic Check - Resistance should be > Current Price (mostly)
                 if resistance and isinstance(resistance, (int, float)):
                     if resistance < current_price * 0.98:
                         logger.warning(f"⚠️ Logical Swing: Resistance ${resistance} is < Current ${current_price}")
                         parsed["resistance_reason"] = f"(Invalid: < Spot ${current_price:.2f}) " + (parsed.get("resistance_reason") or "")
                         parsed["resistance"] = None
                         
        except Exception as e:
             logger.debug(f"Sanity verify failed: {e}")
             
        return response

    def _parse_price(self, val: str) -> Optional[float]:
        """Parse price value from various formats."""
        if not val or not isinstance(val, str):
            return None
        
        if val.lower() in ["unknown", "n/a", "none", "-", "unclear"]:
            return None
        
        try:
            # Remove currency symbols and cleanup
            cleaned = val.replace("$", "").replace(",", "").strip()
            # Extract first number found
            match = re.search(r'[\d.]+', cleaned)
            if match:
                return float(match.group())
        except (ValueError, AttributeError):
            pass
        
        return None

"""
Risk Manager Agent (Dave) - Evaluates trade risk using ATR and volatility rules.
"""

import logging
from typing import Dict, Any, Optional

from src.agents.specialists.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RiskManagerAgent(BaseAgent):
    """Specialist agent for risk assessment (Dave)."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            prompt_file="risk_manager.yaml",
            api_key=api_key,
        )
    
    def analyze(self, pattern_data: Dict[str, Any], trend_data: Dict[str, Any], current_price: float, atr_value: float) -> Dict[str, Any]:
        """
        Analyze risk based on pattern, trend, and ATR.
        
        Args:
            pattern_data: Output from PatternDetector
            trend_data: Output from TrendAnalyst
            current_price: Current market price
            atr_value: Current Average True Range
        """
        # Ensure data is not None
        pattern_data = pattern_data or {}
        trend_data = trend_data or {}
        
        # Calculate ATR percent safely
        atr_percent = (atr_value / current_price * 100) if current_price and current_price > 0 else 0
        
        context = f"""
        PROPOSED_PATTERN: {pattern_data.get('pattern', 'None')}
        PATTERN_CONFIDENCE: {pattern_data.get('confidence', 0.0)}
        
        CURRENT_PRICE: {current_price or 'Unknown'}
        ATR_VALUE: {atr_value or 0}
        ATR_PERCENT: {atr_percent:.2f}%
        
        TREND: {trend_data.get('trend', 'Unknown')}
        MARKET_PHASE: {trend_data.get('phase', 'Unknown')}
        
        Please provide a risk assessment.
        """
        
        # Validate context is meaningful
        if not current_price or (not pattern_data and not trend_data):
            return {
                "risk_assessment": "UNKNOWN",
                "stop_loss": "N/A",
                "position_size": "0%",
                "reasoning": "Insufficient data for risk analysis"
            }
        
        try:
            # Call Gemini with text only
            response = self.model.generate_content([context])
            return self._parse_response(response.text)
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {
                "risk_assessment": "ERROR",
                "stop_loss": "N/A",
                "position_size": "0%",
                "reasoning": f"Analysis failed: {str(e)}"
            }

    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse Risk Manager response."""
        result = {
            "risk_assessment": "UNKNOWN",
            "stop_loss": "N/A",
            "position_size": "0%",
            "reasoning": ""
        }
        
        lines = raw_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("RISK_ASSESSMENT:"):
                result["risk_assessment"] = line.replace("RISK_ASSESSMENT:", "").strip()
            elif line.startswith("STOP_LOSS_LEVEL:") or line.startswith("STOP_LOSS:"):
                result["stop_loss"] = line.split(":", 1)[1].strip()
            elif line.startswith("POSITION_SIZING:") or line.startswith("POSITION_SIZE:"):
                result["position_size"] = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()
        
        return result

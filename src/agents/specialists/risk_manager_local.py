"""
Risk Manager Agent (Local) - Dave - Uses local LLM for risk assessment.
This agent uses text-only input (no images) to evaluate trade risk.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RiskManagerAgentLocal:
    """Local risk manager that calculates risk without LLM for speed."""
    
    def __init__(self, model_name: str = None):
        """Initialize risk manager.
        
        Note: This agent uses rule-based logic instead of LLM for speed,
        since risk calculations are deterministic based on ATR.
        """
        self.model_name = model_name  # Reserved for future LLM-based reasoning
        logger.info("RiskManagerAgentLocal initialized (rule-based)")
    
    def analyze(
        self,
        pattern_data: Dict[str, Any],
        trend_data: Dict[str, Any],
        current_price: float,
        atr_value: float,
        rvol: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Analyze risk based on pattern, trend, ATR and RVOL.
        
        Args:
            pattern_data: Output from PatternDetector
            trend_data: Output from TrendAnalyst
            current_price: Current market price
            atr_value: Current Average True Range
            rvol: Relative Volume (current / average)
            
        Returns:
            Risk assessment with stop loss and position sizing
        """
        try:
            pattern = pattern_data.get("pattern", "none")
            pattern_conf = pattern_data.get("confidence", 0.0)
            trend = trend_data.get("trend", "unknown")
            trend_strength = trend_data.get("strength", "unknown")
            
            # Calculate ATR as percentage of price
            atr_percent = (atr_value / current_price * 100) if current_price > 0 else 0
            
            # ========== RISK RULES (from ATR Guide) ==========
            
            # Rule 1: ATR-based Risk Classification
            if atr_percent > 5.0:
                risk_assessment = "DANGEROUS"
                position_size = "0%"
                reasoning = f"Extreme volatility: ATR is {atr_percent:.1f}% of price. Market too noisy for safe entry."
            elif atr_percent > 3.0:
                risk_assessment = "CAUTION"
                position_size = "50%"
                reasoning = f"High volatility: ATR is {atr_percent:.1f}% of price. Reduce position size."
            else:
                risk_assessment = "SAFE"
                position_size = "100%"
                reasoning = f"Normal volatility: ATR is {atr_percent:.1f}% of price. Standard position size."
            
            # Rule 2: RVOL Check (False Breakout Filter)
            if pattern and "breakout" in pattern.lower():
                if rvol < 1.5:
                    risk_assessment = "DANGEROUS"
                    position_size = "0%"
                    reasoning = f"Low volume breakout (RVOL={rvol:.2f}). Likely fakeout. Avoid."
                elif rvol < 2.0:
                    if risk_assessment == "SAFE":
                        risk_assessment = "CAUTION"
                        position_size = "50%"
                    reasoning += f" Breakout volume marginal (RVOL={rvol:.2f})."
                else:
                    reasoning += f" Strong volume confirmation (RVOL={rvol:.2f})."
            
            # Rule 3: Trend Conflict Check
            bullish_patterns = ["double bottom", "inverse head", "bullish engulfing", "hammer", "ascending triangle"]
            bearish_patterns = ["double top", "head and shoulders", "bearish engulfing", "shooting star", "descending triangle"]
            
            is_bullish_pattern = any(bp in pattern.lower() for bp in bullish_patterns) if pattern else False
            is_bearish_pattern = any(bp in pattern.lower() for bp in bearish_patterns) if pattern else False
            
            if is_bullish_pattern and trend == "down" and trend_strength == "strong":
                if risk_assessment == "SAFE":
                    risk_assessment = "CAUTION"
                    position_size = "50%"
                reasoning += " Bullish pattern against strong downtrend - counter-trend risk."
            
            if is_bearish_pattern and trend == "up" and trend_strength == "strong":
                if risk_assessment == "SAFE":
                    risk_assessment = "CAUTION"
                    position_size = "50%"
                reasoning += " Bearish pattern against strong uptrend - counter-trend risk."
            
            # ========== STOP LOSS CALCULATION ==========
            # Standard: 2x ATR from entry
            # High volatility: 3x ATR
            
            atr_multiplier = 3.0 if risk_assessment == "CAUTION" else 2.0
            
            if is_bullish_pattern or (not is_bearish_pattern and trend == "up"):
                # Long trade: Stop below entry
                stop_loss_price = current_price - (atr_value * atr_multiplier)
                stop_loss = f"${stop_loss_price:,.2f} ({atr_multiplier}x ATR below entry)"
            else:
                # Short trade: Stop above entry
                stop_loss_price = current_price + (atr_value * atr_multiplier)
                stop_loss = f"${stop_loss_price:,.2f} ({atr_multiplier}x ATR above entry)"
            
            return {
                "risk_assessment": risk_assessment,
                "stop_loss": stop_loss,
                "stop_loss_price": stop_loss_price,
                "position_size": position_size,
                "atr_value": atr_value,
                "atr_percent": round(atr_percent, 2),
                "rvol": round(rvol, 2),
                "reasoning": reasoning.strip(),
            }
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {
                "risk_assessment": "ERROR",
                "stop_loss": "N/A",
                "stop_loss_price": 0,
                "position_size": "0%",
                "reasoning": f"Analysis failed: {str(e)}",
            }


def get_risk_manager_local(model_name: str = None) -> RiskManagerAgentLocal:
    """Factory function for RiskManagerAgentLocal."""
    return RiskManagerAgentLocal(model_name=model_name)

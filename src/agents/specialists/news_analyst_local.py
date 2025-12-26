"""
Technical Sentiment Analyst Agent (Local) - Analyzes sentiment from technical indicators.
Renamed from "NewsAnalystLocal" to clarify that it uses technical signals, not actual news.

For actual news integration, consider adding:
- NewsAPI (https://newsapi.org/)
- Alpha Vantage News Sentiment
- FinBERT for financial sentiment analysis
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TechnicalSentimentAnalystLocal:
    """Analyzes market sentiment from technical indicators.
    
    Uses RSI, MACD, and day-of-week effects to determine sentiment.
    For speed, uses rule-based heuristics instead of LLM.
    
    Note: This is NOT a news analyst. For actual news:
    - Use NewsAPI, Alpha Vantage, or similar
    - Or integrate FinBERT for headline sentiment
    """
    
    def __init__(self, model_name: str = None):
        """Initialize sentiment analyst."""
        self.model_name = model_name  # Reserved for future LLM-based reasoning
        logger.info("TechnicalSentimentAnalystLocal initialized (rule-based)")
    
    def analyze(
        self,
        market_context: str,
        current_date: str,
        symbol: str,
        rsi: Optional[float] = None,
        macd: Optional[float] = None,
        interpretation: str = "momentum",  # "momentum" or "reversal"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment based on technical indicators.
        
        Args:
            market_context: Text description of market conditions
            current_date: Today's date (includes day of week)
            symbol: Ticker symbol
            rsi: RSI value from screener (optional)
            macd: MACD value from screener (optional)
            interpretation: How to interpret RSI
                - "momentum": High RSI = bullish (trend following)
                - "reversal": High RSI = bearish (mean reversion)
            
        Returns:
            Sentiment analysis with score and label
        """
        try:
            sentiment_score = 0.0
            key_drivers = []
            
            # ========== RSI INTERPRETATION ==========
            # Two schools of thought:
            # 1. Momentum: High RSI = strong trend = bullish continuation
            # 2. Reversal: High RSI = overbought = bearish reversal risk
            
            if rsi is not None:
                if interpretation == "reversal":
                    # Reversal interpretation (contrarian)
                    if rsi < 30:
                        sentiment_score += 0.3  # Oversold = bullish opportunity
                        key_drivers.append(f"RSI oversold ({rsi:.1f}) - bullish reversal signal")
                    elif rsi > 70:
                        sentiment_score -= 0.3  # Overbought = bearish reversal risk
                        key_drivers.append(f"RSI overbought ({rsi:.1f}) - bearish reversal risk")
                    else:
                        key_drivers.append(f"RSI neutral ({rsi:.1f})")
                else:
                    # Momentum interpretation (trend following) - DEFAULT
                    if rsi < 30:
                        sentiment_score -= 0.3  # Weak momentum = bearish
                        key_drivers.append(f"RSI weak ({rsi:.1f}) - bearish momentum")
                    elif rsi > 70:
                        sentiment_score += 0.3  # Strong momentum = bullish
                        key_drivers.append(f"RSI strong ({rsi:.1f}) - bullish momentum")
                    elif 40 < rsi < 60:
                        key_drivers.append(f"RSI neutral ({rsi:.1f})")
            
            # ========== MACD INTERPRETATION ==========
            if macd is not None:
                if macd > 0:
                    sentiment_score += 0.2
                    key_drivers.append("MACD bullish (above zero)")
                else:
                    sentiment_score -= 0.2
                    key_drivers.append("MACD bearish (below zero)")
            
            # ========== DAY OF WEEK EFFECTS ==========
            date_lower = current_date.lower()
            
            if "friday" in date_lower or "viernes" in date_lower:
                key_drivers.append("Friday effect: Weekend risk, lower conviction")
                # Friday doesn't change sentiment, just adds context
            
            if "monday" in date_lower or "lunes" in date_lower:
                key_drivers.append("Monday effect: Watch for weekend gaps")
            
            # ========== CONTEXT KEYWORDS ==========
            context_lower = market_context.lower() if market_context else ""
            
            bullish_keywords = ["rally", "breakout", "bullish", "upgrade", "beat", "strong", "growth", "surge"]
            bearish_keywords = ["crash", "breakdown", "bearish", "downgrade", "miss", "weak", "decline", "plunge"]
            
            bullish_count = sum(1 for kw in bullish_keywords if kw in context_lower)
            bearish_count = sum(1 for kw in bearish_keywords if kw in context_lower)
            
            if bullish_count > 0:
                sentiment_score += 0.1 * bullish_count
                key_drivers.append(f"Bullish keywords detected ({bullish_count})")
            
            if bearish_count > 0:
                sentiment_score -= 0.1 * bearish_count
                key_drivers.append(f"Bearish keywords detected ({bearish_count})")
            
            # ========== FINAL CLASSIFICATION ==========
            sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1, 1]
            
            if sentiment_score >= 0.3:
                sentiment_label = "Bullish"
            elif sentiment_score <= -0.3:
                sentiment_label = "Bearish"
            else:
                sentiment_label = "Neutral"
            
            # Build summary
            if not key_drivers:
                key_drivers = ["No strong signals detected"]
            
            summary = f"Technical sentiment for {symbol}: {sentiment_label} (score: {sentiment_score:.2f}). "
            summary += "Factors: " + ", ".join(key_drivers[:3])
            
            return {
                "sentiment_score": round(sentiment_score, 2),
                "sentiment_label": sentiment_label,
                "key_drivers": key_drivers[:5],
                "summary": summary,
                "interpretation_mode": interpretation,
                "is_veto": sentiment_score <= -0.5,  # Strong negative = veto power
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "key_drivers": ["Error processing sentiment"],
                "summary": "Could not analyze sentiment.",
                "interpretation_mode": interpretation,
                "is_veto": False,
            }


# Backward compatibility alias
NewsAnalystAgentLocal = TechnicalSentimentAnalystLocal


def get_technical_sentiment_analyst_local(model_name: str = None) -> TechnicalSentimentAnalystLocal:
    """Factory function for TechnicalSentimentAnalystLocal."""
    return TechnicalSentimentAnalystLocal(model_name=model_name)


# Backward compatibility
def get_news_analyst_local(model_name: str = None) -> TechnicalSentimentAnalystLocal:
    """Deprecated: Use get_technical_sentiment_analyst_local instead."""
    return TechnicalSentimentAnalystLocal(model_name=model_name)

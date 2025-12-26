"""
News Analyst Agent (Emily) - Analyzes market sentiment from text.
"""

import logging
from typing import Dict, Any, Optional

from src.agents.specialists.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class NewsAnalystAgent(BaseAgent):
    """Specialist agent for news and sentiment analysis (Emily)."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            prompt_file="news_analyst.yaml",
            api_key=api_key,
        )
    
    def analyze(self, news_text: str, current_date: str, symbol: str) -> Dict[str, Any]:
        """
        Analyze sentiment from news text.
        
        Args:
            news_text: Raw text of news/headlines
            current_date: Today's date string
            symbol: Ticker symbol (e.g. BTC, AAPL)
        """
        # Build context with fallbacks to avoid empty content
        news_text = news_text or "No specific news available."
        current_date = current_date or "Today"
        symbol = symbol or "Unknown"
        
        context = f"""
        MARKET_CONTEXT: {news_text}
        CURRENT_DATE: {current_date}
        ASSET: {symbol}
        """
        
        # Validate context is not effectively empty
        if not context.strip() or len(context.strip()) < 20:
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "key_drivers": ["No data available"],
                "summary": "Insufficient data for sentiment analysis."
            }
        
        try:
            # Call Gemini with text only
            response = self.model.generate_content([context])
            return self._parse_response(response.text)
        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "key_drivers": ["Error processing news"],
                "summary": "Could not analyze news."
            }

    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse News Analyst response."""
        result = {
            "sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "key_drivers": [],
            "summary": ""
        }
        
        lines = raw_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("SENTIMENT_SCORE:"):
                try:
                    result["sentiment_score"] = float(line.replace("SENTIMENT_SCORE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("SENTIMENT_LABEL:"):
                result["sentiment_label"] = line.replace("SENTIMENT_LABEL:", "").strip()
            elif line.startswith("KEY_DRIVERS:"):
                drivers_str = line.replace("KEY_DRIVERS:", "").strip()
                result["key_drivers"] = [d.strip() for d in drivers_str.split(",")]
            elif line.startswith("SUMMARY:"):
                result["summary"] = line.replace("SUMMARY:", "").strip()
        
        return result

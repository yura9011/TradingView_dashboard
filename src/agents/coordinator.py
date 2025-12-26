"""
Coordinator Agent (Otto) - Orchestrates specialist agents (Bob, Emily, Dave) and synthesizes results.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from src.agents.specialists.pattern_detector import PatternDetectorAgent
from src.agents.specialists.trend_analyst import TrendAnalystAgent
from src.agents.specialists.levels_calculator import LevelsCalculatorAgent
from src.agents.specialists.risk_manager import RiskManagerAgent
from src.agents.specialists.news_analyst import NewsAnalystAgent
from src.agents.specialists.base_agent import AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class CoordinatedAnalysis:
    """Result from coordinated multi-agent analysis (QuantAgents Standard)."""
    
    # Final decision (Otto)
    signal_type: str  # candidate, pending, not_candidate
    overall_confidence: float
    
    # Pattern data (Pattern Detector)
    pattern: str
    pattern_confidence: float
    pattern_box: Optional[tuple]
    
    # Trend data (Bob/Trend Analyst)
    trend: str
    trend_strength: str  # Restored
    phase: str
    wave: Optional[str]  # Restored
    
    # Levels data (Levels Calculator) - Restored for compatibility
    support: Optional[float]
    resistance: Optional[float]
    fibonacci: Optional[str]
    key_level: Optional[float]
    
    # Risk data (Dave)
    risk_assessment: str
    stop_loss: str
    position_size: str
    
    # News data (Emily)
    news_sentiment: float
    news_label: str
    
    # Summary
    summary: str
    detailed_reasoning: Optional[str]  # Full JSON
    
    # Raw responses
    raw_pattern: Dict[str, Any]
    raw_trend: Dict[str, Any]
    raw_levels: Dict[str, Any]
    raw_risk: Dict[str, Any]
    raw_news: Dict[str, Any]


class CoordinatorAgent:
    """Orchestrates specialist agents (Otto, Bob, Dave, Emily) for comprehensive analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize coordinator with all specialist agents."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Initialize specialists
        logger.info("Initializing QuantAgents Team...")
        self.pattern_detector = PatternDetectorAgent(api_key=self.api_key)
        self.trend_analyst = TrendAnalystAgent(api_key=self.api_key)
        self.levels_calculator = LevelsCalculatorAgent(api_key=self.api_key)
        self.risk_manager = RiskManagerAgent(api_key=self.api_key)
        self.news_analyst = NewsAnalystAgent(api_key=self.api_key)
        
        # Market data client
        from src.screener.client import ScreenerClient
        from src.models import Market
        self.screener = ScreenerClient(market=Market.AMERICA)
        
        # Load Otto's prompt (Coordinator)
        self.model_name = "models/gemini-2.0-flash-exp" # Using Flash for synthesis
        import google.generativeai as genai
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self._load_prompt("coordinator.yaml")
            )
        
        logger.info("All agents initialized.")

    def _load_prompt(self, prompt_file: str) -> str:
        """Load prompt from YAML."""
        # coordinator.py is in src/agents/, prompts is in root/prompts/
        # Path: src/agents/coordinator.py -> ../.. -> root -> prompts/
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / prompt_file
        import yaml
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f).get("system_prompt", "")
        return ""

    def _get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time market data via Screener/MCP."""
        if not symbol:
            return {}
            
        # Try fetching from America first, then Crypto
        data = self.screener.get_symbol_data(symbol)
        if not data:
            # Quick hack: try switching market context if not found
            # Ideally ScreenerClient should handle this or we iterate markets
            pass
            
        return data or {}

    def analyze(self, image_path: str, symbol: str = "") -> CoordinatedAnalysis:
        """Run full multi-agent analysis pipeline."""
        logger.info(f"Starting QuantAgents analysis for {symbol or 'chart'}")
        
        # 0. Gather Context (Real-time Data)
        now = datetime.datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        day_of_week = now.strftime("%A")
        
        market_data = self._get_market_context(symbol)
        current_price = market_data.get('close', 0.0)
        atr_value = market_data.get('ATR', 0.0)
        volume = market_data.get('volume', 0)
        sector = market_data.get('sector', 'Unknown')
        
        context_str = f"""
        ANALYSIS CONTEXT:
        - Symbol: {symbol}
        - Date: {current_date} ({day_of_week})
        - Sector: {sector}
        - Current Price: {current_price}
        - ATR (Volatility): {atr_value}
        - Volume: {volume}
        """
        
        logger.info(f"Context gathered: {context_str.strip()}")
        
        # --- PARALLEL EXECUTION SIMULATION (Sequential for now) ---
        
        # 1. Visual Agents (Eyes)
        logger.info("🔍 Step 1: Technical Analysis (Pattern, Trend, Levels)...")
        pattern_result = self.pattern_detector.analyze(image_path, context_str)
        trend_result = self.trend_analyst.analyze(image_path, context_str)
        levels_result = self.levels_calculator.analyze(image_path, context_str)
        
        # 2. News Analysis (Emily)
        logger.info("📰 Step 2: News Analysis (Emily)...")
        # In a real system, we'd fetch news here using an MCP tool
        news_context = f"Market analysis for {symbol} on {current_date}. Technical indicators: RSI={market_data.get('RSI', 'N/A')}, MACD={market_data.get('MACD.macd', 'N/A')}."
        news_result = self.news_analyst.analyze(news_context, current_date, symbol)
        
        # 3. Risk Analysis (Dave)
        logger.info("🛡️ Step 3: Risk Analysis (Dave)...")
        
        try:
            # Fallback for price/atr if not found in screener
            if current_price == 0:
                current_price = levels_result.parsed.get("key_level", 0.0) or 100.0
            
            if atr_value == 0:
                support = levels_result.parsed.get("support", 0.0)
                resistance = levels_result.parsed.get("resistance", 0.0)
                if support and resistance:
                    atr_value = (resistance - support) / 10 
                else:
                    atr_value = current_price * 0.02 
                
            risk_result = self.risk_manager.analyze(
                pattern_data=pattern_result.parsed,
                trend_data=trend_result.parsed,
                current_price=current_price,
                atr_value=atr_value
            )
        except Exception as e:
            logger.warning(f"Risk analysis fallback: {e}")
            risk_result = {"risk_assessment": "UNKNOWN", "stop_loss": "N/A", "position_size": "0%"}

        # 4. Final Synthesis (Otto)
        logger.info("🧠 Step 4: Final Decision (Otto)...")
        analysis = self._synthesize(
            pattern_result,
            trend_result,
            levels_result,
            risk_result,
            news_result,
            symbol,
            context_str
        )
        
        logger.info(f"Otto's Decision: {analysis.signal_type}")
        return analysis

    def _synthesize(
        self,
        pattern: AgentResponse,
        trend: AgentResponse,
        levels: AgentResponse,
        risk: Dict[str, Any],
        news: Dict[str, Any],
        symbol: str,
        context_str: str,
    ) -> CoordinatedAnalysis:
        """Synthesize findings using Otto's prompt."""
        
        input_text = f"""
        {context_str}
        
        === PATTERN ANALYSIS ===
        {json.dumps(pattern.parsed, indent=2)}
        
        === TREND ANALYSIS ===
        {json.dumps(trend.parsed, indent=2)}
        
        === LEVELS ANALYSIS ===
        {json.dumps(levels.parsed, indent=2)}
        
        === RISK ANALYSIS (Dave) ===
        {json.dumps(risk, indent=2)}
        
        === NEWS ANALYSIS (Emily) ===
        {json.dumps(news, indent=2)}
        
        SYMBOL: {symbol}
        """
        
        try:
            response = self.model.generate_content([input_text])
            raw_text = response.text
            parsed = self._parse_otto_response(raw_text)
        except Exception as e:
            logger.error(f"Otto synthesis failed: {e}")
            parsed = {}
            raw_text = str(e)

        # Create the result object
        return CoordinatedAnalysis(
            signal_type=parsed.get("signal_type", "not_candidate"),
            overall_confidence=parsed.get("overall_confidence", 0.0),
            
            pattern=parsed.get("pattern", "none"),
            pattern_confidence=parsed.get("pattern_confidence", 0.0),
            pattern_box=pattern.parsed.get("pattern_box"),
            
            trend=parsed.get("trend", "unknown"),
            trend_strength=trend.parsed.get("strength", "unknown"),
            phase=parsed.get("phase", "unknown"),
            wave=trend.parsed.get("wave"),
            
            support=levels.parsed.get("support"),
            resistance=levels.parsed.get("resistance"),
            fibonacci=levels.parsed.get("fibonacci"),
            key_level=levels.parsed.get("key_level"),
            
            risk_assessment=parsed.get("risk_assessment", "UNKNOWN"),
            stop_loss=parsed.get("stop_loss", "N/A"),
            position_size=parsed.get("position_size", "0%"),
            
            news_sentiment=news.get("sentiment_score", 0.0),
            news_label=news.get("sentiment_label", "Neutral"),
            
            summary=parsed.get("summary", "Analysis failed."),
            detailed_reasoning=json.dumps(parsed),
            
            raw_pattern=pattern.parsed,
            raw_trend=trend.parsed,
            raw_levels=levels.parsed,
            raw_risk=risk,
            raw_news=news
        )

    def _parse_otto_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse Otto's response format."""
        result = {}
        lines = raw_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if ": " in line:
                key, val = line.split(": ", 1)
                key = key.lower().replace(" ", "_")
                
                if key == "overall_confidence" or key == "pattern_confidence":
                    try:
                        result[key] = float(val.replace("%", "")) / 100 if "%" in val else float(val)
                    except:
                        result[key] = 0.0
                else:
                    result[key] = val
            elif "SUMMARY:" in line:
                 pass
        
        if "SUMMARY:" in raw_text:
            parts = raw_text.split("SUMMARY:")
            if len(parts) > 1:
                result["summary"] = parts[1].strip()
        
        return result


def get_coordinator(api_key: str = None) -> CoordinatorAgent:
    """Factory function for CoordinatorAgent."""
    return CoordinatorAgent(api_key=api_key)

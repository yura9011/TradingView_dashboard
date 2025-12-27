"""
Coordinator Agent (Local) - Orchestrates specialist agents using Qwen2-VL.
Enhanced with Market Context, Risk (Dave), and News (Emily) integration.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from src.agents.specialists.pattern_detector_local import PatternDetectorAgentLocal
from src.agents.specialists.trend_analyst_local import TrendAnalystAgentLocal
from src.agents.specialists.levels_calculator_local import LevelsCalculatorAgentLocal
from src.agents.specialists.risk_manager_local import RiskManagerAgentLocal
from src.agents.specialists.news_analyst_local import NewsAnalystAgentLocal
from src.agents.specialists.base_agent_local import AgentResponse
from src.screener.client import ScreenerClient
from src.models import Market

# Optional YOLO import (try/except for backwards compatibility)
try:
    from src.agents.specialists.pattern_detector_yolo import YOLOPatternDetectorAgent
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default model (2B for RTX 3070 / 8GB VRAM compatibility)
DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


@dataclass
class CoordinatedAnalysis:
    """Result from coordinated multi-agent analysis (Enhanced QuantAgents)."""
    
    # Final decision
    signal_type: str  # candidate, pending, not_candidate
    overall_confidence: float
    
    # Pattern data
    pattern: str
    pattern_confidence: float
    pattern_box: Optional[tuple]
    pattern_description: Optional[str]
    
    # Trend data
    trend: str
    trend_strength: str
    phase: str
    wave: Optional[str]
    trend_description: Optional[str]
    
    # Levels data
    support: Optional[float]
    resistance: Optional[float]
    support_secondary: Optional[float]
    resistance_secondary: Optional[float]
    fibonacci: Optional[str]
    key_level: Optional[float]
    levels_description: Optional[str]
    
    # Risk data (Dave)
    risk_assessment: str
    stop_loss: str
    stop_loss_price: float
    position_size: str
    
    # Sentiment data (Emily)
    sentiment_score: float
    sentiment_label: str
    
    # Market context
    current_price: float
    atr_value: float
    atr_percent: float
    rvol: float
    sector: str
    
    # Summary
    summary: str
    detailed_reasoning: Optional[str]
    veto_reason: Optional[str]  # If trade was vetoed, why?
    
    # Raw responses
    raw_pattern: Dict[str, Any]
    raw_trend: Dict[str, Any]
    raw_levels: Dict[str, Any]
    raw_risk: Dict[str, Any]
    raw_news: Dict[str, Any]


class CoordinatorAgentLocal:
    """Orchestrates specialist agents using local Qwen2-VL model.
    
    Enhanced flow:
    1. Gather Context (MCP/Screener)
    2. Visual Analysis (Pattern, Trend, Levels) 
    3. News/Sentiment (Emily)
    4. Risk Analysis (Dave) - with VETO power
    5. Final Synthesis (Otto)
    """
    
    def __init__(self, model_name: str = None, market: Market = Market.AMERICA, use_yolo: bool = True):
        """Initialize coordinator with all specialist agents.
        
        Args:
            model_name: HuggingFace model name (default: Qwen/Qwen2-VL-2B-Instruct)
            market: Target market (AMERICA, CRYPTO, FOREX)
            use_yolo: If True, use YOLOv8 for pattern detection (more accurate)
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.market = market
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        
        logger.info("Initializing QuantAgents-Local Team...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Market: {market.value}")
        logger.info(f"YOLO Pattern Detection: {'ENABLED' if self.use_yolo else 'DISABLED (using VLM)'}")
        
        # Pattern detector: Use YOLO if available and enabled, else VLM
        if self.use_yolo:
            self.pattern_detector = YOLOPatternDetectorAgent(confidence_threshold=0.15)
            logger.info("  📊 Pattern Detector: YOLOv8 (foduucom/stockmarket-pattern-detection)")
        else:
            self.pattern_detector = PatternDetectorAgentLocal(model_name=self.model_name)
            logger.info("  📊 Pattern Detector: Qwen2-VL")
        
        # Visual agents (use Qwen2-VL for trend and levels)
        self.trend_analyst = TrendAnalystAgentLocal(model_name=self.model_name)
        self.levels_calculator = LevelsCalculatorAgentLocal(model_name=self.model_name)
        
        # Non-visual agents (rule-based for speed)
        self.risk_manager = RiskManagerAgentLocal(model_name=self.model_name)
        self.news_analyst = NewsAnalystAgentLocal(model_name=self.model_name)
        
        # Market data client (configurable market)
        self.screener = ScreenerClient(market=market)
        
        logger.info("All local specialist agents initialized")
    
    def _get_market_context(self, symbol: str) -> Dict[str, Any]:
        """Fetch real-time market data via Screener.
        
        Returns:
            Dict with price, ATR, volume, sector, RSI, MACD
        """
        if not symbol:
            return {}
        
        try:
            data = self.screener.get_symbol_data(symbol)
            if data:
                # Calculate RVOL if we have volume data
                current_vol = data.get('volume', 0)
                avg_vol = data.get('average_volume_10d_calc', current_vol)
                rvol = current_vol / avg_vol if avg_vol > 0 else 1.0
                data['rvol'] = rvol
                return data
        except Exception as e:
            logger.warning(f"Could not fetch market data for {symbol}: {e}")
        
        return {}
    
    def _build_enhanced_summary(
        self,
        signal_type: str,
        pattern_name: str,
        pattern_conf: float,
        has_pattern: bool,
        trend_dir: str,
        trend_strength: str,
        phase: str,
        risk_assessment: str,
        sentiment_label: str,
        stop_loss: str,
        position_size: str,
        veto_reason: Optional[str],
        support: Optional[float],
        resistance: Optional[float],
    ) -> str:
        """Build an enhanced, detailed summary combining all analysis."""
        
        parts = []
        
        # Signal type and confidence
        if signal_type == "candidate":
            parts.append(f"✅ CANDIDATE: Ready to trade.")
        elif signal_type == "pending":
            parts.append(f"⏳ PENDING: Monitor for confirmation.")
        else:
            parts.append(f"❌ NOT CANDIDATE: Skip this trade.")
        
        # Veto reason (if any)
        if veto_reason:
            parts.append(f"[VETO: {veto_reason}]")
        
        # Pattern
        if has_pattern and pattern_name != "none":
            parts.append(f"Pattern: {pattern_name.title()} ({pattern_conf:.0%} conf).")
        else:
            parts.append("No clear pattern.")
        
        # Trend
        if trend_dir != "unknown":
            parts.append(f"Trend: {trend_dir} ({trend_strength}), Phase: {phase}.")
        
        # Levels
        if support and resistance:
            parts.append(f"Range: ${support:,.0f} - ${resistance:,.0f}.")
        
        # Risk
        parts.append(f"Risk: {risk_assessment}, SL: {stop_loss}, Size: {position_size}.")
        
        # Sentiment
        parts.append(f"Sentiment: {sentiment_label}.")
        
        return " ".join(parts)
    
    def analyze(self, image_path: str, symbol: str = "", additional_context: str = "") -> CoordinatedAnalysis:
        """Run full multi-agent analysis pipeline.
        
        Args:
            image_path: Path to chart image
            symbol: Optional symbol for context
            additional_context: Additional context like price range from OCR
            
        Returns:
            CoordinatedAnalysis with all findings
        """
        logger.info(f"Starting QuantAgents-Local analysis for {symbol or 'chart'}")
        
        # ========== STEP 0: GATHER CONTEXT ==========
        now = datetime.datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        day_of_week = now.strftime("%A")
        
        logger.info("📊 Step 0: Gathering market context...")
        market_data = self._get_market_context(symbol)
        
        current_price = market_data.get('close', 0.0)
        atr_value = market_data.get('ATR', 0.0)
        volume = market_data.get('volume', 0)
        sector = market_data.get('sector', 'Unknown')
        rvol = market_data.get('rvol', 1.0)
        rsi = market_data.get('RSI')
        macd = market_data.get('MACD.macd')
        
        context_str = f"""
        ANALYSIS CONTEXT:
        - Symbol: {symbol}
        - Date: {current_date} ({day_of_week})
        - Sector: {sector}
        - Current Price: ${current_price:,.2f}
        - ATR (Volatility): {atr_value:.2f}
        - RVOL: {rvol:.2f}
        - Volume: {volume:,}
        """
        
        if additional_context:
            context_str += f"\n        - Additional: {additional_context}"
        
        logger.info(f"Context gathered: Price=${current_price:.2f}, ATR={atr_value:.2f}, RVOL={rvol:.2f}")
        
        # Build full context for visual agents
        full_context = context_str.strip()
        if additional_context:
            full_context += f" | {additional_context}"
        
        # ========== STEP 1: VISUAL ANALYSIS ==========
        logger.info("🔍 Step 1/4: Pattern Detection (local)...")
        pattern_result = self.pattern_detector.analyze(image_path, full_context)
        
        logger.info("📈 Step 2/4: Trend Analysis (local)...")
        trend_result = self.trend_analyst.analyze(image_path, full_context)
        
        logger.info("📊 Step 3/4: Levels Calculation (local)...")
        levels_result = self.levels_calculator.analyze(image_path, full_context)
        
        # ========== STEP 2: NEWS/SENTIMENT (Emily) ==========
        logger.info("📰 Step 4a: News/Sentiment Analysis (Emily)...")
        news_result = self.news_analyst.analyze(
            market_context=f"Technical analysis for {symbol}. RSI: {rsi}, MACD: {macd}",
            current_date=f"{current_date} ({day_of_week})",
            symbol=symbol,
            rsi=rsi,
            macd=macd,
        )
        
        # ========== STEP 3: RISK ANALYSIS (Dave) ==========
        logger.info("🛡️ Step 4b: Risk Analysis (Dave)...")
        
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
            atr_value=atr_value,
            rvol=rvol,
        )
        
        # ========== STEP 4: SYNTHESIS (Otto) ==========
        logger.info("🧠 Step 5: Final Synthesis (Otto)...")
        analysis = self._synthesize(
            pattern=pattern_result,
            trend=trend_result,
            levels=levels_result,
            risk=risk_result,
            news=news_result,
            symbol=symbol,
            current_price=current_price,
            atr_value=atr_value,
            rvol=rvol,
            sector=sector,
        )
        
        logger.info(f"Analysis complete: {analysis.signal_type} (confidence: {analysis.overall_confidence:.0%})")
        if analysis.veto_reason:
            logger.warning(f"Trade VETOED: {analysis.veto_reason}")
        
        return analysis
    
    def _synthesize(
        self,
        pattern: AgentResponse,
        trend: AgentResponse,
        levels: AgentResponse,
        risk: Dict[str, Any],
        news: Dict[str, Any],
        symbol: str,
        current_price: float,
        atr_value: float,
        rvol: float,
        sector: str,
    ) -> CoordinatedAnalysis:
        """Synthesize findings from all agents into final decision."""
        
        p = pattern.parsed
        t = trend.parsed
        l = levels.parsed
        
        pattern_conf = p.get("confidence", 0.0)
        has_pattern = p.get("pattern", "none") != "none"
        trend_dir = t.get("trend", "unknown")
        trend_strength = t.get("strength", "unknown")
        phase = t.get("phase", "unclear")
        
        pattern_name = p.get("pattern", "none")
        risk_assessment = risk.get("risk_assessment", "UNKNOWN")
        sentiment_label = news.get("sentiment_label", "Neutral")
        sentiment_score = news.get("sentiment_score", 0.0)
        
        # ========== DECISION LOGIC ==========
        
        veto_reason = None
        
        # VETO CHECK 1: Dave's Risk Assessment
        if risk_assessment == "DANGEROUS":
            signal_type = "not_candidate"
            overall_confidence = 0.1
            veto_reason = f"RISK VETO: {risk.get('reasoning', 'Dangerous volatility')}"
        
        # VETO CHECK 2: Emily's Negative Sentiment (for Long trades)
        elif news.get("is_veto", False):
            bullish_patterns = ["double bottom", "inverse head", "bullish engulfing", "hammer"]
            is_bullish_pattern = any(bp in pattern_name.lower() for bp in bullish_patterns) if pattern_name else False
            
            if is_bullish_pattern or trend_dir == "up":
                signal_type = "not_candidate"
                overall_confidence = 0.1
                veto_reason = f"NEWS VETO: Strong negative sentiment ({sentiment_score:.2f}) against bullish setup"
        
        # VETO CHECK 3: Low RVOL on Breakout
        elif "breakout" in pattern_name.lower() and rvol < 1.5:
            signal_type = "not_candidate"
            overall_confidence = 0.2
            veto_reason = f"FAKEOUT ALERT: Breakout with low volume (RVOL={rvol:.2f})"
        
        # Normal decision logic
        else:
            if pattern_conf >= 0.7 and has_pattern and risk_assessment != "CAUTION":
                signal_type = "candidate"
                overall_confidence = pattern_conf
            elif pattern_conf >= 0.4 or (trend_strength == "strong" and trend_dir != "sideways"):
                signal_type = "pending"
                overall_confidence = max(pattern_conf, 0.5) if has_pattern else 0.4
            else:
                signal_type = "not_candidate"
                overall_confidence = pattern_conf
            
            # Adjust confidence for risk
            if risk_assessment == "CAUTION" and signal_type == "candidate":
                signal_type = "pending"
                overall_confidence *= 0.8
        
        # Build summary
        summary = self._build_enhanced_summary(
            signal_type=signal_type,
            pattern_name=pattern_name,
            pattern_conf=pattern_conf,
            has_pattern=has_pattern,
            trend_dir=trend_dir,
            trend_strength=trend_strength,
            phase=phase,
            risk_assessment=risk_assessment,
            sentiment_label=sentiment_label,
            stop_loss=risk.get("stop_loss", "N/A"),
            position_size=risk.get("position_size", "0%"),
            veto_reason=veto_reason,
            support=l.get("support"),
            resistance=l.get("resistance"),
        )
        
        # Build detailed reasoning
        atr_percent = risk.get("atr_percent", 0.0)
        
        detailed_reasoning = json.dumps({
            "pattern": {
                "name": pattern_name,
                "confidence": pattern_conf,
                "description": p.get("description", ""),
                "box": p.get("pattern_box"),
            },
            "trend": {
                "direction": trend_dir,
                "strength": trend_strength,
                "phase": phase,
                "wave": t.get("wave"),
            },
            "levels": {
                "support": l.get("support"),
                "resistance": l.get("resistance"),
                "key_level": l.get("key_level"),
                "fibonacci": l.get("fibonacci"),
            },
            "risk": risk,
            "sentiment": news,
            "market_context": {
                "price": current_price,
                "atr": atr_value,
                "atr_percent": atr_percent,
                "rvol": rvol,
                "sector": sector,
            },
            "veto_reason": veto_reason,
        }, ensure_ascii=False)
        
        return CoordinatedAnalysis(
            signal_type=signal_type,
            overall_confidence=overall_confidence,
            pattern=pattern_name,
            pattern_confidence=pattern_conf,
            pattern_box=p.get("pattern_box"),
            pattern_description=p.get("description"),
            trend=trend_dir,
            trend_strength=trend_strength,
            phase=phase,
            wave=t.get("wave"),
            trend_description=t.get("description"),
            support=l.get("support"),
            resistance=l.get("resistance"),
            support_secondary=l.get("support_secondary"),
            resistance_secondary=l.get("resistance_secondary"),
            fibonacci=l.get("fibonacci"),
            key_level=l.get("key_level"),
            levels_description=l.get("description"),
            risk_assessment=risk_assessment,
            stop_loss=risk.get("stop_loss", "N/A"),
            stop_loss_price=risk.get("stop_loss_price", 0.0),
            position_size=risk.get("position_size", "0%"),
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            current_price=current_price,
            atr_value=atr_value,
            atr_percent=risk.get("atr_percent", 0.0),
            rvol=rvol,
            sector=sector,
            summary=summary,
            detailed_reasoning=detailed_reasoning,
            veto_reason=veto_reason,
            raw_pattern=p,
            raw_trend=t,
            raw_levels=l,
            raw_risk=risk,
            raw_news=news,
        )


def get_coordinator_local(model_name: str = None, market: Market = Market.AMERICA, use_yolo: bool = True) -> CoordinatorAgentLocal:
    """Factory function for CoordinatorAgentLocal.
    
    Args:
        model_name: HuggingFace model name
        market: Target market (AMERICA, CRYPTO, FOREX)
        use_yolo: If True, use YOLOv8 for pattern detection (default: True)
    """
    return CoordinatorAgentLocal(model_name=model_name, market=market, use_yolo=use_yolo)


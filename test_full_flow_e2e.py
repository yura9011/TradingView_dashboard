"""
Full End-to-End Flow Test - Simulates Client Experience
Uses moondream2 (CPU) to run complete pipeline:
1. Capture chart from TradingView
2. Run all agents (Pattern, Trend, Levels, Risk, Sentiment)
3. Coordinator synthesis with veto system
4. Save signal to database
5. Generate report

Run: python test_full_flow_e2e.py --symbol AAPL
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# DATACLASSES
# ============================================================

@dataclass
class AgentResponse:
    """Mock of the agent response structure."""
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class FullAnalysisResult:
    """Complete analysis result matching CoordinatedAnalysis."""
    signal_type: str = "not_candidate"
    overall_confidence: float = 0.0
    pattern: str = "none"
    pattern_confidence: float = 0.0
    pattern_box: Optional[str] = None
    trend: str = "unknown"
    trend_strength: str = "unknown"
    phase: str = "unclear"
    wave: Optional[str] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    fibonacci: Optional[str] = None
    key_level: Optional[float] = None
    risk_assessment: str = "UNKNOWN"
    stop_loss: str = "N/A"
    position_size: str = "0%"
    sentiment_score: float = 0.0
    sentiment_label: str = "Neutral"
    veto_reason: Optional[str] = None
    summary: str = ""
    detailed_reasoning: str = ""


# ============================================================
# MOONDREAM2 VISION MODEL
# ============================================================

class MoondreamVisionModel:
    """Moondream2 model for CPU inference."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load()
    
    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Loading moondream2 model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model.eval()
        logger.info("Model loaded!")
    
    def analyze(self, image_path: str, prompt: str) -> str:
        from PIL import Image
        
        image = Image.open(image_path).convert("RGB")
        
        # Resize for speed
        max_size = 384
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size)
        
        enc_image = self.model.encode_image(image)
        
        start = time.time()
        response = self.model.answer_question(enc_image, prompt, self.tokenizer)
        elapsed = time.time() - start
        
        logger.info(f"Inference time: {elapsed:.1f}s")
        return response


# ============================================================
# FULL PIPELINE
# ============================================================

class FullFlowOrchestrator:
    """Complete pipeline orchestrator using moondream2."""
    
    def __init__(self):
        logger.info("Initializing Full Flow Orchestrator...")
        self.vision_model = MoondreamVisionModel()
        
        # Import actual components
        from src.screener.client import ScreenerClient
        from src.models import Market
        from src.database import get_signal_repository
        from src.visual import get_report_generator
        
        self.screener = ScreenerClient(market=Market.AMERICA)
        self.db = get_signal_repository()
        self.report_gen = get_report_generator()
        
        logger.info("Orchestrator ready!")
    
    def capture_chart(self, symbol: str, exchange: str = "") -> str:
        """Step 1: Capture chart from TradingView."""
        logger.info(f"📸 Capturing chart for {symbol} (daily, 1 month)...")
        
        from src.screener.chart_capture import get_chart_capture
        
        capture = get_chart_capture()
        chart_path, price_range = capture.capture_sync(
            symbol=symbol,
            exchange=exchange,
            interval="D",
            range_months=1,
        )
        
        logger.info(f"   Chart saved: {chart_path}")
        return str(chart_path)
    
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data from screener."""
        logger.info(f"📊 Fetching market data for {symbol}...")
        
        try:
            data = self.screener.get_symbol_data(symbol)
            if data:
                price = data.get('close', 0)
                atr = data.get('average_true_range', price * 0.02)
                vol = data.get('volume', 0)
                avg_vol = data.get('average_volume_10d_calc', vol)
                rvol = vol / avg_vol if avg_vol > 0 else 1.0
                
                logger.info(f"   Price: ${price:.2f}, ATR: {atr:.2f}, RVOL: {rvol:.2f}")
                
                return {
                    "close": price,
                    "atr": atr,
                    "rvol": rvol,
                    "rsi": data.get('RSI'),
                    "macd": data.get('MACD.macd'),
                    "sector": data.get('sector', 'Unknown'),
                    "volume": vol,
                }
        except Exception as e:
            logger.warning(f"   Screener error: {e}, using defaults")
        
        return {"close": 100.0, "atr": 2.0, "rvol": 1.0, "rsi": 50, "macd": 0, "sector": "Unknown"}
    
    def analyze_pattern(self, image_path: str, symbol: str, price: float) -> AgentResponse:
        """Step 2a: Pattern detection."""
        logger.info("🔍 Pattern Detector analyzing...")
        
        prompt = f"""Look at this stock chart for {symbol} (price ${price:.2f}).
Identify any chart pattern you see. Answer in this format:
PATTERN: [name or none]
CONFIDENCE: [0.0-1.0]
DESCRIPTION: [one sentence]"""
        
        response = self.vision_model.analyze(image_path, prompt)
        
        # Parse
        parsed = {"pattern": "none", "confidence": 0.0, "description": ""}
        for line in response.split('\n'):
            if "pattern:" in line.lower():
                parsed["pattern"] = line.split(":", 1)[1].strip()
            elif "confidence:" in line.lower():
                try:
                    parsed["confidence"] = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif "description:" in line.lower():
                parsed["description"] = line.split(":", 1)[1].strip()
        
        logger.info(f"   Pattern: {parsed['pattern']} ({parsed['confidence']:.0%})")
        return AgentResponse(raw_text=response, parsed=parsed)
    
    def analyze_trend(self, image_path: str, symbol: str) -> AgentResponse:
        """Step 2b: Trend analysis."""
        logger.info("📈 Trend Analyst analyzing...")
        
        prompt = f"""Look at this stock chart for {symbol}.
What is the overall trend? Answer:
TREND: [up/down/sideways]
STRENGTH: [strong/moderate/weak]
PHASE: [accumulation/markup/distribution/markdown]"""
        
        response = self.vision_model.analyze(image_path, prompt)
        
        parsed = {"trend": "unknown", "strength": "unknown", "phase": "unclear"}
        for line in response.split('\n'):
            line_lower = line.lower()
            if "trend:" in line_lower:
                parsed["trend"] = line.split(":", 1)[1].strip().lower()
            elif "strength:" in line_lower:
                parsed["strength"] = line.split(":", 1)[1].strip().lower()
            elif "phase:" in line_lower:
                parsed["phase"] = line.split(":", 1)[1].strip().lower()
        
        logger.info(f"   Trend: {parsed['trend']} ({parsed['strength']})")
        return AgentResponse(raw_text=response, parsed=parsed)
    
    def analyze_levels(self, image_path: str, symbol: str, price: float) -> AgentResponse:
        """Step 2c: Levels calculation."""
        logger.info("📏 Levels Calculator analyzing...")
        
        prompt = f"""Look at this stock chart for {symbol}, current price ${price:.2f}.
Identify key price levels. Answer:
SUPPORT: [price]
RESISTANCE: [price]
KEY_LEVEL: [most important price]"""
        
        response = self.vision_model.analyze(image_path, prompt)
        
        parsed = {"support": None, "resistance": None, "key_level": None}
        for line in response.split('\n'):
            line_lower = line.lower()
            try:
                if "support:" in line_lower:
                    val = line.split(":", 1)[1].strip().replace('$', '').replace(',', '')
                    parsed["support"] = float(val)
                elif "resistance:" in line_lower:
                    val = line.split(":", 1)[1].strip().replace('$', '').replace(',', '')
                    parsed["resistance"] = float(val)
                elif "key_level:" in line_lower:
                    val = line.split(":", 1)[1].strip().replace('$', '').replace(',', '')
                    parsed["key_level"] = float(val)
            except:
                pass
        
        logger.info(f"   Support: {parsed['support']}, Resistance: {parsed['resistance']}")
        return AgentResponse(raw_text=response, parsed=parsed)
    
    def analyze_risk(self, pattern: Dict, trend: Dict, price: float, atr: float, rvol: float) -> Dict[str, Any]:
        """Step 3: Risk analysis (Dave - rule based)."""
        logger.info("🛡️ Risk Manager (Dave) analyzing...")
        
        pattern_name = pattern.get("pattern", "none") or "none"
        atr_pct = (atr / price * 100) if price > 0 else 0
        
        if atr_pct > 5.0:
            risk = "DANGEROUS"
            size = "0%"
            reasoning = f"Extreme volatility: ATR {atr_pct:.1f}%"
        elif atr_pct > 3.0:
            risk = "CAUTION"
            size = "50%"
            reasoning = f"High volatility: ATR {atr_pct:.1f}%"
        else:
            risk = "SAFE"
            size = "100%"
            reasoning = f"Normal volatility: ATR {atr_pct:.1f}%"
        
        # RVOL fakeout check
        if "breakout" in pattern_name.lower() and rvol < 1.5:
            risk = "DANGEROUS"
            size = "0%"
            reasoning = f"FAKEOUT: Low volume breakout (RVOL={rvol:.2f})"
        
        stop = price - (atr * (3.0 if risk == "CAUTION" else 2.0))
        
        result = {
            "risk_assessment": risk,
            "stop_loss": f"${stop:.2f}",
            "position_size": size,
            "atr_percent": atr_pct,
            "rvol": rvol,
            "reasoning": reasoning,
        }
        
        logger.info(f"   Risk: {risk}, SL: ${stop:.2f}, Size: {size}")
        return result
    
    def analyze_sentiment(self, rsi: float, macd: float) -> Dict[str, Any]:
        """Step 4: Technical sentiment (Emily - rule based)."""
        logger.info("📰 Technical Sentiment (Emily) analyzing...")
        
        score = 0.0
        drivers = []
        
        if rsi is not None:
            if rsi < 30:
                score -= 0.3
                drivers.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                score += 0.3
                drivers.append(f"RSI strong ({rsi:.0f})")
            else:
                drivers.append(f"RSI neutral ({rsi:.0f})")
        
        if macd is not None:
            if macd > 0:
                score += 0.2
                drivers.append("MACD bullish")
            else:
                score -= 0.2
                drivers.append("MACD bearish")
        
        score = max(-1.0, min(1.0, score))
        
        if score >= 0.3:
            label = "Bullish"
        elif score <= -0.3:
            label = "Bearish"
        else:
            label = "Neutral"
        
        result = {
            "sentiment_score": score,
            "sentiment_label": label,
            "key_drivers": drivers,
            "is_veto": score <= -0.5,
        }
        
        logger.info(f"   Sentiment: {label} ({score:.2f})")
        return result
    
    def synthesize(self, pattern: AgentResponse, trend: AgentResponse, levels: AgentResponse,
                   risk: Dict, sentiment: Dict, symbol: str, market_data: Dict) -> FullAnalysisResult:
        """Step 5: Otto's synthesis with veto system."""
        logger.info("🧠 Coordinator (Otto) synthesizing...")
        
        p = pattern.parsed
        t = trend.parsed
        l = levels.parsed
        
        pattern_name = p.get("pattern", "none")
        pattern_conf = p.get("confidence", 0.0)
        has_pattern = pattern_name.lower() not in ["none", "n/a", ""]
        trend_dir = t.get("trend", "unknown")
        
        veto_reason = None
        
        # VETO 1: Risk
        if risk["risk_assessment"] == "DANGEROUS":
            signal_type = "not_candidate"
            confidence = 0.1
            veto_reason = f"RISK VETO: {risk['reasoning']}"
        # VETO 2: Sentiment
        elif sentiment["is_veto"] and (has_pattern or trend_dir == "up"):
            signal_type = "not_candidate"
            confidence = 0.1
            veto_reason = "SENTIMENT VETO: Negative against bullish"
        # Normal logic
        else:
            if pattern_conf >= 0.7 and has_pattern and risk["risk_assessment"] != "CAUTION":
                signal_type = "candidate"
                confidence = pattern_conf
            elif pattern_conf >= 0.4 or t.get("strength") == "strong":
                signal_type = "pending"
                confidence = max(pattern_conf, 0.4)
            else:
                signal_type = "not_candidate"
                confidence = pattern_conf
            
            if risk["risk_assessment"] == "CAUTION" and signal_type == "candidate":
                signal_type = "pending"
                confidence *= 0.8
        
        # Build summary
        if signal_type == "candidate":
            summary = f"✅ CANDIDATE: {pattern_name} with {confidence:.0%} confidence. Ready to trade."
        elif signal_type == "pending":
            summary = f"⏳ PENDING: Monitor {pattern_name} for confirmation."
        else:
            summary = f"❌ NOT_CANDIDATE: Skip this trade."
        
        if veto_reason:
            summary += f" [{veto_reason}]"
        
        price = market_data.get("close", 0)
        
        result = FullAnalysisResult(
            signal_type=signal_type,
            overall_confidence=confidence,
            pattern=pattern_name,
            pattern_confidence=pattern_conf,
            trend=trend_dir,
            trend_strength=t.get("strength", "unknown"),
            phase=t.get("phase", "unclear"),
            support=l.get("support"),
            resistance=l.get("resistance"),
            key_level=l.get("key_level") or price,
            risk_assessment=risk["risk_assessment"],
            stop_loss=risk["stop_loss"],
            position_size=risk["position_size"],
            sentiment_score=sentiment["sentiment_score"],
            sentiment_label=sentiment["sentiment_label"],
            veto_reason=veto_reason,
            summary=summary,
            detailed_reasoning=json.dumps({
                "pattern": p,
                "trend": t,
                "levels": l,
                "risk": risk,
                "sentiment": sentiment,
            }),
        )
        
        logger.info(f"   Decision: {signal_type.upper()}")
        return result
    
    def save_to_database(self, result: FullAnalysisResult, symbol: str, chart_path: str) -> int:
        """Step 6: Save to database."""
        logger.info("💾 Saving to database...")
        
        from src.models import Signal, SignalType, PatternType
        
        # Map pattern name to enum
        pattern_map = {
            "double bottom": PatternType.DOUBLE_BOTTOM,
            "double top": PatternType.DOUBLE_TOP,
            "head and shoulders": PatternType.HEAD_SHOULDERS,
            "triangle": PatternType.TRIANGLE,
            "wedge": PatternType.WEDGE,
        }
        
        pattern_type = PatternType.NONE
        for name, ptype in pattern_map.items():
            if name in result.pattern.lower():
                pattern_type = ptype
                break
        
        signal = Signal(
            symbol=symbol,
            signal_type=SignalType(result.signal_type),
            pattern_detected=pattern_type,
            pattern_confidence=result.pattern_confidence,
            trend=result.trend,
            trend_strength=result.trend_strength,
            market_phase=result.phase,
            support_level=result.support,
            resistance_level=result.resistance,
            analysis_summary=result.summary,
            detailed_reasoning=result.detailed_reasoning,
            chart_image_path=chart_path,
        )
        
        signal_id = self.db.create(signal)
        logger.info(f"   Signal ID: {signal_id}")
        return signal_id
    
    def generate_report(self, signal_id: int, chart_path: str) -> str:
        """Step 7: Generate visual report."""
        logger.info("📄 Generating report...")
        
        signal = self.db.get_by_id(signal_id)
        
        report_path = self.report_gen.generate(
            signal=signal,
            chart_image_path=chart_path,
            annotate=True,
        )
        
        logger.info(f"   Report: {report_path}")
        return report_path
    
    def run_full_flow(self, symbol: str, exchange: str = "") -> FullAnalysisResult:
        """Execute complete pipeline."""
        print("\n" + "=" * 60)
        print(f"🚀 FULL E2E FLOW: {exchange}:{symbol}")
        print("=" * 60)
        
        total_start = time.time()
        
        # Step 1: Capture chart
        chart_path = self.capture_chart(symbol, exchange)
        
        # Step 2: Get market data
        market_data = self.get_market_data(symbol)
        price = market_data["close"]
        
        # Step 3: Visual analysis (3 agents)
        pattern = self.analyze_pattern(chart_path, symbol, price)
        trend = self.analyze_trend(chart_path, symbol)
        levels = self.analyze_levels(chart_path, symbol, price)
        
        # Step 4: Risk analysis
        risk = self.analyze_risk(
            pattern.parsed, trend.parsed,
            price, market_data["atr"], market_data["rvol"]
        )
        
        # Step 5: Sentiment
        sentiment = self.analyze_sentiment(
            market_data.get("rsi"),
            market_data.get("macd")
        )
        
        # Step 6: Synthesize
        result = self.synthesize(pattern, trend, levels, risk, sentiment, symbol, market_data)
        
        # Step 7: Save to DB
        signal_id = self.save_to_database(result, symbol, chart_path)
        
        # Step 8: Generate report
        report_path = self.generate_report(signal_id, chart_path)
        
        total_elapsed = time.time() - total_start
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 FINAL RESULT")
        print("=" * 60)
        print(f"  Symbol: {symbol}")
        print(f"  Signal: {result.signal_type.upper()}")
        print(f"  Confidence: {result.overall_confidence:.0%}")
        print(f"  Pattern: {result.pattern} ({result.pattern_confidence:.0%})")
        print(f"  Trend: {result.trend} ({result.trend_strength})")
        print(f"  Phase: {result.phase}")
        print(f"  Risk: {result.risk_assessment}")
        print(f"  Stop Loss: {result.stop_loss}")
        print(f"  Position: {result.position_size}")
        print(f"  Sentiment: {result.sentiment_label} ({result.sentiment_score:.2f})")
        if result.veto_reason:
            print(f"  ⚠️ VETO: {result.veto_reason}")
        print("-" * 60)
        print(f"  {result.summary}")
        print("-" * 60)
        print(f"  Chart: {chart_path}")
        print(f"  Report: {report_path}")
        print(f"  DB ID: {signal_id}")
        print(f"  Total Time: {total_elapsed:.1f}s")
        print("=" * 60)
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Full E2E Flow Test")
    parser.add_argument("--symbol", "-s", default="AAPL", help="Symbol")
    parser.add_argument("--exchange", "-e", default="", help="Exchange (empty for auto-detect)")
    args = parser.parse_args()
    
    # Ensure directories
    Path("data/charts").mkdir(parents=True, exist_ok=True)
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    
    orchestrator = FullFlowOrchestrator()
    result = orchestrator.run_full_flow(args.symbol, args.exchange)
    
    print("\n✅ Full E2E flow completed successfully!\n")


if __name__ == "__main__":
    main()

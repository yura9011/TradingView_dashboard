"""
Multi-Agent Analysis Test Script
Run: python main_multiagent.py --symbol MELI
"""

import os
import sys
import yaml
import asyncio
import logging
import argparse
from pathlib import Path

# Fix Windows encoding for emojis - only if not already wrapped
if sys.platform == "win32":
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except AttributeError:
            pass  # Already wrapped or no buffer available

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.screener.chart_capture import get_chart_capture
from src.agents.coordinator import get_coordinator
from src.visual import get_report_generator, get_annotator
from src.database import get_signal_repository
from src.models import Signal, SignalType, PatternType
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Pattern mapping with variations
PATTERN_ALIASES = {
    PatternType.HEAD_SHOULDERS: [
        "head and shoulders", "head & shoulders", "h&s", "head shoulders",
        "hch", "head-and-shoulders", "cabeza y hombros"
    ],
    PatternType.DOUBLE_TOP: [
        "double top", "double-top", "doubletop", "m pattern", "m top",
        "doble techo"
    ],
    PatternType.DOUBLE_BOTTOM: [
        "double bottom", "double-bottom", "doublebottom", "w pattern", "w bottom",
        "doble suelo", "doble piso"
    ],
    PatternType.BULLISH_ENGULFING: [
        "bullish engulfing", "bullish-engulfing", "envolvente alcista"
    ],
    PatternType.BEARISH_ENGULFING: [
        "bearish engulfing", "bearish-engulfing", "envolvente bajista"
    ],
    PatternType.TRIANGLE: [
        "triangle", "ascending triangle", "descending triangle", 
        "symmetrical triangle", "triángulo", "triangulo"
    ],
    PatternType.WEDGE: [
        "wedge", "rising wedge", "falling wedge", "cuña"
    ],
}


def _map_pattern_to_enum(pattern_name: str) -> PatternType:
    """Map pattern name string to PatternType enum with fuzzy matching."""
    if not pattern_name or pattern_name.lower() in ["none", "n/a", "no pattern"]:
        return PatternType.NONE
    
    pattern_lower = pattern_name.lower().strip()
    
    # Check against all aliases
    for pattern_type, aliases in PATTERN_ALIASES.items():
        for alias in aliases:
            if alias in pattern_lower or pattern_lower in alias:
                return pattern_type
    
    # Fallback: try direct enum matching
    for pt in PatternType:
        if pt.value.replace("_", " ") in pattern_lower:
            return pt
    
    logger.warning(f"Unknown pattern '{pattern_name}' - defaulting to NONE")
    return PatternType.NONE


def load_api_key():
    """Load API key from config or environment."""
    # Try environment first
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    
    # Try config file
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        api_key = config.get("gemini", {}).get("api_key")
        if api_key and api_key != "YOUR_GEMINI_API_KEY":
            os.environ["GEMINI_API_KEY"] = api_key
            return api_key
    
    raise ValueError("GEMINI_API_KEY not found in environment or config.yaml")


async def analyze_with_multiagent(symbol: str, exchange: str = "NASDAQ"):
    """Run analysis using multi-agent architecture."""
    
    logger.info("=" * 60)
    logger.info(f"🚀 Multi-Agent Analysis: {exchange}:{symbol}")
    logger.info("=" * 60)
    
    # Step 1: Capture chart (daily, 1 month)
    logger.info("📸 Capturing chart (daily, 1 month)...")
    chart_capture = get_chart_capture()
    chart_path, price_range = await asyncio.to_thread(
        chart_capture.capture_sync,
        symbol=symbol,
        exchange=exchange,
        interval="D",
        range_months=1,
    )
    logger.info(f"   Chart saved: {chart_path}")
    
    # Build price context from OCR if available and valid
    price_context = ""
    if price_range and price_range.get("ocr_success"):
        price_context = f"IMPORTANT - Price range visible on chart Y-axis: {price_range['price_range_text']}. "
        if price_range.get("current_price"):
            price_context += f"Approximate current price: ${price_range['current_price']:.2f}. "
        price_context += "Use these values as reference for support/resistance levels."
        logger.info(f"   ✅ OCR Price Range: {price_range['price_range_text']}")
    else:
        logger.info("   ⚠️  OCR not available - model will estimate prices from chart visuals")
    
    # Step 2: Multi-agent analysis
    logger.info("\n🤖 Running Multi-Agent Analysis...")
    coordinator = get_coordinator()
    analysis = coordinator.analyze(str(chart_path), symbol)
    
    print("\n" + "=" * 60)
    print("📊 ANALYSIS RESULTS")
    print("=" * 60)
    print(f"  SIGNAL TYPE: {analysis.signal_type.upper()}")
    print(f"  OVERALL CONFIDENCE: {analysis.overall_confidence:.0%}")
    print("-" * 60)
    print("  PATTERN:")
    print(f"    Name: {analysis.pattern}")
    print(f"    Confidence: {analysis.pattern_confidence:.0%}")
    print(f"    Box: {analysis.pattern_box}")
    print("-" * 60)
    print("  TREND:")
    print(f"    Direction: {analysis.trend}")
    print(f"    Strength: {analysis.trend_strength}")
    print(f"    Phase: {analysis.phase}")
    if analysis.wave:
        print(f"    Wave: {analysis.wave}")
    print("-" * 60)
    print("  LEVELS:")
    print(f"    Support: ${analysis.support or 0:,.2f}")
    print(f"    Resistance: ${analysis.resistance or 0:,.2f}")
    print(f"    Fibonacci: {analysis.fibonacci or 'N/A'}")
    print(f"    Key Level: ${analysis.key_level or 0:,.2f}")
    print("-" * 60)
    print("  SUMMARY:")
    print(f"  {analysis.summary}")
    print("=" * 60)
    
    # Step 4: Create Signal and save
    pattern_type = _map_pattern_to_enum(analysis.pattern)
    
    signal = Signal(
        symbol=symbol,
        signal_type=SignalType(analysis.signal_type),
        pattern_detected=pattern_type,
        pattern_confidence=analysis.pattern_confidence,
        trend=analysis.trend,
        trend_strength=analysis.trend_strength,
        market_phase=analysis.phase,
        elliott_wave=analysis.wave,
        support_level=analysis.support,
        resistance_level=analysis.resistance,
        fibonacci_level=analysis.fibonacci,
        analysis_summary=analysis.summary,
        detailed_reasoning=analysis.detailed_reasoning,
        chart_image_path=str(chart_path),
        notes=json.dumps({"pattern_box": analysis.pattern_box}) if analysis.pattern_box else None,
    )
    
    # Save to database
    repo = get_signal_repository()
    signal_id = repo.create(signal)
    signal.id = signal_id
    logger.info(f"✅ Signal saved with ID: {signal_id}")
    
    # Step 5: Generate report with annotations
    logger.info("📄 Generating annotated report...")
    report_gen = get_report_generator()
    report_path = report_gen.generate(
        signal=signal,
        chart_image_path=str(chart_path),
        annotate=True,
    )
    logger.info(f"   Report: {report_path}")
    
    # Update signal with annotated chart path (signal.chart_image_path was updated by report_gen)
    if signal.chart_image_path and signal.chart_image_path != str(chart_path):
        repo.update_chart_path(signal_id, signal.chart_image_path)
        logger.info(f"   Updated chart path: {signal.chart_image_path}")
    
    logger.info("\n✅ Multi-Agent Analysis Complete!")
    
    return signal


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Trading Analysis")
    parser.add_argument("--symbol", "-s", type=str, default="MELI", help="Stock symbol")
    parser.add_argument("--exchange", "-e", type=str, default="NASDAQ", help="Exchange")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    
    # Load API key from config
    load_api_key()
    
    asyncio.run(analyze_with_multiagent(args.symbol, args.exchange))


if __name__ == "__main__":
    main()

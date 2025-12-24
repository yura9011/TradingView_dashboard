"""
Multi-Agent Analysis with Local Phi-3.5-Vision Model
Run: python main_multiagent_local.py --symbol MELI

This script uses a local Phi-3.5-vision-instruct model instead of Gemini API.
Requires: GPU with 8GB+ VRAM (recommended) or CPU (slow)
"""

import os
import sys
import yaml
import asyncio
import logging
import argparse
from pathlib import Path

# Fix Windows encoding for emojis
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.screener.chart_capture import get_chart_capture
from src.agents.coordinator_local import get_coordinator_local
from src.visual import get_report_generator
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
    
    for pattern_type, aliases in PATTERN_ALIASES.items():
        for alias in aliases:
            if alias in pattern_lower or pattern_lower in alias:
                return pattern_type
    
    for pt in PatternType:
        if pt.value.replace("_", " ") in pattern_lower:
            return pt
    
    logger.warning(f"Unknown pattern '{pattern_name}' - defaulting to NONE")
    return PatternType.NONE


def check_system_requirements():
    """Check if system meets requirements for local model."""
    import torch
    
    print("\n" + "=" * 60)
    print("🔍 SYSTEM CHECK")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA Available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {device_name}")
        print(f"  VRAM: {total_memory:.1f} GB")
        
        if total_memory < 8:
            print("  ⚠️  Warning: Less than 8GB VRAM. Model may run slowly or fail.")
    else:
        print("  ⚠️  Warning: No GPU detected. Running on CPU will be very slow.")
        print("  💡 Tip: Install CUDA toolkit and PyTorch with CUDA support.")
    
    print("=" * 60 + "\n")
    
    return cuda_available


async def analyze_with_local_model(
    symbol: str,
    exchange: str = "NASDAQ",
    model_name: str = "microsoft/Phi-3.5-vision-instruct",
):
    """Run analysis using local Phi-3.5-vision model."""
    
    logger.info("=" * 60)
    logger.info(f"🚀 Local Multi-Agent Analysis: {exchange}:{symbol}")
    logger.info(f"📦 Model: {model_name}")
    logger.info("=" * 60)
    
    # Step 1: Capture chart
    logger.info("📸 Capturing chart (weekly timeframe)...")
    chart_capture = get_chart_capture()
    chart_path = await asyncio.to_thread(
        chart_capture.capture_sync,
        symbol=symbol,
        exchange=exchange,
    )
    logger.info(f"   Chart saved: {chart_path}")
    
    # Step 2: Multi-agent analysis with local model
    logger.info("\n🤖 Running Local Multi-Agent Analysis...")
    logger.info("   (First run will download the model ~8GB)")
    
    coordinator = get_coordinator_local(model_name=model_name)
    analysis = coordinator.analyze(str(chart_path), symbol)
    
    print("\n" + "=" * 60)
    print("📊 ANALYSIS RESULTS (Local Model)")
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
    print(f"    Phase (Wyckoff): {analysis.phase}")
    if analysis.wave:
        print(f"    Wave (Elliott): {analysis.wave}")
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
    
    # Step 3: Create Signal and save
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
        notes=json.dumps({"pattern_box": analysis.pattern_box, "model": "local-phi"}) if analysis.pattern_box else json.dumps({"model": "local-phi"}),
    )
    
    # Save to database
    repo = get_signal_repository()
    signal_id = repo.create(signal)
    signal.id = signal_id
    logger.info(f"✅ Signal saved with ID: {signal_id}")
    
    # Step 4: Generate report
    logger.info("📄 Generating annotated report...")
    report_gen = get_report_generator()
    report_path = report_gen.generate(
        signal=signal,
        chart_image_path=str(chart_path),
        annotate=True,
    )
    logger.info(f"   Report: {report_path}")
    
    if signal.chart_image_path and signal.chart_image_path != str(chart_path):
        repo.update_chart_path(signal_id, signal.chart_image_path)
    
    logger.info("\n✅ Local Multi-Agent Analysis Complete!")
    
    return signal


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Trading Analysis with Local Phi-3.5-Vision Model"
    )
    parser.add_argument("--symbol", "-s", type=str, default="MELI", help="Stock symbol")
    parser.add_argument("--exchange", "-e", type=str, default="NASDAQ", help="Exchange")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="microsoft/Phi-3.5-vision-instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip system requirements check"
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("data/charts").mkdir(parents=True, exist_ok=True)
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    
    # Check system requirements
    if not args.skip_check:
        check_system_requirements()
    
    # Run analysis
    asyncio.run(analyze_with_local_model(
        args.symbol,
        args.exchange,
        args.model,
    ))


if __name__ == "__main__":
    main()

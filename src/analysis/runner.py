"""
Analysis Runner - Orchestrates the full analysis pipeline.

Updated to support:
- 1 Year (1Y) timeframe with Daily (1D) candles
- EnhancedPreprocessor for region detection and auto-cropping
- Timeframe-aware configuration
"""

import json
import asyncio
import logging
from typing import Optional, Tuple
from pathlib import Path

from src.screener.chart_capture import get_chart_capture
from src.visual import get_report_generator
from src.database import get_signal_repository
from src.models import Signal, SignalType, PatternType

from .analyzer import ChartAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)

# Default timeframe configuration
DEFAULT_TIMEFRAME = "1Y"
DEFAULT_RANGE_MONTHS = 12


# Pattern mapping with variations
PATTERN_ALIASES = {
    PatternType.HEAD_SHOULDERS: [
        "head and shoulders", "head & shoulders", "h&s", "head shoulders",
        "hch", "head-and-shoulders", "cabeza y hombros", "head_shoulders",
        "inverse_head_shoulders", "inverse head shoulders"
    ],
    PatternType.DOUBLE_TOP: [
        "double top", "double-top", "doubletop", "m pattern", "m top",
        "doble techo", "double_top"
    ],
    PatternType.DOUBLE_BOTTOM: [
        "double bottom", "double-bottom", "doublebottom", "w pattern", "w bottom",
        "doble suelo", "doble piso", "double_bottom"
    ],
    PatternType.BULLISH_ENGULFING: [
        "bullish engulfing", "bullish-engulfing", "envolvente alcista"
    ],
    PatternType.BEARISH_ENGULFING: [
        "bearish engulfing", "bearish-engulfing", "envolvente bajista"
    ],
    PatternType.TRIANGLE: [
        "triangle", "ascending triangle", "descending triangle", 
        "symmetrical triangle", "triángulo", "triangulo",
        "symmetric_triangle", "ascending_triangle", "descending_triangle"
    ],
    PatternType.WEDGE: [
        "wedge", "rising wedge", "falling wedge", "cuña",
        "rising_wedge", "falling_wedge"
    ],
}


def _map_pattern_to_enum(pattern_name: str) -> PatternType:
    """Map pattern name string to PatternType enum."""
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


async def capture_chart(
    symbol: str,
    exchange: str = "",
    interval: str = "D",
    range_months: int = DEFAULT_RANGE_MONTHS,
) -> Tuple[str, dict]:
    """Capture chart from TradingView.
    
    Args:
        symbol: Stock symbol
        exchange: Exchange (empty for auto-detect)
        interval: Chart interval (D=daily)
        range_months: Months of data to show (default: 12 for 1Y)
        
    Returns:
        Tuple of (chart_path, price_range_data)
    """
    chart_capture = get_chart_capture()
    return await asyncio.to_thread(
        chart_capture.capture_sync,
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        range_months=range_months,
    )


def build_price_context(price_range: dict) -> str:
    """Build price context string from OCR data."""
    if not price_range or not price_range.get("ocr_success"):
        return ""
    
    context = f"IMPORTANT - Price range visible on chart Y-axis: {price_range['price_range_text']}. "
    if price_range.get("current_price"):
        context += f"Approximate current price: ${price_range['current_price']:.2f}. "
    context += "Use these values as reference for support/resistance levels."
    return context


def create_signal(
    symbol: str,
    analysis: AnalysisResult,
    chart_path: str,
    model_type: str = "local",
) -> Signal:
    """Create Signal object from analysis result."""
    pattern_type = _map_pattern_to_enum(analysis.pattern)
    
    notes_data = {"model": model_type}
    if analysis.pattern_box:
        notes_data["pattern_box"] = analysis.pattern_box
    
    return Signal(
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
        chart_image_path=chart_path,
        notes=json.dumps(notes_data),
    )


async def run_analysis(
    symbol: str,
    exchange: str = "",
    use_local: bool = True,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    api_key: Optional[str] = None,
    save_to_db: bool = True,
    generate_report: bool = True,
    range_months: int = DEFAULT_RANGE_MONTHS,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> Signal:
    """Run full analysis pipeline.
    
    Args:
        symbol: Stock symbol to analyze
        exchange: Exchange (empty for auto-detect)
        use_local: Use local model (True) or Gemini (False)
        model_name: HuggingFace model for local mode
        api_key: Gemini API key for cloud mode
        save_to_db: Save signal to database
        generate_report: Generate visual report
        range_months: Months of data to show (default: 12 for 1Y)
        timeframe: Timeframe configuration (default: "1Y")
        
    Returns:
        Signal with analysis results
    """
    symbol = symbol.strip().upper()
    
    logger.info("=" * 60)
    logger.info(f"🚀 Analysis: {symbol}")
    logger.info(f"📦 Model: {'Local - ' + model_name if use_local else 'Gemini API'}")
    logger.info(f"📊 Timeframe: {timeframe} ({range_months} months, daily candles)")
    logger.info("=" * 60)
    
    # Step 1: Capture chart with configured timeframe
    logger.info(f"📸 Capturing chart (daily, {range_months} months / {timeframe})...")
    chart_path, price_range = await capture_chart(
        symbol, exchange, range_months=range_months
    )
    logger.info(f"   Chart saved: {chart_path}")
    
    # Step 2: Apply enhanced preprocessing (region detection + auto-crop)
    preprocessed_path = chart_path
    preprocess_metadata = None
    
    try:
        from src.pattern_analysis.pipeline import EnhancedPreprocessor
        from src.pattern_analysis.config import get_config_manager
        import cv2
        
        # Load configuration
        config_manager = get_config_manager()
        config = config_manager.config
        
        # Initialize enhanced preprocessor
        preprocessor = EnhancedPreprocessor(config)
        
        # Process the chart image
        logger.info("🔍 Running enhanced preprocessing (region detection + auto-crop)...")
        preprocess_result = preprocessor.process_with_timeframe(
            chart_path,
            timeframe=None,  # Use default from config (1Y)
            config={"denoise": True}
        )
        
        # Save preprocessed image if cropping was applied
        if preprocess_result.coverage_percentage < 100.0:
            preprocessed_path = chart_path.replace(".png", "_preprocessed.png")
            # Convert RGB back to BGR for OpenCV saving
            preprocessed_bgr = cv2.cvtColor(preprocess_result.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(preprocessed_path, preprocessed_bgr)
            logger.info(f"   Preprocessed chart saved: {preprocessed_path}")
            logger.info(f"   Coverage: {preprocess_result.coverage_percentage:.1f}%")
            
            if preprocess_result.crop_result and preprocess_result.crop_result.excluded_regions:
                excluded = [r.region_type.value for r in preprocess_result.crop_result.excluded_regions]
                logger.info(f"   Excluded regions: {excluded}")
        
        if preprocess_result.needs_review:
            logger.warning("   ⚠️ Low coverage - result flagged for review")
        
        preprocess_metadata = preprocess_result.to_dict()
        
    except ImportError as e:
        logger.warning(f"Enhanced preprocessing not available: {e}")
    except Exception as e:
        logger.warning(f"Enhanced preprocessing failed, using original image: {e}")
    
    # Build price context
    price_context = build_price_context(price_range)
    if price_context:
        logger.info(f"   ✅ OCR Price Range: {price_range['price_range_text']}")
    else:
        logger.info("   ⚠️  OCR not available")
    
    # Add timeframe context to price context
    timeframe_context = f"TIMEFRAME: {timeframe} with daily (1D) candles. "
    price_context = timeframe_context + price_context
    
    # Step 3: Run analysis on preprocessed image
    logger.info("🤖 Running analysis...")
    analyzer = ChartAnalyzer(
        use_local=use_local,
        model_name=model_name,
        api_key=api_key,
    )
    analysis = analyzer.analyze(preprocessed_path, symbol, price_context)
    
    # Step 4: Create signal
    model_type = "local-qwen" if use_local else "gemini"
    signal = create_signal(symbol, analysis, chart_path, model_type)
    
    # Add preprocessing metadata to notes
    if preprocess_metadata:
        notes_data = json.loads(signal.notes) if signal.notes else {}
        notes_data["preprocessing"] = {
            "coverage_percentage": preprocess_metadata.get("coverage_percentage"),
            "needs_review": preprocess_metadata.get("needs_review"),
            "timeframe": timeframe,
        }
        signal.notes = json.dumps(notes_data)
    
    # Step 5: Save to database
    if save_to_db:
        repo = get_signal_repository()
        signal_id = repo.create(signal)
        signal.id = signal_id
        logger.info(f"✅ Signal saved with ID: {signal_id}")
    
    # Step 6: Generate report
    if generate_report:
        logger.info("📄 Generating report...")
        report_gen = get_report_generator()
        report_path = report_gen.generate(
            signal=signal,
            chart_image_path=chart_path,
            annotate=True,
        )
        logger.info(f"   Report: {report_path}")
        
        if save_to_db and signal.chart_image_path != chart_path:
            repo.update_chart_path(signal.id, signal.chart_image_path)
    
    # Print results
    _print_results(analysis)
    
    logger.info("✅ Analysis Complete!")
    return signal


def _print_results(analysis: AnalysisResult):
    """Print analysis results to console."""
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
    print("  RISK:")
    print(f"    Assessment: {analysis.risk_assessment}")
    print(f"    Stop Loss: {analysis.stop_loss}")
    print(f"    Position Size: {analysis.position_size}")
    print("-" * 60)
    print("  SENTIMENT:")
    print(f"    Score: {analysis.sentiment_score:.2f}")
    print(f"    Label: {analysis.sentiment_label}")
    if analysis.veto_reason:
        print("-" * 60)
        print(f"  ⚠️ VETO: {analysis.veto_reason}")
    print("-" * 60)
    print("  SUMMARY:")
    print(f"  {analysis.summary}")
    print("=" * 60)

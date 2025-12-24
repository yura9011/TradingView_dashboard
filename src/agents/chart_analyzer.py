"""
Chart Analyzer - Orchestrates the analysis pipeline.
Connects screener results with Gemini analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.agents.gemini_client import GeminiClient, AnalysisResult, get_gemini_client
from src.models import Signal, SignalType, PatternType, Asset
from src.database import get_signal_repository

logger = logging.getLogger(__name__)


# Pattern mapping with variations (same as main_multiagent.py)
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


class ChartAnalyzer:
    """Orchestrates chart analysis using Gemini."""
    
    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize analyzer.
        
        Args:
            gemini_client: Pre-configured GeminiClient
            api_key: API key if creating new client
        """
        self.client = gemini_client or get_gemini_client(api_key=api_key)
        self.signal_repo = get_signal_repository()
    
    def analyze_chart_image(
        self,
        image_path: str,
        symbol: str,
        asset: Optional[Asset] = None,
        timeframe: str = "1D",
        save_signal: bool = True,
    ) -> Signal:
        """Analyze a chart image and create a signal.
        
        Args:
            image_path: Path to chart screenshot
            symbol: Ticker symbol
            asset: Optional Asset with additional metadata
            timeframe: Chart timeframe
            save_signal: Whether to save signal to database
            
        Returns:
            Signal with analysis results
        """
        logger.info(f"Analyzing chart for {symbol}")
        
        # Build additional context from asset if available
        context = ""
        if asset:
            context = f"Current price: ${asset.price:.2f}, "
            context += f"Change: {asset.change_percent:+.2f}%, "
            context += f"Volume: {asset.volume:,.0f}"
            if asset.sector:
                context += f", Sector: {asset.sector}"
        
        # Run Gemini analysis
        result = self.client.analyze_chart(
            image_path=image_path,
            symbol=symbol,
            timeframe=timeframe,
            additional_context=context,
        )
        
        # Convert to Signal
        signal = self._result_to_signal(result, symbol, image_path)
        
        # Save to database
        if save_signal:
            signal_id = self.signal_repo.create(signal)
            signal.id = signal_id
            logger.info(f"Saved signal {signal_id} for {symbol}")
        
        return signal
    
    def _result_to_signal(
        self,
        result: AnalysisResult,
        symbol: str,
        image_path: str,
    ) -> Signal:
        """Convert AnalysisResult to Signal model."""
        
        # Map pattern string to enum using improved mapping
        pattern = _map_pattern_to_enum(result.pattern_detected)
        
        # Determine signal type based on confidence
        if result.pattern_confidence >= 0.7:
            signal_type = SignalType.CANDIDATE
        elif result.pattern_confidence >= 0.4:
            signal_type = SignalType.PENDING
        else:
            signal_type = SignalType.NOT_CANDIDATE
        
        # Store pattern_box in notes as JSON for later use by annotator
        pattern_box_json = None
        if result.pattern_box:
            pattern_box_json = json.dumps({"pattern_box": result.pattern_box})
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            pattern_detected=pattern,
            pattern_confidence=result.pattern_confidence,
            trend=result.trend,
            support_level=result.support_level,
            resistance_level=result.resistance_level,
            fibonacci_level=result.fibonacci_level,
            analysis_summary=result.analysis_summary,
            chart_image_path=str(image_path),
            notes=pattern_box_json,
        )


def get_chart_analyzer(api_key: Optional[str] = None) -> ChartAnalyzer:
    """Factory function for ChartAnalyzer.
    
    Args:
        api_key: Gemini API key
        
    Returns:
        Configured ChartAnalyzer
    """
    return ChartAnalyzer(api_key=api_key)

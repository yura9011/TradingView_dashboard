"""
Integration module for Chart Pattern Analysis Framework.

Provides adapters and utilities to integrate the new pattern analysis
framework with the existing trading analysis system.

Requirements:
- 4.2: Integrate new components without requiring changes to existing components
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .factory import create_analyzer, ChartPatternAnalyzer
from .models.dataclasses import AnalysisResult as PatternAnalysisResult
from .models.dataclasses import PatternDetection, BoundingBox
from .models.enums import PatternCategory, PatternType as FrameworkPatternType


logger = logging.getLogger(__name__)


# Mapping from framework PatternType to existing system PatternType values
PATTERN_TYPE_MAPPING = {
    FrameworkPatternType.HEAD_SHOULDERS: "head_shoulders",
    FrameworkPatternType.INVERSE_HEAD_SHOULDERS: "head_shoulders",
    FrameworkPatternType.DOUBLE_TOP: "double_top",
    FrameworkPatternType.DOUBLE_BOTTOM: "double_bottom",
    FrameworkPatternType.TRIPLE_TOP: "double_top",  # Map to closest
    FrameworkPatternType.TRIPLE_BOTTOM: "double_bottom",  # Map to closest
    FrameworkPatternType.ASCENDING_TRIANGLE: "triangle",
    FrameworkPatternType.DESCENDING_TRIANGLE: "triangle",
    FrameworkPatternType.SYMMETRICAL_TRIANGLE: "triangle",
    FrameworkPatternType.RISING_WEDGE: "wedge",
    FrameworkPatternType.FALLING_WEDGE: "wedge",
    FrameworkPatternType.BULL_FLAG: "triangle",  # Map to closest
    FrameworkPatternType.BEAR_FLAG: "triangle",  # Map to closest
    FrameworkPatternType.CUP_AND_HANDLE: "double_bottom",  # Map to closest
    FrameworkPatternType.CHANNEL_UP: "triangle",  # Map to closest
    FrameworkPatternType.CHANNEL_DOWN: "triangle",  # Map to closest
}


@dataclass
class LegacyAnalysisResult:
    """
    Analysis result compatible with the existing GeminiClient AnalysisResult.
    
    This provides backward compatibility with the existing system while
    using the new pattern analysis framework under the hood.
    """
    pattern_detected: str
    pattern_confidence: float
    trend: str
    support_level: Optional[float]
    resistance_level: Optional[float]
    fibonacci_level: Optional[str]
    pattern_box: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) as percentages
    analysis_summary: str
    raw_response: str
    
    # Additional fields from new framework
    detections: List[PatternDetection] = None
    total_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.detections is None:
            self.detections = []


class PatternAnalysisAdapter:
    """
    Adapter that wraps the new pattern analysis framework to provide
    compatibility with the existing system's interfaces.
    
    This allows the new framework to be used as a drop-in replacement
    for the existing pattern detection while maintaining backward
    compatibility with existing code.
    
    Requirements: 4.2
    
    Example:
        >>> adapter = PatternAnalysisAdapter()
        >>> result = adapter.analyze_chart("chart.png", "AAPL", "1D")
        >>> # result is compatible with existing AnalysisResult
    """
    
    def __init__(
        self,
        analyzer: Optional[ChartPatternAnalyzer] = None,
        config_path: Optional[str] = None,
        ml_model_path: Optional[str] = None,
        enable_cross_validation: bool = True
    ):
        """
        Initialize the adapter.
        
        Args:
            analyzer: Pre-configured ChartPatternAnalyzer (optional)
            config_path: Path to configuration file
            ml_model_path: Path to ML model for detection
            enable_cross_validation: Whether to enable cross-validation
        """
        self.analyzer = analyzer or create_analyzer(
            config_path=config_path,
            ml_model_path=ml_model_path,
            enable_cross_validation=enable_cross_validation
        )
    
    def analyze_chart(
        self,
        image_path: Union[str, Path],
        symbol: str,
        timeframe: str = "1D",
        additional_context: str = "",
    ) -> LegacyAnalysisResult:
        """
        Analyze a chart image for trading patterns.
        
        This method provides the same interface as GeminiClient.analyze_chart()
        but uses the new pattern analysis framework.
        
        Args:
            image_path: Path to chart screenshot
            symbol: Ticker symbol being analyzed
            timeframe: Chart timeframe (e.g., "1D", "4H")
            additional_context: Extra context (not used by framework)
            
        Returns:
            LegacyAnalysisResult compatible with existing system
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Chart image not found: {image_path}")
        
        logger.info(f"Analyzing chart for {symbol} ({timeframe}) using pattern framework")
        
        # Run analysis with new framework
        result = self.analyzer.analyze(str(image_path))
        
        # Convert to legacy format
        return self._convert_to_legacy_result(result, image_path)
    
    def _convert_to_legacy_result(
        self,
        result: PatternAnalysisResult,
        image_path: Path
    ) -> LegacyAnalysisResult:
        """
        Convert PatternAnalysisResult to LegacyAnalysisResult.
        
        Args:
            result: Result from new framework
            image_path: Path to the analyzed image
            
        Returns:
            LegacyAnalysisResult compatible with existing system
        """
        # Get the highest confidence detection
        pattern_detected = "none"
        pattern_confidence = 0.0
        pattern_box = None
        
        if result.detections:
            top_detection = result.detections[0]  # Already sorted by confidence
            pattern_detected = self._map_pattern_type(top_detection.pattern_type)
            pattern_confidence = top_detection.confidence
            pattern_box = self._convert_bbox_to_percentages(
                top_detection.bounding_box,
                image_path
            )
        
        # Determine trend from patterns
        trend = self._infer_trend(result.detections)
        
        # Extract support/resistance from feature map
        support_level, resistance_level = self._extract_levels(result)
        
        # Generate summary
        summary = self._generate_summary(result)
        
        # Generate raw response (JSON representation)
        raw_response = result.to_json(validate=False)
        
        return LegacyAnalysisResult(
            pattern_detected=pattern_detected,
            pattern_confidence=pattern_confidence,
            trend=trend,
            support_level=support_level,
            resistance_level=resistance_level,
            fibonacci_level=None,  # Not extracted by framework
            pattern_box=pattern_box,
            analysis_summary=summary,
            raw_response=raw_response,
            detections=result.detections,
            total_time_ms=result.total_time_ms
        )
    
    def _map_pattern_type(self, pattern_type: FrameworkPatternType) -> str:
        """Map framework PatternType to legacy pattern string."""
        return PATTERN_TYPE_MAPPING.get(pattern_type, "none")
    
    def _convert_bbox_to_percentages(
        self,
        bbox: BoundingBox,
        image_path: Path
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Convert bounding box coordinates to percentages of image dimensions.
        
        Args:
            bbox: Bounding box with pixel coordinates
            image_path: Path to image for getting dimensions
            
        Returns:
            Tuple of (x1, y1, x2, y2) as percentages (0-100)
        """
        try:
            import cv2
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            h, w = image.shape[:2]
            
            x1_pct = int((bbox.x1 / w) * 100)
            y1_pct = int((bbox.y1 / h) * 100)
            x2_pct = int((bbox.x2 / w) * 100)
            y2_pct = int((bbox.y2 / h) * 100)
            
            return (x1_pct, y1_pct, x2_pct, y2_pct)
        except Exception as e:
            logger.warning(f"Failed to convert bbox to percentages: {e}")
            return None
    
    def _infer_trend(self, detections: List[PatternDetection]) -> str:
        """
        Infer trend direction from detected patterns.
        
        Args:
            detections: List of pattern detections
            
        Returns:
            Trend string: "up", "down", or "sideways"
        """
        if not detections:
            return "sideways"
        
        bullish_count = 0
        bearish_count = 0
        
        bullish_patterns = {
            FrameworkPatternType.INVERSE_HEAD_SHOULDERS,
            FrameworkPatternType.DOUBLE_BOTTOM,
            FrameworkPatternType.TRIPLE_BOTTOM,
            FrameworkPatternType.ASCENDING_TRIANGLE,
            FrameworkPatternType.BULL_FLAG,
            FrameworkPatternType.CUP_AND_HANDLE,
            FrameworkPatternType.FALLING_WEDGE,
            FrameworkPatternType.CHANNEL_UP,
        }
        
        bearish_patterns = {
            FrameworkPatternType.HEAD_SHOULDERS,
            FrameworkPatternType.DOUBLE_TOP,
            FrameworkPatternType.TRIPLE_TOP,
            FrameworkPatternType.DESCENDING_TRIANGLE,
            FrameworkPatternType.BEAR_FLAG,
            FrameworkPatternType.RISING_WEDGE,
            FrameworkPatternType.CHANNEL_DOWN,
        }
        
        for det in detections:
            if det.pattern_type in bullish_patterns:
                bullish_count += det.confidence
            elif det.pattern_type in bearish_patterns:
                bearish_count += det.confidence
        
        if bullish_count > bearish_count + 0.2:
            return "up"
        elif bearish_count > bullish_count + 0.2:
            return "down"
        else:
            return "sideways"
    
    def _extract_levels(
        self,
        result: PatternAnalysisResult
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract support and resistance levels from analysis result.
        
        Note: The framework extracts zones, not specific price levels.
        This returns None as the framework doesn't have price data.
        
        Args:
            result: Analysis result
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        # The framework detects zones visually but doesn't have price data
        # Return None to indicate levels should be obtained from other sources
        return None, None
    
    def _generate_summary(self, result: PatternAnalysisResult) -> str:
        """
        Generate a human-readable summary of the analysis.
        
        Args:
            result: Analysis result
            
        Returns:
            Summary string
        """
        if not result.detections:
            return "No significant chart patterns detected."
        
        patterns = []
        for det in result.detections[:3]:  # Top 3 patterns
            name = det.pattern_type.value.replace("_", " ").title()
            conf = int(det.confidence * 100)
            patterns.append(f"{name} ({conf}%)")
        
        pattern_str = ", ".join(patterns)
        
        validated_count = sum(
            1 for vr in result.validated_detections
            if vr.is_confirmed
        )
        
        if validated_count > 0:
            validation_str = f"{validated_count} pattern(s) confirmed by cross-validation."
        else:
            validation_str = "Patterns pending validation."
        
        return f"Detected patterns: {pattern_str}. {validation_str}"


def create_pattern_adapter(
    config_path: Optional[str] = None,
    ml_model_path: Optional[str] = None,
    enable_cross_validation: bool = True
) -> PatternAnalysisAdapter:
    """
    Factory function to create a PatternAnalysisAdapter.
    
    This provides a drop-in replacement for GeminiClient for pattern detection.
    
    Args:
        config_path: Path to configuration file
        ml_model_path: Path to ML model
        enable_cross_validation: Whether to enable cross-validation
        
    Returns:
        Configured PatternAnalysisAdapter
        
    Example:
        >>> # Replace GeminiClient with pattern framework
        >>> adapter = create_pattern_adapter()
        >>> result = adapter.analyze_chart("chart.png", "AAPL")
    """
    return PatternAnalysisAdapter(
        config_path=config_path,
        ml_model_path=ml_model_path,
        enable_cross_validation=enable_cross_validation
    )


class HybridChartAnalyzer:
    """
    Hybrid analyzer that combines the new pattern framework with
    existing Gemini analysis for comprehensive chart analysis.
    
    Uses the pattern framework for visual pattern detection and
    optionally combines with Gemini for additional context.
    
    Example:
        >>> analyzer = HybridChartAnalyzer()
        >>> result = analyzer.analyze("chart.png", "AAPL")
    """
    
    def __init__(
        self,
        pattern_adapter: Optional[PatternAnalysisAdapter] = None,
        gemini_client: Optional[Any] = None,
        use_gemini: bool = False
    ):
        """
        Initialize the hybrid analyzer.
        
        Args:
            pattern_adapter: Pattern analysis adapter
            gemini_client: Optional GeminiClient for additional analysis
            use_gemini: Whether to use Gemini for additional context
        """
        self.pattern_adapter = pattern_adapter or create_pattern_adapter()
        self.gemini_client = gemini_client
        self.use_gemini = use_gemini and gemini_client is not None
    
    def analyze(
        self,
        image_path: Union[str, Path],
        symbol: str,
        timeframe: str = "1D",
        additional_context: str = ""
    ) -> LegacyAnalysisResult:
        """
        Analyze a chart using the hybrid approach.
        
        Args:
            image_path: Path to chart image
            symbol: Ticker symbol
            timeframe: Chart timeframe
            additional_context: Additional context for analysis
            
        Returns:
            LegacyAnalysisResult with combined analysis
        """
        # Run pattern framework analysis
        result = self.pattern_adapter.analyze_chart(
            image_path, symbol, timeframe, additional_context
        )
        
        # Optionally enhance with Gemini
        if self.use_gemini and self.gemini_client:
            try:
                gemini_result = self.gemini_client.analyze_chart(
                    image_path, symbol, timeframe, additional_context
                )
                # Merge results (prefer framework patterns, use Gemini for levels)
                result = self._merge_results(result, gemini_result)
            except Exception as e:
                logger.warning(f"Gemini analysis failed, using framework only: {e}")
        
        return result
    
    def _merge_results(
        self,
        framework_result: LegacyAnalysisResult,
        gemini_result: Any
    ) -> LegacyAnalysisResult:
        """
        Merge framework and Gemini results.
        
        Prefers framework patterns but uses Gemini for price levels.
        """
        # Use framework pattern if detected, otherwise use Gemini
        if framework_result.pattern_detected == "none" and gemini_result.pattern_detected != "none":
            framework_result.pattern_detected = gemini_result.pattern_detected
            framework_result.pattern_confidence = gemini_result.pattern_confidence
            framework_result.pattern_box = gemini_result.pattern_box
        
        # Use Gemini for price levels (framework doesn't have price data)
        if framework_result.support_level is None:
            framework_result.support_level = gemini_result.support_level
        if framework_result.resistance_level is None:
            framework_result.resistance_level = gemini_result.resistance_level
        if framework_result.fibonacci_level is None:
            framework_result.fibonacci_level = gemini_result.fibonacci_level
        
        # Combine summaries
        if gemini_result.analysis_summary:
            framework_result.analysis_summary = (
                f"{framework_result.analysis_summary} "
                f"Additional context: {gemini_result.analysis_summary}"
            )
        
        return framework_result

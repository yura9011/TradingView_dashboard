"""
ChartAnalyzer - Unified chart analysis using local or cloud models.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from chart analysis."""
    # Final decision
    signal_type: str  # candidate, pending, not_candidate
    overall_confidence: float
    
    # Pattern data
    pattern: str
    pattern_confidence: float
    pattern_box: Optional[tuple]
    
    # Trend data
    trend: str
    trend_strength: str
    phase: str
    wave: Optional[str]
    
    # Levels
    support: Optional[float]
    resistance: Optional[float]
    fibonacci: Optional[str]
    key_level: Optional[float]
    
    # Risk
    risk_assessment: str
    stop_loss: str
    position_size: str
    
    # Sentiment
    sentiment_score: float
    sentiment_label: str
    
    # Summary
    summary: str
    detailed_reasoning: Optional[str]
    veto_reason: Optional[str] = None


class ChartAnalyzer:
    """Unified chart analyzer supporting local and cloud models."""
    
    def __init__(
        self,
        use_local: bool = True,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        api_key: Optional[str] = None,
    ):
        """Initialize analyzer.
        
        Args:
            use_local: Use local model (True) or Gemini API (False)
            model_name: HuggingFace model name for local mode
            api_key: Gemini API key for cloud mode
        """
        self.use_local = use_local
        self.model_name = model_name
        self.api_key = api_key
        self._coordinator = None
    
    def _get_coordinator(self):
        """Lazy load coordinator."""
        if self._coordinator is None:
            if self.use_local:
                from src.agents.coordinator_local import get_coordinator_local
                self._coordinator = get_coordinator_local(model_name=self.model_name)
            else:
                from src.agents.coordinator import get_coordinator
                self._coordinator = get_coordinator(api_key=self.api_key)
        return self._coordinator
    
    def analyze(
        self,
        image_path: str,
        symbol: str,
        additional_context: str = "",
    ) -> AnalysisResult:
        """Run analysis on chart image.
        
        Args:
            image_path: Path to chart image
            symbol: Stock symbol
            additional_context: Additional context for analysis
            
        Returns:
            AnalysisResult with all analysis data
        """
        coordinator = self._get_coordinator()
        
        if self.use_local:
            result = coordinator.analyze(image_path, symbol, additional_context)
        else:
            result = coordinator.analyze(image_path, symbol)
        
        return AnalysisResult(
            signal_type=result.signal_type,
            overall_confidence=result.overall_confidence,
            pattern=result.pattern,
            pattern_confidence=result.pattern_confidence,
            pattern_box=result.pattern_box,
            trend=result.trend,
            trend_strength=result.trend_strength,
            phase=result.phase,
            wave=result.wave,
            support=result.support,
            resistance=result.resistance,
            fibonacci=result.fibonacci,
            key_level=result.key_level,
            risk_assessment=result.risk_assessment,
            stop_loss=result.stop_loss,
            position_size=result.position_size,
            sentiment_score=getattr(result, 'sentiment_score', 0.0) or getattr(result, 'news_sentiment', 0.0),
            sentiment_label=getattr(result, 'sentiment_label', 'Neutral') or getattr(result, 'news_label', 'Neutral'),
            summary=result.summary,
            detailed_reasoning=result.detailed_reasoning,
            veto_reason=getattr(result, 'veto_reason', None),
        )

"""
Coordinator Agent (Local) - Orchestrates specialist agents using Qwen2-VL.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from src.agents.specialists.pattern_detector_local import PatternDetectorAgentLocal
from src.agents.specialists.trend_analyst_local import TrendAnalystAgentLocal
from src.agents.specialists.levels_calculator_local import LevelsCalculatorAgentLocal
from src.agents.specialists.base_agent_local import AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class CoordinatedAnalysis:
    """Result from coordinated multi-agent analysis."""
    
    signal_type: str
    overall_confidence: float
    pattern: str
    pattern_confidence: float
    pattern_box: Optional[tuple]
    pattern_description: Optional[str]
    trend: str
    trend_strength: str
    phase: str
    wave: Optional[str]
    trend_description: Optional[str]
    support: Optional[float]
    resistance: Optional[float]
    support_secondary: Optional[float]
    resistance_secondary: Optional[float]
    fibonacci: Optional[str]
    key_level: Optional[float]
    levels_description: Optional[str]
    summary: str
    detailed_reasoning: Optional[str]
    raw_pattern: Dict[str, Any]
    raw_trend: Dict[str, Any]
    raw_levels: Dict[str, Any]


class CoordinatorAgentLocal:
    """Orchestrates specialist agents using local Qwen2-VL model."""
    
    def __init__(self, model_name: str = None):
        """Initialize coordinator with all specialist agents.
        
        Args:
            model_name: HuggingFace model name (default: Qwen/Qwen2-VL-7B-Instruct)
        """
        self.model_name = model_name or "Qwen/Qwen2-VL-7B-Instruct"
        
        logger.info("Initializing local specialist agents...")
        logger.info(f"Model: {self.model_name}")
        
        # Initialize specialists (they share the same model via LocalModelManager)
        self.pattern_detector = PatternDetectorAgentLocal(model_name=self.model_name)
        self.trend_analyst = TrendAnalystAgentLocal(model_name=self.model_name)
        self.levels_calculator = LevelsCalculatorAgentLocal(model_name=self.model_name)
        
        logger.info("All local specialist agents initialized")
    
    def _build_enhanced_summary(
        self,
        pattern_name: str,
        pattern_conf: float,
        has_pattern: bool,
        trend_dir: str,
        trend_strength: str,
        phase: str,
        wyckoff_event: Optional[str],
        wave: Optional[str],
        wave_count: Optional[str],
        support: Optional[float],
        resistance: Optional[float],
        key_level: Optional[float],
        key_level_reason: Optional[str],
        target: Optional[str],
        invalidation: Optional[str],
        fibonacci_confluence: Optional[str],
    ) -> str:
        """Build an enhanced, detailed summary combining all analysis."""
        
        parts = []
        
        if has_pattern and pattern_name != "none":
            pattern_text = f"{pattern_name.title()} pattern detected ({pattern_conf:.0%} confidence)"
            if target:
                pattern_text += f". Target: {target}"
            if invalidation:
                pattern_text += f". Invalidation: {invalidation}"
            parts.append(pattern_text + ".")
        else:
            parts.append("No clear chart pattern identified.")
        
        trend_text = ""
        if trend_dir != "unknown":
            trend_text = f"Trend: {trend_dir} ({trend_strength})"
            if phase and phase != "unclear":
                trend_text += f", Wyckoff phase: {phase}"
                if wyckoff_event:
                    trend_text += f" ({wyckoff_event} event)"
            parts.append(trend_text + ".")
        
        if wave or wave_count:
            wave_text = "Elliott Wave: "
            if wave:
                wave_text += wave
            if wave_count:
                wave_text += f" ({wave_count})"
            parts.append(wave_text + ".")
        
        levels_parts = []
        if key_level:
            level_text = f"Key level: ${key_level:,.0f}"
            if key_level_reason:
                level_text += f" - {key_level_reason}"
            levels_parts.append(level_text)
        
        if support and resistance:
            levels_parts.append(f"Range: ${support:,.0f} (support) to ${resistance:,.0f} (resistance)")
        
        if fibonacci_confluence:
            levels_parts.append(f"Fibonacci: {fibonacci_confluence}")
        
        if levels_parts:
            parts.append(" | ".join(levels_parts) + ".")
        
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
        logger.info(f"Starting local coordinated analysis for {symbol or 'chart'}")
        
        # Build context with symbol and any additional info (like OCR price range)
        context_parts = []
        if symbol:
            context_parts.append(f"Symbol: {symbol}")
        if additional_context:
            context_parts.append(additional_context)
        context = " | ".join(context_parts)
        
        # Step 1: Pattern Detection
        logger.info("🔍 Step 1/3: Pattern Detection (local)...")
        pattern_result = self.pattern_detector.analyze(image_path, context)
        
        # Step 2: Trend Analysis
        logger.info("📈 Step 2/3: Trend Analysis (local)...")
        trend_result = self.trend_analyst.analyze(image_path, context)
        
        # Step 3: Levels Calculation
        logger.info("📊 Step 3/3: Levels Calculation (local)...")
        levels_result = self.levels_calculator.analyze(image_path, context)
        
        # Synthesize results
        logger.info("🧠 Synthesizing findings...")
        analysis = self._synthesize(
            pattern_result,
            trend_result,
            levels_result,
            symbol,
        )
        
        logger.info(f"Analysis complete: {analysis.signal_type} (confidence: {analysis.overall_confidence:.0%})")
        
        return analysis
    
    def _synthesize(
        self,
        pattern: AgentResponse,
        trend: AgentResponse,
        levels: AgentResponse,
        symbol: str,
    ) -> CoordinatedAnalysis:
        """Synthesize findings from all agents into final decision."""
        
        # Log raw responses for debugging
        logger.info(f"Pattern agent success: {pattern.success}, error: {pattern.error}")
        logger.info(f"Trend agent success: {trend.success}, error: {trend.error}")
        logger.info(f"Levels agent success: {levels.success}, error: {levels.error}")
        
        p = pattern.parsed
        t = trend.parsed
        l = levels.parsed
        
        logger.info(f"Parsed pattern: {p}")
        logger.info(f"Parsed trend: {t}")
        logger.info(f"Parsed levels: {l}")
        
        pattern_conf = p.get("confidence", 0.0)
        has_pattern = p.get("pattern", "none") != "none"
        trend_dir = t.get("trend", "unknown")
        trend_strength = t.get("strength", "unknown")
        phase = t.get("phase", "unclear")
        
        if pattern_conf >= 0.7 and has_pattern:
            signal_type = "candidate"
            overall_confidence = pattern_conf
        elif pattern_conf >= 0.4 or (trend_strength == "strong" and trend_dir != "sideways"):
            signal_type = "pending"
            overall_confidence = max(pattern_conf, 0.5) if has_pattern else 0.4
        else:
            signal_type = "not_candidate"
            overall_confidence = pattern_conf
        
        pattern_name = p.get("pattern", "none")
        if has_pattern:
            bullish_patterns = ["double bottom", "inverse head", "bullish engulfing", "hammer"]
            bearish_patterns = ["double top", "head and shoulders", "bearish engulfing", "shooting star"]
            
            is_bullish_pattern = any(bp in pattern_name for bp in bullish_patterns)
            is_bearish_pattern = any(bp in pattern_name for bp in bearish_patterns)
            
            if (is_bullish_pattern and trend_dir == "down") or (is_bearish_pattern and trend_dir == "up"):
                overall_confidence *= 0.9
        
        summary = self._build_enhanced_summary(
            pattern_name=pattern_name,
            pattern_conf=pattern_conf,
            has_pattern=has_pattern,
            trend_dir=trend_dir,
            trend_strength=trend_strength,
            phase=phase,
            wyckoff_event=t.get("wyckoff_event"),
            wave=t.get("wave"),
            wave_count=t.get("wave_count"),
            support=l.get("support"),
            resistance=l.get("resistance"),
            key_level=l.get("key_level"),
            key_level_reason=l.get("key_level_reason"),
            target=p.get("target"),
            invalidation=p.get("invalidation"),
            fibonacci_confluence=l.get("fibonacci_confluence"),
        )
        
        detailed_reasoning = json.dumps({
            "pattern": {
                "name": pattern_name,
                "confidence": pattern_conf,
                "components": p.get("components"),
                "target": p.get("target"),
                "invalidation": p.get("invalidation"),
                "reasoning": p.get("description", ""),
                "box": p.get("pattern_box"),
            },
            "trend": {
                "direction": trend_dir,
                "strength": trend_strength,
                "reasoning": t.get("description", ""),
            },
            "wyckoff": {
                "phase": phase,
                "event": t.get("wyckoff_event"),
                "reasoning": t.get("description", ""),
            },
            "elliott": {
                "wave": t.get("wave"),
                "wave_count": t.get("wave_count"),
                "reasoning": t.get("description", ""),
            },
            "levels": {
                "support": l.get("support"),
                "support_reason": l.get("support_reason"),
                "support_secondary": l.get("support_secondary"),
                "resistance": l.get("resistance"),
                "resistance_reason": l.get("resistance_reason"),
                "resistance_secondary": l.get("resistance_secondary"),
                "fibonacci": l.get("fibonacci"),
                "fibonacci_confluence": l.get("fibonacci_confluence"),
                "key_level": l.get("key_level"),
                "key_level_reason": l.get("key_level_reason"),
                "reasoning": l.get("description", ""),
            },
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
            summary=summary,
            detailed_reasoning=detailed_reasoning,
            raw_pattern=p,
            raw_trend=t,
            raw_levels=l,
        )


def get_coordinator_local(model_name: str = None) -> CoordinatorAgentLocal:
    """Factory function for CoordinatorAgentLocal."""
    return CoordinatorAgentLocal(model_name=model_name)

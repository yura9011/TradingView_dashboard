"""
Coordinator Agent - Orchestrates specialist agents and synthesizes results.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from src.agents.specialists.pattern_detector import PatternDetectorAgent
from src.agents.specialists.trend_analyst import TrendAnalystAgent
from src.agents.specialists.levels_calculator import LevelsCalculatorAgent
from src.agents.specialists.base_agent import AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class CoordinatedAnalysis:
    """Result from coordinated multi-agent analysis."""
    
    # Final decision
    signal_type: str  # candidate, pending, not_candidate
    overall_confidence: float
    
    # Pattern data
    pattern: str
    pattern_confidence: float
    pattern_box: Optional[tuple]
    pattern_description: Optional[str]  # NEW: Why this pattern
    
    # Trend data
    trend: str
    trend_strength: str
    phase: str
    wave: Optional[str]
    trend_description: Optional[str]  # NEW: Why this trend/phase
    
    # Levels data
    support: Optional[float]
    resistance: Optional[float]
    support_secondary: Optional[float]  # NEW
    resistance_secondary: Optional[float]  # NEW
    fibonacci: Optional[str]
    key_level: Optional[float]
    levels_description: Optional[str]  # NEW: Why these levels
    
    # Summary
    summary: str
    detailed_reasoning: Optional[str]  # NEW: Full reasoning JSON
    
    # Raw responses
    raw_pattern: Dict[str, Any]
    raw_trend: Dict[str, Any]
    raw_levels: Dict[str, Any]


class CoordinatorAgent:
    """Orchestrates specialist agents for comprehensive analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize coordinator with all specialist agents.
        
        Args:
            api_key: Gemini API key (shared across agents)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Initialize specialists
        logger.info("Initializing specialist agents...")
        self.pattern_detector = PatternDetectorAgent(api_key=self.api_key)
        self.trend_analyst = TrendAnalystAgent(api_key=self.api_key)
        self.levels_calculator = LevelsCalculatorAgent(api_key=self.api_key)
        
        logger.info("All specialist agents initialized")
    
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
        
        # Pattern section
        if has_pattern and pattern_name != "none":
            pattern_text = f"{pattern_name.title()} pattern detected ({pattern_conf:.0%} confidence)"
            if target:
                pattern_text += f". Target: {target}"
            if invalidation:
                pattern_text += f". Invalidation: {invalidation}"
            parts.append(pattern_text + ".")
        else:
            parts.append("No clear chart pattern identified.")
        
        # Trend and Wyckoff section
        trend_text = ""
        if trend_dir != "unknown":
            trend_text = f"Trend: {trend_dir} ({trend_strength})"
            if phase and phase != "unclear":
                trend_text += f", Wyckoff phase: {phase}"
                if wyckoff_event:
                    trend_text += f" ({wyckoff_event} event)"
            parts.append(trend_text + ".")
        
        # Elliott Wave section
        if wave or wave_count:
            wave_text = "Elliott Wave: "
            if wave:
                wave_text += wave
            if wave_count:
                wave_text += f" ({wave_count})"
            parts.append(wave_text + ".")
        
        # Key levels section
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
    
    def analyze(self, image_path: str, symbol: str = "") -> CoordinatedAnalysis:
        """Run full multi-agent analysis pipeline.
        
        Args:
            image_path: Path to chart image
            symbol: Optional symbol for context
            
        Returns:
            CoordinatedAnalysis with all findings
        """
        logger.info(f"Starting coordinated analysis for {symbol or 'chart'}")
        
        context = f"Symbol: {symbol}" if symbol else ""
        
        # Step 1: Pattern Detection
        logger.info("🔍 Step 1/3: Pattern Detection...")
        pattern_result = self.pattern_detector.analyze(image_path, context)
        
        # Step 2: Trend Analysis
        logger.info("📈 Step 2/3: Trend Analysis...")
        trend_result = self.trend_analyst.analyze(image_path, context)
        
        # Step 3: Levels Calculation
        logger.info("📊 Step 3/3: Levels Calculation...")
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
        
        p = pattern.parsed
        t = trend.parsed
        l = levels.parsed
        
        # Calculate overall confidence and signal type
        pattern_conf = p.get("confidence", 0.0)
        has_pattern = p.get("pattern", "none") != "none"
        trend_dir = t.get("trend", "unknown")
        trend_strength = t.get("strength", "unknown")
        phase = t.get("phase", "unclear")
        
        # Decision logic
        if pattern_conf >= 0.7 and has_pattern:
            signal_type = "candidate"
            overall_confidence = pattern_conf
        elif pattern_conf >= 0.4 or (trend_strength == "strong" and trend_dir != "sideways"):
            signal_type = "pending"
            overall_confidence = max(pattern_conf, 0.5) if has_pattern else 0.4
        else:
            signal_type = "not_candidate"
            overall_confidence = pattern_conf
        
        # Adjust for trend alignment
        pattern_name = p.get("pattern", "none")
        if has_pattern:
            # Check if pattern aligns with trend
            bullish_patterns = ["double bottom", "inverse head", "bullish engulfing", "hammer"]
            bearish_patterns = ["double top", "head and shoulders", "bearish engulfing", "shooting star"]
            
            is_bullish_pattern = any(bp in pattern_name for bp in bullish_patterns)
            is_bearish_pattern = any(bp in pattern_name for bp in bearish_patterns)
            
            if (is_bullish_pattern and trend_dir == "down") or (is_bearish_pattern and trend_dir == "up"):
                # Pattern against trend - could be reversal, slightly reduce confidence
                overall_confidence *= 0.9
        
        # Build enhanced summary
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
        
        # Build detailed reasoning JSON
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


def get_coordinator(api_key: Optional[str] = None) -> CoordinatorAgent:
    """Factory function for CoordinatorAgent."""
    return CoordinatorAgent(api_key=api_key)

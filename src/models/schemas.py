"""
Pydantic models for AI Trading Analysis Agent.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class Market(str, Enum):
    """Supported markets."""
    AMERICA = "america"
    CRYPTO = "crypto"
    FOREX = "forex"
    CFD = "cfd"
    FUTURES = "futures"


class Timeframe(str, Enum):
    """Supported timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1W"
    MO1 = "1M"


class Asset(BaseModel):
    """Represents a tradeable asset/ticker."""
    
    symbol: str = Field(..., description="Ticker symbol (e.g., AAPL, BTCUSD)")
    name: Optional[str] = Field(None, description="Full asset name")
    market: Market = Field(Market.AMERICA, description="Market type")
    exchange: Optional[str] = Field(None, description="Exchange (e.g., NASDAQ)")
    
    # Price data
    price: Optional[float] = Field(None, description="Current price")
    change_percent: Optional[float] = Field(None, description="Price change %")
    volume: Optional[float] = Field(None, description="Trading volume")
    
    # Additional metadata
    sector: Optional[str] = Field(None, description="Industry sector")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    
    model_config = ConfigDict(use_enum_values=True)


class IndicatorValues(BaseModel):
    """Technical indicator values for an asset."""
    
    symbol: str = Field(..., description="Ticker symbol")
    timeframe: Timeframe = Field(Timeframe.D1, description="Analysis timeframe")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # EMAs
    ema_21: Optional[float] = Field(None, description="21-period EMA")
    ema_50: Optional[float] = Field(None, description="50-period EMA")
    ema_200: Optional[float] = Field(None, description="200-period EMA")
    
    # RSI
    rsi: Optional[float] = Field(None, description="14-period RSI")
    rsi_oversold: bool = Field(False, description="RSI < 30")
    rsi_overbought: bool = Field(False, description="RSI > 70")
    
    # MACD
    macd: Optional[float] = Field(None, description="MACD line")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram")
    
    # Volume
    volume_sma_20: Optional[float] = Field(None, description="20-period Volume SMA")
    
    model_config = ConfigDict(use_enum_values=True)


class SignalType(str, Enum):
    """Types of trading signals."""
    CANDIDATE = "candidate"
    NOT_CANDIDATE = "not_candidate"
    PENDING = "pending"


class PatternType(str, Enum):
    """Detected chart patterns."""
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    DOUBLE_BOTTOM = "double_bottom"
    DOUBLE_TOP = "double_top"
    HEAD_SHOULDERS = "head_shoulders"
    TRIANGLE = "triangle"
    WEDGE = "wedge"
    NONE = "none"


class Signal(BaseModel):
    """A trading signal/opportunity for review."""
    
    id: Optional[int] = Field(None, description="Database ID")
    symbol: str = Field(..., description="Ticker symbol")
    signal_type: SignalType = Field(SignalType.PENDING, description="Signal classification")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Analysis results
    pattern_detected: PatternType = Field(PatternType.NONE)
    pattern_confidence: float = Field(0.0, ge=0.0, le=1.0)
    
    # Technical context
    trend: Optional[str] = Field(None, description="Current trend (up/down/sideways)")
    trend_strength: Optional[str] = Field(None, description="Trend strength (strong/moderate/weak)")
    market_phase: Optional[str] = Field(None, description="Wyckoff phase (accumulation/markup/distribution/markdown)")
    elliott_wave: Optional[str] = Field(None, description="Elliott wave context")
    support_level: Optional[float] = Field(None, description="Key support price")
    resistance_level: Optional[float] = Field(None, description="Key resistance price")
    fibonacci_level: Optional[str] = Field(None, description="Relevant Fib level")
    
    # Sentiment (for future use)
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    
    # Reasoning
    analysis_summary: Optional[str] = Field(None, description="AI analysis summary")
    detailed_reasoning: Optional[str] = Field(None, description="Detailed reasoning from all agents (JSON)")
    
    # Visual report
    chart_image_path: Optional[str] = Field(None, description="Path to annotated chart")
    report_path: Optional[str] = Field(None, description="Path to markdown report")
    
    # Status
    reviewed: bool = Field(False, description="Has been reviewed by human")
    notes: Optional[str] = Field(None, description="Human notes/feedback")
    
    model_config = ConfigDict(use_enum_values=True)


class ScreenerResult(BaseModel):
    """Result from a screener query."""
    
    assets: List[Asset] = Field(default_factory=list)
    market: Market = Field(Market.AMERICA)
    timeframe: Timeframe = Field(Timeframe.D1)
    filters_applied: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_count: int = Field(0)
    
    model_config = ConfigDict(use_enum_values=True)

"""
Enumerations for pattern analysis.

Contains PatternCategory, PatternType, RegionType, Timeframe, and CandleInterval
enums as defined in the design document.
"""

from enum import Enum


class RegionType(Enum):
    """Types of chart regions for region detection."""
    PRIMARY_CHART = "primary_chart"      # Main price chart (candlesticks)
    VOLUME_PANEL = "volume_panel"        # Volume bars panel
    INDICATOR_PANEL = "indicator_panel"  # RSI, MACD, etc.
    TOOLBAR = "toolbar"                  # UI elements
    UNKNOWN = "unknown"


class Timeframe(Enum):
    """Supported timeframes for analysis."""
    DAY_1 = "1D"
    WEEK_1 = "1W"
    MONTH_1 = "1M"
    MONTH_3 = "3M"
    MONTH_6 = "6M"
    YTD = "YTD"
    YEAR_1 = "1Y"
    YEAR_5 = "5Y"


class CandleInterval(Enum):
    """Candle interval settings."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1D"
    WEEKLY = "1W"
    MONTHLY = "1M"


class PatternCategory(Enum):
    """Primary pattern categories for chart patterns."""
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    BILATERAL = "bilateral"


class PatternType(Enum):
    """Specific pattern types that can be detected in charts."""
    # Reversal patterns
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    
    # Continuation patterns
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    CUP_AND_HANDLE = "cup_and_handle"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"
    
    # Triangle patterns (can be continuation or bilateral)
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    
    # Wedge patterns
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"

"""
Enumerations for pattern analysis.

Contains PatternCategory and PatternType enums as defined in the design document.
"""

from enum import Enum


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

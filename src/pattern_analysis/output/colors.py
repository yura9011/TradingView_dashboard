"""
Color scheme and utilities for chart annotation.

Feature: chart-pattern-analysis-framework
Requirements: 8.2
"""

from typing import Dict, Any, Tuple

from ..models.enums import PatternCategory, PatternType


# Color scheme: BGR format for OpenCV
DEFAULT_COLORS: Dict[PatternCategory, Any] = {
    PatternCategory.REVERSAL: {
        "bullish": (0, 200, 0),      # Green
        "bearish": (0, 0, 200),      # Red
        "default": (0, 200, 200),    # Yellow
    },
    PatternCategory.CONTINUATION: {
        "bullish": (0, 255, 100),    # Light green
        "bearish": (100, 100, 255),  # Light red
        "default": (0, 200, 200),    # Yellow
    },
    PatternCategory.BILATERAL: (0, 255, 255),  # Yellow
}

# Bullish patterns (typically green)
BULLISH_PATTERNS = {
    PatternType.INVERSE_HEAD_SHOULDERS,
    PatternType.DOUBLE_BOTTOM,
    PatternType.TRIPLE_BOTTOM,
    PatternType.ASCENDING_TRIANGLE,
    PatternType.BULL_FLAG,
    PatternType.CUP_AND_HANDLE,
    PatternType.FALLING_WEDGE,
    PatternType.CHANNEL_UP,
}

# Bearish patterns (typically red)
BEARISH_PATTERNS = {
    PatternType.HEAD_SHOULDERS,
    PatternType.DOUBLE_TOP,
    PatternType.TRIPLE_TOP,
    PatternType.DESCENDING_TRIANGLE,
    PatternType.BEAR_FLAG,
    PatternType.RISING_WEDGE,
    PatternType.CHANNEL_DOWN,
}


def get_color_for_category(
    category: PatternCategory,
    colors: Dict[PatternCategory, Any] = None
) -> Tuple[int, int, int]:
    """Get the default color for a pattern category."""
    colors = colors or DEFAULT_COLORS
    color = colors.get(category, (0, 200, 200))
    if isinstance(color, tuple):
        return color
    return color.get("default", (0, 200, 200))


def get_color_for_pattern(
    category: PatternCategory,
    pattern_type: PatternType,
    colors: Dict[PatternCategory, Any] = None
) -> Tuple[int, int, int]:
    """Get the appropriate color for a pattern based on category and type."""
    colors = colors or DEFAULT_COLORS
    
    # Handle bilateral patterns (always yellow)
    if category == PatternCategory.BILATERAL:
        color = colors.get(PatternCategory.BILATERAL, (0, 255, 255))
        if isinstance(color, tuple):
            return color
        return color.get("default", (0, 255, 255))
    
    # Determine if bullish or bearish
    if pattern_type in BULLISH_PATTERNS:
        direction = "bullish"
    elif pattern_type in BEARISH_PATTERNS:
        direction = "bearish"
    else:
        direction = "default"
    
    # Get color from category colors
    category_colors = colors.get(category, {})
    if isinstance(category_colors, tuple):
        return category_colors
    
    return category_colors.get(direction, (0, 200, 200))


def get_contrasting_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Get a contrasting color (white or black) for text readability."""
    b, g, r = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)

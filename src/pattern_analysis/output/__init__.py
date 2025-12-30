"""
Output generation module.

Contains components for:
- ChartAnnotator: Visual annotation of detected patterns
- ReportGenerator: Markdown and JSON report generation

Requirements:
- 8.1-8.5: Visual annotation of detected patterns
"""

from .annotator import ChartAnnotator
from .colors import (
    DEFAULT_COLORS,
    BULLISH_PATTERNS,
    BEARISH_PATTERNS,
    get_color_for_category,
    get_color_for_pattern,
    get_contrasting_color,
)
from .drawing import DrawingUtils

__all__ = [
    "ChartAnnotator",
    "DrawingUtils",
    "DEFAULT_COLORS",
    "BULLISH_PATTERNS",
    "BEARISH_PATTERNS",
    "get_color_for_category",
    "get_color_for_pattern",
    "get_contrasting_color",
]

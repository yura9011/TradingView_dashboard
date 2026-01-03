"""
Pattern Reference Comparison Module.

Provides functionality to compare detected patterns with reference images
from trading books to validate pattern detection accuracy.
"""

from .models import ReferenceImage, MatchResult, PatternMatch
from .manager import ReferenceManager
from .extractor import RegionExtractor
from .matcher import PatternMatcher
from .logger import ProgressLogger

__all__ = [
    "ReferenceImage",
    "MatchResult",
    "PatternMatch",
    "ReferenceManager",
    "RegionExtractor",
    "PatternMatcher",
    "ProgressLogger",
]

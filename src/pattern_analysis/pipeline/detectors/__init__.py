"""
Pattern detectors module.
"""

from .triangles import TriangleDetector
from .doubles import DoublePatternDetector
from .head_shoulders import HeadShouldersDetector
from .ml_detector import MLDetector
from .merger import DetectionMerger

__all__ = [
    "TriangleDetector",
    "DoublePatternDetector",
    "HeadShouldersDetector",
    "MLDetector",
    "DetectionMerger",
]

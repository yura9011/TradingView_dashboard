"""
Data models for the pattern analysis framework.

Contains enums, dataclasses, and Pydantic models for:
- Pattern categories and types
- Region types, timeframes, and candle intervals
- Bounding boxes and detections
- Feature maps and analysis results
- Chart regions and crop results
"""

from .enums import (
    PatternCategory,
    PatternType,
    RegionType,
    Timeframe,
    CandleInterval,
)
from .dataclasses import (
    BoundingBox,
    PatternDetection,
    FeatureMap,
    PreprocessResult,
    ValidationResult,
    AnalysisResult,
    ChartRegion,
    RegionDetectionResult,
    CropResult,
    TimeframeConfig,
)
from .schemas import (
    ANALYSIS_RESULT_SCHEMA,
    SchemaValidationError,
    validate_against_schema,
)

__all__ = [
    # Enums
    "PatternCategory",
    "PatternType",
    "RegionType",
    "Timeframe",
    "CandleInterval",
    # Dataclasses
    "BoundingBox",
    "PatternDetection",
    "FeatureMap",
    "PreprocessResult",
    "ValidationResult",
    "AnalysisResult",
    "ChartRegion",
    "RegionDetectionResult",
    "CropResult",
    "TimeframeConfig",
    # Schema utilities
    "ANALYSIS_RESULT_SCHEMA",
    "SchemaValidationError",
    "validate_against_schema",
]

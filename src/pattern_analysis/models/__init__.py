"""
Data models for the pattern analysis framework.

Contains enums, dataclasses, and Pydantic models for:
- Pattern categories and types
- Bounding boxes and detections
- Feature maps and analysis results
"""

from .enums import PatternCategory, PatternType
from .dataclasses import (
    BoundingBox,
    PatternDetection,
    FeatureMap,
    PreprocessResult,
    ValidationResult,
    AnalysisResult,
)
from .schemas import (
    ANALYSIS_RESULT_SCHEMA,
    SchemaValidationError,
    validate_against_schema,
)

__all__ = [
    "PatternCategory",
    "PatternType",
    "BoundingBox",
    "PatternDetection",
    "FeatureMap",
    "PreprocessResult",
    "ValidationResult",
    "AnalysisResult",
    "ANALYSIS_RESULT_SCHEMA",
    "SchemaValidationError",
    "validate_against_schema",
]

"""
Chart Pattern Analysis Framework

A modular framework for detecting, classifying, and validating
chart patterns in financial price charts using computer vision
and machine learning techniques.
"""

__version__ = "0.1.0"

# Import registry components
from .registry import (
    PatternRegistry,
    PatternDefinition,
    PatternComponent,
    RegistryValidationError,
)

# Import factory functions for easy access
from .factory import (
    create_pipeline,
    create_analyzer,
    create_preprocessor,
    create_feature_extractor,
    create_classifier,
    create_cross_validator,
    create_annotator,
    create_pattern_registry,
    ChartPatternAnalyzer,
    PatternAnalyzer,
)

# Import integration components
from .integration import (
    PatternAnalysisAdapter,
    LegacyAnalysisResult,
    HybridChartAnalyzer,
    create_pattern_adapter,
)

# Import models for type hints
from .models.dataclasses import (
    AnalysisResult,
    PatternDetection,
    BoundingBox,
    FeatureMap,
    PreprocessResult,
    ValidationResult,
)

from .models.enums import (
    PatternCategory,
    PatternType,
)

__all__ = [
    # Version
    "__version__",
    # Factory functions
    "create_pipeline",
    "create_analyzer",
    "create_preprocessor",
    "create_feature_extractor",
    "create_classifier",
    "create_cross_validator",
    "create_annotator",
    "create_pattern_registry",
    "create_pattern_adapter",
    # Analyzer classes
    "ChartPatternAnalyzer",
    "PatternAnalyzer",
    # Integration
    "PatternAnalysisAdapter",
    "LegacyAnalysisResult",
    "HybridChartAnalyzer",
    # Registry
    "PatternRegistry",
    "PatternDefinition",
    "PatternComponent",
    "RegistryValidationError",
    # Models
    "AnalysisResult",
    "PatternDetection",
    "BoundingBox",
    "FeatureMap",
    "PreprocessResult",
    "ValidationResult",
    # Enums
    "PatternCategory",
    "PatternType",
]

"""
Pipeline components for the pattern analysis framework.

Contains the processing pipeline stages:
- PipelineStage: Base abstract class for all stages
- Preprocessor: Image normalization and enhancement interface
- FeatureExtractor: Visual feature extraction interface
- PatternClassifier: Pattern classification interface
- CrossValidator: Result validation interface
- StandardPreprocessor: Default preprocessor implementation
- EdgeBasedFeatureExtractor: Default feature extractor implementation
- HybridPatternClassifier: Hybrid rule-based and ML classifier
- MultiMethodCrossValidator: Multi-method cross-validation
- PipelineExecutor: Pipeline orchestration and execution

Requirements: 4.1 - THE Pipeline SHALL define a standard interface that all
processing stages must implement.
"""

from .interfaces import (
    PipelineStage,
    Preprocessor,
    FeatureExtractor,
    PatternClassifier,
    CrossValidator,
)
from .preprocessor import (
    StandardPreprocessor,
    ImageNotFoundError,
    ImageCorruptedError,
    ImageTooSmallError,
)
from .region_detector import ChartRegionDetector
from .auto_cropper import AutoCropper
from .enhanced_preprocessor import EnhancedPreprocessor, EnhancedPreprocessResult
from .feature_extractor import EdgeBasedFeatureExtractor
from .classifier import HybridPatternClassifier
from .cross_validator import MultiMethodCrossValidator
from .executor import PipelineExecutor, StageFailureError

__all__ = [
    # Interfaces
    "PipelineStage",
    "Preprocessor",
    "FeatureExtractor",
    "PatternClassifier",
    "CrossValidator",
    # Implementations
    "StandardPreprocessor",
    "ChartRegionDetector",
    "AutoCropper",
    "EnhancedPreprocessor",
    "EnhancedPreprocessResult",
    "EdgeBasedFeatureExtractor",
    "HybridPatternClassifier",
    "MultiMethodCrossValidator",
    "PipelineExecutor",
    # Exceptions
    "ImageNotFoundError",
    "ImageCorruptedError",
    "ImageTooSmallError",
    "StageFailureError",
]

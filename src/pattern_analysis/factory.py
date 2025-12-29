"""
Factory functions for Chart Pattern Analysis Framework.

Provides convenient factory functions to create pre-configured pipeline
components and analyzers for simplified usage.

Requirements: 4.3 - Support configuration-driven stage ordering and parameter tuning
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config.manager import ConfigurationManager, get_config_manager
from .models.dataclasses import AnalysisResult, FeatureMap
from .output.annotator import ChartAnnotator
from .pipeline.classifier import HybridPatternClassifier
from .pipeline.cross_validator import MultiMethodCrossValidator
from .pipeline.executor import PipelineExecutor
from .pipeline.feature_extractor import EdgeBasedFeatureExtractor
from .pipeline.interfaces import (
    CrossValidator,
    FeatureExtractor,
    PatternClassifier,
    Preprocessor,
)
from .pipeline.preprocessor import StandardPreprocessor
from .registry.pattern_registry import PatternRegistry


logger = logging.getLogger(__name__)


# Default paths
DEFAULT_CONFIG_PATH = "config/pattern_analysis.yaml"
DEFAULT_PATTERN_DEFINITIONS_PATH = "config/pattern_definitions.yaml"


def create_preprocessor(config: Optional[Dict[str, Any]] = None) -> StandardPreprocessor:
    """
    Create a configured StandardPreprocessor instance.
    
    Args:
        config: Optional configuration dictionary. If not provided,
               uses default configuration.
               
    Returns:
        Configured StandardPreprocessor instance
    """
    return StandardPreprocessor()


def create_feature_extractor(
    config: Optional[Dict[str, Any]] = None
) -> EdgeBasedFeatureExtractor:
    """
    Create a configured EdgeBasedFeatureExtractor instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured EdgeBasedFeatureExtractor instance
    """
    return EdgeBasedFeatureExtractor()


def create_pattern_registry(
    definitions_path: Optional[str] = None
) -> PatternRegistry:
    """
    Create a PatternRegistry with loaded definitions.
    
    Args:
        definitions_path: Path to pattern definitions YAML file.
                         If not provided, creates empty registry.
                         
    Returns:
        Configured PatternRegistry instance
    """
    registry = PatternRegistry()
    
    if definitions_path and Path(definitions_path).exists():
        try:
            registry.load_from_yaml(definitions_path)
            logger.info(f"Loaded pattern definitions from {definitions_path}")
        except Exception as e:
            logger.warning(f"Failed to load pattern definitions: {e}")
    
    return registry


def create_classifier(
    pattern_registry: Optional[PatternRegistry] = None,
    ml_model_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> HybridPatternClassifier:
    """
    Create a configured HybridPatternClassifier instance.
    
    Args:
        pattern_registry: Optional PatternRegistry for pattern definitions
        ml_model_path: Optional path to ML model (YOLO) for detection
        config: Optional configuration dictionary
        
    Returns:
        Configured HybridPatternClassifier instance
    """
    return HybridPatternClassifier(
        pattern_registry=pattern_registry,
        ml_model_path=ml_model_path
    )


def create_cross_validator(
    validators: Optional[List[PatternClassifier]] = None,
    consensus_threshold: float = 0.5,
    config: Optional[Dict[str, Any]] = None
) -> MultiMethodCrossValidator:
    """
    Create a configured MultiMethodCrossValidator instance.
    
    Args:
        validators: List of PatternClassifier instances for validation
        consensus_threshold: Minimum score for confirmation (0.0-1.0)
        config: Optional configuration dictionary
        
    Returns:
        Configured MultiMethodCrossValidator instance
    """
    return MultiMethodCrossValidator(
        validators=validators or [],
        consensus_threshold=consensus_threshold
    )


def create_annotator(
    overlay_alpha: float = 0.3,
    config: Optional[Dict[str, Any]] = None
) -> ChartAnnotator:
    """
    Create a configured ChartAnnotator instance.
    
    Args:
        overlay_alpha: Transparency for overlays (0.0-1.0)
        config: Optional configuration dictionary
        
    Returns:
        Configured ChartAnnotator instance
    """
    return ChartAnnotator(overlay_alpha=overlay_alpha)


def create_pipeline(
    config_path: Optional[str] = None,
    pattern_definitions_path: Optional[str] = None,
    ml_model_path: Optional[str] = None,
    enable_cross_validation: bool = True,
    consensus_threshold: float = 0.5
) -> PipelineExecutor:
    """
    Create a fully configured pattern analysis pipeline.
    
    Creates and wires together all pipeline components with default
    or custom configuration. This is the main factory function for
    creating a complete analysis pipeline.
    
    Args:
        config_path: Path to configuration YAML file
        pattern_definitions_path: Path to pattern definitions YAML
        ml_model_path: Optional path to ML model for detection
        enable_cross_validation: Whether to enable cross-validation stage
        consensus_threshold: Threshold for cross-validation consensus
        
    Returns:
        Configured PipelineExecutor ready for analysis
        
    Requirements: 4.3
    
    Example:
        >>> pipeline = create_pipeline()
        >>> result = pipeline.execute("chart.png")
        >>> print(f"Found {len(result.detections)} patterns")
    """
    # Load configuration if path provided
    config: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        try:
            manager = ConfigurationManager(config_path=config_path)
            config = manager.config
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
    
    # Create pipeline components
    preprocessor = create_preprocessor(config.get("preprocessing"))
    feature_extractor = create_feature_extractor(config.get("feature_extraction"))
    
    # Create pattern registry
    registry = create_pattern_registry(pattern_definitions_path)
    
    # Create classifier
    classifier = create_classifier(
        pattern_registry=registry,
        ml_model_path=ml_model_path,
        config=config.get("classification")
    )
    
    # Create cross-validator if enabled
    cross_validator: Optional[CrossValidator] = None
    if enable_cross_validation:
        # Create a secondary classifier for validation
        validation_classifier = create_classifier(
            pattern_registry=registry,
            ml_model_path=None,  # Rule-based only for validation
            config=config.get("classification")
        )
        cross_validator = create_cross_validator(
            validators=[validation_classifier],
            consensus_threshold=consensus_threshold,
            config=config.get("cross_validation")
        )
    
    # Create and return pipeline executor
    pipeline = PipelineExecutor(
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        classifier=classifier,
        cross_validator=cross_validator
    )
    
    logger.info(
        f"Created pipeline with stages: {pipeline.stage_ids}, "
        f"cross_validation={'enabled' if cross_validator else 'disabled'}"
    )
    
    return pipeline


class ChartPatternAnalyzer:
    """
    High-level analyzer for chart pattern detection.
    
    Provides a simplified interface for analyzing chart images,
    combining pipeline execution with output generation.
    
    This class wraps the pipeline executor and provides convenient
    methods for common analysis tasks.
    
    Requirements: 4.3
    
    Example:
        >>> analyzer = create_analyzer()
        >>> result = analyzer.analyze("chart.png")
        >>> analyzer.save_annotated("chart.png", result, "annotated_chart.png")
    """
    
    def __init__(
        self,
        pipeline: PipelineExecutor,
        annotator: Optional[ChartAnnotator] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            pipeline: Configured PipelineExecutor
            annotator: Optional ChartAnnotator for visual output
            config: Optional configuration dictionary
        """
        self.pipeline = pipeline
        self.annotator = annotator or create_annotator()
        self.config = config or {}
    
    def analyze(
        self,
        image_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Analyze a chart image for patterns.
        
        Args:
            image_path: Path to the chart image file
            config: Optional configuration overrides
            
        Returns:
            AnalysisResult containing all detections and metadata
        """
        analysis_config = {**self.config, **(config or {})}
        return self.pipeline.execute(image_path, analysis_config)
    
    def analyze_and_annotate(
        self,
        image_path: str,
        output_path: str,
        config: Optional[Dict[str, Any]] = None,
        show_confidence: bool = True,
        show_validation: bool = True
    ) -> AnalysisResult:
        """
        Analyze a chart and save an annotated image.
        
        Args:
            image_path: Path to the chart image file
            output_path: Path to save the annotated image
            config: Optional configuration overrides
            show_confidence: Whether to show confidence scores
            show_validation: Whether to show validation status
            
        Returns:
            AnalysisResult containing all detections and metadata
        """
        import cv2
        
        # Run analysis
        result = self.analyze(image_path, config)
        
        # Load original image for annotation
        image = cv2.imread(image_path)
        if image is not None:
            # Annotate and save
            self.annotator.annotate_and_save(
                image,
                result.detections,
                output_path,
                result.validated_detections,
                show_confidence,
                show_validation
            )
        
        return result
    
    def save_annotated(
        self,
        image_path: str,
        result: AnalysisResult,
        output_path: str,
        show_confidence: bool = True,
        show_validation: bool = True
    ) -> bool:
        """
        Save an annotated version of the analyzed chart.
        
        Args:
            image_path: Path to the original chart image
            result: AnalysisResult from previous analysis
            output_path: Path to save the annotated image
            show_confidence: Whether to show confidence scores
            show_validation: Whether to show validation status
            
        Returns:
            True if save was successful, False otherwise
        """
        import cv2
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        
        return self.annotator.annotate_and_save(
            image,
            result.detections,
            output_path,
            result.validated_detections,
            show_confidence,
            show_validation
        )
    
    def to_json(self, result: AnalysisResult, validate: bool = True) -> str:
        """
        Convert analysis result to JSON string.
        
        Args:
            result: AnalysisResult to serialize
            validate: Whether to validate against schema
            
        Returns:
            JSON string representation
        """
        return result.to_json(validate=validate)
    
    def to_markdown(self, result: AnalysisResult) -> str:
        """
        Convert analysis result to Markdown format.
        
        Args:
            result: AnalysisResult to format
            
        Returns:
            Markdown string representation
        """
        lines = [
            f"# Pattern Analysis Report",
            f"",
            f"**Image:** {result.image_path}",
            f"**Timestamp:** {result.timestamp}",
            f"**Total Time:** {result.total_time_ms:.2f}ms",
            f"",
            f"## Timing Breakdown",
            f"- Preprocessing: {result.preprocessing_time_ms:.2f}ms",
            f"- Feature Extraction: {result.extraction_time_ms:.2f}ms",
            f"- Classification: {result.classification_time_ms:.2f}ms",
            f"- Validation: {result.validation_time_ms:.2f}ms",
            f"",
            f"## Detected Patterns ({len(result.detections)})",
            f"",
        ]
        
        if not result.detections:
            lines.append("No patterns detected.")
        else:
            for i, detection in enumerate(result.detections, 1):
                pattern_name = detection.pattern_type.value.replace("_", " ").title()
                category = detection.category.value.title()
                confidence = detection.confidence * 100
                bbox = detection.bounding_box
                
                # Check if validated
                validated = "N/A"
                for vr in result.validated_detections:
                    if (vr.original_detection and 
                        vr.original_detection.bounding_box == bbox):
                        validated = "✓ Confirmed" if vr.is_confirmed else "? Uncertain"
                        break
                
                lines.extend([
                    f"### {i}. {pattern_name}",
                    f"- **Category:** {category}",
                    f"- **Confidence:** {confidence:.1f}%",
                    f"- **Validation:** {validated}",
                    f"- **Location:** ({bbox.x1}, {bbox.y1}) to ({bbox.x2}, {bbox.y2})",
                    f"- **Detector:** {detection.detector_id}",
                    f"",
                ])
        
        return "\n".join(lines)


def create_analyzer(
    config_path: Optional[str] = None,
    pattern_definitions_path: Optional[str] = None,
    ml_model_path: Optional[str] = None,
    enable_cross_validation: bool = True,
    consensus_threshold: float = 0.5
) -> ChartPatternAnalyzer:
    """
    Create a fully configured ChartPatternAnalyzer for simplified usage.
    
    This is the recommended entry point for most use cases. It creates
    a complete analyzer with all components configured and ready to use.
    
    Args:
        config_path: Path to configuration YAML file
        pattern_definitions_path: Path to pattern definitions YAML
        ml_model_path: Optional path to ML model for detection
        enable_cross_validation: Whether to enable cross-validation
        consensus_threshold: Threshold for cross-validation consensus
        
    Returns:
        Configured ChartPatternAnalyzer ready for analysis
        
    Requirements: 4.3
    
    Example:
        >>> analyzer = create_analyzer()
        >>> result = analyzer.analyze("chart.png")
        >>> print(analyzer.to_markdown(result))
    """
    # Create pipeline
    pipeline = create_pipeline(
        config_path=config_path,
        pattern_definitions_path=pattern_definitions_path,
        ml_model_path=ml_model_path,
        enable_cross_validation=enable_cross_validation,
        consensus_threshold=consensus_threshold
    )
    
    # Create annotator
    annotator = create_annotator()
    
    # Load config for analyzer
    config: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        try:
            manager = ConfigurationManager(config_path=config_path)
            config = manager.config
        except Exception:
            pass
    
    return ChartPatternAnalyzer(
        pipeline=pipeline,
        annotator=annotator,
        config=config
    )


# Convenience aliases
PatternAnalyzer = ChartPatternAnalyzer

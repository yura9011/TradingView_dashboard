"""
Pipeline executor for orchestrating pattern analysis stages.

This module implements the PipelineExecutor class that orchestrates the
sequential execution of pipeline stages, handles errors with graceful
degradation, and collects timing metrics.

Requirements:
- 4.4: Provide detailed error information and allow graceful degradation
- 4.6: Aggregate results from all stages into a unified AnalysisResult
- 7.4: Include timestamps for each processing stage
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models.dataclasses import (
    AnalysisResult,
    FeatureMap,
    PatternDetection,
    PreprocessResult,
    ValidationResult,
)
from .interfaces import (
    CrossValidator,
    FeatureExtractor,
    PatternClassifier,
    PipelineStage,
    Preprocessor,
)


logger = logging.getLogger(__name__)


class StageFailureError(Exception):
    """Raised when a pipeline stage fails during execution."""
    
    def __init__(self, stage_id: str, message: str, original_error: Optional[Exception] = None):
        self.stage_id = stage_id
        self.original_error = original_error
        super().__init__(f"Stage '{stage_id}' failed: {message}")


class PipelineExecutor:
    """
    Orchestrates the execution of pattern analysis pipeline stages.
    
    The executor manages the sequential execution of preprocessing, feature
    extraction, pattern classification, and cross-validation stages. It
    handles errors gracefully, allowing partial results when stages fail,
    and collects timing metrics for each stage.
    
    Requirements: 4.4, 4.6, 7.4
    
    Attributes:
        preprocessor: Image preprocessing stage
        feature_extractor: Feature extraction stage
        classifier: Pattern classification stage
        cross_validator: Cross-validation stage (optional)
    """
    
    def __init__(
        self,
        preprocessor: Preprocessor,
        feature_extractor: FeatureExtractor,
        classifier: PatternClassifier,
        cross_validator: Optional[CrossValidator] = None
    ):
        """
        Initialize the pipeline executor with processing stages.
        
        Args:
            preprocessor: Preprocessor instance for image normalization
            feature_extractor: FeatureExtractor instance for feature extraction
            classifier: PatternClassifier instance for pattern detection
            cross_validator: Optional CrossValidator for result validation
        """
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.cross_validator = cross_validator
        
        self._stages: List[PipelineStage] = [
            preprocessor,
            feature_extractor,
            classifier,
        ]
        if cross_validator:
            self._stages.append(cross_validator)
    
    @property
    def stage_ids(self) -> List[str]:
        """Return list of stage IDs in execution order."""
        return [stage.stage_id for stage in self._stages]
    
    def execute(
        self,
        image_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Execute the full pipeline on an image.
        
        Runs all pipeline stages sequentially, collecting timing metrics
        and handling errors gracefully. If a stage fails, the pipeline
        attempts to continue with fallback values.
        
        Args:
            image_path: Path to the chart image file
            config: Optional configuration dictionary for all stages
            
        Returns:
            AnalysisResult containing all detections, validations, and metrics
            
        Requirements: 4.6, 7.4
        """
        if config is None:
            config = {}
        
        # Initialize timing metrics
        start_time = time.perf_counter()
        preprocessing_time_ms = 0.0
        extraction_time_ms = 0.0
        classification_time_ms = 0.0
        validation_time_ms = 0.0
        
        # Initialize result containers
        preprocess_result: Optional[PreprocessResult] = None
        feature_map: FeatureMap = FeatureMap.empty()
        detections: List[PatternDetection] = []
        validated_detections: List[ValidationResult] = []
        image: Optional[np.ndarray] = None
        
        timestamp = datetime.now().isoformat()
        
        # Stage 1: Preprocessing
        stage_start = time.perf_counter()
        preprocess_result, image = self._execute_preprocessing(image_path, config)
        preprocessing_time_ms = (time.perf_counter() - stage_start) * 1000
        
        # Stage 2: Feature Extraction
        if preprocess_result is not None:
            stage_start = time.perf_counter()
            feature_map = self._execute_feature_extraction(preprocess_result, config)
            extraction_time_ms = (time.perf_counter() - stage_start) * 1000
        
        # Stage 3: Classification
        if image is not None:
            stage_start = time.perf_counter()
            detections = self._execute_classification(feature_map, image, config)
            classification_time_ms = (time.perf_counter() - stage_start) * 1000
        
        # Stage 4: Cross-Validation (optional)
        if self.cross_validator is not None and image is not None and detections:
            stage_start = time.perf_counter()
            validated_detections = self._execute_validation(detections, image, config)
            validation_time_ms = (time.perf_counter() - stage_start) * 1000
        
        # Determine detector status
        detector_status = {
            "rule_based_active": "true",
            "ml_active": "false"
        }
        
        # Check if classifier has ML model loaded
        if hasattr(self.classifier, "ml_model") and self.classifier.ml_model is not None:
            detector_status["ml_active"] = "true"
        
        # Calculate total time
        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Build and return result
        return AnalysisResult(
            image_path=image_path,
            timestamp=timestamp,
            preprocessing_time_ms=preprocessing_time_ms,
            extraction_time_ms=extraction_time_ms,
            classification_time_ms=classification_time_ms,
            validation_time_ms=validation_time_ms,
            total_time_ms=total_time_ms,
            detections=detections,
            validated_detections=validated_detections,
            feature_map=feature_map,
            config_used=config,
            detector_status=detector_status
        )

    def _execute_preprocessing(
        self,
        image_path: str,
        config: Dict[str, Any]
    ) -> Tuple[Optional[PreprocessResult], Optional[np.ndarray]]:
        """
        Execute preprocessing stage with error handling.
        
        Args:
            image_path: Path to the image file
            config: Configuration dictionary
            
        Returns:
            Tuple of (PreprocessResult, image array) or (None, None) on failure
            
        Requirements: 4.4
        """
        try:
            logger.info(f"Starting preprocessing for: {image_path}")
            result = self.preprocessor.process(image_path, config)
            logger.info(
                f"Preprocessing complete: {result.processed_size}, "
                f"quality={result.quality_score:.2f}"
            )
            return result, result.image
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Attempt minimal fallback preprocessing
            fallback_result, fallback_image = self._minimal_preprocess(image_path)
            if fallback_result is not None:
                logger.warning("Using fallback preprocessing")
            return fallback_result, fallback_image
    
    def _execute_feature_extraction(
        self,
        preprocess_result: PreprocessResult,
        config: Dict[str, Any]
    ) -> FeatureMap:
        """
        Execute feature extraction stage with error handling.
        
        Args:
            preprocess_result: Result from preprocessing stage
            config: Configuration dictionary
            
        Returns:
            FeatureMap with extracted features, or empty FeatureMap on failure
            
        Requirements: 4.4
        """
        try:
            logger.info("Starting feature extraction")
            feature_map = self.feature_extractor.process(preprocess_result, config)
            logger.info(
                f"Feature extraction complete: "
                f"{len(feature_map.candlestick_regions)} candlesticks, "
                f"{len(feature_map.trendlines)} trendlines, "
                f"{len(feature_map.support_zones)} support zones, "
                f"{len(feature_map.resistance_zones)} resistance zones"
            )
            return feature_map
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return FeatureMap.empty()
    
    def _execute_classification(
        self,
        feature_map: FeatureMap,
        image: np.ndarray,
        config: Dict[str, Any]
    ) -> List[PatternDetection]:
        """
        Execute classification stage with error handling.
        
        Args:
            feature_map: Extracted features
            image: Preprocessed image array
            config: Configuration dictionary
            
        Returns:
            List of pattern detections, or empty list on failure
            
        Requirements: 4.4
        """
        try:
            logger.info("Starting pattern classification")
            detections = self.classifier.process((feature_map, image), config)
            logger.info(f"Classification complete: {len(detections)} patterns detected")
            return detections
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return []
    
    def _execute_validation(
        self,
        detections: List[PatternDetection],
        image: np.ndarray,
        config: Dict[str, Any]
    ) -> List[ValidationResult]:
        """
        Execute cross-validation stage with error handling.
        
        Args:
            detections: Pattern detections to validate
            image: Preprocessed image array
            config: Configuration dictionary
            
        Returns:
            List of validation results, or unvalidated results on failure
            
        Requirements: 4.4
        """
        if self.cross_validator is None:
            return []
        
        try:
            logger.info(f"Starting cross-validation for {len(detections)} detections")
            validated = self.cross_validator.process((detections, image), config)
            confirmed_count = sum(1 for v in validated if v.is_confirmed)
            logger.info(
                f"Cross-validation complete: {confirmed_count}/{len(validated)} confirmed"
            )
            return validated
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            # Return unvalidated results as fallback
            return [
                ValidationResult(
                    original_detection=d,
                    validation_score=0.0,
                    agreement_count=0,
                    total_validators=0,
                    is_confirmed=False,
                    validator_results={},
                    status="error"
                )
                for d in detections
            ]
    
    def _minimal_preprocess(
        self,
        image_path: str
    ) -> Tuple[Optional[PreprocessResult], Optional[np.ndarray]]:
        """
        Perform minimal preprocessing as fallback when main preprocessing fails.
        
        Attempts to load the image with basic normalization only.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (PreprocessResult, image) or (None, None) on failure
        """
        try:
            import cv2
            import os
            
            if not os.path.exists(image_path):
                return None, None
            
            image = cv2.imread(image_path)
            if image is None:
                return None, None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = (image.shape[1], image.shape[0])
            
            result = PreprocessResult(
                image=image,
                original_size=original_size,
                processed_size=original_size,
                transformations=["bgr_to_rgb", "fallback_minimal"],
                quality_score=0.5,  # Unknown quality
                masked_regions=[]
            )
            
            return result, image
        except Exception as e:
            logger.error(f"Fallback preprocessing also failed: {e}")
            return None, None
    
    def execute_with_fallback(
        self,
        image_path: str,
        config: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Execute pipeline with explicit fallback handling.
        
        This is an alias for execute() that makes the graceful degradation
        behavior explicit in the method name.
        
        Args:
            image_path: Path to the chart image file
            config: Optional configuration dictionary
            
        Returns:
            AnalysisResult with whatever results could be obtained
            
        Requirements: 4.4
        """
        return self.execute(image_path, config)
    
    def get_stage_info(self) -> List[Dict[str, str]]:
        """
        Get information about all pipeline stages.
        
        Returns:
            List of dictionaries with stage_id and stage type information
        """
        return [
            {
                "stage_id": stage.stage_id,
                "type": type(stage).__name__
            }
            for stage in self._stages
        ]

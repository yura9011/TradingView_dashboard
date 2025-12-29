"""
Base interfaces for the pattern analysis pipeline.

This module defines the abstract base classes that all pipeline stages must implement.
Following the design document, these interfaces ensure modularity and allow components
to be upgraded or replaced independently.

Requirements: 4.1 - THE Pipeline SHALL define a standard interface that all processing
stages must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models.dataclasses import (
    BoundingBox,
    FeatureMap,
    PatternDetection,
    PreprocessResult,
    ValidationResult,
)
from ..models.enums import PatternType


class PipelineStage(ABC):
    """
    Base interface for all pipeline stages.
    
    All processing components in the pattern analysis pipeline must inherit from
    this class and implement its abstract methods. This ensures a consistent
    interface for stage orchestration and error handling.
    
    Requirements: 4.1 - Standard interface for all processing stages.
    """
    
    @property
    @abstractmethod
    def stage_id(self) -> str:
        """
        Unique identifier for this stage.
        
        Returns:
            A string that uniquely identifies this stage instance.
            Format: "{component_type}_{version}" (e.g., "standard_preprocessor_v1")
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Any, config: Dict[str, Any]) -> Any:
        """
        Process input data and return output.
        
        This is the main processing method that each stage must implement.
        The input and output types vary by stage type.
        
        Args:
            input_data: The input to process (type depends on stage)
            config: Configuration dictionary for this processing run
            
        Returns:
            Processed output (type depends on stage)
            
        Raises:
            ValueError: If input validation fails
            ProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input before processing.
        
        Should be called before process() to ensure input is valid.
        
        Args:
            input_data: The input to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass


class Preprocessor(PipelineStage):
    """
    Interface for image preprocessing stages.
    
    Preprocessors normalize and enhance chart images before analysis.
    They handle tasks like resizing, color space conversion, denoising,
    and UI element masking.
    
    Requirements: 1.1-1.6 - Image normalization and enhancement.
    """
    
    @abstractmethod
    def normalize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Normalize image to standard dimensions.
        
        Resizes the image to target dimensions while preserving aspect ratio.
        Padding is added as needed to reach exact target size.
        
        Args:
            image: Input image as numpy array (H, W, C)
            target_size: Target dimensions as (width, height)
            
        Returns:
            Normalized image with target dimensions
        """
        pass
    
    @abstractmethod
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction while preserving edges.
        
        Uses edge-preserving filtering to reduce noise without
        blurring important chart features.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Denoised image
        """
        pass
    
    @abstractmethod
    def detect_roi(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detect regions of interest (UI elements to mask).
        
        Identifies non-chart elements like toolbars, legends, and watermarks
        that should be excluded from analysis.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of BoundingBox objects marking UI regions to mask
        """
        pass


class FeatureExtractor(PipelineStage):
    """
    Interface for feature extraction stages.
    
    Feature extractors identify and extract relevant visual features
    from preprocessed chart images, including candlesticks, trendlines,
    and support/resistance zones.
    
    Requirements: 2.1-2.6 - Visual feature extraction.
    """
    
    @abstractmethod
    def extract_candlesticks(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Extract candlestick body regions.
        
        Identifies individual candlestick bodies using color segmentation
        and contour detection.
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            List of BoundingBox objects for each detected candlestick
        """
        pass
    
    @abstractmethod
    def detect_trendlines(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect trendlines using edge detection.
        
        Uses Canny edge detection and Hough Line Transform to identify
        diagonal lines that may represent trendlines.
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            List of trendline dictionaries with keys:
            - start: (x, y) tuple for line start
            - end: (x, y) tuple for line end
            - angle: Line angle in degrees
            - direction: "up" or "down"
            - length: Line length in pixels
        """
        pass
    
    @abstractmethod
    def find_support_resistance(
        self, image: np.ndarray
    ) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        """
        Find support and resistance zones.
        
        Identifies horizontal price levels with high activity that may
        represent support (below mid-point) or resistance (above mid-point).
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            Tuple of (support_zones, resistance_zones) as lists of BoundingBox
        """
        pass


class PatternClassifier(PipelineStage):
    """
    Interface for pattern classification stages.
    
    Pattern classifiers analyze extracted features to identify and classify
    chart patterns. They may use rule-based detection, machine learning,
    or hybrid approaches.
    
    Requirements: 3.1-3.6 - Pattern classification.
    """
    
    @abstractmethod
    def classify(
        self, features: FeatureMap, image: np.ndarray
    ) -> List[PatternDetection]:
        """
        Classify patterns from extracted features.
        
        Analyzes the feature map and image to detect and classify
        chart patterns. Returns all detections above confidence threshold,
        ordered by confidence descending.
        
        Args:
            features: Extracted features from FeatureExtractor
            image: Preprocessed image for additional analysis
            
        Returns:
            List of PatternDetection objects, ordered by confidence descending
        """
        pass
    
    @abstractmethod
    def get_supported_patterns(self) -> List[PatternType]:
        """
        Return list of patterns this classifier can detect.
        
        Returns:
            List of PatternType enum values this classifier supports
        """
        pass


class CrossValidator(PipelineStage):
    """
    Interface for cross-validation stages.
    
    Cross validators verify pattern detections using multiple independent
    methods to minimize false positives and increase confidence.
    
    Requirements: 5.1-5.5 - Cross-validation of results.
    """
    
    @abstractmethod
    def validate(
        self, detections: List[PatternDetection], image: np.ndarray
    ) -> List[ValidationResult]:
        """
        Validate detections using alternative methods.
        
        Each detection is verified by running it through alternative
        detection methods. Results include validation scores and
        agreement information.
        
        Args:
            detections: List of pattern detections to validate
            image: Original image for re-analysis
            
        Returns:
            List of ValidationResult objects with validation details
        """
        pass
    
    @abstractmethod
    def get_consensus(
        self, results: List[ValidationResult]
    ) -> List[ValidationResult]:
        """
        Apply consensus algorithm to validation results.
        
        Filters validation results to return only those that meet
        the consensus threshold for confirmation.
        
        Args:
            results: List of validation results to filter
            
        Returns:
            List of ValidationResult objects that are confirmed
        """
        pass

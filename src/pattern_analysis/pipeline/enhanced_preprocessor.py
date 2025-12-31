"""
Enhanced Preprocessor with region detection and auto-cropping.

This module extends the StandardPreprocessor with capabilities for:
- Detecting and classifying chart regions (primary chart, volume panel, etc.)
- Auto-cropping to focus on the primary chart region
- Timeframe-aware configuration

Feature: chart-analysis-improvements
Requirements: 1.1, 3.1, 4.1, 4.2
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.timeframe_manager import TimeframeConfigManager
from ..models.dataclasses import (
    BoundingBox,
    ChartRegion,
    CropResult,
    PreprocessResult,
    RegionDetectionResult,
    TimeframeConfig,
)
from ..models.enums import Timeframe
from .auto_cropper import AutoCropper
from .preprocessor import (
    ImageCorruptedError,
    ImageNotFoundError,
    ImageTooSmallError,
    StandardPreprocessor,
)
from .region_detector import ChartRegionDetector


logger = logging.getLogger("pattern_analysis.enhanced_preprocessor")


@dataclass
class EnhancedPreprocessResult(PreprocessResult):
    """
    Extended preprocess result with region metadata.
    
    Adds fields for region detection results, crop information,
    and analysis coverage tracking.
    
    Requirements: 4.1, 4.2
    """
    # Region detection results
    region_detection: Optional[RegionDetectionResult] = None
    
    # Auto-crop results
    crop_result: Optional[CropResult] = None
    
    # Analyzed region bounds (after cropping)
    analyzed_region_bounds: Optional[BoundingBox] = None
    
    # Percentage of original image that was analyzed
    coverage_percentage: float = 100.0
    
    # Flag indicating if result needs manual review (coverage < 50%)
    needs_review: bool = False
    
    # Timeframe configuration used
    timeframe_config: Optional[TimeframeConfig] = None
    
    def __post_init__(self):
        """Validate enhanced preprocess result fields."""
        # Call parent validation
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")
        
        # Validate coverage percentage
        if not 0.0 <= self.coverage_percentage <= 100.0:
            raise ValueError("coverage_percentage must be between 0.0 and 100.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excludes image array)."""
        base_dict = {
            "original_size": list(self.original_size),
            "processed_size": list(self.processed_size),
            "transformations": self.transformations,
            "quality_score": self.quality_score,
            "masked_regions": [r.to_dict() for r in self.masked_regions],
            "coverage_percentage": self.coverage_percentage,
            "needs_review": self.needs_review,
        }
        
        # Add region detection if present
        if self.region_detection:
            base_dict["region_detection"] = self.region_detection.to_dict()
        
        # Add crop result if present
        if self.crop_result:
            base_dict["crop_result"] = self.crop_result.to_dict()
        
        # Add analyzed region bounds if present
        if self.analyzed_region_bounds:
            base_dict["analyzed_region_bounds"] = self.analyzed_region_bounds.to_dict()
        
        # Add timeframe config if present
        if self.timeframe_config:
            base_dict["timeframe_config"] = self.timeframe_config.to_dict()
        
        return base_dict


class EnhancedPreprocessor(StandardPreprocessor):
    """
    Extended preprocessor with region detection and auto-cropping.
    
    Integrates ChartRegionDetector and AutoCropper into the
    preprocessing pipeline to:
    - Detect multiple chart regions in an image
    - Identify the primary chart region
    - Auto-crop to focus on the primary chart
    - Track analysis coverage and metadata
    
    Requirements: 1.1, 3.1, 4.1, 4.2
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EnhancedPreprocessor.
        
        Args:
            config: Configuration dictionary with optional keys:
                - region_detection: Dict with region detection settings
                - auto_crop: Dict with auto-crop settings
                - timeframe: Dict with timeframe settings
                - target_size: Tuple of (width, height) for normalization
                - denoise: bool, whether to apply denoising
        """
        super().__init__()
        
        self._config = config or {}
        
        # Initialize region detector with config
        region_config = self._config.get("region_detection", {})
        self.region_detector = ChartRegionDetector(region_config)
        
        # Initialize auto-cropper with config
        auto_crop_config = self._config.get("auto_crop", {})
        self.auto_cropper = AutoCropper(auto_crop_config)
        
        # Initialize timeframe manager with config
        timeframe_config = self._config.get("timeframe", {})
        self.timeframe_manager = TimeframeConfigManager(timeframe_config)
        
        logger.debug(
            f"EnhancedPreprocessor initialized with config: "
            f"region_detection={bool(region_config)}, "
            f"auto_crop={bool(auto_crop_config)}, "
            f"timeframe={self.timeframe_manager.default_timeframe.value}"
        )
    
    @property
    def stage_id(self) -> str:
        """Unique identifier for this preprocessor."""
        return "enhanced_preprocessor_v1"
    
    def process(
        self, 
        input_data: str, 
        config: Dict[str, Any]
    ) -> EnhancedPreprocessResult:
        """
        Process chart image with region detection and cropping.
        
        Extends the standard preprocessing pipeline with:
        1. Region detection to identify chart regions
        2. Auto-cropping to focus on primary chart
        3. Standard preprocessing on cropped image
        4. Metadata enrichment with region information
        
        Args:
            input_data: Path to image file
            config: Processing configuration with optional keys:
                - target_size: (width, height) tuple
                - denoise: bool, whether to apply denoising
                - timeframe: Timeframe enum or string
                
        Returns:
            EnhancedPreprocessResult with region metadata
            
        Raises:
            ImageNotFoundError: If image path does not exist
            ImageCorruptedError: If image cannot be decoded
            ImageTooSmallError: If image is below minimum size
            
        Requirements: 1.1, 3.1, 4.1, 4.2
        """
        # Validate input
        if not self.validate_input(input_data):
            raise ImageNotFoundError(f"Image file not found: {input_data}")
        
        # Load image
        image = cv2.imread(input_data)
        if image is None:
            raise ImageCorruptedError(
                f"Failed to decode image: {input_data}. "
                "The file may be corrupted or in an unsupported format."
            )
        
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Check minimum size
        if original_size[0] < self.MIN_IMAGE_SIZE[0] or original_size[1] < self.MIN_IMAGE_SIZE[1]:
            raise ImageTooSmallError(
                f"Image dimensions {original_size} are below minimum "
                f"requirements {self.MIN_IMAGE_SIZE}. "
                "Please provide a larger image."
            )
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Track transformations
        transformations = ["bgr_to_rgb"]
        
        # 1. Detect regions
        detection_result = self.region_detector.detect_regions(image_rgb)
        transformations.append("region_detection")
        logger.debug(
            f"Detected {len(detection_result.regions)} regions, "
            f"primary: {detection_result.primary_region.region_type.value if detection_result.primary_region else 'None'}"
        )
        
        # 2. Auto-crop if enabled
        crop_result = self.auto_cropper.crop(image_rgb, detection_result)
        if crop_result.coverage_percentage < 100.0:
            transformations.append("auto_crop")
        
        # Log excluded regions (Requirement 3.5)
        if crop_result.excluded_regions:
            excluded_types = [r.region_type.value for r in crop_result.excluded_regions]
            logger.info(f"Excluded regions from analysis: {excluded_types}")
        
        # 3. Apply standard preprocessing to cropped image
        processed_image = crop_result.cropped_image
        
        # Normalize dimensions
        target = config.get("target_size", self.DEFAULT_TARGET_SIZE)
        processed_image = self.normalize(processed_image, target)
        transformations.append(f"resize_to_{target[0]}x{target[1]}")
        
        # Denoise if configured
        if config.get("denoise", True):
            processed_image = self.denoise(processed_image)
            transformations.append("bilateral_denoise")
        
        # 4. Detect and mask UI elements
        masked_regions = self.detect_roi(processed_image)
        if masked_regions:
            transformations.append(f"masked_{len(masked_regions)}_regions")
        
        # 5. Calculate quality score
        quality = self._calculate_quality(processed_image)
        
        # 6. Get timeframe configuration
        timeframe = config.get("timeframe")
        if isinstance(timeframe, str):
            try:
                timeframe = Timeframe(timeframe)
            except ValueError:
                timeframe = None
        timeframe_config = self.timeframe_manager.get_config(timeframe)
        
        # 7. Check if needs review (coverage < 50%)
        needs_review = crop_result.coverage_percentage < 50.0
        if needs_review:
            logger.warning(
                f"Low coverage ({crop_result.coverage_percentage:.1f}%), "
                "result flagged for review"
            )
        
        return EnhancedPreprocessResult(
            image=processed_image,
            original_size=original_size,
            processed_size=(processed_image.shape[1], processed_image.shape[0]),
            transformations=transformations,
            quality_score=quality,
            masked_regions=masked_regions,
            # Enhanced fields
            region_detection=detection_result,
            crop_result=crop_result,
            analyzed_region_bounds=crop_result.crop_bounds,
            coverage_percentage=crop_result.coverage_percentage,
            needs_review=needs_review,
            timeframe_config=timeframe_config,
        )
    
    def process_with_timeframe(
        self,
        input_data: str,
        timeframe: Optional[Timeframe] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> EnhancedPreprocessResult:
        """
        Process chart image with specific timeframe configuration.
        
        Convenience method that sets up timeframe-specific parameters
        before processing.
        
        Args:
            input_data: Path to image file
            timeframe: Timeframe to use (defaults to manager's default)
            config: Additional processing configuration
            
        Returns:
            EnhancedPreprocessResult with timeframe-aware processing
        """
        process_config = config.copy() if config else {}
        process_config["timeframe"] = timeframe or self.timeframe_manager.default_timeframe
        
        return self.process(input_data, process_config)
    
    def get_pattern_params(
        self, 
        timeframe: Optional[Timeframe] = None
    ) -> Dict[str, Any]:
        """
        Get pattern detection parameters for the specified timeframe.
        
        Delegates to the TimeframeConfigManager to get appropriate
        parameters for pattern detection.
        
        Args:
            timeframe: Timeframe to get params for (defaults to manager's default)
            
        Returns:
            Dictionary of pattern detection parameters
        """
        return self.timeframe_manager.get_pattern_params(timeframe)


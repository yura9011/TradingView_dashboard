"""
Auto Cropper for chart images.

This module implements the AutoCropper class that automatically crops
chart images to focus on the primary chart region, removing secondary
charts like volume panels and indicators.

Requirements:
- 3.1: Provide option to automatically crop out secondary chart regions
- 3.2: Remove volume panel and other secondary indicators from image
- 3.3: Preserve full Primary_Chart region without cutting any price data
- 3.4: Support configuration flag auto_remove_secondary_charts (default: true)
- 3.5: Log which regions were excluded
- 3.6: Fall back to full image with warning if auto-crop fails
- 4.4: Calculate and report percentage of original image analyzed
- 4.5: Flag for review if coverage < 50%
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..models.dataclasses import BoundingBox, ChartRegion, CropResult, RegionDetectionResult


logger = logging.getLogger(__name__)


class AutoCropper:
    """
    Automatically crops chart images to focus on the primary chart region.
    
    Removes secondary charts (volume, indicators) while preserving
    the full primary price chart.
    
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.4, 4.5
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoCropper.
        
        Args:
            config: Configuration dictionary with optional keys:
                - auto_remove_secondary_charts: Enable auto-cropping (default: True)
                - crop_padding: Padding around cropped region in pixels (default: 10)
        """
        self.config = config
        self.enabled = config.get("auto_remove_secondary_charts", True)
        self.padding = config.get("crop_padding", 10)
    
    def crop(
        self, 
        image: np.ndarray, 
        detection_result: RegionDetectionResult
    ) -> CropResult:
        """
        Crop image to primary chart region.
        
        Args:
            image: Original image as numpy array (RGB format)
            detection_result: Result from ChartRegionDetector
            
        Returns:
            CropResult with cropped image and metadata
            
        Requirements: 3.1, 3.2, 3.3, 4.4
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to auto-cropper")
            return self._create_empty_result()
        
        h, w = image.shape[:2]
        original_size = (w, h)
        
        # If disabled or no primary region detected, return full image
        if not self.enabled:
            logger.debug("Auto-crop disabled, returning full image")
            return self._create_full_image_result(image, original_size)
        
        if detection_result.primary_region is None:
            logger.warning(
                "No primary region detected, falling back to full image"
            )
            return self._create_full_image_result(image, original_size)
        
        # Get primary region bounds
        primary = detection_result.primary_region
        bbox = primary.bounding_box
        
        # Validate bounding box
        if not bbox.is_valid():
            logger.warning(
                "Invalid primary region bounding box, falling back to full image"
            )
            return self._create_full_image_result(image, original_size)
        
        # Calculate crop bounds with padding
        crop_bounds = self._calculate_crop_bounds(bbox, w, h)
        
        # Perform the crop
        cropped_image = self._perform_crop(image, crop_bounds)
        cropped_size = (cropped_image.shape[1], cropped_image.shape[0])
        
        # Calculate coverage percentage
        coverage = self._calculate_coverage(original_size, cropped_size)
        
        # Log excluded regions (Requirement 3.5)
        excluded_regions = detection_result.secondary_regions
        self._log_excluded_regions(excluded_regions)
        
        logger.info(
            f"Auto-crop complete: {original_size} -> {cropped_size}, "
            f"coverage: {coverage:.1f}%"
        )
        
        return CropResult(
            cropped_image=cropped_image,
            original_size=original_size,
            cropped_size=cropped_size,
            crop_bounds=crop_bounds,
            excluded_regions=excluded_regions,
            coverage_percentage=coverage
        )
    
    def _calculate_crop_bounds(
        self, 
        bbox: BoundingBox, 
        image_width: int, 
        image_height: int
    ) -> BoundingBox:
        """
        Calculate crop bounds with padding while staying within image bounds.
        
        Preserves full width of the image (Requirement 3.2).
        
        Args:
            bbox: Primary region bounding box
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            BoundingBox with crop coordinates
        """
        # Add padding to vertical bounds
        y1 = max(0, bbox.y1 - self.padding)
        y2 = min(image_height, bbox.y2 + self.padding)
        
        # Preserve full width (Requirement 3.2)
        x1 = 0
        x2 = image_width
        
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    
    def _perform_crop(
        self, 
        image: np.ndarray, 
        bounds: BoundingBox
    ) -> np.ndarray:
        """
        Perform the actual image crop.
        
        Args:
            image: Original image
            bounds: Crop bounding box
            
        Returns:
            Cropped image as numpy array
        """
        return image[bounds.y1:bounds.y2, bounds.x1:bounds.x2].copy()
    
    def _calculate_coverage(
        self, 
        original_size: tuple, 
        cropped_size: tuple
    ) -> float:
        """
        Calculate the percentage of original image that was analyzed.
        
        Args:
            original_size: (width, height) of original image
            cropped_size: (width, height) of cropped image
            
        Returns:
            Coverage percentage (0-100)
            
        Requirements: 4.4
        """
        original_area = original_size[0] * original_size[1]
        cropped_area = cropped_size[0] * cropped_size[1]
        
        if original_area == 0:
            return 0.0
        
        return (cropped_area / original_area) * 100.0
    
    def _log_excluded_regions(self, excluded_regions: List[ChartRegion]) -> None:
        """
        Log information about excluded regions.
        
        Args:
            excluded_regions: List of regions that were excluded from analysis
            
        Requirements: 3.5
        """
        if not excluded_regions:
            logger.debug("No regions excluded from analysis")
            return
        
        excluded_types = [r.region_type.value for r in excluded_regions]
        logger.info(f"Excluded regions from analysis: {excluded_types}")
        
        for region in excluded_regions:
            logger.debug(
                f"  - {region.region_type.value}: "
                f"bounds={region.bounding_box.to_dict()}, "
                f"confidence={region.confidence:.2f}"
            )
    
    def _create_full_image_result(
        self, 
        image: np.ndarray, 
        original_size: tuple
    ) -> CropResult:
        """
        Create a CropResult that uses the full image.
        
        Used when auto-crop is disabled or fails.
        
        Args:
            image: Original image
            original_size: (width, height) of image
            
        Returns:
            CropResult with full image
        """
        w, h = original_size
        return CropResult(
            cropped_image=image.copy(),
            original_size=original_size,
            cropped_size=original_size,
            crop_bounds=BoundingBox(x1=0, y1=0, x2=w, y2=h),
            excluded_regions=[],
            coverage_percentage=100.0
        )
    
    def _create_empty_result(self) -> CropResult:
        """
        Create an empty CropResult for edge cases.
        
        Returns:
            Empty CropResult
        """
        return CropResult(
            cropped_image=np.array([], dtype=np.uint8),
            original_size=(0, 0),
            cropped_size=(0, 0),
            crop_bounds=BoundingBox(x1=0, y1=0, x2=0, y2=0),
            excluded_regions=[],
            coverage_percentage=0.0
        )
    
    def needs_review(self, crop_result: CropResult) -> bool:
        """
        Check if the crop result needs manual review.
        
        A result needs review if coverage is below 50%.
        
        Args:
            crop_result: Result from crop operation
            
        Returns:
            True if coverage < 50%, False otherwise
            
        Requirements: 4.5
        """
        return crop_result.coverage_percentage < 50.0

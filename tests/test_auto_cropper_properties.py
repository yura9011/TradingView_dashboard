"""
Property-based tests for AutoCropper.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document for chart-analysis-improvements.

Feature: chart-analysis-improvements
Property 2: Auto-crop Preserves Primary Chart
Validates: Requirements 3.2, 3.3
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from src.pattern_analysis.pipeline.auto_cropper import AutoCropper
from src.pattern_analysis.pipeline.region_detector import ChartRegionDetector
from src.pattern_analysis.models.dataclasses import (
    BoundingBox,
    ChartRegion,
    RegionDetectionResult,
)
from src.pattern_analysis.models.enums import RegionType


# Default configurations for tests
DEFAULT_CROPPER_CONFIG = {
    "auto_remove_secondary_charts": True,
    "crop_padding": 10
}

DEFAULT_DETECTOR_CONFIG = {
    "min_region_height_pct": 0.1,
    "volume_position_threshold": 0.7,
    "separator_threshold": 0.3
}


def create_cropper(enabled: bool = True, padding: int = 10) -> AutoCropper:
    """Create an AutoCropper with specified config."""
    return AutoCropper({
        "auto_remove_secondary_charts": enabled,
        "crop_padding": padding
    })


def create_detector() -> ChartRegionDetector:
    """Create a ChartRegionDetector with default config."""
    return ChartRegionDetector(DEFAULT_DETECTOR_CONFIG)


# =============================================================================
# Strategies for generating test data
# =============================================================================

@st.composite
def valid_primary_region(draw, image_height: int, image_width: int):
    """Generate a valid primary region within image bounds."""
    # Primary region should be in the upper portion and large
    y1 = draw(st.integers(min_value=0, max_value=int(image_height * 0.1)))
    y2 = draw(st.integers(min_value=int(image_height * 0.5), max_value=int(image_height * 0.85)))
    
    bbox = BoundingBox(x1=0, y1=y1, x2=image_width, y2=y2)
    
    return ChartRegion(
        region_type=RegionType.PRIMARY_CHART,
        bounding_box=bbox,
        confidence=0.9,
        characteristics={"height_ratio": (y2 - y1) / image_height}
    )


@st.composite
def valid_secondary_region(draw, image_height: int, image_width: int, primary_y2: int):
    """Generate a valid secondary region (volume panel) below primary."""
    y1 = draw(st.integers(min_value=primary_y2 + 1, max_value=int(image_height * 0.9)))
    y2 = draw(st.integers(min_value=y1 + 10, max_value=image_height))
    
    bbox = BoundingBox(x1=0, y1=y1, x2=image_width, y2=y2)
    
    return ChartRegion(
        region_type=RegionType.VOLUME_PANEL,
        bounding_box=bbox,
        confidence=0.8,
        characteristics={"height_ratio": (y2 - y1) / image_height}
    )


@st.composite
def multi_region_detection_result(draw):
    """Generate a RegionDetectionResult with primary and secondary regions."""
    height = draw(st.integers(min_value=400, max_value=800))
    width = draw(st.integers(min_value=400, max_value=800))
    
    # Generate primary region
    primary = draw(valid_primary_region(height, width))
    
    # Generate secondary region below primary
    secondary = draw(valid_secondary_region(height, width, primary.bounding_box.y2))
    
    return RegionDetectionResult(
        regions=[primary, secondary],
        primary_region=primary,
        secondary_regions=[secondary],
        original_size=(width, height),
        detection_method="horizontal_projection"
    ), height, width


@st.composite
def multi_region_chart_image(draw):
    """
    Generate a synthetic chart image with multiple regions:
    - Primary chart (top, large, with candlestick colors)
    - Volume panel (bottom, smaller)
    """
    total_height = draw(st.integers(min_value=400, max_value=800))
    width = draw(st.integers(min_value=400, max_value=800))
    
    # Primary chart takes 70-80% of height
    primary_ratio = draw(st.floats(min_value=0.65, max_value=0.80))
    primary_height = int(total_height * primary_ratio)
    volume_height = total_height - primary_height
    
    # Create primary chart region with candlestick colors
    primary_region = np.ones((primary_height, width, 3), dtype=np.uint8) * 240
    
    # Add green and red candles to primary region
    num_candles = draw(st.integers(min_value=10, max_value=30))
    for i in range(num_candles):
        x = int((i / num_candles) * (width - 20)) + 5
        y = draw(st.integers(min_value=20, max_value=primary_height - 60))
        h = draw(st.integers(min_value=30, max_value=50))
        w = draw(st.integers(min_value=5, max_value=12))
        color = [0, 180, 0] if draw(st.booleans()) else [180, 0, 0]
        primary_region[y:y+h, x:x+w] = color
    
    # Create volume region (darker, with bars)
    volume_region = np.ones((volume_height, width, 3), dtype=np.uint8) * 220
    
    # Add volume bars
    for i in range(num_candles):
        x = int((i / num_candles) * (width - 20)) + 5
        bar_height = draw(st.integers(min_value=5, max_value=volume_height - 10))
        w = draw(st.integers(min_value=5, max_value=12))
        volume_region[volume_height - bar_height:, x:x+w] = [100, 100, 150]
    
    # Add separator line between regions
    separator = np.ones((3, width, 3), dtype=np.uint8) * 50
    
    # Combine regions
    image = np.vstack([primary_region, separator, volume_region])
    
    return image, primary_height, volume_height


# =============================================================================
# Property Tests for Auto-crop Preservation (Property 2)
# =============================================================================

class TestAutoCropPreservesPrimaryChart:
    """
    Property tests for auto-crop preservation.
    
    Feature: chart-analysis-improvements
    Property 2: Auto-crop Preserves Primary Chart
    Validates: Requirements 3.2, 3.3
    
    For any image with auto-crop enabled, the cropping operation SHALL
    preserve the complete primary chart region without cutting any price data,
    while removing secondary chart regions.
    """
    
    @given(data=multi_region_detection_result())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_crop_preserves_primary_region_bounds(self, data):
        """
        Feature: chart-analysis-improvements
        Property 2: Auto-crop Preserves Primary Chart
        Validates: Requirements 3.2, 3.3
        
        For any image with a detected primary region, the cropped image
        SHALL fully contain the primary region bounds.
        """
        detection_result, height, width = data
        cropper = create_cropper(enabled=True, padding=10)
        
        # Create a test image
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Perform crop
        crop_result = cropper.crop(image, detection_result)
        
        # Get primary region bounds
        primary_bbox = detection_result.primary_region.bounding_box
        crop_bbox = crop_result.crop_bounds
        
        # Crop bounds should fully contain primary region (with padding)
        assert crop_bbox.y1 <= primary_bbox.y1, \
            f"Crop y1 ({crop_bbox.y1}) should be <= primary y1 ({primary_bbox.y1})"
        assert crop_bbox.y2 >= primary_bbox.y2, \
            f"Crop y2 ({crop_bbox.y2}) should be >= primary y2 ({primary_bbox.y2})"
        
        # Full width should be preserved
        assert crop_bbox.x1 == 0, "Crop should preserve full width (x1=0)"
        assert crop_bbox.x2 == width, f"Crop should preserve full width (x2={width})"
    
    @given(data=multi_region_detection_result())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_crop_excludes_secondary_regions(self, data):
        """
        Feature: chart-analysis-improvements
        Property 2: Auto-crop Preserves Primary Chart
        Validates: Requirements 3.2
        
        For any image with secondary regions, the crop result SHALL
        list those regions as excluded.
        """
        detection_result, height, width = data
        cropper = create_cropper(enabled=True, padding=10)
        
        # Create a test image
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Perform crop
        crop_result = cropper.crop(image, detection_result)
        
        # Excluded regions should match secondary regions
        assert len(crop_result.excluded_regions) == len(detection_result.secondary_regions), \
            "Excluded regions should match secondary regions count"
        
        for excluded in crop_result.excluded_regions:
            assert excluded.region_type != RegionType.PRIMARY_CHART, \
                "Primary chart should not be in excluded regions"
    
    @given(data=multi_region_detection_result())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_cropped_image_dimensions_match_bounds(self, data):
        """
        Feature: chart-analysis-improvements
        Property 2: Auto-crop Preserves Primary Chart
        Validates: Requirements 3.3
        
        The cropped image dimensions SHALL match the crop bounds.
        """
        detection_result, height, width = data
        cropper = create_cropper(enabled=True, padding=10)
        
        # Create a test image
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Perform crop
        crop_result = cropper.crop(image, detection_result)
        
        # Verify dimensions match
        expected_height = crop_result.crop_bounds.y2 - crop_result.crop_bounds.y1
        expected_width = crop_result.crop_bounds.x2 - crop_result.crop_bounds.x1
        
        actual_height, actual_width = crop_result.cropped_image.shape[:2]
        
        assert actual_height == expected_height, \
            f"Cropped height ({actual_height}) should match bounds ({expected_height})"
        assert actual_width == expected_width, \
            f"Cropped width ({actual_width}) should match bounds ({expected_width})"
    
    @given(data=multi_region_detection_result())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_coverage_percentage_is_valid(self, data):
        """
        Feature: chart-analysis-improvements
        Property 2: Auto-crop Preserves Primary Chart
        Validates: Requirements 4.4
        
        Coverage percentage SHALL be between 0 and 100.
        """
        detection_result, height, width = data
        cropper = create_cropper(enabled=True, padding=10)
        
        # Create a test image
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Perform crop
        crop_result = cropper.crop(image, detection_result)
        
        assert 0.0 <= crop_result.coverage_percentage <= 100.0, \
            f"Coverage ({crop_result.coverage_percentage}) should be between 0 and 100"
    
    @given(data=multi_region_detection_result())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_coverage_calculation_is_correct(self, data):
        """
        Feature: chart-analysis-improvements
        Property 2: Auto-crop Preserves Primary Chart
        Validates: Requirements 4.4
        
        Coverage percentage SHALL equal (cropped_area / original_area) * 100.
        """
        detection_result, height, width = data
        cropper = create_cropper(enabled=True, padding=10)
        
        # Create a test image
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Perform crop
        crop_result = cropper.crop(image, detection_result)
        
        # Calculate expected coverage
        original_area = crop_result.original_size[0] * crop_result.original_size[1]
        cropped_area = crop_result.cropped_size[0] * crop_result.cropped_size[1]
        expected_coverage = (cropped_area / original_area) * 100.0
        
        assert abs(crop_result.coverage_percentage - expected_coverage) < 0.01, \
            f"Coverage ({crop_result.coverage_percentage}) should equal calculated ({expected_coverage})"
    
    @given(image_data=multi_region_chart_image())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_end_to_end_crop_preserves_primary(self, image_data):
        """
        Feature: chart-analysis-improvements
        Property 2: Auto-crop Preserves Primary Chart
        Validates: Requirements 3.2, 3.3
        
        End-to-end test: detect regions then crop, verifying primary is preserved.
        """
        image, primary_height, volume_height = image_data
        
        detector = create_detector()
        cropper = create_cropper(enabled=True, padding=10)
        
        # Detect regions
        detection_result = detector.detect_regions(image)
        
        # Skip if no primary region detected
        if detection_result.primary_region is None:
            return
        
        # Perform crop
        crop_result = cropper.crop(image, detection_result)
        
        # Cropped image should not be empty
        assert crop_result.cropped_image.size > 0, "Cropped image should not be empty"
        
        # Cropped image should be smaller or equal to original
        original_area = image.shape[0] * image.shape[1]
        cropped_area = crop_result.cropped_image.shape[0] * crop_result.cropped_image.shape[1]
        assert cropped_area <= original_area, \
            "Cropped area should be <= original area"


class TestAutoCropDisabled:
    """Tests for when auto-crop is disabled."""
    
    @given(data=multi_region_detection_result())
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_disabled_returns_full_image(self, data):
        """
        When auto-crop is disabled, the full image should be returned.
        """
        detection_result, height, width = data
        cropper = create_cropper(enabled=False)
        
        # Create a test image
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Perform crop
        crop_result = cropper.crop(image, detection_result)
        
        # Should return full image
        assert crop_result.coverage_percentage == 100.0, \
            "Disabled cropper should return 100% coverage"
        assert crop_result.cropped_size == (width, height), \
            "Disabled cropper should return original size"
        assert len(crop_result.excluded_regions) == 0, \
            "Disabled cropper should not exclude any regions"


class TestAutoCropNeedsReview:
    """Tests for the needs_review functionality."""
    
    def test_needs_review_below_50_percent(self):
        """
        Coverage below 50% should flag for review.
        Validates: Requirements 4.5
        """
        cropper = create_cropper()
        
        # Create a crop result with low coverage
        from src.pattern_analysis.models.dataclasses import CropResult
        
        low_coverage_result = CropResult(
            cropped_image=np.ones((100, 200, 3), dtype=np.uint8),
            original_size=(400, 400),
            cropped_size=(200, 100),
            crop_bounds=BoundingBox(x1=0, y1=0, x2=200, y2=100),
            excluded_regions=[],
            coverage_percentage=12.5  # 20000 / 160000 = 12.5%
        )
        
        assert cropper.needs_review(low_coverage_result) is True, \
            "Coverage < 50% should need review"
    
    def test_no_review_above_50_percent(self):
        """
        Coverage above 50% should not flag for review.
        Validates: Requirements 4.5
        """
        cropper = create_cropper()
        
        from src.pattern_analysis.models.dataclasses import CropResult
        
        high_coverage_result = CropResult(
            cropped_image=np.ones((300, 400, 3), dtype=np.uint8),
            original_size=(400, 400),
            cropped_size=(400, 300),
            crop_bounds=BoundingBox(x1=0, y1=0, x2=400, y2=300),
            excluded_regions=[],
            coverage_percentage=75.0
        )
        
        assert cropper.needs_review(high_coverage_result) is False, \
            "Coverage >= 50% should not need review"


class TestAutoCropEdgeCases:
    """Unit tests for edge cases in auto-cropping."""
    
    def test_empty_image_returns_empty_result(self):
        """Empty images should return empty results gracefully."""
        cropper = create_cropper()
        empty_image = np.array([], dtype=np.uint8)
        
        detection_result = RegionDetectionResult(
            regions=[],
            primary_region=None,
            secondary_regions=[],
            original_size=(0, 0),
            detection_method="horizontal_projection"
        )
        
        crop_result = cropper.crop(empty_image, detection_result)
        
        assert crop_result.coverage_percentage == 0.0
        assert crop_result.cropped_image.size == 0
    
    def test_no_primary_region_returns_full_image(self):
        """No primary region should return full image with warning."""
        cropper = create_cropper()
        
        height, width = 400, 600
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        detection_result = RegionDetectionResult(
            regions=[],
            primary_region=None,
            secondary_regions=[],
            original_size=(width, height),
            detection_method="horizontal_projection"
        )
        
        crop_result = cropper.crop(image, detection_result)
        
        assert crop_result.coverage_percentage == 100.0
        assert crop_result.cropped_size == (width, height)
    
    def test_cropper_initialization(self):
        """Test that cropper initializes with correct config values."""
        cropper = create_cropper(enabled=True, padding=15)
        
        assert cropper.enabled is True
        assert cropper.padding == 15
    
    def test_cropper_with_default_config(self):
        """Test cropper uses defaults with empty config."""
        cropper = AutoCropper({})
        
        assert cropper.enabled is True
        assert cropper.padding == 10

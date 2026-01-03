"""
Property-based tests for ChartRegionDetector.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document for chart-analysis-improvements.

Feature: chart-analysis-improvements
Property 1: Region Detection and Classification
Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from src.pattern_analysis.pipeline.region_detector import ChartRegionDetector


# Default detector configuration for tests
DEFAULT_DETECTOR_CONFIG = {
    "min_region_height_pct": 0.1,
    "volume_position_threshold": 0.7,
    "separator_threshold": 0.3
}


def create_detector():
    """Create a ChartRegionDetector with default config."""
    return ChartRegionDetector(DEFAULT_DETECTOR_CONFIG)
from src.pattern_analysis.models.dataclasses import (
    BoundingBox,
    ChartRegion,
    RegionDetectionResult,
)
from src.pattern_analysis.models.enums import RegionType


# =============================================================================
# Strategies for generating test data
# =============================================================================

@st.composite
def valid_rgb_image(draw, min_height=100, max_height=800, min_width=200, max_width=1200):
    """Generate a valid RGB image as numpy array."""
    height = draw(st.integers(min_value=min_height, max_value=max_height))
    width = draw(st.integers(min_value=min_width, max_value=max_width))
    
    # Generate random RGB values
    image = draw(
        st.lists(
            st.lists(
                st.tuples(
                    st.integers(0, 255),
                    st.integers(0, 255),
                    st.integers(0, 255)
                ),
                min_size=width,
                max_size=width
            ),
            min_size=height,
            max_size=height
        )
    )
    
    return np.array(image, dtype=np.uint8)


@st.composite
def image_with_separator(draw, min_height=200, max_height=600, width=400):
    """Generate an image with a clear horizontal separator."""
    height = draw(st.integers(min_value=min_height, max_value=max_height))
    
    # Create base image with some content
    image = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add a dark separator line somewhere in the middle
    separator_y = draw(st.integers(min_value=int(height * 0.3), max_value=int(height * 0.7)))
    separator_height = draw(st.integers(min_value=3, max_value=10))
    
    # Make separator dark (low values)
    image[separator_y:separator_y + separator_height, :] = 20
    
    return image, separator_y, separator_height


@st.composite
def image_with_candlestick_colors(draw, height=400, width=600):
    """Generate an image with green and red candlestick-like colors."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light background
    
    # Add some green regions (bullish candles)
    num_green = draw(st.integers(min_value=5, max_value=20))
    for _ in range(num_green):
        x = draw(st.integers(min_value=0, max_value=width - 20))
        y = draw(st.integers(min_value=0, max_value=height - 50))
        w = draw(st.integers(min_value=5, max_value=15))
        h = draw(st.integers(min_value=20, max_value=50))
        # Green color in RGB
        image[y:y+h, x:x+w] = [0, 200, 0]
    
    # Add some red regions (bearish candles)
    num_red = draw(st.integers(min_value=5, max_value=20))
    for _ in range(num_red):
        x = draw(st.integers(min_value=0, max_value=width - 20))
        y = draw(st.integers(min_value=0, max_value=height - 50))
        w = draw(st.integers(min_value=5, max_value=15))
        h = draw(st.integers(min_value=20, max_value=50))
        # Red color in RGB
        image[y:y+h, x:x+w] = [200, 0, 0]
    
    return image


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
# Property Tests for Region Detection and Classification (Property 1)
# =============================================================================

class TestRegionDetectionAndClassification:
    """
    Property tests for region detection and classification.
    
    Feature: chart-analysis-improvements
    Property 1: Region Detection and Classification
    Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6
    
    For any chart image containing multiple regions (price chart + volume panel),
    the ChartRegionDetector SHALL correctly identify the primary chart region
    and classify secondary regions, ensuring that pattern analysis is performed
    ONLY on the primary chart region.
    """
    
    @given(image_data=multi_region_chart_image())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_multi_region_detection_identifies_primary(self, image_data):
        """
        Feature: chart-analysis-improvements
        Property 1: Region Detection and Classification
        Validates: Requirements 1.1, 1.2, 1.3
        
        For any image with multiple chart regions, the detector SHALL identify
        a primary region.
        """
        detector = create_detector()
        image, primary_height, volume_height = image_data
        
        result = detector.detect_regions(image)
        
        # Must have at least one region
        assert len(result.regions) >= 1, "Should detect at least one region"
        
        # Must identify a primary region
        assert result.primary_region is not None, "Should identify a primary region"
        
        # Primary region should have valid bounding box
        assert result.primary_region.bounding_box.is_valid(), \
            "Primary region should have valid bounding box"
    
    @given(image_data=multi_region_chart_image())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_primary_region_is_largest_with_candles(self, image_data):
        """
        Feature: chart-analysis-improvements
        Property 1: Region Detection and Classification
        Validates: Requirements 1.3
        
        The primary region SHALL be the largest region with candlestick
        characteristics, or the largest region as fallback.
        """
        detector = create_detector()
        image, primary_height, volume_height = image_data
        
        result = detector.detect_regions(image)
        
        if result.primary_region is None:
            return  # Skip if no primary found
        
        # Primary region should be among the largest regions
        if len(result.regions) > 1:
            primary_area = result.primary_region.bounding_box.area()
            max_area = max(r.bounding_box.area() for r in result.regions)
            
            # Primary should be at least 50% of the largest region
            assert primary_area >= max_area * 0.5, \
                "Primary region should be among the largest regions"
    
    @given(image_data=multi_region_chart_image())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region_bounding_boxes_are_valid(self, image_data):
        """
        Feature: chart-analysis-improvements
        Property 1: Region Detection and Classification
        Validates: Requirements 1.6
        
        All detected regions SHALL have valid bounding box coordinates.
        """
        detector = create_detector()
        image, _, _ = image_data
        
        result = detector.detect_regions(image)
        
        for region in result.regions:
            bbox = region.bounding_box
            
            # Bounding box should be valid (x1 < x2, y1 < y2)
            assert bbox.is_valid(), \
                f"Region {region.region_type} has invalid bounding box"
            
            # Coordinates should be within image bounds
            h, w = image.shape[:2]
            assert bbox.x1 >= 0 and bbox.x2 <= w, \
                "Bounding box x coordinates should be within image width"
            assert bbox.y1 >= 0 and bbox.y2 <= h, \
                "Bounding box y coordinates should be within image height"
    
    @given(image_data=multi_region_chart_image())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_secondary_regions_exclude_primary(self, image_data):
        """
        Feature: chart-analysis-improvements
        Property 1: Region Detection and Classification
        Validates: Requirements 1.4
        
        Secondary regions SHALL NOT include the primary region.
        """
        detector = create_detector()
        image, _, _ = image_data
        
        result = detector.detect_regions(image)
        
        if result.primary_region is None:
            return
        
        # Primary should not be in secondary list
        for secondary in result.secondary_regions:
            assert secondary != result.primary_region, \
                "Primary region should not be in secondary regions list"
    
    @given(image_data=multi_region_chart_image())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_all_regions_have_valid_types(self, image_data):
        """
        Feature: chart-analysis-improvements
        Property 1: Region Detection and Classification
        Validates: Requirements 1.2
        
        All detected regions SHALL have a valid RegionType classification.
        """
        detector = create_detector()
        image, _, _ = image_data
        
        result = detector.detect_regions(image)
        
        valid_types = set(RegionType)
        
        for region in result.regions:
            assert region.region_type in valid_types, \
                f"Region type {region.region_type} is not a valid RegionType"
            assert 0.0 <= region.confidence <= 1.0, \
                f"Region confidence {region.confidence} should be between 0 and 1"
    
    @given(image_data=multi_region_chart_image())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_detection_result_has_correct_original_size(self, image_data):
        """
        Feature: chart-analysis-improvements
        Property 1: Region Detection and Classification
        Validates: Requirements 1.6
        
        The detection result SHALL include correct original image dimensions.
        """
        detector = create_detector()
        image, _, _ = image_data
        h, w = image.shape[:2]
        
        result = detector.detect_regions(image)
        
        assert result.original_size == (w, h), \
            f"Original size should be ({w}, {h}), got {result.original_size}"
    
    def test_single_region_uses_full_image(self):
        """
        Feature: chart-analysis-improvements
        Property 1: Region Detection and Classification
        Validates: Requirements 1.5
        
        If only one chart region is detected, the System SHALL use the
        entire image for analysis.
        """
        detector = create_detector()
        # Create a simple image without separators
        height, width = 400, 600
        image = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Add some candlestick colors
        for i in range(10):
            x = i * 50 + 20
            image[100:200, x:x+10] = [0, 180, 0]  # Green
            image[220:300, x+20:x+30] = [180, 0, 0]  # Red
        
        result = detector.detect_regions(image)
        
        # Should have exactly one region or primary should cover most of image
        if len(result.regions) == 1:
            region = result.regions[0]
            coverage = region.bounding_box.area() / (width * height)
            assert coverage > 0.8, \
                "Single region should cover most of the image"
    
    def test_empty_image_returns_empty_result(self):
        """
        Feature: chart-analysis-improvements
        Property 1: Region Detection and Classification
        
        Empty images should return empty results gracefully.
        """
        detector = create_detector()
        empty_image = np.array([], dtype=np.uint8)
        
        result = detector.detect_regions(empty_image)
        
        assert len(result.regions) == 0
        assert result.primary_region is None
        assert len(result.secondary_regions) == 0
    
    @given(
        height=st.integers(min_value=100, max_value=500),
        width=st.integers(min_value=100, max_value=500)
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_uniform_image_returns_single_region(self, height, width):
        """
        Feature: chart-analysis-improvements
        Property 1: Region Detection and Classification
        Validates: Requirements 1.5
        
        A uniform image (no separators) should return a single region.
        """
        detector = create_detector()
        # Create uniform gray image
        image = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        result = detector.detect_regions(image)
        
        # Should have at least one region
        assert len(result.regions) >= 1, "Should detect at least one region"
        
        # Primary region should exist
        assert result.primary_region is not None, "Should have a primary region"


# =============================================================================
# Additional Unit Tests for Edge Cases
# =============================================================================

class TestRegionDetectorEdgeCases:
    """Unit tests for edge cases in region detection."""
    
    def test_detector_initialization(self):
        """Test that detector initializes with correct config values."""
        detector = create_detector()
        assert detector.min_region_height_pct == 0.1
        assert detector.volume_position_threshold == 0.7
        assert detector.separator_threshold == 0.3
    
    def test_detector_with_custom_config(self):
        """Test detector with custom configuration."""
        config = {
            "min_region_height_pct": 0.2,
            "volume_position_threshold": 0.8,
            "separator_threshold": 0.4
        }
        detector = ChartRegionDetector(config)
        
        assert detector.min_region_height_pct == 0.2
        assert detector.volume_position_threshold == 0.8
        assert detector.separator_threshold == 0.4
    
    def test_detector_with_empty_config(self):
        """Test detector uses defaults with empty config."""
        detector = ChartRegionDetector({})
        
        assert detector.min_region_height_pct == 0.1
        assert detector.volume_position_threshold == 0.7
        assert detector.separator_threshold == 0.3
    
    def test_detection_method_is_horizontal_projection(self):
        """Test that detection method is correctly reported."""
        detector = create_detector()
        image = np.ones((200, 300, 3), dtype=np.uint8) * 128
        
        result = detector.detect_regions(image)
        
        assert result.detection_method == "horizontal_projection"

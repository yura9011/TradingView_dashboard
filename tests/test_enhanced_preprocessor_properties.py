"""
Property-based tests for the EnhancedPreprocessor.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.

Feature: chart-analysis-improvements
Property 3: Analysis Metadata Completeness
Validates: Requirements 4.1, 4.2, 4.4, 4.5
"""

import os
import tempfile
from typing import Tuple

import cv2
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from src.pattern_analysis.pipeline.enhanced_preprocessor import (
    EnhancedPreprocessor,
    EnhancedPreprocessResult,
)
from src.pattern_analysis.models.dataclasses import BoundingBox


# =============================================================================
# Custom Strategies for Image Generation
# =============================================================================

@st.composite
def chart_image_strategy(draw, min_size=200, max_size=800):
    """
    Generate random chart-like images for testing.
    
    Creates images with characteristics similar to chart images:
    - Background color (typically dark or light)
    - Candlestick-like colored regions (green/red)
    - Optional volume panel at bottom
    
    Args:
        draw: Hypothesis draw function
        min_size: Minimum dimension size
        max_size: Maximum dimension size
        
    Returns:
        Random chart-like RGB image as numpy array
    """
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Create base image with chart-like background
    bg_brightness = draw(st.integers(min_value=20, max_value=60))
    image = np.full((height, width, 3), bg_brightness, dtype=np.uint8)
    
    # Add candlestick-like colored regions (green and red)
    num_candles = draw(st.integers(min_value=5, max_value=20))
    candle_width = max(5, width // (num_candles * 2))
    
    for i in range(num_candles):
        x = draw(st.integers(min_value=10, max_value=width - candle_width - 10))
        y1 = draw(st.integers(min_value=10, max_value=height // 2))
        y2 = draw(st.integers(min_value=y1 + 20, max_value=min(y1 + 100, height - 50)))
        
        # Randomly choose green or red
        if draw(st.booleans()):
            # Green candle (bullish)
            color = (0, draw(st.integers(min_value=150, max_value=255)), 0)
        else:
            # Red candle (bearish)
            color = (draw(st.integers(min_value=150, max_value=255)), 0, 0)
        
        cv2.rectangle(image, (x, y1), (x + candle_width, y2), color, -1)
    
    return image


@st.composite
def multi_region_chart_image_strategy(draw, min_size=300, max_size=800):
    """
    Generate chart images with multiple regions (price chart + volume panel).
    
    Creates images that simulate real chart screenshots with:
    - Primary price chart region (top, larger)
    - Volume panel region (bottom, smaller)
    - Separator line between regions
    
    Args:
        draw: Hypothesis draw function
        min_size: Minimum dimension size
        max_size: Maximum dimension size
        
    Returns:
        Tuple of (image, expected_regions_count)
    """
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Create base image
    bg_brightness = draw(st.integers(min_value=20, max_value=50))
    image = np.full((height, width, 3), bg_brightness, dtype=np.uint8)
    
    # Define region boundaries
    # Primary chart takes ~70% of height
    primary_end = int(height * 0.7)
    separator_height = 5
    volume_start = primary_end + separator_height
    
    # Add separator line (dark line between regions)
    cv2.rectangle(
        image, 
        (0, primary_end), 
        (width, volume_start), 
        (10, 10, 10), 
        -1
    )
    
    # Add candlesticks to primary region
    num_candles = draw(st.integers(min_value=10, max_value=30))
    candle_width = max(3, width // (num_candles * 2))
    
    for i in range(num_candles):
        x = 10 + i * (candle_width + 5)
        if x + candle_width >= width - 10:
            break
            
        y1 = draw(st.integers(min_value=20, max_value=primary_end // 3))
        y2 = draw(st.integers(min_value=y1 + 30, max_value=min(y1 + 150, primary_end - 20)))
        
        # Green or red candle
        if draw(st.booleans()):
            color = (0, draw(st.integers(min_value=150, max_value=255)), 0)
        else:
            color = (draw(st.integers(min_value=150, max_value=255)), 0, 0)
        
        cv2.rectangle(image, (x, y1), (x + candle_width, y2), color, -1)
    
    # Add volume bars to volume region
    for i in range(num_candles):
        x = 10 + i * (candle_width + 5)
        if x + candle_width >= width - 10:
            break
            
        bar_height = draw(st.integers(min_value=10, max_value=height - volume_start - 10))
        y1 = height - bar_height
        y2 = height - 5
        
        # Volume bars typically blue or gray
        color = (100, 100, draw(st.integers(min_value=150, max_value=200)))
        cv2.rectangle(image, (x, y1), (x + candle_width, y2), color, -1)
    
    return image


# =============================================================================
# Property Tests for Analysis Metadata Completeness
# =============================================================================

class TestAnalysisMetadataCompleteness:
    """
    Property tests for analysis metadata completeness.
    
    Feature: chart-analysis-improvements
    Property 3: Analysis Metadata Completeness
    Validates: Requirements 4.1, 4.2, 4.4, 4.5
    """
    
    @given(chart_image_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_result_includes_valid_bounding_box(self, image: np.ndarray):
        """
        Feature: chart-analysis-improvements
        Property 3: Analysis Metadata Completeness
        Validates: Requirements 4.1, 4.2
        
        For any analysis result, the output SHALL include valid bounding box
        coordinates of the analyzed region.
        """
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        try:
            preprocessor = EnhancedPreprocessor()
            config = {"target_size": (640, 480), "denoise": True}
            
            result = preprocessor.process(temp_path, config)
            
            # Check that analyzed_region_bounds is present
            assert result.analyzed_region_bounds is not None, (
                "analyzed_region_bounds should not be None"
            )
            
            # Check that bounding box is valid
            bbox = result.analyzed_region_bounds
            assert isinstance(bbox, BoundingBox), (
                f"analyzed_region_bounds should be BoundingBox, got {type(bbox)}"
            )
            
            # Check bounding box has valid coordinates
            assert bbox.x1 >= 0, f"x1 should be >= 0, got {bbox.x1}"
            assert bbox.y1 >= 0, f"y1 should be >= 0, got {bbox.y1}"
            assert bbox.x2 > bbox.x1, f"x2 ({bbox.x2}) should be > x1 ({bbox.x1})"
            assert bbox.y2 > bbox.y1, f"y2 ({bbox.y2}) should be > y1 ({bbox.y1})"
            
            # Check bounding box is within original image bounds
            orig_w, orig_h = result.original_size
            assert bbox.x2 <= orig_w, f"x2 ({bbox.x2}) should be <= original width ({orig_w})"
            assert bbox.y2 <= orig_h, f"y2 ({bbox.y2}) should be <= original height ({orig_h})"
            
        finally:
            os.unlink(temp_path)
    
    @given(chart_image_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_result_includes_coverage_percentage(self, image: np.ndarray):
        """
        Feature: chart-analysis-improvements
        Property 3: Analysis Metadata Completeness
        Validates: Requirements 4.4
        
        For any analysis result, the output SHALL include the coverage percentage
        in valid range [0.0, 100.0].
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        try:
            preprocessor = EnhancedPreprocessor()
            result = preprocessor.process(temp_path, {})
            
            # Check coverage_percentage is present and valid
            assert hasattr(result, 'coverage_percentage'), (
                "Result should have coverage_percentage attribute"
            )
            assert 0.0 <= result.coverage_percentage <= 100.0, (
                f"coverage_percentage should be in [0.0, 100.0], got {result.coverage_percentage}"
            )
            
        finally:
            os.unlink(temp_path)
    
    @given(chart_image_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_result_includes_needs_review_flag(self, image: np.ndarray):
        """
        Feature: chart-analysis-improvements
        Property 3: Analysis Metadata Completeness
        Validates: Requirements 4.5
        
        For any analysis result, the output SHALL include the needs_review flag
        that is True when coverage is below 50%.
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        try:
            preprocessor = EnhancedPreprocessor()
            result = preprocessor.process(temp_path, {})
            
            # Check needs_review flag is present
            assert hasattr(result, 'needs_review'), (
                "Result should have needs_review attribute"
            )
            assert isinstance(result.needs_review, bool), (
                f"needs_review should be bool, got {type(result.needs_review)}"
            )
            
            # Verify needs_review is consistent with coverage
            if result.coverage_percentage < 50.0:
                assert result.needs_review is True, (
                    f"needs_review should be True when coverage ({result.coverage_percentage}%) < 50%"
                )
            else:
                assert result.needs_review is False, (
                    f"needs_review should be False when coverage ({result.coverage_percentage}%) >= 50%"
                )
            
        finally:
            os.unlink(temp_path)
    
    @given(chart_image_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_result_includes_region_detection_metadata(self, image: np.ndarray):
        """
        Feature: chart-analysis-improvements
        Property 3: Analysis Metadata Completeness
        Validates: Requirements 4.1
        
        For any analysis result, the output SHALL include region detection metadata.
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        try:
            preprocessor = EnhancedPreprocessor()
            result = preprocessor.process(temp_path, {})
            
            # Check region_detection is present
            assert result.region_detection is not None, (
                "region_detection should not be None"
            )
            
            # Check region_detection has required fields
            assert hasattr(result.region_detection, 'regions'), (
                "region_detection should have regions attribute"
            )
            assert hasattr(result.region_detection, 'primary_region'), (
                "region_detection should have primary_region attribute"
            )
            assert hasattr(result.region_detection, 'original_size'), (
                "region_detection should have original_size attribute"
            )
            
        finally:
            os.unlink(temp_path)
    
    @given(chart_image_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_result_includes_crop_result_metadata(self, image: np.ndarray):
        """
        Feature: chart-analysis-improvements
        Property 3: Analysis Metadata Completeness
        Validates: Requirements 4.2
        
        For any analysis result, the output SHALL include crop result metadata.
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        try:
            preprocessor = EnhancedPreprocessor()
            result = preprocessor.process(temp_path, {})
            
            # Check crop_result is present
            assert result.crop_result is not None, (
                "crop_result should not be None"
            )
            
            # Check crop_result has required fields
            assert hasattr(result.crop_result, 'original_size'), (
                "crop_result should have original_size attribute"
            )
            assert hasattr(result.crop_result, 'cropped_size'), (
                "crop_result should have cropped_size attribute"
            )
            assert hasattr(result.crop_result, 'crop_bounds'), (
                "crop_result should have crop_bounds attribute"
            )
            assert hasattr(result.crop_result, 'coverage_percentage'), (
                "crop_result should have coverage_percentage attribute"
            )
            
        finally:
            os.unlink(temp_path)
    
    @given(multi_region_chart_image_strategy(min_size=300, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_coverage_percentage_matches_crop_bounds(self, image: np.ndarray):
        """
        Feature: chart-analysis-improvements
        Property 3: Analysis Metadata Completeness
        Validates: Requirements 4.4
        
        For any analysis result, the coverage percentage SHALL be consistent
        with the crop bounds and original image size.
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        try:
            preprocessor = EnhancedPreprocessor()
            result = preprocessor.process(temp_path, {})
            
            # Calculate expected coverage from bounds
            orig_w, orig_h = result.original_size
            original_area = orig_w * orig_h
            
            if result.crop_result:
                crop_w, crop_h = result.crop_result.cropped_size
                cropped_area = crop_w * crop_h
                expected_coverage = (cropped_area / original_area) * 100.0
                
                # Coverage should match within tolerance
                assert abs(result.coverage_percentage - expected_coverage) < 0.1, (
                    f"Coverage mismatch: result={result.coverage_percentage}, "
                    f"expected={expected_coverage}"
                )
            
        finally:
            os.unlink(temp_path)
    
    @given(chart_image_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_result_serialization_includes_all_metadata(self, image: np.ndarray):
        """
        Feature: chart-analysis-improvements
        Property 3: Analysis Metadata Completeness
        Validates: Requirements 4.1, 4.2
        
        For any analysis result, serialization to dict SHALL include all
        region metadata fields.
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        try:
            preprocessor = EnhancedPreprocessor()
            result = preprocessor.process(temp_path, {})
            
            # Serialize to dict
            result_dict = result.to_dict()
            
            # Check required fields are present
            assert "coverage_percentage" in result_dict, (
                "Serialized result should include coverage_percentage"
            )
            assert "needs_review" in result_dict, (
                "Serialized result should include needs_review"
            )
            assert "region_detection" in result_dict, (
                "Serialized result should include region_detection"
            )
            assert "crop_result" in result_dict, (
                "Serialized result should include crop_result"
            )
            assert "analyzed_region_bounds" in result_dict, (
                "Serialized result should include analyzed_region_bounds"
            )
            
        finally:
            os.unlink(temp_path)


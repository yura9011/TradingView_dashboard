"""
Property-based tests for the EdgeBasedFeatureExtractor.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.

Feature: chart-pattern-analysis-framework
Property 4: Feature Extraction Completeness
Validates: Requirements 2.1, 2.2, 2.3, 2.5
"""

import numpy as np
import cv2
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from typing import Tuple

from src.pattern_analysis.pipeline.feature_extractor import EdgeBasedFeatureExtractor
from src.pattern_analysis.models import BoundingBox, FeatureMap, PreprocessResult


# =============================================================================
# Custom Strategies for Test Data Generation
# =============================================================================

@st.composite
def rgb_image_strategy(draw, min_size=200, max_size=800):
    """
    Generate random valid RGB images for testing.
    
    Args:
        draw: Hypothesis draw function
        min_size: Minimum dimension size
        max_size: Maximum dimension size
        
    Returns:
        Random RGB image as numpy array (height, width, 3)
    """
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


@st.composite
def quality_score_strategy(draw):
    """Generate valid quality scores in [0.0, 1.0]."""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def preprocess_result_strategy(draw, min_size=200, max_size=800):
    """
    Generate valid PreprocessResult objects for testing.
    
    Creates a PreprocessResult with a random RGB image and valid metadata.
    """
    image = draw(rgb_image_strategy(min_size=min_size, max_size=max_size))
    height, width = image.shape[:2]
    quality_score = draw(quality_score_strategy())
    
    return PreprocessResult(
        image=image,
        original_size=(width, height),
        processed_size=(width, height),
        transformations=["bgr_to_rgb"],
        quality_score=quality_score,
        masked_regions=[]
    )


@st.composite
def chart_like_image_strategy(draw, min_size=300, max_size=800):
    """
    Generate images that resemble financial charts with candlesticks and lines.
    
    Creates images with:
    - Dark background
    - Green and red vertical rectangles (candlesticks)
    - Diagonal lines (trendlines)
    - Horizontal lines (support/resistance)
    """
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Dark background (typical chart background)
    bg_color = draw(st.integers(min_value=10, max_value=50))
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Add candlestick-like shapes (green and red vertical rectangles)
    num_candles = draw(st.integers(min_value=5, max_value=20))
    candle_width = max(3, width // (num_candles * 2))
    
    for i in range(num_candles):
        x = draw(st.integers(min_value=10, max_value=width - candle_width - 10))
        candle_height = draw(st.integers(min_value=20, max_value=height // 3))
        y = draw(st.integers(min_value=10, max_value=height - candle_height - 10))
        
        # Alternate between green (bullish) and red (bearish)
        if draw(st.booleans()):
            # Green candle (HSV: H=60, high S and V)
            color = (0, 200, 0)  # RGB green
        else:
            # Red candle
            color = (200, 0, 0)  # RGB red
        
        cv2.rectangle(image, (x, y), (x + candle_width, y + candle_height), color, -1)
    
    # Add diagonal lines (trendlines)
    num_lines = draw(st.integers(min_value=2, max_value=6))
    for _ in range(num_lines):
        x1 = draw(st.integers(min_value=10, max_value=width - 10))
        y1 = draw(st.integers(min_value=10, max_value=height - 10))
        x2 = draw(st.integers(min_value=10, max_value=width - 10))
        y2 = draw(st.integers(min_value=10, max_value=height - 10))
        
        # Ensure line is diagonal (not horizontal or vertical)
        if abs(x2 - x1) > 50 and abs(y2 - y1) > 20:
            color = (200, 200, 200)  # Light gray
            cv2.line(image, (x1, y1), (x2, y2), color, 2)
    
    # Add horizontal lines (support/resistance)
    num_horizontal = draw(st.integers(min_value=1, max_value=4))
    for _ in range(num_horizontal):
        y = draw(st.integers(min_value=20, max_value=height - 20))
        color = (150, 150, 150)  # Gray
        cv2.line(image, (0, y), (width, y), color, 1)
    
    return image


@st.composite
def chart_preprocess_result_strategy(draw, min_size=300, max_size=800):
    """
    Generate PreprocessResult with chart-like images.
    """
    image = draw(chart_like_image_strategy(min_size=min_size, max_size=max_size))
    height, width = image.shape[:2]
    quality_score = draw(quality_score_strategy())
    
    return PreprocessResult(
        image=image,
        original_size=(width, height),
        processed_size=(width, height),
        transformations=["bgr_to_rgb"],
        quality_score=quality_score,
        masked_regions=[]
    )


# =============================================================================
# Property Tests for Feature Extraction Completeness
# =============================================================================

class TestFeatureExtractionCompleteness:
    """
    Property tests for feature extraction completeness.
    
    Feature: chart-pattern-analysis-framework
    Property 4: Feature Extraction Completeness
    Validates: Requirements 2.1, 2.2, 2.3, 2.5
    """
    
    @given(preprocess_result_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_feature_map_has_all_required_fields(self, preprocess_result: PreprocessResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.1, 2.2, 2.3, 2.5
        
        For any preprocessed image, the Feature_Extractor SHALL produce a valid
        FeatureMap containing all required fields:
        - candlestick_regions (list, may be empty)
        - trendlines (list, may be empty)
        - support_zones (list, may be empty)
        - resistance_zones (list, may be empty)
        - quality_score (float in [0, 1])
        """
        extractor = EdgeBasedFeatureExtractor()
        config = {"extract_volume": True}
        
        feature_map = extractor.process(preprocess_result, config)
        
        # Verify FeatureMap is returned
        assert isinstance(feature_map, FeatureMap), (
            f"Expected FeatureMap, got {type(feature_map)}"
        )
        
        # Verify candlestick_regions is a list
        assert isinstance(feature_map.candlestick_regions, list), (
            f"candlestick_regions should be a list, got {type(feature_map.candlestick_regions)}"
        )
        
        # Verify trendlines is a list
        assert isinstance(feature_map.trendlines, list), (
            f"trendlines should be a list, got {type(feature_map.trendlines)}"
        )
        
        # Verify support_zones is a list
        assert isinstance(feature_map.support_zones, list), (
            f"support_zones should be a list, got {type(feature_map.support_zones)}"
        )
        
        # Verify resistance_zones is a list
        assert isinstance(feature_map.resistance_zones, list), (
            f"resistance_zones should be a list, got {type(feature_map.resistance_zones)}"
        )
        
        # Verify quality_score is in valid range [0, 1]
        assert 0.0 <= feature_map.quality_score <= 1.0, (
            f"quality_score {feature_map.quality_score} not in [0.0, 1.0]"
        )
    
    @given(preprocess_result_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_candlestick_regions_are_valid_bounding_boxes(self, preprocess_result: PreprocessResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.1
        
        For any preprocessed image, all candlestick_regions SHALL be valid
        BoundingBox objects with x1 < x2 and y1 < y2.
        """
        extractor = EdgeBasedFeatureExtractor()
        feature_map = extractor.process(preprocess_result, {})
        
        for i, bbox in enumerate(feature_map.candlestick_regions):
            assert isinstance(bbox, BoundingBox), (
                f"candlestick_regions[{i}] should be BoundingBox, got {type(bbox)}"
            )
            assert bbox.is_valid(), (
                f"candlestick_regions[{i}] is invalid: x1={bbox.x1}, y1={bbox.y1}, "
                f"x2={bbox.x2}, y2={bbox.y2}"
            )
    
    @given(preprocess_result_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_trendlines_have_required_structure(self, preprocess_result: PreprocessResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.2
        
        For any preprocessed image, all trendlines SHALL be dictionaries with
        required keys: start, end, angle, direction, length.
        """
        extractor = EdgeBasedFeatureExtractor()
        feature_map = extractor.process(preprocess_result, {})
        
        required_keys = {"start", "end", "angle", "direction", "length"}
        
        for i, trendline in enumerate(feature_map.trendlines):
            assert isinstance(trendline, dict), (
                f"trendlines[{i}] should be dict, got {type(trendline)}"
            )
            
            missing_keys = required_keys - set(trendline.keys())
            assert not missing_keys, (
                f"trendlines[{i}] missing keys: {missing_keys}"
            )
            
            # Verify start and end are tuples of 2 integers
            assert isinstance(trendline["start"], tuple) and len(trendline["start"]) == 2, (
                f"trendlines[{i}]['start'] should be (x, y) tuple"
            )
            assert isinstance(trendline["end"], tuple) and len(trendline["end"]) == 2, (
                f"trendlines[{i}]['end'] should be (x, y) tuple"
            )
            
            # Verify angle is a number
            assert isinstance(trendline["angle"], (int, float)), (
                f"trendlines[{i}]['angle'] should be numeric"
            )
            
            # Verify direction is "up" or "down"
            assert trendline["direction"] in ("up", "down"), (
                f"trendlines[{i}]['direction'] should be 'up' or 'down', got {trendline['direction']}"
            )
            
            # Verify length is positive
            assert isinstance(trendline["length"], (int, float)) and trendline["length"] > 0, (
                f"trendlines[{i}]['length'] should be positive number"
            )
    
    @given(preprocess_result_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_support_resistance_zones_are_valid_bounding_boxes(self, preprocess_result: PreprocessResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.3
        
        For any preprocessed image, all support_zones and resistance_zones SHALL
        be valid BoundingBox objects.
        """
        extractor = EdgeBasedFeatureExtractor()
        feature_map = extractor.process(preprocess_result, {})
        
        # Check support zones
        for i, bbox in enumerate(feature_map.support_zones):
            assert isinstance(bbox, BoundingBox), (
                f"support_zones[{i}] should be BoundingBox, got {type(bbox)}"
            )
            assert bbox.is_valid(), (
                f"support_zones[{i}] is invalid: x1={bbox.x1}, y1={bbox.y1}, "
                f"x2={bbox.x2}, y2={bbox.y2}"
            )
        
        # Check resistance zones
        for i, bbox in enumerate(feature_map.resistance_zones):
            assert isinstance(bbox, BoundingBox), (
                f"resistance_zones[{i}] should be BoundingBox, got {type(bbox)}"
            )
            assert bbox.is_valid(), (
                f"resistance_zones[{i}] is invalid: x1={bbox.x1}, y1={bbox.y1}, "
                f"x2={bbox.x2}, y2={bbox.y2}"
            )
    
    @given(preprocess_result_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_quality_score_preserved_from_input(self, preprocess_result: PreprocessResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.5
        
        For any preprocessed image, the FeatureMap quality_score SHALL match
        the input PreprocessResult quality_score.
        """
        extractor = EdgeBasedFeatureExtractor()
        feature_map = extractor.process(preprocess_result, {})
        
        assert feature_map.quality_score == preprocess_result.quality_score, (
            f"quality_score mismatch: input={preprocess_result.quality_score}, "
            f"output={feature_map.quality_score}"
        )
    
    @given(preprocess_result_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_volume_profile_structure_when_enabled(self, preprocess_result: PreprocessResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.5
        
        When volume extraction is enabled, if volume_profile is not None,
        it SHALL have the required structure.
        """
        extractor = EdgeBasedFeatureExtractor()
        config = {"extract_volume": True}
        feature_map = extractor.process(preprocess_result, config)
        
        if feature_map.volume_profile is not None:
            required_keys = {"region", "distribution", "avg_volume", "max_volume"}
            missing_keys = required_keys - set(feature_map.volume_profile.keys())
            
            assert not missing_keys, (
                f"volume_profile missing keys: {missing_keys}"
            )
            
            # Verify region is a BoundingBox
            assert isinstance(feature_map.volume_profile["region"], BoundingBox), (
                "volume_profile['region'] should be BoundingBox"
            )
            
            # Verify distribution is a list
            assert isinstance(feature_map.volume_profile["distribution"], list), (
                "volume_profile['distribution'] should be a list"
            )
            
            # Verify avg_volume and max_volume are numbers
            assert isinstance(feature_map.volume_profile["avg_volume"], (int, float)), (
                "volume_profile['avg_volume'] should be numeric"
            )
            assert isinstance(feature_map.volume_profile["max_volume"], (int, float)), (
                "volume_profile['max_volume'] should be numeric"
            )
    
    @given(preprocess_result_strategy(min_size=200, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_volume_profile_none_when_disabled(self, preprocess_result: PreprocessResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.5
        
        When volume extraction is disabled, volume_profile SHALL be None.
        """
        extractor = EdgeBasedFeatureExtractor()
        config = {"extract_volume": False}
        feature_map = extractor.process(preprocess_result, config)
        
        assert feature_map.volume_profile is None, (
            "volume_profile should be None when extract_volume=False"
        )
    
    @given(chart_preprocess_result_strategy(min_size=300, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_candlesticks_sorted_by_x_coordinate(self, preprocess_result: PreprocessResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.1
        
        For any preprocessed image, candlestick_regions SHALL be sorted by
        x-coordinate (left to right, representing time order).
        """
        extractor = EdgeBasedFeatureExtractor()
        feature_map = extractor.process(preprocess_result, {})
        
        if len(feature_map.candlestick_regions) > 1:
            for i in range(len(feature_map.candlestick_regions) - 1):
                current = feature_map.candlestick_regions[i]
                next_candle = feature_map.candlestick_regions[i + 1]
                assert current.x1 <= next_candle.x1, (
                    f"Candlesticks not sorted by x: [{i}].x1={current.x1} > [{i+1}].x1={next_candle.x1}"
                )
    
    @given(chart_preprocess_result_strategy(min_size=300, max_size=600))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_trendlines_sorted_by_length(self, preprocess_result: PreprocessResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.2
        
        For any preprocessed image, trendlines SHALL be sorted by length
        in descending order (longest first).
        """
        extractor = EdgeBasedFeatureExtractor()
        feature_map = extractor.process(preprocess_result, {})
        
        if len(feature_map.trendlines) > 1:
            for i in range(len(feature_map.trendlines) - 1):
                current_length = feature_map.trendlines[i]["length"]
                next_length = feature_map.trendlines[i + 1]["length"]
                assert current_length >= next_length, (
                    f"Trendlines not sorted by length: [{i}].length={current_length} < [{i+1}].length={next_length}"
                )


# =============================================================================
# Property Tests for Invalid Input Handling
# =============================================================================

class TestFeatureExtractorInputValidation:
    """
    Property tests for feature extractor input validation.
    
    Feature: chart-pattern-analysis-framework
    Property 4: Feature Extraction Completeness
    Validates: Requirements 2.5 (empty feature set with appropriate status)
    """
    
    def test_invalid_input_returns_empty_feature_map(self):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.5
        
        When input is invalid, the Feature_Extractor SHALL return an empty
        FeatureMap with appropriate status.
        """
        extractor = EdgeBasedFeatureExtractor()
        
        # Test with None input
        result = extractor.process(None, {})
        assert isinstance(result, FeatureMap), "Should return FeatureMap for invalid input"
        assert result.candlestick_regions == [], "Should have empty candlestick_regions"
        assert result.trendlines == [], "Should have empty trendlines"
        assert result.support_zones == [], "Should have empty support_zones"
        assert result.resistance_zones == [], "Should have empty resistance_zones"
    
    def test_non_rgb_image_returns_empty_feature_map(self):
        """
        Feature: chart-pattern-analysis-framework
        Property 4: Feature Extraction Completeness
        Validates: Requirements 2.5
        
        When input image is not RGB (3 channels), the Feature_Extractor SHALL
        return an empty FeatureMap.
        """
        extractor = EdgeBasedFeatureExtractor()
        
        # Create grayscale image (2D array)
        gray_image = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        
        invalid_preprocess = PreprocessResult(
            image=gray_image,
            original_size=(200, 200),
            processed_size=(200, 200),
            transformations=[],
            quality_score=0.5,
            masked_regions=[]
        )
        
        result = extractor.process(invalid_preprocess, {})
        assert isinstance(result, FeatureMap), "Should return FeatureMap for invalid input"
        assert result.quality_score == 0.0, "Empty FeatureMap should have quality_score=0.0"


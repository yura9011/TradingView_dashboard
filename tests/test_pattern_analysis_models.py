"""
Property-based tests for pattern analysis data models.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from src.pattern_analysis.models import (
    PatternCategory,
    PatternType,
    BoundingBox,
    PatternDetection,
    FeatureMap,
    AnalysisResult,
)


# =============================================================================
# Custom Strategies for Domain Objects
# =============================================================================

@st.composite
def bounding_box_strategy(draw, max_dim=1000):
    """
    Generate valid bounding boxes where x1 < x2 and y1 < y2.
    """
    x1 = draw(st.integers(min_value=0, max_value=max_dim - 10))
    y1 = draw(st.integers(min_value=0, max_value=max_dim - 10))
    x2 = draw(st.integers(min_value=x1 + 1, max_value=max_dim))
    y2 = draw(st.integers(min_value=y1 + 1, max_value=max_dim))
    return BoundingBox(x1, y1, x2, y2)


@st.composite
def invalid_bounding_box_strategy(draw, max_dim=1000):
    """
    Generate invalid bounding boxes where x1 >= x2 or y1 >= y2.
    """
    # Generate coordinates that violate the validity constraint
    choice = draw(st.integers(min_value=0, max_value=2))
    
    if choice == 0:
        # x1 >= x2
        x2 = draw(st.integers(min_value=0, max_value=max_dim - 1))
        x1 = draw(st.integers(min_value=x2, max_value=max_dim))
        y1 = draw(st.integers(min_value=0, max_value=max_dim - 10))
        y2 = draw(st.integers(min_value=y1 + 1, max_value=max_dim))
    elif choice == 1:
        # y1 >= y2
        x1 = draw(st.integers(min_value=0, max_value=max_dim - 10))
        x2 = draw(st.integers(min_value=x1 + 1, max_value=max_dim))
        y2 = draw(st.integers(min_value=0, max_value=max_dim - 1))
        y1 = draw(st.integers(min_value=y2, max_value=max_dim))
    else:
        # Both invalid
        x2 = draw(st.integers(min_value=0, max_value=max_dim - 1))
        x1 = draw(st.integers(min_value=x2, max_value=max_dim))
        y2 = draw(st.integers(min_value=0, max_value=max_dim - 1))
        y1 = draw(st.integers(min_value=y2, max_value=max_dim))
    
    return BoundingBox(x1, y1, x2, y2)


# =============================================================================
# Property Tests for BoundingBox
# =============================================================================

class TestBoundingBoxValidity:
    """
    Property tests for BoundingBox validity.
    
    Feature: chart-pattern-analysis-framework
    Property 5: Pattern Detection Validity (partial)
    Validates: Requirements 3.5
    """
    
    @given(bounding_box_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_bounding_box_has_x1_less_than_x2(self, bbox: BoundingBox):
        """
        Feature: chart-pattern-analysis-framework
        Property 5: Pattern Detection Validity (partial)
        Validates: Requirements 3.5
        
        For any valid bounding box, x1 SHALL be less than x2.
        """
        assert bbox.x1 < bbox.x2, f"Expected x1 ({bbox.x1}) < x2 ({bbox.x2})"
    
    @given(bounding_box_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_bounding_box_has_y1_less_than_y2(self, bbox: BoundingBox):
        """
        Feature: chart-pattern-analysis-framework
        Property 5: Pattern Detection Validity (partial)
        Validates: Requirements 3.5
        
        For any valid bounding box, y1 SHALL be less than y2.
        """
        assert bbox.y1 < bbox.y2, f"Expected y1 ({bbox.y1}) < y2 ({bbox.y2})"
    
    @given(bounding_box_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_bounding_box_is_valid_returns_true(self, bbox: BoundingBox):
        """
        Feature: chart-pattern-analysis-framework
        Property 5: Pattern Detection Validity (partial)
        Validates: Requirements 3.5
        
        For any bounding box generated with valid constraints, is_valid() SHALL return True.
        """
        assert bbox.is_valid(), f"BoundingBox({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2}) should be valid"
    
    @given(invalid_bounding_box_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_invalid_bounding_box_is_valid_returns_false(self, bbox: BoundingBox):
        """
        Feature: chart-pattern-analysis-framework
        Property 5: Pattern Detection Validity (partial)
        Validates: Requirements 3.5
        
        For any bounding box with x1 >= x2 or y1 >= y2, is_valid() SHALL return False.
        """
        assert not bbox.is_valid(), f"BoundingBox({bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2}) should be invalid"
    
    @given(bounding_box_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_bounding_box_has_positive_area(self, bbox: BoundingBox):
        """
        Feature: chart-pattern-analysis-framework
        Property 5: Pattern Detection Validity (partial)
        Validates: Requirements 3.5
        
        For any valid bounding box, area() SHALL return a positive value.
        """
        area = bbox.area()
        assert area > 0, f"Expected positive area, got {area}"
        expected_area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
        assert area == expected_area, f"Expected area {expected_area}, got {area}"


# =============================================================================
# Custom Strategies for Serialization Tests
# =============================================================================

@st.composite
def pattern_detection_strategy(draw):
    """Generate valid pattern detections."""
    bbox = draw(bounding_box_strategy())
    return PatternDetection(
        pattern_type=draw(st.sampled_from(list(PatternType))),
        category=draw(st.sampled_from(list(PatternCategory))),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        bounding_box=bbox,
        metadata={},  # Simplified for speed
        detector_id=draw(st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'))
    )


@st.composite
def feature_map_strategy(draw):
    """Generate valid feature maps."""
    return FeatureMap(
        candlestick_regions=draw(st.lists(bounding_box_strategy(), min_size=0, max_size=2)),
        trendlines=[],  # Simplified for speed
        support_zones=draw(st.lists(bounding_box_strategy(), min_size=0, max_size=2)),
        resistance_zones=draw(st.lists(bounding_box_strategy(), min_size=0, max_size=2)),
        volume_profile=None,  # Simplified for speed
        quality_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    )


@st.composite
def analysis_result_strategy(draw):
    """Generate valid analysis results for round-trip testing."""
    from datetime import datetime
    
    detections = draw(st.lists(pattern_detection_strategy(), min_size=0, max_size=3))
    
    # Generate timing values
    preprocessing_time = draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
    extraction_time = draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
    classification_time = draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
    validation_time = draw(st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False))
    total_time = draw(st.floats(min_value=0, max_value=5000, allow_nan=False, allow_infinity=False))
    
    return AnalysisResult(
        image_path=draw(st.text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_.')),
        timestamp=datetime.now().isoformat(),
        preprocessing_time_ms=preprocessing_time,
        extraction_time_ms=extraction_time,
        classification_time_ms=classification_time,
        validation_time_ms=validation_time,
        total_time_ms=total_time,
        detections=detections,
        validated_detections=[],  # Keep simple for round-trip test
        feature_map=draw(feature_map_strategy()),
        config_used={}  # Simplified for speed
    )


# =============================================================================
# Property Tests for Serialization Round-Trip
# =============================================================================

class TestSerializationRoundTrip:
    """
    Property tests for serialization round-trip.
    
    Feature: chart-pattern-analysis-framework
    Property 11: Serialization Round-Trip
    Validates: Requirements 7.6
    """
    
    @given(analysis_result_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_json_round_trip_preserves_data(self, result: AnalysisResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 11: Serialization Round-Trip
        Validates: Requirements 7.6
        
        For any valid AnalysisResult object R, serializing R to JSON and then
        deserializing SHALL produce an object R' such that R == R' (structural equality).
        """
        # Serialize to JSON (with validation disabled to avoid schema strictness)
        json_str = result.to_json(validate=False)
        
        # Deserialize back (with validation disabled)
        restored = AnalysisResult.from_json(json_str, validate=False)
        
        # Verify structural equality
        assert result == restored, (
            f"Round-trip should preserve all data.\n"
            f"Original image_path: {result.image_path}\n"
            f"Restored image_path: {restored.image_path}\n"
            f"Original detections count: {len(result.detections)}\n"
            f"Restored detections count: {len(restored.detections)}"
        )
    
    @given(analysis_result_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_json_round_trip_preserves_detection_count(self, result: AnalysisResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 11: Serialization Round-Trip
        Validates: Requirements 7.6
        
        For any valid AnalysisResult, the number of detections SHALL be preserved
        after serialization and deserialization.
        """
        json_str = result.to_json(validate=False)
        restored = AnalysisResult.from_json(json_str, validate=False)
        
        assert len(result.detections) == len(restored.detections), (
            f"Detection count should be preserved. "
            f"Original: {len(result.detections)}, Restored: {len(restored.detections)}"
        )
    
    @given(analysis_result_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_json_round_trip_preserves_timing_metrics(self, result: AnalysisResult):
        """
        Feature: chart-pattern-analysis-framework
        Property 11: Serialization Round-Trip
        Validates: Requirements 7.6
        
        For any valid AnalysisResult, all timing metrics SHALL be preserved
        within floating point tolerance after round-trip.
        """
        json_str = result.to_json(validate=False)
        restored = AnalysisResult.from_json(json_str, validate=False)
        
        tolerance = 0.001
        assert abs(result.preprocessing_time_ms - restored.preprocessing_time_ms) < tolerance
        assert abs(result.extraction_time_ms - restored.extraction_time_ms) < tolerance
        assert abs(result.classification_time_ms - restored.classification_time_ms) < tolerance
        assert abs(result.validation_time_ms - restored.validation_time_ms) < tolerance
        assert abs(result.total_time_ms - restored.total_time_ms) < tolerance
    
    @given(bounding_box_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_bounding_box_round_trip(self, bbox: BoundingBox):
        """
        Feature: chart-pattern-analysis-framework
        Property 11: Serialization Round-Trip (component)
        Validates: Requirements 7.6
        
        For any valid BoundingBox, to_dict() then from_dict() SHALL produce
        an equivalent BoundingBox.
        """
        data = bbox.to_dict()
        restored = BoundingBox.from_dict(data)
        
        assert bbox.x1 == restored.x1
        assert bbox.y1 == restored.y1
        assert bbox.x2 == restored.x2
        assert bbox.y2 == restored.y2
    
    @given(pattern_detection_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_pattern_detection_round_trip(self, detection: PatternDetection):
        """
        Feature: chart-pattern-analysis-framework
        Property 11: Serialization Round-Trip (component)
        Validates: Requirements 7.6
        
        For any valid PatternDetection, to_dict() then from_dict() SHALL produce
        an equivalent PatternDetection.
        """
        data = detection.to_dict()
        restored = PatternDetection.from_dict(data)
        
        assert detection.pattern_type == restored.pattern_type
        assert detection.category == restored.category
        assert abs(detection.confidence - restored.confidence) < 0.001
        assert detection.detector_id == restored.detector_id
        assert detection.bounding_box.x1 == restored.bounding_box.x1
        assert detection.bounding_box.y1 == restored.bounding_box.y1
        assert detection.bounding_box.x2 == restored.bounding_box.x2
        assert detection.bounding_box.y2 == restored.bounding_box.y2

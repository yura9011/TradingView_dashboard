"""
Property-based tests for HybridPatternClassifier.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.

Feature: chart-pattern-analysis-framework
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
import numpy as np

from src.pattern_analysis.models import (
    PatternCategory,
    PatternType,
    BoundingBox,
    PatternDetection,
    FeatureMap,
)
from src.pattern_analysis.pipeline import HybridPatternClassifier


# =============================================================================
# Custom Strategies for Domain Objects
# =============================================================================

@st.composite
def bounding_box_strategy(draw, max_dim=500):
    """Generate valid bounding boxes where x1 < x2 and y1 < y2."""
    x1 = draw(st.integers(min_value=0, max_value=max_dim - 10))
    y1 = draw(st.integers(min_value=0, max_value=max_dim - 10))
    x2 = draw(st.integers(min_value=x1 + 1, max_value=max_dim))
    y2 = draw(st.integers(min_value=y1 + 1, max_value=max_dim))
    return BoundingBox(x1, y1, x2, y2)


@st.composite
def trendline_strategy(draw, max_dim=500):
    """Generate valid trendline dictionaries."""
    x1 = draw(st.integers(min_value=0, max_value=max_dim - 100))
    y1 = draw(st.integers(min_value=0, max_value=max_dim))
    x2 = draw(st.integers(min_value=x1 + 50, max_value=max_dim))
    y2 = draw(st.integers(min_value=0, max_value=max_dim))
    
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    direction = "up" if angle < 0 else "down"
    
    return {
        "start": (x1, y1),
        "end": (x2, y2),
        "angle": float(angle),
        "direction": direction,
        "length": float(length)
    }


@st.composite
def feature_map_strategy(draw):
    """Generate valid feature maps for testing."""
    candlesticks = draw(st.lists(bounding_box_strategy(), min_size=0, max_size=5))
    trendlines = draw(st.lists(trendline_strategy(), min_size=0, max_size=3))
    support = draw(st.lists(bounding_box_strategy(), min_size=0, max_size=3))
    resistance = draw(st.lists(bounding_box_strategy(), min_size=0, max_size=3))
    
    return FeatureMap(
        candlestick_regions=candlesticks,
        trendlines=trendlines,
        support_zones=support,
        resistance_zones=resistance,
        volume_profile=None,
        quality_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    )


@st.composite
def image_strategy(draw, min_dim=200, max_dim=500):
    """Generate random valid images."""
    width = draw(st.integers(min_value=min_dim, max_value=max_dim))
    height = draw(st.integers(min_value=min_dim, max_value=max_dim))
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


@st.composite
def pattern_detection_strategy(draw):
    """Generate valid pattern detections."""
    bbox = draw(bounding_box_strategy())
    return PatternDetection(
        pattern_type=draw(st.sampled_from(list(PatternType))),
        category=draw(st.sampled_from(list(PatternCategory))),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        bounding_box=bbox,
        metadata={},
        detector_id=draw(st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'))
    )


# =============================================================================
# Property Tests for Pattern Detection Validity (Property 5)
# =============================================================================

class TestPatternDetectionValidity:
    """
    Property tests for pattern detection validity.
    
    Feature: chart-pattern-analysis-framework
    Property 5: Pattern Detection Validity
    Validates: Requirements 3.1, 3.2, 3.3, 3.5, 3.6
    """
    
    @given(feature_map_strategy(), image_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_detections_have_valid_pattern_type(
        self,
        features: FeatureMap,
        image: np.ndarray
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 5: Pattern Detection Validity
        Validates: Requirements 3.1, 3.2
        
        For any pattern detection returned by the classifier,
        pattern_type SHALL be a valid PatternType enum value.
        """
        classifier = HybridPatternClassifier()
        detections = classifier.classify(features, image)
        
        for detection in detections:
            assert isinstance(detection.pattern_type, PatternType), (
                f"pattern_type must be PatternType enum, got {type(detection.pattern_type)}"
            )
    
    @given(feature_map_strategy(), image_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_detections_have_valid_category(
        self,
        features: FeatureMap,
        image: np.ndarray
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 5: Pattern Detection Validity
        Validates: Requirements 3.1, 3.2
        
        For any pattern detection returned by the classifier,
        category SHALL be a valid PatternCategory enum value.
        """
        classifier = HybridPatternClassifier()
        detections = classifier.classify(features, image)
        
        for detection in detections:
            assert isinstance(detection.category, PatternCategory), (
                f"category must be PatternCategory enum, got {type(detection.category)}"
            )
    
    @given(feature_map_strategy(), image_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_detections_have_valid_confidence(
        self,
        features: FeatureMap,
        image: np.ndarray
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 5: Pattern Detection Validity
        Validates: Requirements 3.3
        
        For any pattern detection returned by the classifier,
        confidence SHALL be in range [0.0, 1.0].
        """
        classifier = HybridPatternClassifier()
        detections = classifier.classify(features, image)
        
        for detection in detections:
            assert 0.0 <= detection.confidence <= 1.0, (
                f"confidence must be in [0.0, 1.0], got {detection.confidence}"
            )
    
    @given(feature_map_strategy(), image_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_detections_have_valid_bounding_box(
        self,
        features: FeatureMap,
        image: np.ndarray
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 5: Pattern Detection Validity
        Validates: Requirements 3.5
        
        For any pattern detection returned by the classifier,
        bounding_box SHALL have x1 < x2 and y1 < y2.
        """
        classifier = HybridPatternClassifier()
        detections = classifier.classify(features, image)
        
        for detection in detections:
            bbox = detection.bounding_box
            assert bbox.x1 < bbox.x2, (
                f"bounding_box must have x1 < x2, got x1={bbox.x1}, x2={bbox.x2}"
            )
            assert bbox.y1 < bbox.y2, (
                f"bounding_box must have y1 < y2, got y1={bbox.y1}, y2={bbox.y2}"
            )
    
    @given(feature_map_strategy(), image_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_detections_have_non_null_metadata(
        self,
        features: FeatureMap,
        image: np.ndarray
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 5: Pattern Detection Validity
        Validates: Requirements 3.6
        
        For any pattern detection returned by the classifier,
        metadata SHALL be a non-null dictionary.
        """
        classifier = HybridPatternClassifier()
        detections = classifier.classify(features, image)
        
        for detection in detections:
            assert detection.metadata is not None, "metadata must not be None"
            assert isinstance(detection.metadata, dict), (
                f"metadata must be a dict, got {type(detection.metadata)}"
            )



# =============================================================================
# Property Tests for Detection Ordering (Property 6)
# =============================================================================

class TestDetectionOrdering:
    """
    Property tests for detection ordering by confidence.
    
    Feature: chart-pattern-analysis-framework
    Property 6: Detection Ordering by Confidence
    Validates: Requirements 3.4
    """
    
    @given(feature_map_strategy(), image_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_detections_ordered_by_confidence_descending(
        self,
        features: FeatureMap,
        image: np.ndarray
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 6: Detection Ordering by Confidence
        Validates: Requirements 3.4
        
        For any list of pattern detections returned by the classifier,
        the detections SHALL be ordered by confidence in descending order.
        """
        classifier = HybridPatternClassifier()
        detections = classifier.classify(features, image)
        
        # Verify descending confidence order
        for i in range(len(detections) - 1):
            assert detections[i].confidence >= detections[i + 1].confidence, (
                f"Detections must be ordered by confidence descending. "
                f"Detection {i} has confidence {detections[i].confidence}, "
                f"but detection {i+1} has confidence {detections[i+1].confidence}"
            )
    
    @given(st.lists(pattern_detection_strategy(), min_size=1, max_size=20))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_merge_detections_preserves_confidence_ordering(
        self,
        detections: list
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 6: Detection Ordering by Confidence
        Validates: Requirements 3.4
        
        For any list of detections passed to _merge_detections,
        the output SHALL be ordered by confidence in descending order.
        """
        classifier = HybridPatternClassifier()
        merged = classifier._merge_detections(detections)
        
        # Verify descending confidence order
        for i in range(len(merged) - 1):
            assert merged[i].confidence >= merged[i + 1].confidence, (
                f"Merged detections must be ordered by confidence descending. "
                f"Detection {i} has confidence {merged[i].confidence}, "
                f"but detection {i+1} has confidence {merged[i+1].confidence}"
            )
    
    @given(st.lists(pattern_detection_strategy(), min_size=2, max_size=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_merge_keeps_highest_confidence_for_overlapping(
        self,
        detections: list
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 6: Detection Ordering by Confidence
        Validates: Requirements 3.4
        
        When merging overlapping detections, the one with highest
        confidence SHALL be kept.
        """
        classifier = HybridPatternClassifier()
        merged = classifier._merge_detections(detections)
        
        # The first detection in merged should have the highest confidence
        # among all input detections (since we sort by confidence first)
        if merged:
            max_input_confidence = max(d.confidence for d in detections)
            assert merged[0].confidence == max_input_confidence, (
                f"First merged detection should have highest confidence. "
                f"Expected {max_input_confidence}, got {merged[0].confidence}"
            )

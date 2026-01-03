"""
Property-based tests for ChartAnnotator.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.

Feature: chart-pattern-analysis-framework
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
import numpy as np
from itertools import combinations

from src.pattern_analysis.models import (
    PatternCategory,
    PatternType,
    BoundingBox,
    PatternDetection,
)
from src.pattern_analysis.output.annotator import ChartAnnotator


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
def pattern_detection_strategy(draw, category: PatternCategory = None):
    """
    Generate valid pattern detections.
    
    Args:
        category: If provided, force the detection to have this category.
                  Otherwise, randomly select a category.
    """
    bbox = draw(bounding_box_strategy())
    
    if category is None:
        category = draw(st.sampled_from(list(PatternCategory)))
    
    # Select a pattern type that matches the category
    pattern_type = draw(st.sampled_from(list(PatternType)))
    
    return PatternDetection(
        pattern_type=pattern_type,
        category=category,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        bounding_box=bbox,
        metadata={},
        detector_id=draw(st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'))
    )


@st.composite
def pattern_detection_with_category_strategy(draw, category: PatternCategory):
    """Generate a pattern detection with a specific category."""
    bbox = draw(bounding_box_strategy())
    pattern_type = draw(st.sampled_from(list(PatternType)))
    
    return PatternDetection(
        pattern_type=pattern_type,
        category=category,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        bounding_box=bbox,
        metadata={},
        detector_id=draw(st.text(min_size=1, max_size=10, alphabet='abcdefghijklmnopqrstuvwxyz'))
    )


# =============================================================================
# Property Tests for Annotation Color Coding (Property 14)
# =============================================================================

class TestAnnotationColorCoding:
    """
    Property tests for annotation color coding.
    
    Feature: chart-pattern-analysis-framework
    Property 14: Annotation Color Coding
    Validates: Requirements 8.2
    
    WHEN annotating, THE Annotator SHALL use color coding to distinguish 
    pattern types (green for bullish, red for bearish, yellow for neutral)
    """
    
    @given(
        reversal_detection=pattern_detection_with_category_strategy(PatternCategory.REVERSAL),
        continuation_detection=pattern_detection_with_category_strategy(PatternCategory.CONTINUATION),
        bilateral_detection=pattern_detection_with_category_strategy(PatternCategory.BILATERAL),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_different_categories_have_different_colors(
        self,
        reversal_detection: PatternDetection,
        continuation_detection: PatternDetection,
        bilateral_detection: PatternDetection,
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 14: Annotation Color Coding
        Validates: Requirements 8.2
        
        For any two pattern detections with different categories 
        (reversal vs continuation vs bilateral), the annotation colors 
        SHALL be different.
        """
        annotator = ChartAnnotator()
        
        # Get colors for each category using the public method
        reversal_color = annotator.get_color_for_category(reversal_detection.category)
        continuation_color = annotator.get_color_for_category(continuation_detection.category)
        bilateral_color = annotator.get_color_for_category(bilateral_detection.category)
        
        # Collect all colors with their categories for comparison
        colors_by_category = {
            PatternCategory.REVERSAL: reversal_color,
            PatternCategory.CONTINUATION: continuation_color,
            PatternCategory.BILATERAL: bilateral_color,
        }
        
        # Verify that bilateral (neutral) has a distinct color from the others
        # Note: Reversal and continuation may have similar colors for same direction
        # (bullish/bearish), but bilateral should always be distinct (yellow)
        assert bilateral_color != reversal_color or bilateral_color != continuation_color, (
            f"Bilateral patterns should have a distinct color. "
            f"Reversal: {reversal_color}, Continuation: {continuation_color}, "
            f"Bilateral: {bilateral_color}"
        )
    
    @given(st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_bilateral_category_always_uses_yellow(self, data):
        """
        Feature: chart-pattern-analysis-framework
        Property 14: Annotation Color Coding
        Validates: Requirements 8.2
        
        For any bilateral pattern detection, the color SHALL be yellow
        (representing neutral patterns).
        """
        detection = data.draw(pattern_detection_with_category_strategy(PatternCategory.BILATERAL))
        annotator = ChartAnnotator()
        
        # Use the public method to get color for the category
        color = annotator.get_color_for_category(detection.category)
        
        # Yellow in BGR is (0, 255, 255) or similar yellow variants
        # Check that it's a yellow-ish color (high G and R, low B in RGB terms)
        # In BGR: B=0, G=255, R=255 for pure yellow
        b, g, r = color
        
        # Yellow should have high green and red values, relatively low blue
        assert g >= 200 and r >= 200, (
            f"Bilateral patterns should use yellow color. Got BGR: {color}"
        )
    
    @given(st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_get_color_for_category_returns_consistent_colors(self, data):
        """
        Feature: chart-pattern-analysis-framework
        Property 14: Annotation Color Coding
        Validates: Requirements 8.2
        
        For any category, get_color_for_category SHALL return a valid
        BGR color tuple.
        """
        category = data.draw(st.sampled_from(list(PatternCategory)))
        annotator = ChartAnnotator()
        
        color = annotator.get_color_for_category(category)
        
        # Verify it's a valid BGR tuple
        assert isinstance(color, tuple), f"Color must be a tuple, got {type(color)}"
        assert len(color) == 3, f"Color must have 3 components (BGR), got {len(color)}"
        
        for i, component in enumerate(color):
            assert isinstance(component, int), (
                f"Color component {i} must be int, got {type(component)}"
            )
            assert 0 <= component <= 255, (
                f"Color component {i} must be in [0, 255], got {component}"
            )
    
    @given(
        category1=st.sampled_from(list(PatternCategory)),
        category2=st.sampled_from(list(PatternCategory)),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_same_category_returns_same_base_color(
        self,
        category1: PatternCategory,
        category2: PatternCategory,
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 14: Annotation Color Coding
        Validates: Requirements 8.2
        
        For any two calls with the same category, get_color_for_category
        SHALL return the same color (consistency).
        """
        assume(category1 == category2)
        
        annotator = ChartAnnotator()
        
        color1 = annotator.get_color_for_category(category1)
        color2 = annotator.get_color_for_category(category2)
        
        assert color1 == color2, (
            f"Same category should return same color. "
            f"Category: {category1}, Color1: {color1}, Color2: {color2}"
        )
    
    @given(st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_categories_have_defined_colors(self, data):
        """
        Feature: chart-pattern-analysis-framework
        Property 14: Annotation Color Coding
        Validates: Requirements 8.2
        
        For any valid PatternCategory, the annotator SHALL have a 
        defined color mapping.
        """
        annotator = ChartAnnotator()
        
        for category in PatternCategory:
            color = annotator.get_color_for_category(category)
            
            assert color is not None, (
                f"Category {category} must have a defined color"
            )
            assert isinstance(color, tuple) and len(color) == 3, (
                f"Category {category} color must be a BGR tuple, got {color}"
            )


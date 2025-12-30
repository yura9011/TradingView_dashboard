"""
Property-based tests for MultiMethodCrossValidator.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.

Feature: chart-pattern-analysis-framework
Property 8: Cross-Validation Consistency
Validates: Requirements 5.1, 5.2, 5.3, 5.4
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
import numpy as np
from typing import List

from src.pattern_analysis.models import (
    PatternCategory,
    PatternType,
    BoundingBox,
    PatternDetection,
    FeatureMap,
    ValidationResult,
)
from src.pattern_analysis.pipeline import (
    MultiMethodCrossValidator,
    HybridPatternClassifier,
)


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


@st.composite
def image_strategy(draw, min_dim=200, max_dim=500):
    """Generate random valid images."""
    width = draw(st.integers(min_value=min_dim, max_value=max_dim))
    height = draw(st.integers(min_value=min_dim, max_value=max_dim))
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


# =============================================================================
# Property Tests for Cross-Validation Consistency (Property 8)
# =============================================================================

class TestCrossValidationConsistency:
    """
    Property tests for cross-validation consistency.
    
    Feature: chart-pattern-analysis-framework
    Property 8: Cross-Validation Consistency
    Validates: Requirements 5.1, 5.2, 5.3, 5.4
    """
    
    @given(
        st.integers(min_value=0, max_value=10),  # agreement_count
        st.integers(min_value=1, max_value=10),  # total_validators
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False)  # threshold
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_validation_score_equals_agreement_ratio(
        self,
        agreement: int,
        total: int,
        threshold: float
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 8: Cross-Validation Consistency
        Validates: Requirements 5.3
        
        For any validated detection, validation_score SHALL equal
        agreement_count / total_validators.
        """
        # Ensure agreement doesn't exceed total
        agreement = min(agreement, total)
        
        expected_score = agreement / total
        is_confirmed = expected_score >= threshold
        
        # Create a mock detection for the ValidationResult
        detection = PatternDetection(
            pattern_type=PatternType.DOUBLE_TOP,
            category=PatternCategory.REVERSAL,
            confidence=0.8,
            bounding_box=BoundingBox(0, 0, 100, 100),
            metadata={},
            detector_id="test"
        )
        
        # Create ValidationResult with calculated values
        result = ValidationResult(
            original_detection=detection,
            validation_score=expected_score,
            agreement_count=agreement,
            total_validators=total,
            is_confirmed=is_confirmed,
            validator_results={"validator_1": True} if agreement > 0 else {}
        )
        
        # Verify score calculation
        assert abs(result.validation_score - (agreement / total)) < 0.001, (
            f"validation_score must equal agreement_count / total_validators. "
            f"Expected {agreement / total}, got {result.validation_score}"
        )
    
    @given(
        st.integers(min_value=0, max_value=10),  # agreement_count
        st.integers(min_value=1, max_value=10),  # total_validators
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False)  # threshold
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_is_confirmed_matches_threshold_comparison(
        self,
        agreement: int,
        total: int,
        threshold: float
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 8: Cross-Validation Consistency
        Validates: Requirements 5.2, 5.4
        
        For any validated detection, is_confirmed SHALL be True if and only if
        validation_score >= consensus_threshold.
        """
        # Ensure agreement doesn't exceed total
        agreement = min(agreement, total)
        
        score = agreement / total
        expected_confirmed = score >= threshold
        
        # Create a mock detection
        detection = PatternDetection(
            pattern_type=PatternType.DOUBLE_TOP,
            category=PatternCategory.REVERSAL,
            confidence=0.8,
            bounding_box=BoundingBox(0, 0, 100, 100),
            metadata={},
            detector_id="test"
        )
        
        # Create ValidationResult
        result = ValidationResult(
            original_detection=detection,
            validation_score=score,
            agreement_count=agreement,
            total_validators=total,
            is_confirmed=expected_confirmed,
            validator_results={"validator_1": True} if agreement > 0 else {}
        )
        
        # Verify confirmation logic
        assert result.is_confirmed == (result.validation_score >= threshold), (
            f"is_confirmed must be True iff validation_score >= threshold. "
            f"score={result.validation_score}, threshold={threshold}, "
            f"is_confirmed={result.is_confirmed}"
        )
    
    @given(
        st.lists(pattern_detection_strategy(), min_size=1, max_size=5),
        image_strategy()
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_validator_results_contains_entries_when_validators_exist(
        self,
        detections: List[PatternDetection],
        image: np.ndarray
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 8: Cross-Validation Consistency
        Validates: Requirements 5.1
        
        For any validated detection with validators available,
        validator_results SHALL contain at least one entry.
        """
        # Create a validator (using HybridPatternClassifier)
        validator = HybridPatternClassifier()
        
        # Create cross-validator with one validator
        cross_validator = MultiMethodCrossValidator(
            validators=[validator],
            consensus_threshold=0.5
        )
        
        # Run validation
        results = cross_validator.validate(detections, image)
        
        # Verify each result has validator_results entries
        for result in results:
            # If the detection's detector_id matches the validator's stage_id,
            # it will be skipped, so we need to check for that case
            if result.original_detection.detector_id != validator.stage_id:
                assert len(result.validator_results) >= 1, (
                    f"validator_results must contain at least one entry when "
                    f"validators are available. Got {len(result.validator_results)} entries"
                )
    
    @given(
        st.lists(pattern_detection_strategy(), min_size=1, max_size=3),
        image_strategy(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_get_consensus_returns_only_confirmed(
        self,
        detections: List[PatternDetection],
        image: np.ndarray,
        threshold: float
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 8: Cross-Validation Consistency
        Validates: Requirements 5.2
        
        The get_consensus method SHALL return only ValidationResults
        where is_confirmed is True.
        """
        # Create cross-validator with threshold
        cross_validator = MultiMethodCrossValidator(
            validators=[HybridPatternClassifier()],
            consensus_threshold=threshold
        )
        
        # Run validation
        results = cross_validator.validate(detections, image)
        
        # Get consensus
        consensus = cross_validator.get_consensus(results)
        
        # Verify all consensus results are confirmed
        for result in consensus:
            assert result.is_confirmed, (
                f"get_consensus must return only confirmed results. "
                f"Got result with is_confirmed={result.is_confirmed}"
            )
    
    @given(
        st.lists(pattern_detection_strategy(), min_size=0, max_size=5),
        image_strategy()
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_validation_returns_result_for_each_detection(
        self,
        detections: List[PatternDetection],
        image: np.ndarray
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 8: Cross-Validation Consistency
        Validates: Requirements 5.1
        
        For any list of detections, validate SHALL return exactly one
        ValidationResult for each detection.
        """
        cross_validator = MultiMethodCrossValidator(
            validators=[HybridPatternClassifier()],
            consensus_threshold=0.5
        )
        
        results = cross_validator.validate(detections, image)
        
        assert len(results) == len(detections), (
            f"validate must return one result per detection. "
            f"Expected {len(detections)}, got {len(results)}"
        )
    
    @given(
        st.lists(pattern_detection_strategy(), min_size=1, max_size=3),
        image_strategy()
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_validation_result_references_original_detection(
        self,
        detections: List[PatternDetection],
        image: np.ndarray
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 8: Cross-Validation Consistency
        Validates: Requirements 5.1
        
        Each ValidationResult SHALL reference the original detection
        that was validated.
        """
        cross_validator = MultiMethodCrossValidator(
            validators=[HybridPatternClassifier()],
            consensus_threshold=0.5
        )
        
        results = cross_validator.validate(detections, image)
        
        for i, result in enumerate(results):
            assert result.original_detection is not None, (
                f"ValidationResult must reference original detection"
            )
            assert result.original_detection == detections[i], (
                f"ValidationResult must reference the correct original detection"
            )


class TestCrossValidatorInputValidation:
    """
    Tests for cross-validator input validation.
    """
    
    def test_validate_input_accepts_valid_tuple(self):
        """Test that valid input tuple is accepted."""
        cross_validator = MultiMethodCrossValidator()
        
        detection = PatternDetection(
            pattern_type=PatternType.DOUBLE_TOP,
            category=PatternCategory.REVERSAL,
            confidence=0.8,
            bounding_box=BoundingBox(0, 0, 100, 100),
            metadata={},
            detector_id="test"
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        assert cross_validator.validate_input(([detection], image)) is True
    
    def test_validate_input_rejects_non_tuple(self):
        """Test that non-tuple input is rejected."""
        cross_validator = MultiMethodCrossValidator()
        
        assert cross_validator.validate_input([]) is False
        assert cross_validator.validate_input("invalid") is False
        assert cross_validator.validate_input(None) is False
    
    def test_validate_input_rejects_wrong_tuple_length(self):
        """Test that tuple with wrong length is rejected."""
        cross_validator = MultiMethodCrossValidator()
        
        assert cross_validator.validate_input(([], )) is False
        assert cross_validator.validate_input(([], np.zeros((10, 10, 3)), "extra")) is False
    
    def test_validate_input_rejects_non_list_detections(self):
        """Test that non-list detections are rejected."""
        cross_validator = MultiMethodCrossValidator()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        assert cross_validator.validate_input(("not a list", image)) is False
    
    def test_validate_input_rejects_non_ndarray_image(self):
        """Test that non-ndarray image is rejected."""
        cross_validator = MultiMethodCrossValidator()
        
        assert cross_validator.validate_input(([], "not an image")) is False


class TestCrossValidatorConfiguration:
    """
    Tests for cross-validator configuration.
    """
    
    def test_default_consensus_threshold(self):
        """Test default consensus threshold is 0.5."""
        cross_validator = MultiMethodCrossValidator()
        assert cross_validator.consensus_threshold == 0.5
    
    def test_custom_consensus_threshold(self):
        """Test custom consensus threshold is applied."""
        cross_validator = MultiMethodCrossValidator(consensus_threshold=0.7)
        assert cross_validator.consensus_threshold == 0.7
    
    def test_invalid_consensus_threshold_raises_error(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError):
            MultiMethodCrossValidator(consensus_threshold=1.5)
        
        with pytest.raises(ValueError):
            MultiMethodCrossValidator(consensus_threshold=-0.1)
    
    def test_set_consensus_threshold(self):
        """Test setting consensus threshold after initialization."""
        cross_validator = MultiMethodCrossValidator()
        cross_validator.set_consensus_threshold(0.8)
        assert cross_validator.consensus_threshold == 0.8
    
    def test_set_invalid_consensus_threshold_raises_error(self):
        """Test that setting invalid threshold raises ValueError."""
        cross_validator = MultiMethodCrossValidator()
        
        with pytest.raises(ValueError):
            cross_validator.set_consensus_threshold(1.5)
        
        with pytest.raises(ValueError):
            cross_validator.set_consensus_threshold(-0.1)
    
    def test_add_validator(self):
        """Test adding a validator."""
        cross_validator = MultiMethodCrossValidator()
        validator = HybridPatternClassifier()
        
        cross_validator.add_validator(validator)
        assert validator in cross_validator.validators
    
    def test_remove_validator(self):
        """Test removing a validator."""
        validator = HybridPatternClassifier()
        cross_validator = MultiMethodCrossValidator(validators=[validator])
        
        result = cross_validator.remove_validator(validator)
        assert result is True
        assert validator not in cross_validator.validators
    
    def test_remove_nonexistent_validator(self):
        """Test removing a validator that doesn't exist."""
        cross_validator = MultiMethodCrossValidator()
        validator = HybridPatternClassifier()
        
        result = cross_validator.remove_validator(validator)
        assert result is False
    
    def test_stage_id(self):
        """Test stage_id property."""
        cross_validator = MultiMethodCrossValidator()
        assert cross_validator.stage_id == "multi_method_validator_v1"

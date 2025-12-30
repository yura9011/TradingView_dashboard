"""
Property-based tests for PipelineExecutor.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.

Feature: chart-pattern-analysis-framework
Property 7: Pipeline Result Aggregation
Validates: Requirements 4.6
"""

import os
import tempfile
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
import numpy as np

from src.pattern_analysis.models import (
    AnalysisResult,
    FeatureMap,
    PatternDetection,
    ValidationResult,
)
from src.pattern_analysis.pipeline import (
    PipelineExecutor,
    StandardPreprocessor,
    EdgeBasedFeatureExtractor,
    HybridPatternClassifier,
    MultiMethodCrossValidator,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def pipeline_executor():
    """Create a pipeline executor with all stages."""
    preprocessor = StandardPreprocessor()
    feature_extractor = EdgeBasedFeatureExtractor()
    classifier = HybridPatternClassifier()
    cross_validator = MultiMethodCrossValidator()
    
    return PipelineExecutor(
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        classifier=classifier,
        cross_validator=cross_validator
    )


@pytest.fixture
def pipeline_executor_no_validator():
    """Create a pipeline executor without cross-validator."""
    preprocessor = StandardPreprocessor()
    feature_extractor = EdgeBasedFeatureExtractor()
    classifier = HybridPatternClassifier()
    
    return PipelineExecutor(
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        classifier=classifier,
        cross_validator=None
    )


@pytest.fixture
def sample_image_path():
    """Get path to a sample chart image for testing."""
    # Use an existing chart image from the data directory
    image_path = "data/charts/AAPL_20251224_164830.png"
    if os.path.exists(image_path):
        return image_path
    
    # Fallback: create a temporary test image
    return create_temp_test_image()


def create_temp_test_image():
    """Create a temporary test image for testing."""
    import cv2
    
    # Create a simple chart-like image
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Add some green and red rectangles to simulate candlesticks
    for i in range(10):
        x = 100 + i * 100
        color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
        cv2.rectangle(img, (x, 200), (x + 20, 400), color, -1)
    
    # Add some lines to simulate trendlines
    cv2.line(img, (100, 400), (900, 200), (255, 255, 255), 2)
    cv2.line(img, (100, 200), (900, 400), (255, 255, 255), 2)
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    cv2.imwrite(temp_file.name, img)
    return temp_file.name


# =============================================================================
# Custom Strategies for Property-Based Testing
# =============================================================================

@st.composite
def image_strategy(draw, min_dim=200, max_dim=500):
    """Generate random valid images."""
    width = draw(st.integers(min_value=min_dim, max_value=max_dim))
    height = draw(st.integers(min_value=min_dim, max_value=max_dim))
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


@st.composite
def config_strategy(draw):
    """Generate valid configuration dictionaries."""
    config = {}
    
    if draw(st.booleans()):
        config["target_size"] = (
            draw(st.integers(min_value=640, max_value=1920)),
            draw(st.integers(min_value=480, max_value=1080))
        )
    
    if draw(st.booleans()):
        config["denoise"] = draw(st.booleans())
    
    if draw(st.booleans()):
        config["confidence_threshold"] = draw(
            st.floats(min_value=0.1, max_value=0.9, allow_nan=False)
        )
    
    return config


# =============================================================================
# Property Tests for Pipeline Result Aggregation (Property 7)
# =============================================================================

class TestPipelineResultAggregation:
    """
    Property tests for pipeline result aggregation.
    
    Feature: chart-pattern-analysis-framework
    Property 7: Pipeline Result Aggregation
    Validates: Requirements 4.6
    """
    
    def test_result_contains_all_timing_metrics(self, pipeline_executor, sample_image_path):
        """
        Feature: chart-pattern-analysis-framework
        Property 7: Pipeline Result Aggregation
        Validates: Requirements 4.6
        
        For any successful pipeline execution, the AnalysisResult SHALL contain
        all timing metrics (preprocessing_time_ms, extraction_time_ms,
        classification_time_ms, validation_time_ms, total_time_ms).
        """
        result = pipeline_executor.execute(sample_image_path, {})
        
        # All timing metrics must be present and non-negative
        assert hasattr(result, 'preprocessing_time_ms'), "Missing preprocessing_time_ms"
        assert hasattr(result, 'extraction_time_ms'), "Missing extraction_time_ms"
        assert hasattr(result, 'classification_time_ms'), "Missing classification_time_ms"
        assert hasattr(result, 'validation_time_ms'), "Missing validation_time_ms"
        assert hasattr(result, 'total_time_ms'), "Missing total_time_ms"
        
        assert result.preprocessing_time_ms >= 0, "preprocessing_time_ms must be non-negative"
        assert result.extraction_time_ms >= 0, "extraction_time_ms must be non-negative"
        assert result.classification_time_ms >= 0, "classification_time_ms must be non-negative"
        assert result.validation_time_ms >= 0, "validation_time_ms must be non-negative"
        assert result.total_time_ms >= 0, "total_time_ms must be non-negative"
    
    def test_total_time_greater_than_or_equal_to_sum_of_stages(
        self, pipeline_executor, sample_image_path
    ):
        """
        Feature: chart-pattern-analysis-framework
        Property 7: Pipeline Result Aggregation
        Validates: Requirements 4.6
        
        For any successful pipeline execution, total_time_ms SHALL be
        greater than or equal to the sum of individual stage times.
        """
        result = pipeline_executor.execute(sample_image_path, {})
        
        sum_of_stages = (
            result.preprocessing_time_ms +
            result.extraction_time_ms +
            result.classification_time_ms +
            result.validation_time_ms
        )
        
        # Total time should be >= sum of stages (may include overhead)
        assert result.total_time_ms >= sum_of_stages - 0.1, (
            f"total_time_ms ({result.total_time_ms}) should be >= "
            f"sum of stages ({sum_of_stages})"
        )
    
    def test_config_used_is_non_empty_dict(self, pipeline_executor, sample_image_path):
        """
        Feature: chart-pattern-analysis-framework
        Property 7: Pipeline Result Aggregation
        Validates: Requirements 4.6
        
        For any successful pipeline execution, config_used SHALL be
        a non-null dictionary.
        """
        config = {"test_key": "test_value"}
        result = pipeline_executor.execute(sample_image_path, config)
        
        assert result.config_used is not None, "config_used must not be None"
        assert isinstance(result.config_used, dict), "config_used must be a dict"
    
    @given(config_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_config_passed_through_to_result(self, generated_config):
        """
        Feature: chart-pattern-analysis-framework
        Property 7: Pipeline Result Aggregation
        Validates: Requirements 4.6
        
        For any configuration passed to execute(), the same configuration
        SHALL be present in config_used of the result.
        """
        # Create pipeline executor inline to avoid fixture conflict
        preprocessor = StandardPreprocessor()
        feature_extractor = EdgeBasedFeatureExtractor()
        classifier = HybridPatternClassifier()
        cross_validator = MultiMethodCrossValidator()
        
        pipeline_executor = PipelineExecutor(
            preprocessor=preprocessor,
            feature_extractor=feature_extractor,
            classifier=classifier,
            cross_validator=cross_validator
        )
        
        # Use existing sample image
        sample_image_path = "data/charts/AAPL_20251224_164830.png"
        if not os.path.exists(sample_image_path):
            sample_image_path = create_temp_test_image()
        
        result = pipeline_executor.execute(sample_image_path, generated_config)
        
        # The config should be preserved in the result
        assert result.config_used == generated_config, (
            f"config_used should match input config. "
            f"Expected {generated_config}, got {result.config_used}"
        )
    
    def test_result_has_valid_timestamp(self, pipeline_executor, sample_image_path):
        """
        Feature: chart-pattern-analysis-framework
        Property 7: Pipeline Result Aggregation
        Validates: Requirements 4.6
        
        For any successful pipeline execution, the timestamp SHALL be
        a valid ISO format string.
        """
        from datetime import datetime
        
        result = pipeline_executor.execute(sample_image_path, {})
        
        assert result.timestamp is not None, "timestamp must not be None"
        assert isinstance(result.timestamp, str), "timestamp must be a string"
        
        # Should be parseable as ISO format
        try:
            datetime.fromisoformat(result.timestamp.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"timestamp '{result.timestamp}' is not valid ISO format")
    
    def test_result_has_correct_image_path(self, pipeline_executor, sample_image_path):
        """
        Feature: chart-pattern-analysis-framework
        Property 7: Pipeline Result Aggregation
        Validates: Requirements 4.6
        
        For any successful pipeline execution, the image_path in the result
        SHALL match the input image path.
        """
        result = pipeline_executor.execute(sample_image_path, {})
        
        assert result.image_path == sample_image_path, (
            f"image_path should match input. "
            f"Expected {sample_image_path}, got {result.image_path}"
        )
    
    def test_result_has_feature_map(self, pipeline_executor, sample_image_path):
        """
        Feature: chart-pattern-analysis-framework
        Property 7: Pipeline Result Aggregation
        Validates: Requirements 4.6
        
        For any successful pipeline execution, the result SHALL contain
        a valid FeatureMap.
        """
        result = pipeline_executor.execute(sample_image_path, {})
        
        assert result.feature_map is not None, "feature_map must not be None"
        assert isinstance(result.feature_map, FeatureMap), (
            f"feature_map must be FeatureMap, got {type(result.feature_map)}"
        )
    
    def test_result_detections_is_list(self, pipeline_executor, sample_image_path):
        """
        Feature: chart-pattern-analysis-framework
        Property 7: Pipeline Result Aggregation
        Validates: Requirements 4.6
        
        For any successful pipeline execution, detections SHALL be a list.
        """
        result = pipeline_executor.execute(sample_image_path, {})
        
        assert result.detections is not None, "detections must not be None"
        assert isinstance(result.detections, list), (
            f"detections must be a list, got {type(result.detections)}"
        )
    
    def test_result_validated_detections_is_list(self, pipeline_executor, sample_image_path):
        """
        Feature: chart-pattern-analysis-framework
        Property 7: Pipeline Result Aggregation
        Validates: Requirements 4.6
        
        For any successful pipeline execution, validated_detections SHALL be a list.
        """
        result = pipeline_executor.execute(sample_image_path, {})
        
        assert result.validated_detections is not None, "validated_detections must not be None"
        assert isinstance(result.validated_detections, list), (
            f"validated_detections must be a list, got {type(result.validated_detections)}"
        )


# =============================================================================
# Additional Property Tests for Graceful Degradation
# =============================================================================

class TestPipelineGracefulDegradation:
    """
    Tests for graceful degradation when stages fail.
    
    Feature: chart-pattern-analysis-framework
    Validates: Requirements 4.4
    """
    
    def test_nonexistent_image_returns_result_with_empty_detections(
        self, pipeline_executor
    ):
        """
        When preprocessing fails due to nonexistent image,
        the pipeline SHALL return a result with empty detections.
        """
        result = pipeline_executor.execute("nonexistent_image.png", {})
        
        # Should still return a valid AnalysisResult
        assert isinstance(result, AnalysisResult)
        assert result.detections == []
        assert result.image_path == "nonexistent_image.png"
    
    def test_pipeline_without_validator_skips_validation(
        self, pipeline_executor_no_validator, sample_image_path
    ):
        """
        When cross_validator is None, validation_time_ms SHALL be 0
        and validated_detections SHALL be empty.
        """
        result = pipeline_executor_no_validator.execute(sample_image_path, {})
        
        assert result.validation_time_ms == 0.0, (
            "validation_time_ms should be 0 when no validator"
        )
        assert result.validated_detections == [], (
            "validated_detections should be empty when no validator"
        )


# =============================================================================
# Property Tests for Stage Information
# =============================================================================

class TestPipelineStageInfo:
    """
    Tests for pipeline stage information retrieval.
    """
    
    def test_stage_ids_returns_all_stages(self, pipeline_executor):
        """
        stage_ids property SHALL return IDs for all configured stages.
        """
        stage_ids = pipeline_executor.stage_ids
        
        assert len(stage_ids) == 4, f"Expected 4 stages, got {len(stage_ids)}"
        assert "standard_preprocessor_v1" in stage_ids
        assert "edge_feature_extractor_v1" in stage_ids
        assert "hybrid_classifier_v1" in stage_ids
        assert "multi_method_validator_v1" in stage_ids
    
    def test_stage_ids_without_validator(self, pipeline_executor_no_validator):
        """
        stage_ids property SHALL return IDs for configured stages only.
        """
        stage_ids = pipeline_executor_no_validator.stage_ids
        
        assert len(stage_ids) == 3, f"Expected 3 stages, got {len(stage_ids)}"
        assert "multi_method_validator_v1" not in stage_ids
    
    def test_get_stage_info_returns_correct_structure(self, pipeline_executor):
        """
        get_stage_info SHALL return list of dicts with stage_id and type.
        """
        stage_info = pipeline_executor.get_stage_info()
        
        assert isinstance(stage_info, list)
        assert len(stage_info) == 4
        
        for info in stage_info:
            assert "stage_id" in info
            assert "type" in info
            assert isinstance(info["stage_id"], str)
            assert isinstance(info["type"], str)

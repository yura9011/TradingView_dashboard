"""
Integration tests for Chart Pattern Analysis Framework.

Tests end-to-end functionality including:
- Pipeline execution with real images
- Output format verification (JSON, Markdown, annotated images)
- Factory function integration
- CLI interface

Requirements: 7.5 - Generate both machine-readable (JSON) and human-readable (Markdown) formats
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import numpy as np

# Import the pattern analysis framework
from src.pattern_analysis import (
    create_pipeline,
    create_analyzer,
    create_pattern_adapter,
    ChartPatternAnalyzer,
    PatternAnalysisAdapter,
    AnalysisResult,
    PatternDetection,
    BoundingBox,
    PatternCategory,
    PatternType,
)
from src.pattern_analysis.factory import (
    create_preprocessor,
    create_feature_extractor,
    create_classifier,
    create_annotator,
)
from src.pattern_analysis.cli import main as cli_main, create_parser


# Test data paths
TEST_DATA_DIR = Path("data/charts")
SAMPLE_IMAGES = list(TEST_DATA_DIR.glob("*.png"))[:3] if TEST_DATA_DIR.exists() else []


def get_test_image() -> Optional[str]:
    """Get a test image path if available."""
    if SAMPLE_IMAGES:
        return str(SAMPLE_IMAGES[0])
    return None


def create_synthetic_image(width: int = 800, height: int = 600) -> np.ndarray:
    """Create a synthetic test image with chart-like features."""
    # Create a white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some green and red rectangles (simulating candlesticks)
    import cv2
    for i in range(10):
        x = 50 + i * 70
        if i % 2 == 0:
            # Green candle
            cv2.rectangle(image, (x, 200), (x + 20, 350), (0, 200, 0), -1)
        else:
            # Red candle
            cv2.rectangle(image, (x, 250), (x + 20, 400), (0, 0, 200), -1)
    
    # Add some diagonal lines (simulating trendlines)
    cv2.line(image, (50, 400), (700, 200), (100, 100, 100), 2)
    cv2.line(image, (50, 150), (700, 350), (100, 100, 100), 2)
    
    return image


class TestFactoryFunctions:
    """Test factory function creation."""
    
    def test_create_preprocessor(self):
        """Test preprocessor creation."""
        preprocessor = create_preprocessor()
        assert preprocessor is not None
        assert preprocessor.stage_id == "standard_preprocessor_v1"
    
    def test_create_feature_extractor(self):
        """Test feature extractor creation."""
        extractor = create_feature_extractor()
        assert extractor is not None
        assert extractor.stage_id == "edge_feature_extractor_v1"
    
    def test_create_classifier(self):
        """Test classifier creation."""
        classifier = create_classifier()
        assert classifier is not None
        assert classifier.stage_id == "hybrid_classifier_v1"
    
    def test_create_annotator(self):
        """Test annotator creation."""
        annotator = create_annotator()
        assert annotator is not None
        assert annotator.overlay_alpha == 0.3
    
    def test_create_pipeline(self):
        """Test pipeline creation with default settings."""
        pipeline = create_pipeline(enable_cross_validation=False)
        assert pipeline is not None
        assert len(pipeline.stage_ids) >= 3  # preprocessor, extractor, classifier
    
    def test_create_pipeline_with_validation(self):
        """Test pipeline creation with cross-validation enabled."""
        pipeline = create_pipeline(enable_cross_validation=True)
        assert pipeline is not None
        assert len(pipeline.stage_ids) >= 4  # includes validator
    
    def test_create_analyzer(self):
        """Test analyzer creation."""
        analyzer = create_analyzer(enable_cross_validation=False)
        assert analyzer is not None
        assert isinstance(analyzer, ChartPatternAnalyzer)


class TestPipelineExecution:
    """Test pipeline execution with images."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        return create_pipeline(enable_cross_validation=False)
    
    @pytest.fixture
    def synthetic_image_path(self):
        """Create a synthetic test image and return its path."""
        import cv2
        image = create_synthetic_image()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, image)
            yield f.name
        
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_pipeline_executes_without_error(self, pipeline, synthetic_image_path):
        """Test that pipeline executes without raising exceptions."""
        result = pipeline.execute(synthetic_image_path)
        assert result is not None
        assert isinstance(result, AnalysisResult)
    
    def test_pipeline_returns_valid_result(self, pipeline, synthetic_image_path):
        """Test that pipeline returns a valid AnalysisResult."""
        result = pipeline.execute(synthetic_image_path)
        
        # Check required fields
        assert result.image_path == synthetic_image_path
        assert result.timestamp is not None
        assert result.total_time_ms >= 0
        assert result.preprocessing_time_ms >= 0
        assert result.extraction_time_ms >= 0
        assert result.classification_time_ms >= 0
        assert isinstance(result.detections, list)
        assert result.feature_map is not None
    
    def test_pipeline_timing_consistency(self, pipeline, synthetic_image_path):
        """Test that total time >= sum of stage times."""
        result = pipeline.execute(synthetic_image_path)
        
        stage_sum = (
            result.preprocessing_time_ms +
            result.extraction_time_ms +
            result.classification_time_ms +
            result.validation_time_ms
        )
        
        # Allow small tolerance for timing variations
        assert result.total_time_ms >= stage_sum * 0.9
    
    @pytest.mark.skipif(not SAMPLE_IMAGES, reason="No test images available")
    def test_pipeline_with_real_image(self, pipeline):
        """Test pipeline with a real chart image."""
        image_path = str(SAMPLE_IMAGES[0])
        result = pipeline.execute(image_path)
        
        assert result is not None
        assert result.image_path == image_path


class TestOutputFormats:
    """Test output format generation."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an analyzer for testing."""
        return create_analyzer(enable_cross_validation=False)
    
    @pytest.fixture
    def synthetic_image_path(self):
        """Create a synthetic test image."""
        import cv2
        image = create_synthetic_image()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, image)
            yield f.name
        
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_json_output_is_valid(self, analyzer, synthetic_image_path):
        """Test that JSON output is valid JSON."""
        result = analyzer.analyze(synthetic_image_path)
        json_str = analyzer.to_json(result, validate=False)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "image_path" in parsed
        assert "timestamp" in parsed
        assert "detections" in parsed
    
    def test_json_output_contains_required_fields(self, analyzer, synthetic_image_path):
        """Test that JSON output contains all required fields."""
        result = analyzer.analyze(synthetic_image_path)
        json_str = analyzer.to_json(result, validate=False)
        parsed = json.loads(json_str)
        
        required_fields = [
            "image_path",
            "timestamp",
            "total_time_ms",
            "preprocessing_time_ms",
            "extraction_time_ms",
            "classification_time_ms",
            "validation_time_ms",
            "detections",
        ]
        
        for field in required_fields:
            assert field in parsed, f"Missing required field: {field}"
    
    def test_markdown_output_is_valid(self, analyzer, synthetic_image_path):
        """Test that Markdown output is generated correctly."""
        result = analyzer.analyze(synthetic_image_path)
        md_str = analyzer.to_markdown(result)
        
        assert isinstance(md_str, str)
        assert "# Pattern Analysis Report" in md_str
        assert "**Image:**" in md_str
        assert "## Timing Breakdown" in md_str
    
    def test_annotated_image_saved(self, analyzer, synthetic_image_path):
        """Test that annotated image can be saved."""
        result = analyzer.analyze(synthetic_image_path)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name
        
        try:
            success = analyzer.save_annotated(
                synthetic_image_path,
                result,
                output_path
            )
            
            assert success
            assert os.path.exists(output_path)
            
            # Check file is a valid image
            import cv2
            img = cv2.imread(output_path)
            assert img is not None
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestIntegrationAdapter:
    """Test the integration adapter for backward compatibility."""
    
    @pytest.fixture
    def adapter(self):
        """Create an adapter for testing."""
        return create_pattern_adapter(enable_cross_validation=False)
    
    @pytest.fixture
    def synthetic_image_path(self):
        """Create a synthetic test image."""
        import cv2
        image = create_synthetic_image()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, image)
            yield f.name
        
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_adapter_returns_legacy_result(self, adapter, synthetic_image_path):
        """Test that adapter returns LegacyAnalysisResult."""
        from src.pattern_analysis.integration import LegacyAnalysisResult
        
        result = adapter.analyze_chart(synthetic_image_path, "TEST", "1D")
        
        assert isinstance(result, LegacyAnalysisResult)
    
    def test_adapter_result_has_required_fields(self, adapter, synthetic_image_path):
        """Test that adapter result has all required fields."""
        result = adapter.analyze_chart(synthetic_image_path, "TEST", "1D")
        
        # Check required fields exist
        assert hasattr(result, "pattern_detected")
        assert hasattr(result, "pattern_confidence")
        assert hasattr(result, "trend")
        assert hasattr(result, "support_level")
        assert hasattr(result, "resistance_level")
        assert hasattr(result, "analysis_summary")
        assert hasattr(result, "raw_response")
    
    def test_adapter_confidence_in_range(self, adapter, synthetic_image_path):
        """Test that confidence is in valid range."""
        result = adapter.analyze_chart(synthetic_image_path, "TEST", "1D")
        
        assert 0.0 <= result.pattern_confidence <= 1.0
    
    def test_adapter_trend_is_valid(self, adapter, synthetic_image_path):
        """Test that trend is a valid value."""
        result = adapter.analyze_chart(synthetic_image_path, "TEST", "1D")
        
        assert result.trend in ["up", "down", "sideways"]


class TestCLI:
    """Test CLI interface."""
    
    @pytest.fixture
    def synthetic_image_path(self):
        """Create a synthetic test image."""
        import cv2
        image = create_synthetic_image()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, image)
            yield f.name
        
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_cli_parser_creation(self):
        """Test that CLI parser is created correctly."""
        parser = create_parser()
        assert parser is not None
    
    def test_cli_help_does_not_error(self):
        """Test that --help doesn't raise an error."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--help"])
        assert exc_info.value.code == 0
    
    def test_cli_analyze_json_output(self, synthetic_image_path):
        """Test CLI analyze command with JSON output."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        
        try:
            exit_code = cli_main([
                "analyze",
                synthetic_image_path,
                "--format", "json",
                "--output", output_path,
                "--no-validation",
                "--quiet"
            ])
            
            assert exit_code == 0
            assert os.path.exists(output_path)
            
            # Verify JSON is valid
            with open(output_path) as f:
                data = json.load(f)
            assert "image_path" in data
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_cli_analyze_markdown_output(self, synthetic_image_path):
        """Test CLI analyze command with Markdown output."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            output_path = f.name
        
        try:
            exit_code = cli_main([
                "analyze",
                synthetic_image_path,
                "--format", "markdown",
                "--output", output_path,
                "--no-validation",
                "--quiet"
            ])
            
            assert exit_code == 0
            assert os.path.exists(output_path)
            
            # Verify Markdown content
            with open(output_path) as f:
                content = f.read()
            assert "# Pattern Analysis Report" in content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_cli_analyze_annotated_output(self, synthetic_image_path):
        """Test CLI analyze command with annotated image output."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name
        
        try:
            exit_code = cli_main([
                "analyze",
                synthetic_image_path,
                "--format", "annotated",
                "--output", output_path,
                "--no-validation",
                "--quiet"
            ])
            
            assert exit_code == 0
            assert os.path.exists(output_path)
            
            # Verify image is valid
            import cv2
            img = cv2.imread(output_path)
            assert img is not None
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_cli_nonexistent_file_returns_error(self):
        """Test that CLI returns error for nonexistent file."""
        exit_code = cli_main([
            "analyze",
            "nonexistent_file.png",
            "--format", "json",
            "--quiet"
        ])
        
        assert exit_code != 0


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.skipif(not SAMPLE_IMAGES, reason="No test images available")
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow with real image."""
        image_path = str(SAMPLE_IMAGES[0])
        
        # Create analyzer
        analyzer = create_analyzer(enable_cross_validation=False)
        
        # Run analysis
        result = analyzer.analyze(image_path)
        
        # Verify result
        assert result is not None
        assert result.image_path == image_path
        assert result.total_time_ms > 0
        
        # Generate outputs
        json_output = analyzer.to_json(result, validate=False)
        md_output = analyzer.to_markdown(result)
        
        assert len(json_output) > 0
        assert len(md_output) > 0
        
        # Verify JSON is parseable
        parsed = json.loads(json_output)
        assert "detections" in parsed
    
    @pytest.mark.skipif(not SAMPLE_IMAGES, reason="No test images available")
    def test_adapter_with_real_image(self):
        """Test adapter with real chart image."""
        image_path = str(SAMPLE_IMAGES[0])
        
        adapter = create_pattern_adapter(enable_cross_validation=False)
        result = adapter.analyze_chart(image_path, "TEST", "1D")
        
        assert result is not None
        assert result.trend in ["up", "down", "sideways"]
        assert 0.0 <= result.pattern_confidence <= 1.0

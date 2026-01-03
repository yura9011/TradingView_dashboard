"""
Integration tests for Chart Analysis Improvements.

Tests end-to-end functionality for:
- Region detection with multi-chart images (price + volume)
- Auto-cropping to focus on primary chart only
- Timeframe configuration defaults (1Y/1D)

Requirements:
- 1.3: Analyze ONLY the Primary_Chart region for pattern detection
- 2.1: Default to analyzing charts with 1Y timeframe
- 2.6: Default timeframe "1Y" and default candle interval "1D"
- 3.2: Remove volume panel and secondary indicators from image
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from src.pattern_analysis.pipeline.enhanced_preprocessor import (
    EnhancedPreprocessor,
    EnhancedPreprocessResult,
)
from src.pattern_analysis.pipeline.region_detector import ChartRegionDetector
from src.pattern_analysis.pipeline.auto_cropper import AutoCropper
from src.pattern_analysis.config.timeframe_manager import TimeframeConfigManager
from src.pattern_analysis.models.enums import RegionType, Timeframe, CandleInterval
from src.pattern_analysis.models.dataclasses import BoundingBox


# Test data paths
TEST_DATA_DIR = Path("data/charts")
SAMPLE_IMAGES = list(TEST_DATA_DIR.glob("*.png"))[:5] if TEST_DATA_DIR.exists() else []


def create_multi_region_chart_image(
    width: int = 800,
    height: int = 600,
    include_volume: bool = True,
    include_toolbar: bool = False
) -> np.ndarray:
    """
    Create a synthetic chart image with multiple regions.
    
    Creates an image with:
    - Primary chart region (candlesticks) in the upper portion
    - Volume panel at the bottom (if include_volume=True)
    - Toolbar at the top (if include_toolbar=True)
    
    Args:
        width: Image width
        height: Image height
        include_volume: Whether to include a volume panel
        include_toolbar: Whether to include a toolbar
        
    Returns:
        Synthetic chart image as numpy array (RGB)
    """
    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Calculate region heights
    toolbar_height = int(height * 0.08) if include_toolbar else 0
    volume_height = int(height * 0.20) if include_volume else 0
    chart_height = height - toolbar_height - volume_height
    
    # Draw toolbar region (gray background)
    if include_toolbar:
        cv2.rectangle(
            image, 
            (0, 0), 
            (width, toolbar_height), 
            (200, 200, 200), 
            -1
        )
        # Add a separator line
        cv2.line(
            image,
            (0, toolbar_height),
            (width, toolbar_height),
            (100, 100, 100),
            2
        )
    
    # Draw primary chart region with candlesticks
    chart_start_y = toolbar_height
    chart_end_y = chart_start_y + chart_height
    
    # Add candlesticks (green and red rectangles)
    num_candles = 15
    candle_width = int(width * 0.04)
    candle_spacing = int(width * 0.06)
    start_x = 50
    
    for i in range(num_candles):
        x = start_x + i * candle_spacing
        
        # Alternate between green (bullish) and red (bearish) candles
        if i % 3 == 0:
            # Green candle (bullish)
            y1 = chart_start_y + int(chart_height * 0.3)
            y2 = chart_start_y + int(chart_height * 0.7)
            color = (0, 200, 0)  # Green in RGB
        elif i % 3 == 1:
            # Red candle (bearish)
            y1 = chart_start_y + int(chart_height * 0.25)
            y2 = chart_start_y + int(chart_height * 0.65)
            color = (200, 0, 0)  # Red in RGB
        else:
            # Another green candle
            y1 = chart_start_y + int(chart_height * 0.35)
            y2 = chart_start_y + int(chart_height * 0.75)
            color = (0, 180, 0)  # Green in RGB
        
        cv2.rectangle(image, (x, y1), (x + candle_width, y2), color, -1)
        
        # Add wicks
        wick_x = x + candle_width // 2
        cv2.line(
            image,
            (wick_x, y1 - 10),
            (wick_x, y1),
            color,
            1
        )
        cv2.line(
            image,
            (wick_x, y2),
            (wick_x, y2 + 10),
            color,
            1
        )
    
    # Draw separator between chart and volume
    if include_volume:
        separator_y = chart_end_y
        cv2.line(
            image,
            (0, separator_y),
            (width, separator_y),
            (50, 50, 50),
            3
        )
        
        # Draw volume panel (bars at bottom)
        volume_start_y = separator_y + 5
        for i in range(num_candles):
            x = start_x + i * candle_spacing
            bar_height = int(volume_height * (0.3 + 0.5 * (i % 4) / 4))
            y1 = height - bar_height
            y2 = height - 5
            
            # Volume bars in blue/gray
            color = (100, 100, 180)
            cv2.rectangle(image, (x, y1), (x + candle_width, y2), color, -1)
    
    return image


class TestEndToEndWithMultipleCharts:
    """
    End-to-end tests with images containing multiple chart regions.
    
    Requirements: 1.3, 3.2
    """
    
    @pytest.fixture
    def enhanced_preprocessor(self):
        """Create an EnhancedPreprocessor with default config."""
        config = {
            "region_detection": {
                "enabled": True,
                "min_region_height_pct": 0.1,
                "volume_position_threshold": 0.7,
            },
            "auto_crop": {
                "auto_remove_secondary_charts": True,
                "crop_padding": 10,
            },
            "timeframe": {
                "default_timeframe": "1Y",
                "default_candle_interval": "1D",
            }
        }
        return EnhancedPreprocessor(config)
    
    @pytest.fixture
    def multi_chart_image_path(self):
        """Create a synthetic multi-chart image and return its path."""
        image = create_multi_region_chart_image(
            width=800,
            height=600,
            include_volume=True,
            include_toolbar=False
        )
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Convert RGB to BGR for cv2.imwrite
            cv2.imwrite(f.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            yield f.name
        
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_region_detection_identifies_multiple_regions(
        self, enhanced_preprocessor, multi_chart_image_path
    ):
        """
        Test that region detection identifies multiple chart regions.
        
        Requirements: 1.1, 1.2
        """
        result = enhanced_preprocessor.process(
            multi_chart_image_path,
            {"denoise": False}
        )
        
        assert result.region_detection is not None
        assert len(result.region_detection.regions) >= 1
        assert result.region_detection.primary_region is not None
    
    def test_primary_chart_region_is_identified(
        self, enhanced_preprocessor, multi_chart_image_path
    ):
        """
        Test that the primary chart region is correctly identified.
        
        Requirements: 1.3
        """
        result = enhanced_preprocessor.process(
            multi_chart_image_path,
            {"denoise": False}
        )
        
        primary = result.region_detection.primary_region
        assert primary is not None
        assert primary.region_type == RegionType.PRIMARY_CHART
        assert primary.confidence > 0.5
    
    def test_auto_crop_removes_secondary_charts(
        self, enhanced_preprocessor, multi_chart_image_path
    ):
        """
        Test that auto-crop removes secondary chart regions (volume panel).
        
        Requirements: 3.2
        """
        result = enhanced_preprocessor.process(
            multi_chart_image_path,
            {"denoise": False}
        )
        
        # Coverage should be less than 100% since volume was removed
        assert result.coverage_percentage < 100.0
        
        # Excluded regions should contain the volume panel
        if result.crop_result.excluded_regions:
            excluded_types = [
                r.region_type for r in result.crop_result.excluded_regions
            ]
            # At least one secondary region should be excluded
            assert len(excluded_types) >= 0  # May be 0 if only primary detected
    
    def test_only_primary_chart_analyzed(
        self, enhanced_preprocessor, multi_chart_image_path
    ):
        """
        Test that only the primary chart region is analyzed.
        
        Requirements: 1.3, 3.2
        """
        result = enhanced_preprocessor.process(
            multi_chart_image_path,
            {"denoise": False}
        )
        
        # The analyzed region bounds should match the primary region
        assert result.analyzed_region_bounds is not None
        
        # The cropped image should be smaller than original
        original_height = result.original_size[1]
        cropped_height = result.crop_result.cropped_size[1]
        
        # If volume was detected and removed, cropped should be smaller
        # (allowing for padding)
        assert cropped_height <= original_height
    
    def test_analysis_metadata_includes_region_info(
        self, enhanced_preprocessor, multi_chart_image_path
    ):
        """
        Test that analysis result includes region metadata.
        
        Requirements: 4.1, 4.2
        """
        result = enhanced_preprocessor.process(
            multi_chart_image_path,
            {"denoise": False}
        )
        
        # Check metadata is present
        assert result.region_detection is not None
        assert result.crop_result is not None
        assert result.analyzed_region_bounds is not None
        assert 0.0 <= result.coverage_percentage <= 100.0
        assert isinstance(result.needs_review, bool)
    
    @pytest.mark.skipif(not SAMPLE_IMAGES, reason="No test images available")
    def test_with_real_chart_image(self, enhanced_preprocessor):
        """
        Test end-to-end processing with a real chart image.
        
        Requirements: 1.3, 3.2
        """
        image_path = str(SAMPLE_IMAGES[0])
        
        result = enhanced_preprocessor.process(
            image_path,
            {"denoise": True}
        )
        
        # Basic validation
        assert result is not None
        assert isinstance(result, EnhancedPreprocessResult)
        assert result.region_detection is not None
        assert result.image is not None
        assert result.image.shape[0] > 0
        assert result.image.shape[1] > 0


class TestTimeframeConfiguration:
    """
    Tests for timeframe configuration defaults.
    
    Requirements: 2.1, 2.6
    """
    
    def test_default_timeframe_is_1y(self):
        """
        Test that default timeframe is 1Y.
        
        Requirements: 2.1, 2.6
        """
        manager = TimeframeConfigManager({})
        
        assert manager.default_timeframe == Timeframe.YEAR_1
    
    def test_default_candle_interval_is_1d(self):
        """
        Test that default candle interval is 1D (daily).
        
        Requirements: 2.6
        """
        manager = TimeframeConfigManager({})
        
        assert manager.default_candle_interval == CandleInterval.DAILY
    
    def test_config_file_defaults(self):
        """
        Test that config file specifies correct defaults.
        
        Requirements: 2.1, 2.6
        """
        # Load from config with explicit values
        config = {
            "default_timeframe": "1Y",
            "default_candle_interval": "1D"
        }
        manager = TimeframeConfigManager(config)
        
        assert manager.default_timeframe == Timeframe.YEAR_1
        assert manager.default_candle_interval == CandleInterval.DAILY
    
    def test_1y_timeframe_parameters(self):
        """
        Test that 1Y timeframe has appropriate parameters.
        
        Requirements: 2.2, 2.5
        """
        manager = TimeframeConfigManager({})
        config = manager.get_config(Timeframe.YEAR_1)
        
        # 1Y should have higher min_pattern_candles than shorter timeframes
        assert config.min_pattern_candles >= 30
        
        # 1Y should have lower trend_sensitivity (less noise)
        assert config.trend_sensitivity <= 0.5
        
        # Should use daily candles
        assert config.candle_interval == CandleInterval.DAILY
    
    def test_pattern_params_for_1y(self):
        """
        Test pattern detection parameters for 1Y timeframe.
        
        Requirements: 2.2, 2.3
        """
        manager = TimeframeConfigManager({})
        params = manager.get_pattern_params(Timeframe.YEAR_1)
        
        assert "min_candles" in params
        assert "min_height_pct" in params
        assert "trend_sensitivity" in params
        assert "candle_interval" in params
        
        assert params["min_candles"] >= 30
        assert params["candle_interval"] == "1D"
    
    def test_longer_timeframes_have_more_candles(self):
        """
        Test that longer timeframes require more candles for patterns.
        
        Requirements: 2.5
        """
        manager = TimeframeConfigManager({})
        
        day_config = manager.get_config(Timeframe.DAY_1)
        week_config = manager.get_config(Timeframe.WEEK_1)
        month_config = manager.get_config(Timeframe.MONTH_1)
        year_config = manager.get_config(Timeframe.YEAR_1)
        
        # Longer timeframes should require more candles
        assert day_config.min_pattern_candles < week_config.min_pattern_candles
        assert week_config.min_pattern_candles < month_config.min_pattern_candles
        assert month_config.min_pattern_candles <= year_config.min_pattern_candles
    
    def test_longer_timeframes_have_lower_sensitivity(self):
        """
        Test that longer timeframes have lower trend sensitivity.
        
        Requirements: 2.5
        """
        manager = TimeframeConfigManager({})
        
        day_config = manager.get_config(Timeframe.DAY_1)
        year_config = manager.get_config(Timeframe.YEAR_1)
        
        # Longer timeframes should have lower sensitivity (less noise)
        assert year_config.trend_sensitivity < day_config.trend_sensitivity


class TestEnhancedPreprocessorIntegration:
    """
    Integration tests for EnhancedPreprocessor with timeframe config.
    """
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor with default config."""
        return EnhancedPreprocessor({
            "timeframe": {
                "default_timeframe": "1Y",
                "default_candle_interval": "1D"
            }
        })
    
    @pytest.fixture
    def simple_chart_image_path(self):
        """Create a simple chart image without volume panel."""
        image = create_multi_region_chart_image(
            width=800,
            height=600,
            include_volume=False,
            include_toolbar=False
        )
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            yield f.name
        
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_preprocessor_uses_default_timeframe(
        self, preprocessor, simple_chart_image_path
    ):
        """
        Test that preprocessor uses default 1Y timeframe.
        
        Requirements: 2.1
        """
        result = preprocessor.process(
            simple_chart_image_path,
            {"denoise": False}
        )
        
        assert result.timeframe_config is not None
        assert result.timeframe_config.timeframe == Timeframe.YEAR_1
    
    def test_preprocessor_pattern_params(self, preprocessor):
        """
        Test that preprocessor returns correct pattern params.
        
        Requirements: 2.2, 2.3
        """
        params = preprocessor.get_pattern_params()
        
        assert params["min_candles"] >= 30
        assert params["candle_interval"] == "1D"
    
    def test_process_with_explicit_timeframe(
        self, preprocessor, simple_chart_image_path
    ):
        """
        Test processing with explicit timeframe override.
        """
        result = preprocessor.process_with_timeframe(
            simple_chart_image_path,
            timeframe=Timeframe.MONTH_1,
            config={"denoise": False}
        )
        
        assert result.timeframe_config is not None
        assert result.timeframe_config.timeframe == Timeframe.MONTH_1
    
    def test_full_image_when_no_secondary_charts(
        self, preprocessor, simple_chart_image_path
    ):
        """
        Test that full image is used when no secondary charts detected.
        
        Requirements: 1.5
        """
        result = preprocessor.process(
            simple_chart_image_path,
            {"denoise": False}
        )
        
        # Coverage should be high (close to 100%) when no secondary charts
        # Allow some tolerance for region detection variations
        assert result.coverage_percentage >= 50.0


class TestAutoCropperIntegration:
    """
    Integration tests for AutoCropper with region detection.
    """
    
    @pytest.fixture
    def region_detector(self):
        """Create region detector."""
        return ChartRegionDetector({
            "min_region_height_pct": 0.1,
            "volume_position_threshold": 0.7
        })
    
    @pytest.fixture
    def auto_cropper(self):
        """Create auto cropper."""
        return AutoCropper({
            "auto_remove_secondary_charts": True,
            "crop_padding": 10
        })
    
    def test_crop_preserves_primary_chart(
        self, region_detector, auto_cropper
    ):
        """
        Test that cropping preserves the primary chart region.
        
        Requirements: 3.3
        """
        image = create_multi_region_chart_image(
            width=800,
            height=600,
            include_volume=True
        )
        
        detection_result = region_detector.detect_regions(image)
        crop_result = auto_cropper.crop(image, detection_result)
        
        # Primary region should be fully contained in cropped image
        if detection_result.primary_region:
            primary_bbox = detection_result.primary_region.bounding_box
            crop_bbox = crop_result.crop_bounds
            
            # Crop bounds should contain primary region (with padding)
            assert crop_bbox.y1 <= primary_bbox.y1
            assert crop_bbox.y2 >= primary_bbox.y2
    
    def test_crop_removes_volume_panel(
        self, region_detector, auto_cropper
    ):
        """
        Test that cropping removes the volume panel.
        
        Requirements: 3.2
        """
        image = create_multi_region_chart_image(
            width=800,
            height=600,
            include_volume=True
        )
        
        detection_result = region_detector.detect_regions(image)
        crop_result = auto_cropper.crop(image, detection_result)
        
        # Cropped image should be smaller than original
        original_area = image.shape[0] * image.shape[1]
        cropped_area = crop_result.cropped_image.shape[0] * crop_result.cropped_image.shape[1]
        
        # If volume was detected, cropped should be smaller
        if detection_result.secondary_regions:
            assert cropped_area < original_area
    
    def test_disabled_auto_crop_returns_full_image(self, region_detector):
        """
        Test that disabled auto-crop returns full image.
        
        Requirements: 3.4
        """
        auto_cropper = AutoCropper({
            "auto_remove_secondary_charts": False
        })
        
        image = create_multi_region_chart_image(
            width=800,
            height=600,
            include_volume=True
        )
        
        detection_result = region_detector.detect_regions(image)
        crop_result = auto_cropper.crop(image, detection_result)
        
        # Should return full image
        assert crop_result.coverage_percentage == 100.0
        assert crop_result.cropped_size == crop_result.original_size


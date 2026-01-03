"""
Property-based tests for Region Detection and Timeframe Configuration.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document for chart-analysis-improvements.

Feature: chart-analysis-improvements
Property 5: Default Configuration Values
Validates: Requirements 2.1, 2.6
"""

import os
import tempfile
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
import yaml

from src.pattern_analysis.models import (
    RegionType,
    Timeframe,
    CandleInterval,
    TimeframeConfig,
)


# =============================================================================
# Property Tests for Default Configuration Values (Property 5)
# =============================================================================

class TestDefaultConfigurationValues:
    """
    Property tests for default configuration values.
    
    Feature: chart-analysis-improvements
    Property 5: Default Configuration Values
    Validates: Requirements 2.1, 2.6
    
    For any new system instance without explicit configuration, the default 
    timeframe SHALL be "1Y" and the default candle interval SHALL be "1D".
    """
    
    CONFIG_PATH = "config/pattern_analysis.yaml"
    
    def test_default_timeframe_is_1y(self):
        """
        Feature: chart-analysis-improvements
        Property 5: Default Configuration Values
        Validates: Requirements 2.1, 2.6
        
        The default timeframe in configuration SHALL be "1Y".
        """
        with open(self.CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        timeframe_config = config.get("timeframe", {})
        default_timeframe = timeframe_config.get("default_timeframe")
        
        assert default_timeframe == "1Y", \
            f"Default timeframe should be '1Y', but got '{default_timeframe}'"
    
    def test_default_candle_interval_is_1d(self):
        """
        Feature: chart-analysis-improvements
        Property 5: Default Configuration Values
        Validates: Requirements 2.1, 2.6
        
        The default candle interval in configuration SHALL be "1D".
        """
        with open(self.CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        timeframe_config = config.get("timeframe", {})
        default_interval = timeframe_config.get("default_candle_interval")
        
        assert default_interval == "1D", \
            f"Default candle interval should be '1D', but got '{default_interval}'"
    
    def test_default_values_are_valid_enums(self):
        """
        Feature: chart-analysis-improvements
        Property 5: Default Configuration Values
        Validates: Requirements 2.1, 2.6
        
        The default timeframe and candle interval SHALL be valid enum values.
        """
        with open(self.CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        timeframe_config = config.get("timeframe", {})
        default_timeframe = timeframe_config.get("default_timeframe")
        default_interval = timeframe_config.get("default_candle_interval")
        
        # Verify they can be converted to valid enums
        timeframe_enum = Timeframe(default_timeframe)
        interval_enum = CandleInterval(default_interval)
        
        assert timeframe_enum == Timeframe.YEAR_1, \
            f"Default timeframe should map to Timeframe.YEAR_1, got {timeframe_enum}"
        assert interval_enum == CandleInterval.DAILY, \
            f"Default interval should map to CandleInterval.DAILY, got {interval_enum}"
    
    @given(
        timeframe=st.sampled_from(list(Timeframe)),
        candle_interval=st.sampled_from(list(CandleInterval)),
        min_candles=st.integers(min_value=1, max_value=100),
        height_pct=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        sensitivity=st.floats(min_value=0.01, max_value=0.99, allow_nan=False)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_timeframe_config_creation_roundtrip(
        self, 
        timeframe: Timeframe, 
        candle_interval: CandleInterval,
        min_candles: int,
        height_pct: float,
        sensitivity: float
    ):
        """
        Feature: chart-analysis-improvements
        Property 5: Default Configuration Values
        Validates: Requirements 2.1, 2.6
        
        For any valid TimeframeConfig, serializing to dict and back SHALL 
        produce an equivalent configuration.
        """
        # Create TimeframeConfig
        config = TimeframeConfig(
            timeframe=timeframe,
            candle_interval=candle_interval,
            min_pattern_candles=min_candles,
            min_pattern_height_pct=height_pct,
            trend_sensitivity=sensitivity
        )
        
        # Serialize to dict
        config_dict = config.to_dict()
        
        # Deserialize back
        restored_config = TimeframeConfig.from_dict(config_dict)
        
        # Verify equality
        assert restored_config.timeframe == config.timeframe
        assert restored_config.candle_interval == config.candle_interval
        assert restored_config.min_pattern_candles == config.min_pattern_candles
        assert abs(restored_config.min_pattern_height_pct - config.min_pattern_height_pct) < 0.0001
        assert abs(restored_config.trend_sensitivity - config.trend_sensitivity) < 0.0001
    
    def test_auto_remove_secondary_charts_default_true(self):
        """
        Feature: chart-analysis-improvements
        Property 5: Default Configuration Values
        Validates: Requirements 3.4
        
        The auto_remove_secondary_charts setting SHALL default to true.
        """
        with open(self.CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        auto_crop_config = config.get("preprocessing", {}).get("auto_crop", {})
        auto_remove = auto_crop_config.get("auto_remove_secondary_charts")
        
        assert auto_remove is True, \
            f"auto_remove_secondary_charts should be True, but got '{auto_remove}'"
    
    def test_region_detection_enabled_by_default(self):
        """
        Feature: chart-analysis-improvements
        Property 5: Default Configuration Values
        Validates: Requirements 1.1
        
        Region detection SHALL be enabled by default.
        """
        with open(self.CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        region_config = config.get("preprocessing", {}).get("region_detection", {})
        enabled = region_config.get("enabled")
        
        assert enabled is True, \
            f"region_detection.enabled should be True, but got '{enabled}'"
    
    def test_timeframe_adjustments_exist_for_1y(self):
        """
        Feature: chart-analysis-improvements
        Property 5: Default Configuration Values
        Validates: Requirements 2.5
        
        Timeframe adjustments SHALL exist for the default 1Y timeframe.
        """
        with open(self.CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        adjustments = config.get("timeframe", {}).get("adjustments", {})
        year_1_adjustments = adjustments.get("1Y", {})
        
        assert "min_pattern_candles" in year_1_adjustments, \
            "1Y adjustments should include min_pattern_candles"
        assert "min_pattern_height_pct" in year_1_adjustments, \
            "1Y adjustments should include min_pattern_height_pct"
        assert "trend_sensitivity" in year_1_adjustments, \
            "1Y adjustments should include trend_sensitivity"
        
        # Verify 1Y has longer minimum pattern thresholds than shorter intervals
        day_1_adjustments = adjustments.get("1D", {})
        if day_1_adjustments:
            assert year_1_adjustments.get("min_pattern_candles", 0) >= \
                   day_1_adjustments.get("min_pattern_candles", 0), \
                "1Y should have >= min_pattern_candles than 1D"


# =============================================================================
# Property Tests for Enum Values
# =============================================================================

class TestEnumValues:
    """
    Property tests for enum value consistency.
    
    Feature: chart-analysis-improvements
    Validates: Requirements 2.4
    """
    
    def test_region_type_values(self):
        """
        RegionType enum SHALL contain all required region types.
        """
        required_types = ["primary_chart", "volume_panel", "indicator_panel", "toolbar", "unknown"]
        actual_values = [e.value for e in RegionType]
        
        for required in required_types:
            assert required in actual_values, \
                f"RegionType should contain '{required}'"
    
    def test_timeframe_values(self):
        """
        Timeframe enum SHALL contain all required timeframes.
        """
        required_timeframes = ["1D", "1W", "1M", "3M", "6M", "YTD", "1Y", "5Y"]
        actual_values = [e.value for e in Timeframe]
        
        for required in required_timeframes:
            assert required in actual_values, \
                f"Timeframe should contain '{required}'"
    
    def test_candle_interval_values(self):
        """
        CandleInterval enum SHALL contain all required intervals.
        """
        required_intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1D", "1W", "1M"]
        actual_values = [e.value for e in CandleInterval]
        
        for required in required_intervals:
            assert required in actual_values, \
                f"CandleInterval should contain '{required}'"

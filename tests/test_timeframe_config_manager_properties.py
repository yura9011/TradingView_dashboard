"""
Property-based tests for TimeframeConfigManager.

Uses Hypothesis for property-based testing to verify correctness properties
defined in the design document.

Feature: chart-analysis-improvements
Property 4: Timeframe Parameter Adjustment
Validates: Requirements 2.2, 2.3, 2.5
"""

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from src.pattern_analysis.config.timeframe_manager import TimeframeConfigManager
from src.pattern_analysis.models.enums import Timeframe, CandleInterval
from src.pattern_analysis.models.dataclasses import TimeframeConfig


# =============================================================================
# Custom Strategies for Timeframe Testing
# =============================================================================

# Strategy for generating valid timeframes
timeframe_strategy = st.sampled_from(list(Timeframe))

# Strategy for generating pairs of timeframes for comparison
@st.composite
def timeframe_pair_strategy(draw):
    """Generate a pair of timeframes where one is shorter than the other."""
    # Define timeframe ordering (shorter to longer)
    timeframe_order = [
        Timeframe.DAY_1,
        Timeframe.WEEK_1,
        Timeframe.MONTH_1,
        Timeframe.MONTH_3,
        Timeframe.MONTH_6,
        Timeframe.YTD,
        Timeframe.YEAR_1,
        Timeframe.YEAR_5,
    ]
    
    # Pick two different indices
    idx1 = draw(st.integers(min_value=0, max_value=len(timeframe_order) - 2))
    idx2 = draw(st.integers(min_value=idx1 + 1, max_value=len(timeframe_order) - 1))
    
    return timeframe_order[idx1], timeframe_order[idx2]


# Strategy for generating valid config dictionaries
@st.composite
def config_dict_strategy(draw):
    """Generate a valid configuration dictionary for TimeframeConfigManager."""
    config = {}
    
    # Optionally include default_timeframe
    if draw(st.booleans()):
        config["default_timeframe"] = draw(st.sampled_from(["1D", "1W", "1M", "1Y", "YTD"]))
    
    # Optionally include default_candle_interval
    if draw(st.booleans()):
        config["default_candle_interval"] = draw(st.sampled_from(["1m", "5m", "1h", "1D", "1W"]))
    
    return config


# =============================================================================
# Property Tests for Timeframe Parameter Adjustment (Property 4)
# =============================================================================

class TestTimeframeParameterAdjustment:
    """
    Property tests for timeframe parameter adjustment.
    
    Feature: chart-analysis-improvements
    Property 4: Timeframe Parameter Adjustment
    Validates: Requirements 2.2, 2.3, 2.5
    """
    
    @given(timeframe_pair=timeframe_pair_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_longer_timeframe_has_more_min_candles(self, timeframe_pair):
        """
        Feature: chart-analysis-improvements
        Property 4: Timeframe Parameter Adjustment
        Validates: Requirements 2.2, 2.3, 2.5
        
        For any two timeframes where tf1 is shorter than tf2,
        the min_pattern_candles for tf2 SHALL be greater than or equal to tf1.
        """
        shorter_tf, longer_tf = timeframe_pair
        
        mgr = TimeframeConfigManager()
        
        shorter_config = mgr.get_config(shorter_tf)
        longer_config = mgr.get_config(longer_tf)
        
        assert longer_config.min_pattern_candles >= shorter_config.min_pattern_candles, \
            f"Longer timeframe {longer_tf.value} should have >= min_pattern_candles " \
            f"({longer_config.min_pattern_candles}) than shorter timeframe {shorter_tf.value} " \
            f"({shorter_config.min_pattern_candles})"
    
    @given(timeframe_pair=timeframe_pair_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_longer_timeframe_has_lower_trend_sensitivity(self, timeframe_pair):
        """
        Feature: chart-analysis-improvements
        Property 4: Timeframe Parameter Adjustment
        Validates: Requirements 2.2, 2.3, 2.5
        
        For any two timeframes where tf1 is shorter than tf2,
        the trend_sensitivity for tf2 SHALL be less than or equal to tf1.
        """
        shorter_tf, longer_tf = timeframe_pair
        
        mgr = TimeframeConfigManager()
        
        shorter_config = mgr.get_config(shorter_tf)
        longer_config = mgr.get_config(longer_tf)
        
        assert longer_config.trend_sensitivity <= shorter_config.trend_sensitivity, \
            f"Longer timeframe {longer_tf.value} should have <= trend_sensitivity " \
            f"({longer_config.trend_sensitivity}) than shorter timeframe {shorter_tf.value} " \
            f"({shorter_config.trend_sensitivity})"
    
    @given(timeframe=timeframe_strategy)
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_get_config_returns_valid_timeframe_config(self, timeframe):
        """
        Feature: chart-analysis-improvements
        Property 4: Timeframe Parameter Adjustment
        Validates: Requirements 2.2, 2.3
        
        For any valid timeframe, get_config() SHALL return a valid TimeframeConfig
        with all required fields properly set.
        """
        mgr = TimeframeConfigManager()
        config = mgr.get_config(timeframe)
        
        # Verify it's a TimeframeConfig instance
        assert isinstance(config, TimeframeConfig), \
            f"get_config should return TimeframeConfig, got {type(config)}"
        
        # Verify all fields are valid
        assert isinstance(config.timeframe, Timeframe), \
            f"timeframe should be Timeframe enum, got {type(config.timeframe)}"
        assert isinstance(config.candle_interval, CandleInterval), \
            f"candle_interval should be CandleInterval enum, got {type(config.candle_interval)}"
        assert config.min_pattern_candles >= 1, \
            f"min_pattern_candles should be >= 1, got {config.min_pattern_candles}"
        assert 0.0 <= config.min_pattern_height_pct <= 1.0, \
            f"min_pattern_height_pct should be in [0, 1], got {config.min_pattern_height_pct}"
        assert 0.0 <= config.trend_sensitivity <= 1.0, \
            f"trend_sensitivity should be in [0, 1], got {config.trend_sensitivity}"
    
    @given(timeframe=timeframe_strategy)
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_get_pattern_params_returns_valid_dict(self, timeframe):
        """
        Feature: chart-analysis-improvements
        Property 4: Timeframe Parameter Adjustment
        Validates: Requirements 2.2, 2.3
        
        For any valid timeframe, get_pattern_params() SHALL return a dictionary
        with all required keys for pattern detection.
        """
        mgr = TimeframeConfigManager()
        params = mgr.get_pattern_params(timeframe)
        
        # Verify it's a dictionary
        assert isinstance(params, dict), \
            f"get_pattern_params should return dict, got {type(params)}"
        
        # Verify required keys exist
        required_keys = ["min_candles", "min_height_pct", "trend_sensitivity", "candle_interval"]
        for key in required_keys:
            assert key in params, f"Missing required key: {key}"
        
        # Verify values are valid
        assert params["min_candles"] >= 1, \
            f"min_candles should be >= 1, got {params['min_candles']}"
        assert 0.0 <= params["min_height_pct"] <= 1.0, \
            f"min_height_pct should be in [0, 1], got {params['min_height_pct']}"
        assert 0.0 <= params["trend_sensitivity"] <= 1.0, \
            f"trend_sensitivity should be in [0, 1], got {params['trend_sensitivity']}"
        assert isinstance(params["candle_interval"], str), \
            f"candle_interval should be string, got {type(params['candle_interval'])}"
    
    @given(config=config_dict_strategy())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_config_dict_initializes_manager(self, config):
        """
        Feature: chart-analysis-improvements
        Property 4: Timeframe Parameter Adjustment
        Validates: Requirements 2.2, 2.3
        
        For any valid configuration dictionary, TimeframeConfigManager SHALL
        initialize successfully and return valid configurations.
        """
        mgr = TimeframeConfigManager(config)
        
        # Verify manager is functional
        assert mgr.default_timeframe is not None
        assert mgr.default_candle_interval is not None
        
        # Verify get_config works with default
        default_config = mgr.get_config()
        assert isinstance(default_config, TimeframeConfig)


class TestDefaultConfigurationValues:
    """
    Property tests for default configuration values.
    
    Feature: chart-analysis-improvements
    Property 5: Default Configuration Values
    Validates: Requirements 2.1, 2.6
    """
    
    def test_default_timeframe_is_1y(self):
        """
        Feature: chart-analysis-improvements
        Property 5: Default Configuration Values
        Validates: Requirements 2.1, 2.6
        
        For any new TimeframeConfigManager instance without explicit configuration,
        the default timeframe SHALL be "1Y".
        """
        mgr = TimeframeConfigManager()
        
        assert mgr.default_timeframe == Timeframe.YEAR_1, \
            f"Default timeframe should be 1Y, got {mgr.default_timeframe.value}"
    
    def test_default_candle_interval_is_1d(self):
        """
        Feature: chart-analysis-improvements
        Property 5: Default Configuration Values
        Validates: Requirements 2.1, 2.6
        
        For any new TimeframeConfigManager instance without explicit configuration,
        the default candle interval SHALL be "1D".
        """
        mgr = TimeframeConfigManager()
        
        assert mgr.default_candle_interval == CandleInterval.DAILY, \
            f"Default candle interval should be 1D, got {mgr.default_candle_interval.value}"
    
    def test_default_config_uses_1y_parameters(self):
        """
        Feature: chart-analysis-improvements
        Property 5: Default Configuration Values
        Validates: Requirements 2.1, 2.6
        
        For any new TimeframeConfigManager instance, calling get_config() without
        arguments SHALL return the 1Y configuration.
        """
        mgr = TimeframeConfigManager()
        config = mgr.get_config()
        
        assert config.timeframe == Timeframe.YEAR_1, \
            f"Default config timeframe should be 1Y, got {config.timeframe.value}"
        assert config.candle_interval == CandleInterval.DAILY, \
            f"Default config candle_interval should be 1D, got {config.candle_interval.value}"


class TestTimeframeConfigConsistency:
    """
    Property tests for configuration consistency.
    
    Feature: chart-analysis-improvements
    Validates: Requirements 2.2, 2.3, 2.5
    """
    
    @given(timeframe=timeframe_strategy)
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_get_config_and_get_pattern_params_consistent(self, timeframe):
        """
        Feature: chart-analysis-improvements
        Property 4: Timeframe Parameter Adjustment
        Validates: Requirements 2.2, 2.3
        
        For any timeframe, get_config() and get_pattern_params() SHALL return
        consistent values for the same parameters.
        """
        mgr = TimeframeConfigManager()
        
        config = mgr.get_config(timeframe)
        params = mgr.get_pattern_params(timeframe)
        
        assert params["min_candles"] == config.min_pattern_candles, \
            f"min_candles mismatch: params={params['min_candles']}, config={config.min_pattern_candles}"
        assert params["min_height_pct"] == config.min_pattern_height_pct, \
            f"min_height_pct mismatch: params={params['min_height_pct']}, config={config.min_pattern_height_pct}"
        assert params["trend_sensitivity"] == config.trend_sensitivity, \
            f"trend_sensitivity mismatch: params={params['trend_sensitivity']}, config={config.trend_sensitivity}"
        assert params["candle_interval"] == config.candle_interval.value, \
            f"candle_interval mismatch: params={params['candle_interval']}, config={config.candle_interval.value}"
    
    @given(timeframe=timeframe_strategy)
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_calls_return_same_config(self, timeframe):
        """
        Feature: chart-analysis-improvements
        Property 4: Timeframe Parameter Adjustment
        Validates: Requirements 2.2, 2.3
        
        For any timeframe, multiple calls to get_config() SHALL return
        equivalent configurations (idempotent).
        """
        mgr = TimeframeConfigManager()
        
        config1 = mgr.get_config(timeframe)
        config2 = mgr.get_config(timeframe)
        
        assert config1.timeframe == config2.timeframe
        assert config1.candle_interval == config2.candle_interval
        assert config1.min_pattern_candles == config2.min_pattern_candles
        assert config1.min_pattern_height_pct == config2.min_pattern_height_pct
        assert config1.trend_sensitivity == config2.trend_sensitivity

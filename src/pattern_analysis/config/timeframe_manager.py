"""
Timeframe Configuration Manager for Chart Pattern Analysis.

Manages timeframe-specific configuration for pattern analysis,
adjusting detection parameters based on the expected timeframe
and candle interval of the chart being analyzed.

Feature: chart-analysis-improvements
Requirements: 2.2, 2.3, 2.5
"""

import logging
from typing import Any, Dict, Optional

from ..models.enums import Timeframe, CandleInterval
from ..models.dataclasses import TimeframeConfig


logger = logging.getLogger("pattern_analysis.timeframe_manager")


class TimeframeConfigManager:
    """
    Manages timeframe-specific configuration for pattern analysis.
    
    Adjusts detection parameters based on the expected timeframe
    and candle interval of the chart being analyzed.
    
    Key behaviors:
    - Longer timeframes (1Y, YTD) use more candles for pattern detection
    - Longer timeframes have lower trend sensitivity (less noise)
    - Default configuration is 1Y timeframe with daily candles
    
    Requirements: 2.2, 2.3, 2.5
    """
    
    # Default configurations per timeframe
    # Longer timeframes require more candles and have lower sensitivity
    TIMEFRAME_DEFAULTS: Dict[Timeframe, TimeframeConfig] = {
        Timeframe.DAY_1: TimeframeConfig(
            timeframe=Timeframe.DAY_1,
            candle_interval=CandleInterval.MINUTE_5,
            min_pattern_candles=10,
            min_pattern_height_pct=0.02,
            trend_sensitivity=0.8
        ),
        Timeframe.WEEK_1: TimeframeConfig(
            timeframe=Timeframe.WEEK_1,
            candle_interval=CandleInterval.HOUR_1,
            min_pattern_candles=15,
            min_pattern_height_pct=0.03,
            trend_sensitivity=0.7
        ),
        Timeframe.MONTH_1: TimeframeConfig(
            timeframe=Timeframe.MONTH_1,
            candle_interval=CandleInterval.HOUR_4,
            min_pattern_candles=20,
            min_pattern_height_pct=0.04,
            trend_sensitivity=0.6
        ),
        Timeframe.MONTH_3: TimeframeConfig(
            timeframe=Timeframe.MONTH_3,
            candle_interval=CandleInterval.DAILY,
            min_pattern_candles=25,
            min_pattern_height_pct=0.045,
            trend_sensitivity=0.55
        ),
        Timeframe.MONTH_6: TimeframeConfig(
            timeframe=Timeframe.MONTH_6,
            candle_interval=CandleInterval.DAILY,
            min_pattern_candles=28,
            min_pattern_height_pct=0.048,
            trend_sensitivity=0.52
        ),
        Timeframe.YTD: TimeframeConfig(
            timeframe=Timeframe.YTD,
            candle_interval=CandleInterval.DAILY,
            min_pattern_candles=30,
            min_pattern_height_pct=0.05,
            trend_sensitivity=0.5
        ),
        Timeframe.YEAR_1: TimeframeConfig(
            timeframe=Timeframe.YEAR_1,
            candle_interval=CandleInterval.DAILY,
            min_pattern_candles=30,
            min_pattern_height_pct=0.05,
            trend_sensitivity=0.5
        ),
        Timeframe.YEAR_5: TimeframeConfig(
            timeframe=Timeframe.YEAR_5,
            candle_interval=CandleInterval.WEEKLY,
            min_pattern_candles=40,
            min_pattern_height_pct=0.06,
            trend_sensitivity=0.4
        ),
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TimeframeConfigManager.
        
        Args:
            config: Optional configuration dictionary with keys:
                - default_timeframe: Default timeframe string (e.g., "1Y")
                - default_candle_interval: Default candle interval (e.g., "1D")
                - adjustments: Dict of timeframe-specific parameter overrides
        """
        self._config = config or {}
        
        # Parse default timeframe from config or use 1Y
        default_tf_str = self._config.get("default_timeframe", "1Y")
        self._default_timeframe = self._parse_timeframe(default_tf_str)
        
        # Parse default candle interval from config or use 1D
        default_ci_str = self._config.get("default_candle_interval", "1D")
        self._default_interval = self._parse_candle_interval(default_ci_str)
        
        # Apply any custom adjustments from config
        self._custom_configs = self._build_custom_configs()
        
        logger.debug(
            f"TimeframeConfigManager initialized: "
            f"default_timeframe={self._default_timeframe.value}, "
            f"default_interval={self._default_interval.value}"
        )
    
    @property
    def default_timeframe(self) -> Timeframe:
        """Get the default timeframe."""
        return self._default_timeframe
    
    @property
    def default_candle_interval(self) -> CandleInterval:
        """Get the default candle interval."""
        return self._default_interval
    
    def _parse_timeframe(self, value: str) -> Timeframe:
        """Parse a timeframe string to Timeframe enum."""
        try:
            return Timeframe(value)
        except ValueError:
            logger.warning(
                f"Invalid timeframe '{value}', using default '1Y'"
            )
            return Timeframe.YEAR_1
    
    def _parse_candle_interval(self, value: str) -> CandleInterval:
        """Parse a candle interval string to CandleInterval enum."""
        try:
            return CandleInterval(value)
        except ValueError:
            logger.warning(
                f"Invalid candle interval '{value}', using default '1D'"
            )
            return CandleInterval.DAILY
    
    def _build_custom_configs(self) -> Dict[Timeframe, TimeframeConfig]:
        """Build custom configurations from config adjustments."""
        custom = {}
        adjustments = self._config.get("adjustments", {})
        
        for tf_str, params in adjustments.items():
            try:
                tf = self._parse_timeframe(tf_str)
                base_config = self.TIMEFRAME_DEFAULTS.get(tf)
                
                if base_config:
                    # Create new config with overridden values
                    custom[tf] = TimeframeConfig(
                        timeframe=tf,
                        candle_interval=base_config.candle_interval,
                        min_pattern_candles=params.get(
                            "min_pattern_candles",
                            base_config.min_pattern_candles
                        ),
                        min_pattern_height_pct=params.get(
                            "min_pattern_height_pct",
                            base_config.min_pattern_height_pct
                        ),
                        trend_sensitivity=params.get(
                            "trend_sensitivity",
                            base_config.trend_sensitivity
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to parse adjustment for {tf_str}: {e}")
        
        return custom
    
    def get_config(self, timeframe: Optional[Timeframe] = None) -> TimeframeConfig:
        """
        Get configuration for specified or default timeframe.
        
        Args:
            timeframe: Optional timeframe to get config for.
                      If None, uses the default timeframe.
        
        Returns:
            TimeframeConfig with parameters for the timeframe.
        """
        tf = timeframe or self._default_timeframe
        
        # Check custom configs first (from YAML adjustments)
        if tf in self._custom_configs:
            return self._custom_configs[tf]
        
        # Fall back to built-in defaults
        if tf in self.TIMEFRAME_DEFAULTS:
            return self.TIMEFRAME_DEFAULTS[tf]
        
        # Ultimate fallback to 1Y config
        logger.warning(
            f"No config found for timeframe {tf.value}, using 1Y defaults"
        )
        return self.TIMEFRAME_DEFAULTS[Timeframe.YEAR_1]
    
    def get_pattern_params(
        self,
        timeframe: Optional[Timeframe] = None
    ) -> Dict[str, Any]:
        """
        Get pattern detection parameters for timeframe.
        
        Returns a dictionary suitable for passing to pattern detection
        components with the following keys:
        - min_candles: Minimum candles required for pattern
        - min_height_pct: Minimum pattern height as percentage
        - trend_sensitivity: Sensitivity for trend detection
        - candle_interval: The candle interval value string
        
        Args:
            timeframe: Optional timeframe to get params for.
                      If None, uses the default timeframe.
        
        Returns:
            Dictionary of pattern detection parameters.
        """
        config = self.get_config(timeframe)
        
        return {
            "min_candles": config.min_pattern_candles,
            "min_height_pct": config.min_pattern_height_pct,
            "trend_sensitivity": config.trend_sensitivity,
            "candle_interval": config.candle_interval.value
        }
    
    def get_all_timeframes(self) -> list:
        """Get list of all supported timeframes."""
        return list(Timeframe)
    
    def is_long_timeframe(self, timeframe: Optional[Timeframe] = None) -> bool:
        """
        Check if a timeframe is considered "long" (3M or more).
        
        Long timeframes use more conservative detection parameters.
        
        Args:
            timeframe: Timeframe to check. If None, uses default.
        
        Returns:
            True if timeframe is 3M, 6M, YTD, 1Y, or 5Y.
        """
        tf = timeframe or self._default_timeframe
        long_timeframes = {
            Timeframe.MONTH_3,
            Timeframe.MONTH_6,
            Timeframe.YTD,
            Timeframe.YEAR_1,
            Timeframe.YEAR_5
        }
        return tf in long_timeframes
    
    def __repr__(self) -> str:
        return (
            f"TimeframeConfigManager("
            f"default_timeframe={self._default_timeframe.value}, "
            f"default_interval={self._default_interval.value})"
        )

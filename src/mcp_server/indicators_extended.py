"""
Indicators - RVOL (Relative Volume) and Bollinger Bands additions.
Extension to src/mcp_server/indicators.py
"""

from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class RVOLResult:
    """Relative Volume calculation result."""
    current_volume: float
    average_volume: float
    rvol: float  # Ratio: current / average
    is_significant: bool  # rvol > 1.5 (confirms breakout)
    is_low: bool  # rvol < 0.5 (low conviction)


@dataclass
class BollingerResult:
    """Bollinger Bands calculation result."""
    upper: float
    middle: float  # SMA
    lower: float
    bandwidth: float  # (upper - lower) / middle
    percent_b: float  # (price - lower) / (upper - lower)
    is_at_lower: bool  # price within 5% of lower band
    is_at_upper: bool  # price within 5% of upper band
    is_squeeze: bool  # bandwidth < 0.1 (volatility squeeze)


def calculate_rvol(
    volumes: List[float],
    period: int = 20,
) -> RVOLResult:
    """
    Calculate Relative Volume (RVOL).
    
    RVOL = Current Volume / Average Volume
    
    Args:
        volumes: List of volume values (most recent last)
        period: Lookback period for average (default 20)
        
    Returns:
        RVOLResult with RVOL ratio and significance flags
    """
    if len(volumes) < period + 1:
        raise ValueError(f"Need at least {period + 1} volume values")
    
    current_volume = volumes[-1]
    historical_volumes = volumes[-(period + 1):-1]  # Exclude current
    
    average_volume = sum(historical_volumes) / len(historical_volumes)
    
    if average_volume == 0:
        rvol = 0.0
    else:
        rvol = current_volume / average_volume
    
    return RVOLResult(
        current_volume=current_volume,
        average_volume=round(average_volume, 2),
        rvol=round(rvol, 2),
        is_significant=rvol >= 1.5,
        is_low=rvol < 0.5,
    )


def calculate_bollinger(
    prices: List[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> BollingerResult:
    """
    Calculate Bollinger Bands.
    
    - Middle Band: SMA of prices
    - Upper Band: SMA + (std_dev * standard deviation)
    - Lower Band: SMA - (std_dev * standard deviation)
    
    Args:
        prices: List of closing prices (most recent last)
        period: SMA period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
        
    Returns:
        BollingerResult with bands and signals
    """
    if len(prices) < period:
        raise ValueError(f"Need at least {period} prices")
    
    # Use most recent 'period' prices
    recent_prices = prices[-period:]
    current_price = prices[-1]
    
    # Calculate SMA (middle band)
    sma = sum(recent_prices) / period
    
    # Calculate standard deviation
    variance = sum((p - sma) ** 2 for p in recent_prices) / period
    std = variance ** 0.5
    
    # Calculate bands
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    # Calculate bandwidth and %B
    bandwidth = (upper - lower) / sma if sma > 0 else 0
    
    band_range = upper - lower
    if band_range > 0:
        percent_b = (current_price - lower) / band_range
    else:
        percent_b = 0.5
    
    # Determine position relative to bands (within 5%)
    band_tolerance = band_range * 0.05
    is_at_lower = current_price <= lower + band_tolerance
    is_at_upper = current_price >= upper - band_tolerance
    
    # Volatility squeeze detection
    is_squeeze = bandwidth < 0.1
    
    return BollingerResult(
        upper=round(upper, 4),
        middle=round(sma, 4),
        lower=round(lower, 4),
        bandwidth=round(bandwidth, 4),
        percent_b=round(percent_b, 4),
        is_at_lower=is_at_lower,
        is_at_upper=is_at_upper,
        is_squeeze=is_squeeze,
    )

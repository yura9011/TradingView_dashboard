"""
Unit tests for MCP Indicator Tools.
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_server.indicators import IndicatorTools


class TestEMA:
    """Tests for EMA calculation."""
    
    def test_ema_basic(self):
        """Test basic EMA calculation."""
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        result = IndicatorTools.calculate_ema(prices, period=5)
        
        assert result.period == 5
        assert result.prices_used == 10
        assert 105 < result.value < 109  # Should be near recent prices
    
    def test_ema_not_enough_data(self):
        """Test EMA with insufficient data."""
        prices = [100, 102, 101]
        
        with pytest.raises(ValueError) as exc:
            IndicatorTools.calculate_ema(prices, period=5)
        
        assert "Need at least 5 prices" in str(exc.value)
    
    def test_ema_periods(self):
        """Test standard EMA periods (21, 50, 200)."""
        prices = list(range(100, 350))  # 250 prices
        
        ema_21 = IndicatorTools.calculate_ema(prices, 21)
        ema_50 = IndicatorTools.calculate_ema(prices, 50)
        ema_200 = IndicatorTools.calculate_ema(prices, 200)
        
        # Shorter EMAs should be closer to current price (349)
        assert ema_21.value > ema_50.value > ema_200.value


class TestRSI:
    """Tests for RSI calculation."""
    
    def test_rsi_uptrend(self):
        """Test RSI in strong uptrend."""
        # Consistently rising prices
        prices = [100 + i for i in range(20)]
        result = IndicatorTools.calculate_rsi(prices, period=14)
        
        assert result.value > 70, "RSI should be high in uptrend"
        assert result.is_overbought is True
        assert result.is_oversold is False
    
    def test_rsi_downtrend(self):
        """Test RSI in strong downtrend."""
        # Consistently falling prices
        prices = [100 - i for i in range(20)]
        result = IndicatorTools.calculate_rsi(prices, period=14)
        
        assert result.value < 30, "RSI should be low in downtrend"
        assert result.is_oversold is True
        assert result.is_overbought is False
    
    def test_rsi_range(self):
        """Test RSI is always between 0 and 100."""
        prices = [100, 150, 80, 120, 90, 110, 95, 105, 100, 102, 98, 103, 97, 101, 99]
        result = IndicatorTools.calculate_rsi(prices)
        
        assert 0 <= result.value <= 100


class TestMACD:
    """Tests for MACD calculation."""
    
    def test_macd_basic(self):
        """Test basic MACD calculation."""
        # 40 price points
        prices = [100 + i * 0.5 for i in range(40)]
        result = IndicatorTools.calculate_macd(prices)
        
        assert result.macd_line != 0
        assert result.signal_line != 0
        assert result.histogram == round(result.macd_line - result.signal_line, 4)
    
    def test_macd_bullish(self):
        """Test MACD bullish signal (positive MACD line)."""
        # Strong uptrend
        prices = [100 + i for i in range(40)]
        result = IndicatorTools.calculate_macd(prices)
        
        # Fast EMA > Slow EMA in uptrend = positive MACD
        assert result.macd_line > 0, "MACD line should be positive in uptrend"
        # Note: is_bullish depends on histogram (macd - signal), which varies
    
    def test_macd_not_enough_data(self):
        """Test MACD with insufficient data."""
        prices = [100, 101, 102]  # Too few
        
        with pytest.raises(ValueError) as exc:
            IndicatorTools.calculate_macd(prices)
        
        assert "Need at least 35 prices" in str(exc.value)
    
    def test_macd_signal_is_real_ema(self):
        """Verify signal line is calculated as EMA, not approximation."""
        prices = [100 + i for i in range(50)]
        result = IndicatorTools.calculate_macd(prices)
        
        # Signal should NOT be exactly macd * 0.9 (old approximation)
        approximation = result.macd_line * 0.9
        assert abs(result.signal_line - approximation) > 0.01, \
            "Signal line should be real EMA, not approximation"


class TestFibonacci:
    """Tests for Fibonacci calculation."""
    
    def test_fibonacci_uptrend(self):
        """Test Fibonacci levels in uptrend."""
        result = IndicatorTools.calculate_fibonacci(high=200, low=100, is_uptrend=True)
        
        assert result.level_0 == 200     # 0% = high
        assert result.level_100 == 100   # 100% = low
        assert result.level_50 == 150    # 50% midpoint
        assert abs(result.level_618 - 138.2) < 0.1  # 61.8% level
    
    def test_fibonacci_downtrend(self):
        """Test Fibonacci levels in downtrend."""
        result = IndicatorTools.calculate_fibonacci(high=200, low=100, is_uptrend=False)
        
        assert result.level_0 == 100     # 0% = low
        assert result.level_100 == 200   # 100% = high
        assert result.level_50 == 150    # 50% midpoint
    
    def test_fibonacci_as_dict(self):
        """Test Fibonacci dict output."""
        result = IndicatorTools.calculate_fibonacci(high=100, low=0)
        levels = result.as_dict()
        
        assert "0%" in levels
        assert "61.8%" in levels
        assert "100%" in levels
        assert len(levels) == 7


class TestMultipleEMAs:
    """Tests for batch EMA calculation."""
    
    def test_all_emas(self):
        """Test calculating multiple EMAs at once."""
        prices = list(range(100, 350))  # 250 prices
        result = IndicatorTools.calculate_all_emas(prices, [21, 50, 200])
        
        assert "ema_21" in result
        assert "ema_50" in result
        assert "ema_200" in result
        assert result["ema_21"] is not None
    
    def test_emas_insufficient_for_some(self):
        """Test when some periods have insufficient data."""
        prices = list(range(100, 150))  # 50 prices
        result = IndicatorTools.calculate_all_emas(prices, [21, 50, 200])
        
        assert result["ema_21"] is not None
        assert result["ema_50"] is not None
        assert result["ema_200"] is None  # Not enough data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

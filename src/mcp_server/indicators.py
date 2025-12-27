"""
MCP Tools Server - Technical Indicators
Provides calculation tools for EMA, RSI, MACD, Fibonacci.
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EMAResult:
    """EMA calculation result."""
    period: int
    value: float
    prices_used: int


@dataclass
class RSIResult:
    """RSI calculation result."""
    value: float
    is_oversold: bool
    is_overbought: bool
    period: int


@dataclass
class MACDResult:
    """MACD calculation result."""
    macd_line: float
    signal_line: float
    histogram: float
    is_bullish: bool


@dataclass
class ATRResult:
    """ATR calculation result."""
    period: int
    value: float
    prices_used: int


@dataclass
class FibonacciLevels:
    """Fibonacci retracement levels."""
    level_0: float      # 0%
    level_236: float    # 23.6%
    level_382: float    # 38.2%
    level_50: float     # 50%
    level_618: float    # 61.8%
    level_786: float    # 78.6%
    level_100: float    # 100%
    
    def as_dict(self) -> dict:
        return {
            "0%": self.level_0,
            "23.6%": self.level_236,
            "38.2%": self.level_382,
            "50%": self.level_50,
            "61.8%": self.level_618,
            "78.6%": self.level_786,
            "100%": self.level_100,
        }


class IndicatorTools:
    """MCP-style tools for technical indicator calculations."""
    
    @staticmethod
    def calculate_atr(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> ATRResult:
        """Calculate Average True Range (ATR).
        
        Args:
            highs: List of high prices
            lows: List of low prices
            closes: List of closing prices
            period: ATR period (default 14)
            
        Returns:
            ATRResult with calculated value
        """
        if len(highs) != len(lows) or len(lows) != len(closes):
            raise ValueError("Price lists must have same length")
        
        if len(closes) < period + 1:
            raise ValueError(f"Need at least {period + 1} prices for ATR")

        # Calculate True Ranges
        tr_values = []
        for i in range(1, len(closes)):
            h = highs[i]
            l = lows[i]
            pc = closes[i-1]
            
            tr = max(h - l, abs(h - pc), abs(l - pc))
            tr_values.append(tr)
            
        # First ATR is SMA of TRs
        atr = sum(tr_values[:period]) / period
        
        # Subsequent ATRs using Wilder's Smoothing
        # ATR_current = ((ATR_prev * (period - 1)) + TR_current) / period
        for i in range(period, len(tr_values)):
            atr = ((atr * (period - 1)) + tr_values[i]) / period
            
        return ATRResult(
            period=period,
            value=round(atr, 4),
            prices_used=len(closes)
        )

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> EMAResult:
        """Calculate Exponential Moving Average.
        
        Args:
            prices: List of closing prices (oldest first)
            period: EMA period (e.g., 21, 50, 200)
            
        Returns:
            EMAResult with calculated value
        """
        if len(prices) < period:
            raise ValueError(f"Need at least {period} prices, got {len(prices)}")
        
        multiplier = 2 / (period + 1)
        
        # Start with SMA for first EMA value
        sma = sum(prices[:period]) / period
        ema = sma
        
        # Calculate EMA for remaining prices
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return EMAResult(
            period=period,
            value=round(ema, 4),
            prices_used=len(prices),
        )
    
    @staticmethod
    def calculate_rsi(
        prices: List[float],
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> RSIResult:
        """Calculate Relative Strength Index.
        
        Args:
            prices: List of closing prices (oldest first)
            period: RSI period (default 14)
            overbought: Overbought threshold
            oversold: Oversold threshold
            
        Returns:
            RSIResult with value and signals
        """
        if len(prices) < period + 1:
            raise ValueError(f"Need at least {period + 1} prices")
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [max(0, c) for c in changes]
        losses = [abs(min(0, c)) for c in changes]
        
        # Calculate average gain/loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Smooth averages for remaining periods
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        # Calculate RSI
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        return RSIResult(
            value=round(rsi, 2),
            is_oversold=rsi < oversold,
            is_overbought=rsi > overbought,
            period=period,
        )
    
    @staticmethod
    def calculate_macd(
        prices: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> MACDResult:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: List of closing prices (oldest first)
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            MACDResult with MACD line, signal, histogram
        """
        min_prices = slow_period + signal_period
        if len(prices) < min_prices:
            raise ValueError(f"Need at least {min_prices} prices")
        
        # Calculate MACD line values for all prices
        macd_values = []
        
        for i in range(slow_period, len(prices) + 1):
            subset = prices[:i]
            fast_ema = IndicatorTools._calculate_ema_value(subset, fast_period)
            slow_ema = IndicatorTools._calculate_ema_value(subset, slow_period)
            macd_values.append(fast_ema - slow_ema)
        
        # Calculate signal line (EMA of MACD values)
        if len(macd_values) >= signal_period:
            signal_line = IndicatorTools._calculate_ema_value(macd_values, signal_period)
        else:
            signal_line = macd_values[-1] if macd_values else 0.0
        
        macd_line = macd_values[-1] if macd_values else 0.0
        histogram = macd_line - signal_line
        
        logger.debug(f"MACD calculated: line={macd_line:.4f}, signal={signal_line:.4f}")
        
        return MACDResult(
            macd_line=round(macd_line, 4),
            signal_line=round(signal_line, 4),
            histogram=round(histogram, 4),
            is_bullish=histogram > 0,
        )
    
    @staticmethod
    def _calculate_ema_value(prices: List[float], period: int) -> float:
        """Internal helper to calculate raw EMA value."""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (period + 1)
        sma = sum(prices[:period]) / period
        ema = sma
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def calculate_fibonacci(
        high: float,
        low: float,
        is_uptrend: bool = True,
    ) -> FibonacciLevels:
        """Calculate Fibonacci retracement levels.
        
        Args:
            high: Swing high price
            low: Swing low price
            is_uptrend: True for uptrend (levels from low), False for downtrend
            
        Returns:
            FibonacciLevels with all standard levels
        """
        diff = high - low
        
        if is_uptrend:
            # Retracement from high to low
            return FibonacciLevels(
                level_0=high,
                level_236=high - diff * 0.236,
                level_382=high - diff * 0.382,
                level_50=high - diff * 0.5,
                level_618=high - diff * 0.618,
                level_786=high - diff * 0.786,
                level_100=low,
            )
        else:
            # Retracement from low to high
            return FibonacciLevels(
                level_0=low,
                level_236=low + diff * 0.236,
                level_382=low + diff * 0.382,
                level_50=low + diff * 0.5,
                level_618=low + diff * 0.618,
                level_786=low + diff * 0.786,
                level_100=high,
            )
    
    @staticmethod
    def calculate_all_emas(
        prices: List[float],
        periods: List[int] = [21, 50, 200],
    ) -> dict:
        """Calculate multiple EMAs at once.
        
        Args:
            prices: List of closing prices
            periods: EMA periods to calculate
            
        Returns:
            Dict of period -> EMA value
        """
        results = {}
        for period in periods:
            if len(prices) >= period:
                result = IndicatorTools.calculate_ema(prices, period)
                results[f"ema_{period}"] = result.value
            else:
                results[f"ema_{period}"] = None
        return results


# Convenience functions
def get_indicator_tools() -> IndicatorTools:
    """Get IndicatorTools instance."""
    return IndicatorTools()

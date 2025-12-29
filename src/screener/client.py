"""
TradingView Screener Client
Wrapper around tradingview-screener for fetching market data.
"""

import logging
from typing import Optional, List
from datetime import datetime

from tradingview_screener import Query, Column

from src.models import Asset, Market, Timeframe, ScreenerResult

logger = logging.getLogger(__name__)


class ScreenerClient:
    """Client for querying TradingView screener data."""
    
    # Market to screener mapping
    MARKET_MAP = {
        Market.AMERICA: "america",
        Market.CRYPTO: "crypto",
        Market.FOREX: "forex",
    }
    
    def __init__(self, market: Market = Market.AMERICA):
        """Initialize screener client.
        
        Args:
            market: Target market to scan
        """
        self.market = market
        self._market_name = self.MARKET_MAP.get(market, "america")
    
    def get_top_volume(
        self,
        limit: int = 10,
        min_price: float = 5.0,
        max_price: float = 500.0,
        min_volume: float = 1_000_000,
    ) -> ScreenerResult:
        """Get top stocks by volume.
        
        Args:
            limit: Number of results to return
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_volume: Minimum volume filter
            
        Returns:
            ScreenerResult with list of assets
        """
        logger.info(f"Fetching top {limit} by volume from {self._market_name}")
        
        try:
            query = (
                Query()
                .select("name", "close", "change", "volume", "market_cap_basic", "sector", "ATR")
                .where(
                    Column("close") >= min_price,
                    Column("close") <= max_price,
                    Column("volume") >= min_volume,
                )
                .order_by("volume", ascending=False)
                .limit(limit)
            )
            
            # Execute query
            count, df = query.get_scanner_data()
            
            assets = self._parse_dataframe(df)
            
            return ScreenerResult(
                assets=assets,
                market=self.market,
                filters_applied={
                    "min_price": min_price,
                    "max_price": max_price,
                    "min_volume": min_volume,
                },
                total_count=count,
            )
            
        except Exception as e:
            logger.error(f"Screener query failed: {e}")
            return ScreenerResult(market=self.market)

    def get_symbol_data(self, symbol: str) -> Optional[dict]:
        """Get data for a specific symbol.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'BTCUSD')
            
        Returns:
            Dictionary with symbol data or None if not found
        """
        try:
            # Clean symbol (remove exchange if present)
            clean_symbol = symbol.split(':')[-1] if ':' in symbol else symbol
            
            query = (
                Query()
                .select(
                    "name", "close", "change", "volume", 
                    "open", "high", "low",  # Added for VSA (Spread/Close Position)
                    "ATR", "RSI", "MACD.macd", "MACD.signal",
                    "average_volume_10d_calc", "sector"
                )
                .where(Column("name") == clean_symbol)
                .limit(1)
            )
            
            _, df = query.get_scanner_data()
            
            if df is None or df.empty:
                return None
                
            # Convert first row to dict
            data = df.iloc[0].to_dict()
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def get_gappers(
        self,
        limit: int = 10,
        min_gap_percent: float = 3.0,
        min_volume: float = 500_000,
    ) -> ScreenerResult:
        """Get stocks with significant price gaps.
        
        Args:
            limit: Number of results
            min_gap_percent: Minimum gap percentage (positive or negative)
            min_volume: Minimum volume filter
            
        Returns:
            ScreenerResult with gapper stocks
        """
        logger.info(f"Fetching gappers > {min_gap_percent}% from {self._market_name}")
        
        try:
            query = (
                Query()
                .select("name", "close", "change", "volume", "gap", "sector")
                .where(
                    Column("volume") >= min_volume,
                    Column("change").between(-100, -min_gap_percent) | 
                    Column("change").between(min_gap_percent, 100),
                )
                .order_by("change", ascending=False)
                .limit(limit)
            )
            
            count, df = query.get_scanner_data()
            assets = self._parse_dataframe(df)
            
            return ScreenerResult(
                assets=assets,
                market=self.market,
                filters_applied={
                    "min_gap_percent": min_gap_percent,
                    "min_volume": min_volume,
                },
                total_count=count,
            )
            
        except Exception as e:
            logger.error(f"Gapper query failed: {e}")
            return ScreenerResult(market=self.market)
    
    def get_oversold(
        self,
        limit: int = 10,
        rsi_threshold: float = 30.0,
        min_volume: float = 500_000,
    ) -> ScreenerResult:
        """Get oversold stocks (RSI < threshold).
        
        Args:
            limit: Number of results
            rsi_threshold: RSI upper bound for oversold
            min_volume: Minimum volume filter
            
        Returns:
            ScreenerResult with oversold stocks
        """
        logger.info(f"Fetching oversold (RSI < {rsi_threshold}) from {self._market_name}")
        
        try:
            query = (
                Query()
                .select("name", "close", "change", "volume", "RSI", "sector")
                .where(
                    Column("volume") >= min_volume,
                    Column("RSI") <= rsi_threshold,
                    Column("RSI") > 0,  # Valid RSI
                )
                .order_by("RSI", ascending=True)
                .limit(limit)
            )
            
            count, df = query.get_scanner_data()
            assets = self._parse_dataframe(df)
            
            return ScreenerResult(
                assets=assets,
                market=self.market,
                filters_applied={
                    "rsi_threshold": rsi_threshold,
                    "min_volume": min_volume,
                },
                total_count=count,
            )
            
        except Exception as e:
            logger.error(f"Oversold query failed: {e}")
            return ScreenerResult(market=self.market)
    
    def _parse_dataframe(self, df) -> List[Asset]:
        """Parse pandas DataFrame to Asset list.
        
        Args:
            df: DataFrame from screener
            
        Returns:
            List of Asset models
        """
        assets = []
        
        for _, row in df.iterrows():
            ticker = row.get("ticker", "")
            
            # Extract exchange and symbol from ticker (e.g., "NASDAQ:AAPL")
            if ":" in str(ticker):
                exchange, symbol = str(ticker).split(":", 1)
            else:
                exchange = None
                symbol = str(ticker)
            
            asset = Asset(
                symbol=symbol,
                name=row.get("name"),
                market=self.market,
                exchange=exchange,
                price=row.get("close"),
                change_percent=row.get("change"),
                volume=row.get("volume"),
                sector=row.get("sector"),
                market_cap=row.get("market_cap_basic"),
            )
            assets.append(asset)
        
        return assets


def get_screener(market: Market = Market.AMERICA) -> ScreenerClient:
    """Factory function for ScreenerClient.
    
    Args:
        market: Target market
        
    Returns:
        Configured ScreenerClient instance
    """
    return ScreenerClient(market=market)

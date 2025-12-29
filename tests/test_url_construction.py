
import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.screener.chart_capture import ChartCapture

class TestURLConstruction(unittest.TestCase):
    def setUp(self):
        # We don't need real args since we just test the helper
        self.capture = ChartCapture(headless=True)
        
    def test_with_exchange(self):
        symbol = "AAPL"
        exchange = "NASDAQ"
        interval = "D"
        
        expected = "https://www.tradingview.com/chart/?symbol=NASDAQ:AAPL&interval=D"
        actual = self.capture._build_chart_url(symbol, exchange, interval)
        
        print(f"With Exchange: {actual}")
        self.assertEqual(actual, expected)

    def test_without_exchange(self):
        symbol = "MELI"
        exchange = None
        interval = "30"
        
        # When exchange is None, we expect just symbol=MELI
        expected = "https://www.tradingview.com/chart/?symbol=MELI&interval=30"
        actual = self.capture._build_chart_url(symbol, exchange, interval)
        
        print(f"Without Exchange: {actual}")
        self.assertEqual(actual, expected)
        
    def test_empty_exchange(self):
        symbol = "TSLA"
        exchange = ""
        interval = "W"
        
        # Empty string should likely behave same as None (no exchange prefix)
        expected = "https://www.tradingview.com/chart/?symbol=TSLA&interval=W"
        actual = self.capture._build_chart_url(symbol, exchange, interval)
        
        print(f"Empty Exchange: {actual}")
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()

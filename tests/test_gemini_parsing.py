"""
Unit tests for Gemini Client parsing.
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.gemini_client import GeminiClient, AnalysisResult


class TestResponseParsing:
    """Tests for Gemini response parsing."""
    
    def get_parser(self):
        """Get a mock client for testing parsing."""
        # Create minimal client without API call
        class MockClient:
            def _parse_analysis_response(self, raw_text):
                return GeminiClient._parse_analysis_response(self, raw_text)
            
            def _parse_price(self, val):
                return GeminiClient._parse_price(self, val)
        
        return MockClient()
    
    def test_parse_complete_response(self):
        """Test parsing a complete Gemini response."""
        client = self.get_parser()
        
        response = """PATTERN: Double Bottom
CONFIDENCE: 0.85
TREND: up
SUPPORT: $1,976.00
RESISTANCE: $2,004.00
FIBONACCI: 61.8% retracement
SUMMARY: Strong bullish momentum with potential breakout."""
        
        result = client._parse_analysis_response(response)
        
        assert result.pattern_detected == "double bottom"
        assert result.pattern_confidence == 0.85
        assert result.trend == "up"
        assert result.support_level == 1976.0
        assert result.resistance_level == 2004.0
        assert "61.8%" in result.fibonacci_level
        assert "bullish" in result.analysis_summary.lower()
    
    def test_parse_minimal_response(self):
        """Test parsing minimal response with defaults."""
        client = self.get_parser()
        
        response = """PATTERN: none
CONFIDENCE: 0.3
TREND: sideways
SUMMARY: No clear pattern detected."""
        
        result = client._parse_analysis_response(response)
        
        assert result.pattern_detected == "none"
        assert result.pattern_confidence == 0.3
        assert result.trend == "sideways"
        assert result.support_level is None
        assert result.resistance_level is None
    
    def test_parse_confidence_clamping(self):
        """Test confidence is clamped to 0-1 range."""
        client = self.get_parser()
        
        # Test > 1
        response1 = "PATTERN: test\nCONFIDENCE: 1.5\nTREND: up\nSUMMARY: test"
        result1 = client._parse_analysis_response(response1)
        assert result1.pattern_confidence == 1.0
        
        # Test < 0
        response2 = "PATTERN: test\nCONFIDENCE: -0.5\nTREND: up\nSUMMARY: test"
        result2 = client._parse_analysis_response(response2)
        assert result2.pattern_confidence == 0.0
    
    def test_parse_trend_variations(self):
        """Test various trend parsings."""
        client = self.get_parser()
        
        # Uptrend variations
        for trend in ["up", "uptrend", "UP", "UPTREND"]:
            response = f"PATTERN: none\nCONFIDENCE: 0.5\nTREND: {trend}\nSUMMARY: test"
            result = client._parse_analysis_response(response)
            assert result.trend == "up"
        
        # Downtrend variations
        for trend in ["down", "downtrend", "DOWN"]:
            response = f"PATTERN: none\nCONFIDENCE: 0.5\nTREND: {trend}\nSUMMARY: test"
            result = client._parse_analysis_response(response)
            assert result.trend == "down"
        
        # Sideways default
        for trend in ["sideways", "consolidation", "neutral"]:
            response = f"PATTERN: none\nCONFIDENCE: 0.5\nTREND: {trend}\nSUMMARY: test"
            result = client._parse_analysis_response(response)
            assert result.trend == "sideways"


class TestPriceParsing:
    """Tests for price value parsing."""
    
    def get_parser(self):
        class MockClient:
            def _parse_price(self, val):
                return GeminiClient._parse_price(self, val)
        return MockClient()
    
    def test_parse_price_with_dollar(self):
        """Test parsing $1,976.00"""
        client = self.get_parser()
        assert client._parse_price("$1,976.00") == 1976.0
    
    def test_parse_price_without_dollar(self):
        """Test parsing 1976.00"""
        client = self.get_parser()
        assert client._parse_price("1976.00") == 1976.0
    
    def test_parse_price_with_commas(self):
        """Test parsing 1,976"""
        client = self.get_parser()
        assert client._parse_price("1,976") == 1976.0
    
    def test_parse_price_unknown(self):
        """Test unknown returns None."""
        client = self.get_parser()
        assert client._parse_price("unknown") is None
        assert client._parse_price("N/A") is None
        assert client._parse_price("none") is None
        assert client._parse_price("-") is None
    
    def test_parse_price_empty(self):
        """Test empty string returns None."""
        client = self.get_parser()
        assert client._parse_price("") is None
        assert client._parse_price(None) is None


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""
    
    def test_analysis_result_creation(self):
        """Test creating an AnalysisResult."""
        result = AnalysisResult(
            pattern_detected="bullish engulfing",
            pattern_confidence=0.8,
            trend="up",
            support_level=100.0,
            resistance_level=110.0,
            fibonacci_level="61.8%",
            analysis_summary="Test summary",
            raw_response="Raw text",
        )
        
        assert result.pattern_detected == "bullish engulfing"
        assert result.pattern_confidence == 0.8
        assert result.support_level == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

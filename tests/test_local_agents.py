"""
Test script for new local agents and indicators.
Run: python tests/test_local_agents.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_risk_manager():
    """Test RiskManagerAgentLocal."""
    print("\n" + "=" * 50)
    print("TEST: RiskManagerAgentLocal")
    print("=" * 50)
    
    from src.agents.specialists.risk_manager_local import RiskManagerAgentLocal
    
    rm = RiskManagerAgentLocal()
    
    # Test 1: Low RVOL breakout (should be DANGEROUS)
    result = rm.analyze(
        pattern_data={"pattern": "breakout", "confidence": 0.8},
        trend_data={"trend": "up", "strength": "strong"},
        current_price=100,
        atr_value=2.0,
        rvol=1.2,  # Low volume breakout
    )
    
    print(f"\nTest 1: Low RVOL Breakout (1.2)")
    print(f"  Assessment: {result['risk_assessment']}")
    print(f"  Position: {result['position_size']}")
    print(f"  Reasoning: {result['reasoning']}")
    assert result["risk_assessment"] == "DANGEROUS", "Should detect fakeout"
    assert result["position_size"] == "0%", "Should avoid trade"
    print("  ✅ PASSED")
    
    # Test 2: High RVOL breakout (should be SAFE)
    result = rm.analyze(
        pattern_data={"pattern": "breakout", "confidence": 0.8},
        trend_data={"trend": "up", "strength": "strong"},
        current_price=100,
        atr_value=2.0,
        rvol=2.5,  # Strong volume
    )
    
    print(f"\nTest 2: High RVOL Breakout (2.5)")
    print(f"  Assessment: {result['risk_assessment']}")
    print(f"  Position: {result['position_size']}")
    print(f"  Reasoning: {result['reasoning']}")
    assert result["risk_assessment"] == "SAFE", "Should approve trade"
    assert result["position_size"] == "100%", "Full position"
    print("  ✅ PASSED")
    
    # Test 3: High ATR (should be DANGEROUS)
    result = rm.analyze(
        pattern_data={"pattern": "double bottom", "confidence": 0.8},
        trend_data={"trend": "up", "strength": "moderate"},
        current_price=100,
        atr_value=6.0,  # 6% ATR - very high
        rvol=1.5,
    )
    
    print(f"\nTest 3: High Volatility (ATR 6%)")
    print(f"  Assessment: {result['risk_assessment']}")
    print(f"  ATR%: {result['atr_percent']}%")
    assert result["risk_assessment"] == "DANGEROUS", "High volatility = dangerous"
    print("  ✅ PASSED")
    
    print("\n✅ RiskManagerAgentLocal: All tests passed!")


def test_news_analyst():
    """Test TechnicalSentimentAnalystLocal (fka NewsAnalystAgentLocal)."""
    print("\n" + "=" * 50)
    print("TEST: TechnicalSentimentAnalystLocal")
    print("=" * 50)
    
    from src.agents.specialists.news_analyst_local import TechnicalSentimentAnalystLocal
    
    na = TechnicalSentimentAnalystLocal()
    
    # Test 1: Bullish signals (momentum interpretation)
    result = na.analyze(
        market_context="Tech rally continues, AI stocks surge",
        current_date="Monday",
        symbol="NVDA",
        rsi=72,  # High RSI
        macd=1.5,  # Bullish
        interpretation="momentum",  # High RSI = bullish
    )
    
    print(f"\nTest 1: Momentum Interpretation (RSI=72)")
    print(f"  Score: {result['sentiment_score']}")
    print(f"  Label: {result['sentiment_label']}")
    print(f"  Mode: {result['interpretation_mode']}")
    assert result["sentiment_label"] == "Bullish", "Momentum: High RSI should be bullish"
    print("  ✅ PASSED")
    
    # Test 2: Same RSI but reversal interpretation
    result = na.analyze(
        market_context="Tech stocks",
        current_date="Tuesday",
        symbol="NVDA",
        rsi=72,  # High RSI
        macd=0,  # Neutral
        interpretation="reversal",  # High RSI = bearish (overbought)
    )
    
    print(f"\nTest 2: Reversal Interpretation (RSI=72)")
    print(f"  Score: {result['sentiment_score']}")
    print(f"  Label: {result['sentiment_label']}")
    print(f"  Mode: {result['interpretation_mode']}")
    assert result["sentiment_label"] == "Bearish", "Reversal: High RSI should be bearish"
    print("  ✅ PASSED")
    
    # Test 3: Strong veto signal
    result = na.analyze(
        market_context="Market crash, prices plunge",
        current_date="Friday",
        symbol="SPY",
        rsi=25,
        macd=-2.0,
        interpretation="momentum",
    )
    
    print(f"\nTest 3: Bearish Veto Signal")
    print(f"  Score: {result['sentiment_score']}")
    print(f"  Veto: {result['is_veto']}")
    assert result["is_veto"] == True, "Strong negative should trigger veto"
    print("  ✅ PASSED")
    
    print("\n✅ TechnicalSentimentAnalystLocal: All tests passed!")


def test_indicators_extended():
    """Test RVOL and Bollinger Bands."""
    print("\n" + "=" * 50)
    print("TEST: Extended Indicators (RVOL, Bollinger)")
    print("=" * 50)
    
    from src.mcp_server.indicators_extended import calculate_rvol, calculate_bollinger
    
    # Test RVOL
    volumes = [100_000] * 20 + [200_000]  # Current = 2x average
    result = calculate_rvol(volumes)
    
    print(f"\nTest RVOL:")
    print(f"  Current: {result.current_volume:,}")
    print(f"  Average: {result.average_volume:,}")
    print(f"  RVOL: {result.rvol}")
    print(f"  Significant: {result.is_significant}")
    assert result.rvol == 2.0, "RVOL should be 2.0"
    assert result.is_significant == True, "2x volume is significant"
    print("  ✅ PASSED")
    
    # Test Bollinger
    prices = [100 + i * 0.5 for i in range(25)]
    result = calculate_bollinger(prices)
    
    print(f"\nTest Bollinger Bands:")
    print(f"  Upper: ${result.upper:.2f}")
    print(f"  Middle: ${result.middle:.2f}")
    print(f"  Lower: ${result.lower:.2f}")
    print(f"  %B: {result.percent_b:.2f}")
    assert result.upper > result.middle > result.lower, "Bands should be ordered"
    print("  ✅ PASSED")
    
    print("\n✅ Extended Indicators: All tests passed!")


if __name__ == "__main__":
    print("=" * 50)
    print("  LOCAL AGENTS & INDICATORS TEST SUITE")
    print("=" * 50)
    
    try:
        test_risk_manager()
        test_news_analyst()
        test_indicators_extended()
        
        print("\n" + "=" * 50)
        print("  ✅ ALL TESTS PASSED!")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

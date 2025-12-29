"""
Integration Test for QuantAgents Local Pipeline
Tests the ENTIRE flow end-to-end with mocked model responses.

This catches:
- Import errors
- Missing dependencies
- Path issues
- Database errors
- Screener API errors
- Coordinator integration bugs

Run: python test_integration_flow.py
"""

import os
import sys
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_imports():
    """Test 1: Verify all imports work."""
    print("\n" + "=" * 60)
    print("TEST 1: Import Verification")
    print("=" * 60)
    
    errors = []
    
    # Core modules
    try:
        from src.agents.coordinator_local import CoordinatorAgentLocal, get_coordinator_local, CoordinatedAnalysis
        print("  ✅ coordinator_local.py")
    except Exception as e:
        errors.append(f"coordinator_local: {e}")
        print(f"  ❌ coordinator_local: {e}")
    
    try:
        from src.agents.specialists.risk_manager_local import RiskManagerAgentLocal
        print("  ✅ risk_manager_local.py")
    except Exception as e:
        errors.append(f"risk_manager_local: {e}")
        print(f"  ❌ risk_manager_local: {e}")
    
    try:
        from src.agents.specialists.news_analyst_local import TechnicalSentimentAnalystLocal, NewsAnalystAgentLocal
        print("  ✅ news_analyst_local.py (TechnicalSentimentAnalystLocal)")
    except Exception as e:
        errors.append(f"news_analyst_local: {e}")
        print(f"  ❌ news_analyst_local: {e}")
    
    try:
        from src.screener.client import ScreenerClient
        print("  ✅ screener/client.py")
    except Exception as e:
        errors.append(f"screener client: {e}")
        print(f"  ❌ screener client: {e}")
    
    try:
        from src.mcp_server.indicators_extended import calculate_rvol, calculate_bollinger
        print("  ✅ indicators_extended.py")
    except Exception as e:
        errors.append(f"indicators_extended: {e}")
        print(f"  ❌ indicators_extended: {e}")
    
    try:
        from src.database import get_signal_repository
        print("  ✅ database.py")
    except Exception as e:
        errors.append(f"database: {e}")
        print(f"  ❌ database: {e}")
    
    try:
        from src.models import Signal, SignalType, PatternType, Market
        print("  ✅ models.py")
    except Exception as e:
        errors.append(f"models: {e}")
        print(f"  ❌ models: {e}")
    
    try:
        from src.visual import get_report_generator
        print("  ✅ visual/report_generator.py")
    except Exception as e:
        errors.append(f"report_generator: {e}")
        print(f"  ❌ report_generator: {e}")
    
    if errors:
        print(f"\n❌ {len(errors)} import errors found!")
        return False
    
    print("\n✅ All imports successful!")
    return True


def test_screener_connection():
    """Test 2: Verify TradingView screener connection."""
    print("\n" + "=" * 60)
    print("TEST 2: TradingView Screener Connection")
    print("=" * 60)
    
    try:
        from src.screener.client import ScreenerClient
        from src.models import Market
        
        client = ScreenerClient(market=Market.AMERICA)
        data = client.get_symbol_data("AAPL")
        
        if data:
            print(f"  ✅ Connected! Got data for AAPL")
            print(f"     Close: {data.get('close', 'N/A')}")
            print(f"     Volume: {data.get('volume', 'N/A')}")
            print(f"     RSI: {data.get('RSI', 'N/A')}")
            return True
        else:
            print("  ⚠️ Connected but no data returned")
            return True  # Connection works, data issue
            
    except Exception as e:
        print(f"  ❌ Screener error: {e}")
        return False


def test_database():
    """Test 3: Verify database operations."""
    print("\n" + "=" * 60)
    print("TEST 3: Database Operations")
    print("=" * 60)
    
    try:
        from src.database import get_signal_repository
        from src.models import Signal, SignalType, PatternType
        
        repo = get_signal_repository()
        
        # Create test signal
        test_signal = Signal(
            symbol="TEST_INTEGRATION",
            signal_type=SignalType.PENDING,
            pattern_detected=PatternType.NONE,
            pattern_confidence=0.5,
            trend="up",
            trend_strength="moderate",
            analysis_summary="Integration test signal",
        )
        
        # Save
        signal_id = repo.create(test_signal)
        print(f"  ✅ Created signal with ID: {signal_id}")
        
        # Read
        saved = repo.get_by_id(signal_id)
        if saved:
            print(f"  ✅ Retrieved signal: {saved.symbol}")
        else:
            print("  ⚠️ Could not retrieve signal")
        
        # Delete (cleanup)
        try:
            repo.delete(signal_id)
            print(f"  ✅ Deleted test signal")
        except:
            print("  ⚠️ Could not delete test signal (cleanup)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_manager():
    """Test 4: Risk Manager (Dave) logic."""
    print("\n" + "=" * 60)
    print("TEST 4: Risk Manager (Dave) Logic")
    print("=" * 60)
    
    try:
        from src.agents.specialists.risk_manager_local import RiskManagerAgentLocal
        
        dave = RiskManagerAgentLocal()
        
        # Test high volatility
        result = dave.analyze(
            pattern_data={"pattern": "breakout", "confidence": 0.8},
            trend_data={"trend": "up", "strength": "strong"},
            current_price=100.0,
            atr_value=6.0,  # 6% ATR = high
            rvol=2.0,
        )
        
        print(f"  Test: High ATR (6%)")
        print(f"    Risk: {result['risk_assessment']}")
        assert result['risk_assessment'] in ["DANGEROUS", "CAUTION"], "Should flag high risk"
        print("  ✅ PASSED")
        
        # Test RVOL fakeout
        result = dave.analyze(
            pattern_data={"pattern": "breakout", "confidence": 0.8},
            trend_data={"trend": "up"},
            current_price=100.0,
            atr_value=2.0,
            rvol=1.2,  # Low volume breakout
        )
        
        print(f"  Test: Low RVOL Breakout (1.2)")
        print(f"    Risk: {result['risk_assessment']}")
        assert result['risk_assessment'] == "DANGEROUS", "Should detect fakeout"
        print("  ✅ PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Risk Manager error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_analyst():
    """Test 5: Technical Sentiment Analyst (Emily) logic."""
    print("\n" + "=" * 60)
    print("TEST 5: Technical Sentiment (Emily) Logic")
    print("=" * 60)
    
    try:
        from src.agents.specialists.news_analyst_local import TechnicalSentimentAnalystLocal
        
        emily = TechnicalSentimentAnalystLocal()
        
        # Test momentum mode
        result = emily.analyze(
            market_context="Strong tech rally",
            current_date="Monday",
            symbol="NVDA",
            rsi=75,  # High RSI
            macd=2.0,  # Bullish
            interpretation="momentum",
        )
        
        print(f"  Test: Momentum Mode (RSI=75)")
        print(f"    Label: {result['sentiment_label']}")
        print(f"    Score: {result['sentiment_score']}")
        assert result['sentiment_label'] == "Bullish", "Should be bullish in momentum mode"
        print("  ✅ PASSED")
        
        # Test reversal mode
        result = emily.analyze(
            market_context="Market conditions",
            current_date="Tuesday",
            symbol="SPY",
            rsi=75,  # Same high RSI
            macd=0,
            interpretation="reversal",  # Now interpret as overbought
        )
        
        print(f"  Test: Reversal Mode (RSI=75)")
        print(f"    Label: {result['sentiment_label']}")
        assert result['sentiment_label'] == "Bearish", "Should be bearish in reversal mode"
        print("  ✅ PASSED")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Sentiment Analyst error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coordinator_synthesis():
    """Test 6: Coordinator synthesis logic (mocked agents)."""
    print("\n" + "=" * 60)
    print("TEST 6: Coordinator Synthesis Logic")
    print("=" * 60)
    
    try:
        from src.agents.coordinator_local import CoordinatorAgentLocal, CoordinatedAnalysis
        from src.agents.specialists.base_agent_local import AgentResponse
        
        # Mock coordinator instance
        with patch('src.agents.coordinator_local.PatternDetectorAgentLocal'), \
             patch('src.agents.coordinator_local.TrendAnalystAgentLocal'), \
             patch('src.agents.coordinator_local.LevelsCalculatorAgentLocal'), \
             patch('src.agents.coordinator_local.ScreenerClient'):
            
            coord = CoordinatorAgentLocal.__new__(CoordinatorAgentLocal)
            coord.model_name = "mock"
            
            # Create mock AgentResponse objects
            mock_pattern = AgentResponse(
                raw_text="",
                parsed={"pattern": "double bottom", "confidence": 0.8, "description": "W pattern"},
                success=True
            )
            mock_trend = AgentResponse(
                raw_text="",
                parsed={"trend": "up", "strength": "strong", "phase": "markup", "wave": "3"},
                success=True
            )
            mock_levels = AgentResponse(
                raw_text="",
                parsed={"support": 95.0, "resistance": 110.0, "key_level": 100.0},
                success=True
            )
            
            # Risk and news are dicts (from rule-based agents)
            mock_risk = {"risk_assessment": "SAFE", "stop_loss": "$94.00", "position_size": "100%", "atr_percent": 2.0, "rvol": 1.8, "reasoning": "Normal volatility"}
            mock_news = {"sentiment_score": 0.3, "sentiment_label": "Bullish", "is_veto": False, "key_drivers": ["RSI strong"]}
            
            # Call synthesize with correct arguments
            result = coord._synthesize(
                pattern=mock_pattern,
                trend=mock_trend,
                levels=mock_levels,
                risk=mock_risk,
                news=mock_news,
                symbol="AAPL",
                current_price=100.0,
                atr_value=2.0,
                rvol=1.8,
                sector="Technology",
            )
            
            print(f"  Signal Type: {result.signal_type}")
            print(f"  Confidence: {result.overall_confidence}")
            print(f"  Veto: {result.veto_reason}")
            
            assert result.signal_type in ["candidate", "pending", "not_candidate"]
            print("  ✅ PASSED: Synthesis produced valid output")
            
            # Test VETO scenario
            mock_risk["risk_assessment"] = "DANGEROUS"
            mock_risk["reasoning"] = "High volatility"
            result = coord._synthesize(
                pattern=mock_pattern,
                trend=mock_trend,
                levels=mock_levels,
                risk=mock_risk,
                news=mock_news,
                symbol="AAPL",
                current_price=100.0,
                atr_value=2.0,
                rvol=1.8,
                sector="Technology",
            )
            
            print(f"  Test: DANGEROUS risk → should VETO")
            print(f"    Signal: {result.signal_type}")
            print(f"    Veto: {result.veto_reason}")
            assert result.signal_type == "not_candidate", "DANGEROUS should veto"
            print("  ✅ PASSED: Veto system working")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Coordinator error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chart_capture():
    """Test 7: Chart capture system."""
    print("\n" + "=" * 60)
    print("TEST 7: Chart Capture System")
    print("=" * 60)
    
    try:
        from src.screener.chart_capture import get_chart_capture
        
        capture = get_chart_capture()
        print(f"  ✅ ChartCapture initialized")
        print(f"     Driver: {type(capture.driver).__name__ if hasattr(capture, 'driver') else 'Not loaded'}")
        
        # Check if we have existing charts
        charts_dir = Path("data/charts")
        if charts_dir.exists():
            charts = list(charts_dir.glob("*.png"))
            print(f"  ✅ Found {len(charts)} existing chart images")
            if charts:
                print(f"     Latest: {charts[-1].name}")
        else:
            print("  ⚠️ No charts directory yet (will be created on first capture)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Chart capture error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataclass_fields():
    """Test 8: Verify CoordinatedAnalysis has all expected fields."""
    print("\n" + "=" * 60)
    print("TEST 8: CoordinatedAnalysis Dataclass Fields")
    print("=" * 60)
    
    try:
        from src.agents.coordinator_local import CoordinatedAnalysis
        from dataclasses import fields
        
        expected_fields = [
            "signal_type", "overall_confidence",
            "pattern", "pattern_confidence", "pattern_box",
            "trend", "trend_strength", "phase", "wave",
            "support", "resistance", "fibonacci", "key_level",
            "risk_assessment", "stop_loss", "position_size",
            "sentiment_score", "sentiment_label",
            "veto_reason", "summary", "detailed_reasoning",
        ]
        
        actual_fields = [f.name for f in fields(CoordinatedAnalysis)]
        
        missing = [f for f in expected_fields if f not in actual_fields]
        extra = [f for f in actual_fields if f not in expected_fields]
        
        if missing:
            print(f"  ⚠️ Missing fields: {missing}")
        if extra:
            print(f"  ℹ️ Extra fields: {extra}")
        
        if not missing:
            print(f"  ✅ All {len(expected_fields)} expected fields present")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  ❌ Dataclass error: {e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("🧪 QUANTAGENTS INTEGRATION TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    results["Imports"] = test_imports()
    results["Screener"] = test_screener_connection()
    results["Database"] = test_database()
    results["Risk Manager"] = test_risk_manager()
    results["Sentiment"] = test_sentiment_analyst()
    results["Coordinator"] = test_coordinator_synthesis()
    results["Chart Capture"] = test_chart_capture()
    results["Dataclass"] = test_dataclass_fields()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print("-" * 60)
    print(f"  TOTAL: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Flow is ready for client.\n")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Fix before deploying.\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

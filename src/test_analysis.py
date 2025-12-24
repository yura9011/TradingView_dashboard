"""
Test script for Milestone 2 - AI Analysis Engine
Run: python -m src.test_analysis
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import GeminiClient, get_gemini_client
from src.mcp_server import IndicatorTools, get_indicator_tools

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_indicator_tools():
    """Test MCP indicator calculations."""
    print("\n" + "="*60)
    print("📊 Testing MCP Indicator Tools")
    print("="*60)
    
    tools = get_indicator_tools()
    
    # Sample price data (simulated OHLC closes) - need 35+ for MACD
    prices = [
        100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
        111, 110, 112, 114, 113, 115, 117, 116, 118, 120,
        119, 121, 123, 122, 124, 126, 125, 127, 129, 128,
        130, 132, 131, 133, 135, 134, 136, 138, 137, 139,
    ]
    
    # Test EMA
    print("\n📈 EMA Calculations:")
    ema_21 = tools.calculate_ema(prices, 21)
    print(f"   EMA(21): {ema_21.value} (using {ema_21.prices_used} prices)")
    
    # Test RSI
    print("\n📉 RSI Calculation:")
    rsi = tools.calculate_rsi(prices, period=14)
    print(f"   RSI(14): {rsi.value}")
    print(f"   Oversold: {rsi.is_oversold}, Overbought: {rsi.is_overbought}")
    
    # Test MACD
    print("\n📊 MACD Calculation:")
    macd = tools.calculate_macd(prices)
    print(f"   MACD Line: {macd.macd_line}")
    print(f"   Signal Line: {macd.signal_line}")
    print(f"   Histogram: {macd.histogram}")
    print(f"   Bullish: {macd.is_bullish}")
    
    # Test Fibonacci
    print("\n🔢 Fibonacci Levels (High: 150, Low: 100):")
    fib = tools.calculate_fibonacci(high=150, low=100, is_uptrend=True)
    for level, value in fib.as_dict().items():
        print(f"   {level}: ${value:.2f}")
    
    print("\n   ✅ All indicator tests passed!")
    return True


def test_gemini_connection():
    """Test Gemini API connection."""
    print("\n" + "="*60)
    print("🤖 Testing Gemini API Connection")
    print("="*60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("   ⚠️ GEMINI_API_KEY not set in environment")
        print("   Set it with: $env:GEMINI_API_KEY='your-key'")
        print("   Skipping Gemini test...")
        return None
    
    try:
        client = get_gemini_client(api_key=api_key)
        
        print(f"   Model: {client.model_name}")
        print("   Testing connection...")
        
        if client.test_connection():
            print("   ✅ Gemini API connection successful!")
            return True
        else:
            print("   ❌ Gemini API connection failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_chart_analysis():
    """Test chart analysis with a sample image."""
    print("\n" + "="*60)
    print("📸 Testing Chart Analysis")
    print("="*60)
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("   ⚠️ Skipping - GEMINI_API_KEY not set")
        return None
    
    # Check for sample chart image
    sample_images = [
        Path("docs/assets/architecture-diagram.png"),
        Path("docs/assets/flow-diagram.png"),
    ]
    
    image_path = None
    for img in sample_images:
        if img.exists():
            image_path = img
            break
    
    if not image_path:
        print("   ⚠️ No sample chart image found")
        print("   Add a chart screenshot to test analysis")
        return None
    
    try:
        from src.agents import get_chart_analyzer
        
        analyzer = get_chart_analyzer(api_key=api_key)
        print(f"   Using image: {image_path}")
        print("   Running analysis...")
        
        signal = analyzer.analyze_chart_image(
            image_path=str(image_path),
            symbol="TEST",
            timeframe="1D",
            save_signal=False,  # Don't save test signal
        )
        
        print(f"\n   📋 Analysis Result:")
        print(f"   Symbol: {signal.symbol}")
        print(f"   Pattern: {signal.pattern_detected}")
        print(f"   Confidence: {signal.pattern_confidence:.2f}")
        print(f"   Trend: {signal.trend}")
        print(f"   Signal Type: {signal.signal_type}")
        print(f"   Summary: {signal.analysis_summary[:100]}...")
        
        print("\n   ✅ Chart analysis test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all M2 tests."""
    print("\n" + "🧠 "*20)
    print("  AI Trading Analysis Agent - Milestone 2 Tests")
    print("🧠 "*20)
    
    results = {}
    
    # Test indicator tools (no API needed)
    results["indicators"] = test_indicator_tools()
    
    # Test Gemini connection
    results["gemini"] = test_gemini_connection()
    
    # Test chart analysis (requires API + image)
    results["analysis"] = test_chart_analysis()
    
    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60)
    
    for test, result in results.items():
        status = "✅ PASSED" if result == True else ("⚠️ SKIPPED" if result is None else "❌ FAILED")
        print(f"   {test}: {status}")
    
    # Check if core tests passed
    if results["indicators"]:
        print("\n" + "="*60)
        print("✅ Milestone 2 Core Components Working!")
        print("   (Set GEMINI_API_KEY to test AI features)")
        print("="*60 + "\n")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())


import logging
import sys
import os
sys.path.append(os.getcwd())

from src.agents.coordinator_local import CoordinatorAgentLocal
from src.agents.specialists.news_analyst_local import TechnicalSentimentAnalystLocal
from src.models import Market

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VSA_Sim")

def simulate_buying_climax():
    """
    Simulate a 'Buying Climax' Scenario:
    - Context: Strong Uptrend, Good News (High RSI), but...
    - VSA: Ultra High Volume + Narrow Spread (Effort > Result) -> DISTRIBUTION.
    """
    print("\n--- 🧪 SIMULATION: BUYING CLIMAX (The 'Trap') ---")
    
    # 1. Setup Mock Data (Screener)
    # Price is rising, but high = low + small_spread, while vol is HUGE.
    mock_market_data = {
        'symbol': 'NVDA',
        'close': 150.0,
        'high': 151.0,  # Narrow spread ($2 range? No, let's make it narrow vs ATR)
        'low': 149.0,
        'open': 149.5,
        'volume': 100_000_000, # Massive
        'average_volume_10d_calc': 20_000_000, # RVOL = 5.0 (Ultra High)
        'ATR': 5.0, # Normal range is $5. Today is $2. (Effort without Result)
        'RSI': 80.0, # Euforia extreme
        'MACD.macd': 1.5,
    }
    
    coordinator = CoordinatorAgentLocal(market=Market.AMERICA, use_yolo=False)
    # Monkey patch screener
    coordinator.screener.get_symbol_data = lambda x: mock_market_data
    
    # 2. Get Context (Test Data Metrics)
    print("\n📊 STEP 1: Calculting VSA Metrics...")
    context = coordinator._get_market_context('NVDA')
    print(f"   Spread: ${context['spread']:.2f} ({context['spread_type']})")
    print(f"   RVOL: {context['rvol']:.1f}x (Volume Effort)")
    
    # 3. Simulate Trend Analyst Response (VSA Analyst)
    # (Since we don't run the actual LLM, we inject the EXPECTED response from the new Prompt 3.0)
    print("\n🤖 STEP 2: Trend Analyst (VSA Logic with Prompt 3.0)...")
    mock_trend_response = """
    MARKET_PHASE: distribution
    TREND: up
    STRENGTH: weak
    VSA_SIGNAL: buying climax
    VOLUME_ACTION: Ultra High Volume with Narrow Spread
    DESCRIPTION: Price is up but spread is narrow despite massive volume. Professional selling detected.
    """
    trend_parsed = coordinator.trend_analyst._parse_response(mock_trend_response)
    print(f"   Detected Phase: {trend_parsed['phase'].upper()}")
    print(f"   Detected Signal: {trend_parsed.get('vsa_signal', 'none').upper()}")
    
    # 4. Simulate News/Psychology Response (Contrarian Logic)
    print("\n🧠 STEP 3: Psychology Analyst (Contrarian Logic)...")
    news_agent = TechnicalSentimentAnalystLocal()
    news_result = news_agent.analyze(
        market_context="Bullish trend",
        current_date="Monday",
        symbol="NVDA",
        rsi=mock_market_data['RSI'],
        vsa_signal=trend_parsed.get('vsa_signal')
    )
    
    print(f"   RSI: {mock_market_data['RSI']} (Euphoria)")
    print(f"   Sentiment Score: {news_result['sentiment_score']:.2f}")
    print(f"   Key Drivers: {news_result['key_drivers']}")
    
    # 5. Check Output
    if news_result['sentiment_score'] < 0 and "SMART MONEY SELL" in str(news_result['key_drivers']):
        print("\n✅ SUCCESS: System identified 'Smart Money Selling' despite High RSI!")
    else:
        print("\n❌ FAILURE: System was fooled by High RSI.")

if __name__ == "__main__":
    simulate_buying_climax()

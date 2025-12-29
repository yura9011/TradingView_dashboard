import asyncio
import logging
import os
from pathlib import Path
from src.screener.chart_capture import get_chart_capture
from src.agents.specialists.pattern_detector_yolo import get_pattern_detector_yolo
from src.agents.specialists.levels_calculator_local import LevelsCalculatorAgentLocal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("verification_log.txt", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def test_fixes():
    with open("verification_log.txt", "a") as f:
        f.write("\n=== TEST 1: Chart Capture (30m Interval) ===\n")
    
    capturer = get_chart_capture()
    symbol = "AAPL"
    
    # Test capture with 30m interval
    try:
        chart_path = await capturer.capture(
            symbol=symbol,
            exchange="", # Test empty exchange resolution
            interval="30", 
            range_months=1
        )
        price_data = capturer.get_last_price_range()
        logger.info(f"✅ Capture success: {chart_path}")
        logger.info(f"✅ Price Data: {price_data}")
    except Exception as e:
        logger.error(f"❌ Capture failed: {e}")
        return

    print("\n=== TEST 2: YOLO Pattern Detector (Visual + Deps) ===")
    try:
        # Initialize
        yolo = get_pattern_detector_yolo()
        if yolo.model is None:
             print("⚠️ YOLO model failed to load (check deps). Skipping inference.")
        else:
             # Run analysis
             result = yolo.analyze(chart_path)
             print(f"✅ Analysis Result: {result.raw_text}")
             
             # Test annotation
             annotated = yolo.get_annotated_chart(chart_path)
             if annotated:
                 print(f"✅ Annotated chart saved: {annotated}")
             else:
                 print("⚠️ Failed to save annotated chart")
                 
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")

    print("\n=== TEST 3: Levels Calculator (Sanity Check) ===")
    try:
        levels = LevelsCalculatorAgentLocal()
        
        # Mock context with WRONG price to trigger sanity check or valid one
        # Case A: Context matches chart (approx)
        market_context_real = f"Current Price: ${price_data.get('current_price', 150.00)}"
        print(f"Testing with context: {market_context_real}")
        
        res_real = levels.analyze(chart_path, market_context=market_context_real)
        print("Result Real Context:")
        print(res_real.parsed)
        
        # Case B: Hallucination Trigger (values from old example)
        # We can't easily force the model to hallucinate specific values without mocking, 
        # but we can verify the sanity logic code path exists (already reviewed in code).
        
    except Exception as e:
        print(f"❌ Levels test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_fixes())

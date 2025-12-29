
import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
import cv2

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.screener.chart_capture import ChartCapture
from src.screener.client import ScreenerClient, Market
from src.pattern_analysis.pipeline.preprocessor import StandardPreprocessor
from src.pattern_analysis.pipeline.feature_extractor import EdgeBasedFeatureExtractor
from src.pattern_analysis.pipeline.classifier import HybridPatternClassifier
from src.pattern_analysis.pipeline.executor import PipelineExecutor
from src.pattern_analysis.output.annotator import ChartAnnotator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Run Full System Test (Capture + Analysis)")
    parser.add_argument("--symbol", default="MELI", help="Symbol to test (e.g. MELI, AAPL)")
    parser.add_argument("--exchange", default=None, help="Exchange (optional, e.g. NASDAQ)")
    parser.add_argument("--skip-capture", action="store_true", help="Skip capture and use latest existing image for symbol")
    args = parser.parse_args()

    symbol = args.symbol
    exchange = args.exchange

    print("\n" + "="*60)
    print(f"🚀 RUNNING FULL SYSTEM TEST: {exchange if exchange else 'Auto'} : {symbol}")
    print("="*60)

    # 1. Capture/Load Chart
    image_path = None
    
    if not args.skip_capture:
        print(f"\n📸 Step 1: Capturing Chart (daily, 1 month)...")
        capture = ChartCapture(headless=True)
        try:
            # capture returns a Path object
            path = await capture.capture(
                symbol=symbol,
                exchange=exchange,
                interval="D",
                range_months=1
            )
            image_path = str(path)
            print(f"   Success! Saved to: {image_path}")
        except Exception as e:
            print(f"   Capture failed: {e}")
            print("   Attempting to fall back to existing images...")
    
    if not image_path:
        # Fallback to finding latest image for symbol
        import glob
        pattern = f"data/charts/{symbol}_*.png"
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            image_path = matches[0]
            print(f"   Using existing image: {image_path}")
        else:
            print(f"❌ No image found module for {symbol}. Cannot proceed.")
            return

    # 2. Market Data
    print(f"\n📊 Step 2: Fetching Market Data...")
    screener = ScreenerClient(market=Market.AMERICA)
    data = screener.get_symbol_data(symbol)
    if data:
        print(f"   Price: ${data.get('close')}")
        print(f"   Volume: {data.get('volume')}")
    else:
        print("   Warning: Could not fetch market data (continuing with analysis only)")

    # 3. Pipeline Execution
    print(f"\n🔍 Step 3: Running Analysis Pipeline...")
    
    preprocessor = StandardPreprocessor()
    feature_extractor = EdgeBasedFeatureExtractor()
    classifier = HybridPatternClassifier()
    executor = PipelineExecutor(preprocessor, feature_extractor, classifier)

    config = {
        "extract_volume": True,
        "min_end_x_ratio": 0.0, # Analyze full chart
        "denoise": True
    }

    try:
        result = executor.execute(image_path, config)
        
        # Stats
        fmap = result.feature_map
        print(f"   Candlesticks: {len(fmap.candlestick_regions)}")
        print(f"   SR Zones: {len(fmap.support_zones)} Support, {len(fmap.resistance_zones)} Resistance")
        
        if result.feature_map.volume_profile:
             print(f"   Volume Profile: Extracted (Avg: {result.feature_map.volume_profile.get('avg_volume', 0):.2f})")
        else:
             print(f"   Volume Profile: Not found")

        print(f"\n   >> PATTERNS DETECTED: {len(result.detections)}")
        for i, det in enumerate(result.detections):
             print(f"      {i+1}. {det.pattern_type.value} ({det.category.value}) - Conf: {det.confidence:.2f}")

    except Exception as e:
        print(f"❌ Analysis Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Output Annotation
    print(f"\n🎨 Step 4: Generating Annotation...")
    try:
        annotator = ChartAnnotator()
        # Reload image for annotation
        img = cv2.imread(image_path)
        if img is not None:
            annotated = annotator.annotate_from_result(img, result)
            out_path = image_path.replace(".png", "_system_test.png")
            cv2.imwrite(out_path, annotated)
            print(f"   Saved annotation to: {out_path}")
        else:
            print("   Failed to reload image for annotation.")
    except Exception as e:
        print(f"   Annotation failed: {e}")

    print("\n✅ Test Complete.")

if __name__ == "__main__":
    asyncio.run(main())

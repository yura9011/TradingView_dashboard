
import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pattern_analysis.pipeline.feature_extractor import EdgeBasedFeatureExtractor
from src.pattern_analysis.models.dataclasses import FeatureMap, PreprocessResult, BoundingBox

def create_synthetic_blue_white_chart(width=800, height=600):
    """Creates a chart with Blue (Bullish) and White (Bearish) candles and Volume."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (20, 20, 20) # Dark background
    
    # Colors (BGR)
    BLUE = (255, 0, 0)   # Blue
    WHITE = (255, 255, 255) # White
    GRAY = (100, 100, 100) # Volume
    
    # Draw some candles
    candlestick_count = 0
    for i in range(10, width - 20, 30):
        # Random height
        h = np.random.randint(20, 100)
        y_start = np.random.randint(100, height - 200)
        
        # Alternate colors
        color = BLUE if (i // 30) % 2 == 0 else WHITE
        
        # Wick
        cv2.line(img, (i + 10, y_start - 20), (i + 10, y_start + h + 20), (200, 200, 200), 1)
        # Body
        cv2.rectangle(img, (i, y_start), (i + 20, y_start + h), color, -1)
        candlestick_count += 1
        
        # Draw Volume bar at bottom
        vol_h = np.random.randint(10, 80)
        cv2.rectangle(img, (i, height - vol_h), (i + 20, height), GRAY, -1)
        
    print(f"Generated {candlestick_count} candles.")
    return img

def create_preprocess_result(image):
    return PreprocessResult(
        image=image,
        original_size=image.shape[:2],
        processed_size=image.shape[:2],
        transformations=[],
        quality_score=1.0,
        masked_regions=[]
    )

def test_extraction():
    extractor = EdgeBasedFeatureExtractor()
    
    # 2. Test Blue/White Chart
    print("\n--- Testing Synthetic Blue/White Chart ---")
    img_synth = create_synthetic_blue_white_chart()
    # Convert to RGB because pipeline expects RGB
    img_synth_rgb = cv2.cvtColor(img_synth, cv2.COLOR_BGR2RGB)
    
    try:
        input_data_synth = create_preprocess_result(img_synth_rgb)
        
        # Test Candle Extraction
        config = {
            "extract_volume": True,
            "volume_region_ratio": 0.25
        }
        features_synth = extractor.process(input_data_synth, config)
        
        candle_count = len(features_synth.candlestick_regions)
        print(f"Candles detected: {candle_count}")
        
        if candle_count > 10:
            print("SUCCESS: Detected sufficient candles using dynamic color clustering.")
            # Check metadata
            sample_candle = features_synth.candlestick_regions[0]
            print(f"Sample Candle Metadata: {sample_candle.metadata}")
            if "direction" in sample_candle.metadata:
                 print("SUCCESS: Direction metadata present.")
            else:
                 print("FAILURE: Direction metadata missing.")
        else:
            print("FAILURE: Failed to detect candles with non-standard colors.")
            exit(1)
            
        # Test Volume Extraction
        vol_profile = features_synth.volume_profile
        if vol_profile:
            print("SUCCESS: Volume profile extracted.")
            print(f"Avg Volume: {vol_profile.get('avg_volume')}")
            region = vol_profile.get("region")
            if region:
                print(f"Volume region: y={region.y1} to {region.y2}")
        else:
            print("FAILURE: Volume profile NOT extracted.")
            # Depending on config, this might be failure.
            exit(1)
            
    except Exception as e:
        print(f"Extraction error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    test_extraction()

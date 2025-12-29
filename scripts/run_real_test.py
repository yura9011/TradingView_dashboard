
import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pattern_analysis.pipeline.preprocessor import StandardPreprocessor
from src.pattern_analysis.pipeline.feature_extractor import EdgeBasedFeatureExtractor
from src.pattern_analysis.pipeline.classifier import HybridPatternClassifier
from src.pattern_analysis.pipeline.executor import PipelineExecutor
from src.pattern_analysis.models.dataclasses import AnalysisResult

def main():
    # Setup paths
    image_path = r"d:\tareas\tradingview\data\charts\AAPL_20251226_103218.png"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        # Try another one if that fails usually
        image_path = r"d:\tareas\tradingview\data\charts\MELI_20251224_090037.png"
        if not os.path.exists(image_path):
             print(f"Backup image not found either.")
             return

    print(f"Running pipeline on: {image_path}")

    # Initialize components
    preprocessor = StandardPreprocessor()
    feature_extractor = EdgeBasedFeatureExtractor()
    classifier = HybridPatternClassifier()
    # No cross-validator for this simple run, or we could add one mock/real if available
    
    executor = PipelineExecutor(
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        classifier=classifier
    )

    # Config
    config = {
        "extract_volume": True,
        "volume_region_ratio": 0.2, # Hints
        "min_end_x_ratio": 0.0, # Analyze full chart
        "denoise": True
    }

    # Execute
    try:
        result = executor.execute(image_path, config)
        
        # Print Summary
        print("\n--- Analysis Result ---")
        print(f"Time: {result.total_time_ms:.2f} ms")
        print(f"Preprocess: {result.preprocessing_time_ms:.2f} ms")
        print(f"Feature Ext: {result.extraction_time_ms:.2f} ms")
        print(f"Classify:   {result.classification_time_ms:.2f} ms")
        
        print(f"\nDetector Status: {result.detector_status}")
        
        print(f"\nDetected {len(result.detections)} patterns:")
        for i, det in enumerate(result.detections):
            print(f"  {i+1}. {det.pattern_type.value} ({det.category.value}) - Conf: {det.confidence:.2f}")
            print(f"     BBox: {det.bounding_box}")
            print(f"     Metadata: {det.metadata}")

        if result.feature_map.volume_profile:
            vol = result.feature_map.volume_profile
            print(f"\nVolume Profile Extracted:")
            print(f"  Region: y={vol['region'].y1}-{vol['region'].y2}")
            print(f"  Avg Volume: {vol['avg_volume']:.2f}")
        else:
            print("\nVolume Profile: NOT extracted")
            
        # Check Candles
        candles = result.feature_map.candlestick_regions
        print(f"\nCandlesticks detected: {len(candles)}")
        if candles:
            print(f"  First candle: {candles[0].metadata}")
            print(f"  Last candle: {candles[-1].metadata}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

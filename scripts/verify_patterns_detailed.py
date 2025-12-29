
import os
import sys
import glob

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pattern_analysis.pipeline.preprocessor import StandardPreprocessor
from src.pattern_analysis.pipeline.feature_extractor import EdgeBasedFeatureExtractor
from src.pattern_analysis.pipeline.classifier import HybridPatternClassifier
from src.pattern_analysis.pipeline.executor import PipelineExecutor
from src.pattern_analysis.output.annotator import ChartAnnotator
from src.pattern_analysis.models.dataclasses import AnalysisResult

def run_verification_on_file(image_path, executor, annotator):
    print(f"\n{'='*50}")
    print(f"Analyzing: {os.path.basename(image_path)}")
    print(f"{'='*50}")
    
    config = {
        "extract_volume": True,
        "min_end_x_ratio": 0.0, # Analyze full chart for visibility
        "denoise": True
    }
    
    try:
        result = executor.execute(image_path, config)
        
        # 1. Feature Stats
        fmap = result.feature_map
        print(f"Features Detected:")
        print(f"  - Candlesticks: {len(fmap.candlestick_regions)}")
        print(f"  - Trendlines: {len(fmap.trendlines)}")
        print(f"  - Support Zones: {len(fmap.support_zones)}")
        print(f"  - Resistance Zones: {len(fmap.resistance_zones)}")
        
        # 2. Support/Resistance Detail
        if fmap.support_zones:
            print("\n  [Support Zones Sample (Top 3)]")
            for i, z in enumerate(fmap.support_zones[:3]):
                print(f"    {i+1}. y={z.y1}-{z.y2} (Width: {z.x2-z.x1})")
                
        if fmap.resistance_zones:
            print("\n  [Resistance Zones Sample (Top 3)]")
            for i, z in enumerate(fmap.resistance_zones[:3]):
                print(f"    {i+1}. y={z.y1}-{z.y2} (Width: {z.x2-z.x1})")

        # 3. Pattern Detections
        print(f"\nPatterns Detected: {len(result.detections)}")
        for i, det in enumerate(result.detections):
            print(f"  {i+1}. {det.pattern_type.value} ({det.category.value})")
            print(f"     Conf: {det.confidence:.2f}")
            print(f"     BBox: {det.bounding_box}")
            # Print specific details for double patterns if present
            if "level_difference" in det.metadata:
                 print(f"     Level Diff: {det.metadata['level_difference']}")
            if "touch_1" in det.metadata:
                 print(f"     Touch 1: {det.metadata['touch_1']}")
                 print(f"     Touch 2: {det.metadata['touch_2']}")

        # 4. Annotation
        import cv2
        # Reload image for annotation (Executor doesn't return the image array in result)
        image = cv2.imread(image_path)
        if image is not None:
            # Annotator expects RGB/BGR? OpenCV uses BGR. 
            # Let's check Annotator source. It uses cv2.rectangle which works on BGR.
            # But if Annotator expects RGB (pipeline standard), we might need conversion.
            # Looking at Annotator code, it takes "image: np.ndarray" and draws with BGR colors.
            # So passing the BGR image loaded by cv2.imread should be fine if we want to save it with cv2.imwrite.
            
            output_path = image_path.replace(".png", "_debug_annotated.png")
            
            # Annotate
            annotated_image = annotator.annotate_from_result(image, result)
            
            # Save
            cv2.imwrite(output_path, annotated_image)
            print(f"\nGenerated Annotated Chart: {output_path}")
        else:
            print(f"Failed to reload image for annotation: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Setup
    preprocessor = StandardPreprocessor()
    feature_extractor = EdgeBasedFeatureExtractor()
    classifier = HybridPatternClassifier()
    executor = PipelineExecutor(preprocessor, feature_extractor, classifier)
    annotator = ChartAnnotator()
    
    # Select a few representative charts
    base_dir = r"d:\tareas\tradingview\data\charts"
    test_files = [
        "AAPL_20251226_103218.png", # Recent AAPL
        "MELI_20251224_090037.png", # MELI
        "TSLA_20251224_165052.png"  # TSLA
    ]
    
    found_any = False
    for fname in test_files:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            run_verification_on_file(path, executor, annotator)
            found_any = True
        else:
            print(f"File not found: {path}")
            
    if not found_any:
        print("No test files found! Running on whatever is in the directory...")
        all_files = glob.glob(os.path.join(base_dir, "*.png"))
        for path in all_files[:3]:
             run_verification_on_file(path, executor, annotator)

if __name__ == "__main__":
    main()

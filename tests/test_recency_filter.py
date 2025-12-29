
import unittest
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pattern_analysis.pipeline.classifier import HybridPatternClassifier
from src.pattern_analysis.models.dataclasses import FeatureMap, BoundingBox, PatternDetection
from src.pattern_analysis.models.enums import PatternType, PatternCategory

class TestRecencyFilter(unittest.TestCase):
    def setUp(self):
        self.classifier = HybridPatternClassifier()
        self.width = 1000
        self.height = 500
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
    def test_recency_filter(self):
        # Create a Mock FeatureMap with trendlines designed to trigger triangle detection
        # But we can also just mock the _rule_based_detection return if we want to isolate the filter logic?
        # No, let's integration test it properly.
        # Construct trendlines that form triangles.
        
        # Triangle 1: Current (Right side) - ending at x ~ 900
        t1_up = {"start": (800, 400), "end": (900, 300), "angle": 45, "direction": "up", "length": 141}
        t1_down = {"start": (800, 200), "end": (900, 300), "angle": -45, "direction": "down", "length": 141}
        
        # Triangle 2: Old (Left side) - ending at x ~ 300
        t2_up = {"start": (200, 400), "end": (300, 300), "angle": 45, "direction": "up", "length": 141}
        t2_down = {"start": (200, 200), "end": (300, 300), "angle": -45, "direction": "down", "length": 141}
        
        feature_map = FeatureMap.empty()
        feature_map.trendlines = [t1_up, t1_down, t2_up, t2_down]
        
        # Config 1: No filter (min_end_x_ratio = 0.0)
        config_all = {"min_end_x_ratio": 0.0, "confidence_threshold": 0.0}
        detections_all = self.classifier.process((feature_map, self.image), config_all)
        
        print(f"Detections without filter: {len(detections_all)}")
        self.assertTrue(len(detections_all) >= 2, "Should detect at least the 2 main triangles")
        
        # Verify we detected the 'old' one
        old_patterns = [d for d in detections_all if d.bounding_box.x2 < 500]
        self.assertTrue(len(old_patterns) > 0, "Should have detected at least one old pattern")
        
        # Config 2: Recent only (min_end_x_ratio = 0.5)
        # Should keep patterns ending in last 50% (x > 500)
        config_recent = {"min_end_x_ratio": 0.5, "confidence_threshold": 0.0}
        detections_recent = self.classifier.process((feature_map, self.image), config_recent)
        
        print(f"Detections with filter: {len(detections_recent)}")
        
        # Verify filtering happened
        self.assertLess(len(detections_recent), len(detections_all), "Filter should remove old patterns")
        
        # Verify no old patterns remain
        remaining_old = [d for d in detections_recent if d.bounding_box.x2 < 500]
        self.assertEqual(len(remaining_old), 0, "No patterns ending before x=500 should remain")
        
        # Verify recent patterns remain
        self.assertTrue(len(detections_recent) > 0, "Should still have recent patterns")

if __name__ == '__main__':
    unittest.main()

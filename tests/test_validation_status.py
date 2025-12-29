
import unittest
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pattern_analysis.pipeline.cross_validator import MultiMethodCrossValidator
from src.pattern_analysis.models.dataclasses import FeatureMap, BoundingBox, PatternDetection, ValidationResult
from src.pattern_analysis.models.enums import PatternType, PatternCategory

class MockClassifier:
    def __init__(self, should_confirm=True, stage_id="mock_validator"):
        self.stage_id = stage_id
        self.should_confirm = should_confirm
        
    def classify(self, features, image):
        if self.should_confirm:
            # Return a detection that matches the pattern type
            # We assume the validator will be called on a small ROI, 
            # so we just return a detection covering that ROI or simply a matching type.
            return [
                PatternDetection(
                    pattern_type=PatternType.HEAD_SHOULDERS, # Assuming we test H&S
                    category=PatternCategory.REVERSAL,
                    confidence=0.9,
                    bounding_box=BoundingBox(0, 0, 10, 10),
                    metadata={},
                    detector_id=self.stage_id
                )
            ]
        return []

class TestValidationStatus(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.detection = PatternDetection(
            pattern_type=PatternType.HEAD_SHOULDERS,
            category=PatternCategory.REVERSAL,
            confidence=0.8,
            bounding_box=BoundingBox(10, 10, 90, 90),
            metadata={},
            detector_id="original_detector"
        )
        
    def test_status_skipped(self):
        # No validators
        validator = MultiMethodCrossValidator(validators=[])
        results = validator.process(([self.detection], self.image), {})
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "skipped")
        self.assertEqual(results[0].validation_score, 0.0)
        self.assertFalse(results[0].is_confirmed)
        
    def test_status_confirmed(self):
        # Validator confirms
        mock_val = MockClassifier(should_confirm=True)
        validator = MultiMethodCrossValidator(validators=[mock_val], consensus_threshold=0.5)
        results = validator.process(([self.detection], self.image), {})
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "confirmed")
        self.assertEqual(results[0].agreement_count, 1)
        self.assertTrue(results[0].is_confirmed)
        
    def test_status_unconfirmed(self):
        # Validator denies
        mock_val = MockClassifier(should_confirm=False)
        validator = MultiMethodCrossValidator(validators=[mock_val], consensus_threshold=0.5)
        results = validator.process(([self.detection], self.image), {})
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "unconfirmed")
        self.assertEqual(results[0].agreement_count, 0)
        self.assertFalse(results[0].is_confirmed)

if __name__ == '__main__':
    unittest.main()

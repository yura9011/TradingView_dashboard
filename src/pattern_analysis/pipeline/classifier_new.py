"""
Hybrid pattern classifier - orchestrates multiple detectors.

This is the refactored version using modular detectors.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models.dataclasses import BoundingBox, FeatureMap, PatternDetection
from ..models.enums import PatternType
from .interfaces import PatternClassifier
from .detectors import (
    TriangleDetector,
    DoublePatternDetector,
    HeadShouldersDetector,
    MLDetector,
    DetectionMerger,
)


class HybridPatternClassifier(PatternClassifier):
    """
    Pattern classifier combining rule-based and ML approaches.
    
    Uses modular detectors for each pattern type, then merges
    and filters results by confidence.
    """
    
    DEFAULT_CONFIDENCE_THRESHOLD = 0.3
    DEFAULT_IOU_THRESHOLD = 0.5
    
    def __init__(
        self,
        pattern_registry: Optional[Any] = None,
        ml_model_path: Optional[str] = None
    ):
        """Initialize classifier with detectors."""
        self.registry = pattern_registry
        self._config: Dict[str, Any] = {}
        
        # Initialize detectors
        self.triangle_detector = TriangleDetector()
        self.double_detector = DoublePatternDetector()
        self.hs_detector = HeadShouldersDetector()
        self.ml_detector = MLDetector(ml_model_path) if ml_model_path else None
        self.merger = DetectionMerger()
    
    @property
    def stage_id(self) -> str:
        return "hybrid_classifier_v2"
    
    def process(
        self,
        input_data: Tuple[FeatureMap, np.ndarray],
        config: Dict[str, Any]
    ) -> List[PatternDetection]:
        """Process features and image to detect patterns."""
        if not self.validate_input(input_data):
            return []
        
        features, image = input_data
        self._update_config(config)
        return self.classify(features, image)
    
    def classify(
        self,
        features: FeatureMap,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Classify patterns using all detectors."""
        detections: List[PatternDetection] = []
        
        # Rule-based detections
        detections.extend(self.triangle_detector.detect(features, image))
        detections.extend(self.double_detector.detect(features, image))
        detections.extend(self.hs_detector.detect(features, image))
        
        # ML detections
        if self.ml_detector:
            detections.extend(self.ml_detector.detect(features, image))
        
        # Merge overlapping
        merged = self.merger.merge(detections)
        
        # Filter by confidence
        threshold = self._config.get("confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD)
        filtered = [d for d in merged if d.confidence >= threshold]
        
        # Apply recency filter
        filtered = self._apply_recency_filter(filtered, image)
        
        # Sort by confidence
        return sorted(filtered, key=lambda d: d.confidence, reverse=True)
    
    def get_supported_patterns(self) -> List[PatternType]:
        return list(PatternType)
    
    def validate_input(self, input_data: Any) -> bool:
        if not isinstance(input_data, tuple) or len(input_data) != 2:
            return False
        features, image = input_data
        return isinstance(features, FeatureMap) and isinstance(image, np.ndarray)
    
    def _update_config(self, config: Dict[str, Any]):
        """Update config for all detectors."""
        self._config = config
        self.triangle_detector.update_config(config)
        self.double_detector.update_config(config)
        self.hs_detector.update_config(config)
        if self.ml_detector:
            self.ml_detector.update_config(config)
        
        iou = config.get("iou_threshold", self.DEFAULT_IOU_THRESHOLD)
        self.merger.iou_threshold = iou
    
    def _apply_recency_filter(
        self,
        detections: List[PatternDetection],
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Filter to keep only recent patterns."""
        min_end_x_ratio = self._config.get("min_end_x_ratio", 0.0)
        if min_end_x_ratio <= 0.0:
            return detections
        
        h, w = image.shape[:2]
        min_x = w * (1.0 - min_end_x_ratio)
        return [d for d in detections if d.bounding_box.x2 >= min_x]

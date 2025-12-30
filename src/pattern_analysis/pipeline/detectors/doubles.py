"""
Double top/bottom pattern detector.
"""

from typing import Any, Dict, List

import numpy as np

from ...models.dataclasses import BoundingBox, FeatureMap, PatternDetection
from ...models.enums import PatternCategory, PatternType
from .base import BaseDetector


class DoublePatternDetector(BaseDetector):
    """Detects double top and double bottom patterns."""
    
    DEFAULT_LEVEL_TOLERANCE = 20
    
    def detect(
        self,
        features: FeatureMap,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Detect double patterns."""
        detections: List[PatternDetection] = []
        
        tolerance = self.config.get(
            "double_pattern_level_tolerance",
            self.DEFAULT_LEVEL_TOLERANCE
        )
        
        # Double bottoms from support zones
        bottoms = self._find_double_touches(
            features.support_zones,
            tolerance,
            PatternType.DOUBLE_BOTTOM,
            image
        )
        detections.extend(bottoms)
        
        # Double tops from resistance zones
        tops = self._find_double_touches(
            features.resistance_zones,
            tolerance,
            PatternType.DOUBLE_TOP,
            image
        )
        detections.extend(tops)
        
        return detections
    
    def _find_double_touches(
        self,
        zones: List[BoundingBox],
        tolerance: int,
        pattern_type: PatternType,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Find pairs of zones at similar levels."""
        detections: List[PatternDetection] = []
        
        if len(zones) < 2:
            return detections
        
        h, w = image.shape[:2]
        
        for i, zone1 in enumerate(zones):
            for zone2 in zones[i + 1:]:
                y1_center = (zone1.y1 + zone1.y2) // 2
                y2_center = (zone2.y1 + zone2.y2) // 2
                
                if abs(y1_center - y2_center) <= tolerance:
                    x_separation = abs(zone1.x1 - zone2.x1)
                    min_separation = w * 0.1
                    
                    if x_separation >= min_separation:
                        detection = self._create_detection(
                            zone1, zone2, y1_center, y2_center,
                            tolerance, pattern_type
                        )
                        if detection:
                            detections.append(detection)
        
        return detections
    
    def _create_detection(
        self,
        zone1: BoundingBox,
        zone2: BoundingBox,
        y1_center: int,
        y2_center: int,
        tolerance: int,
        pattern_type: PatternType
    ) -> PatternDetection:
        """Create detection from two zones."""
        bbox = BoundingBox(
            min(zone1.x1, zone2.x1),
            min(zone1.y1, zone2.y1),
            max(zone1.x2, zone2.x2),
            max(zone1.y2, zone2.y2)
        )
        
        if not bbox.is_valid():
            return None
        
        level_diff = abs(y1_center - y2_center)
        level_similarity = 1.0 - (level_diff / max(tolerance, 1))
        confidence = 0.5 + 0.3 * level_similarity
        
        return PatternDetection(
            pattern_type=pattern_type,
            category=PatternCategory.REVERSAL,
            confidence=min(confidence, 0.9),
            bounding_box=bbox,
            metadata={
                "touch_1": zone1.to_dict(),
                "touch_2": zone2.to_dict(),
                "level_difference": level_diff,
                "neckline_y": (y1_center + y2_center) // 2
            },
            detector_id="rule_double"
        )

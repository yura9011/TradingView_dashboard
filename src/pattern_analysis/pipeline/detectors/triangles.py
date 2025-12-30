"""
Triangle pattern detector.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...models.dataclasses import BoundingBox, FeatureMap, PatternDetection
from ...models.enums import PatternCategory, PatternType
from .base import BaseDetector


class TriangleDetector(BaseDetector):
    """Detects triangle patterns from converging trendlines."""
    
    def detect(
        self,
        features: FeatureMap,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Detect triangle patterns."""
        detections: List[PatternDetection] = []
        trendlines = features.trendlines
        
        if len(trendlines) < 2:
            return detections
        
        up_lines = [t for t in trendlines if t.get("direction") == "up"]
        down_lines = [t for t in trendlines if t.get("direction") == "down"]
        
        for up_line in up_lines:
            for down_line in down_lines:
                if self._lines_converge(up_line, down_line, image):
                    detection = self._create_detection(up_line, down_line, image)
                    if detection:
                        detections.append(detection)
        
        return detections
    
    def _lines_converge(
        self,
        line1: Dict[str, Any],
        line2: Dict[str, Any],
        image: np.ndarray
    ) -> bool:
        """Check if two lines converge."""
        if line1.get("direction") == line2.get("direction"):
            return False
        
        x1_1, y1_1 = line1["start"]
        x1_2, y1_2 = line1["end"]
        x2_1, y2_1 = line2["start"]
        x2_2, y2_2 = line2["end"]
        
        intersection = self._line_intersection(
            (x1_1, y1_1), (x1_2, y1_2),
            (x2_1, y2_1), (x2_2, y2_2)
        )
        
        if intersection is None:
            return False
        
        ix, iy = intersection
        h, w = image.shape[:2]
        margin = max(w, h) * 0.5
        
        if ix < -margin or ix > w + margin:
            return False
        if iy < -margin or iy > h + margin:
            return False
        
        min_x = min(x1_1, x1_2, x2_1, x2_2)
        return ix >= min_x
    
    def _line_intersection(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two lines."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        
        return (ix, iy)
    
    def _create_detection(
        self,
        up_line: Dict[str, Any],
        down_line: Dict[str, Any],
        image: np.ndarray
    ) -> Optional[PatternDetection]:
        """Create detection from converging lines."""
        pattern_type, category = self._classify_type(up_line, down_line)
        bbox = self._calculate_bbox(up_line, down_line)
        
        if not bbox.is_valid():
            return None
        
        confidence = self._calculate_confidence(up_line, down_line, image)
        apex = self._calculate_apex(up_line, down_line)
        
        return PatternDetection(
            pattern_type=pattern_type,
            category=category,
            confidence=confidence,
            bounding_box=bbox,
            metadata={
                "upper_line": down_line,
                "lower_line": up_line,
                "apex": apex
            },
            detector_id="rule_triangle"
        )
    
    def _classify_type(
        self,
        up_line: Dict[str, Any],
        down_line: Dict[str, Any]
    ) -> Tuple[PatternType, PatternCategory]:
        """Classify triangle type based on angles."""
        up_angle = abs(up_line.get("angle", 0))
        down_angle = abs(down_line.get("angle", 0))
        threshold = 10
        
        if down_angle < threshold and up_angle >= threshold:
            return PatternType.ASCENDING_TRIANGLE, PatternCategory.CONTINUATION
        if up_angle < threshold and down_angle >= threshold:
            return PatternType.DESCENDING_TRIANGLE, PatternCategory.CONTINUATION
        return PatternType.SYMMETRICAL_TRIANGLE, PatternCategory.BILATERAL
    
    def _calculate_bbox(
        self,
        up_line: Dict[str, Any],
        down_line: Dict[str, Any]
    ) -> BoundingBox:
        """Calculate bounding box for triangle."""
        all_x = [
            up_line["start"][0], up_line["end"][0],
            down_line["start"][0], down_line["end"][0]
        ]
        all_y = [
            up_line["start"][1], up_line["end"][1],
            down_line["start"][1], down_line["end"][1]
        ]
        
        x1, y1 = int(min(all_x)), int(min(all_y))
        x2, y2 = int(max(all_x)), int(max(all_y))
        
        if x1 >= x2:
            x2 = x1 + 1
        if y1 >= y2:
            y2 = y1 + 1
        
        return BoundingBox(x1, y1, x2, y2)
    
    def _calculate_confidence(
        self,
        up_line: Dict[str, Any],
        down_line: Dict[str, Any],
        image: np.ndarray
    ) -> float:
        """Calculate confidence score."""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        up_length = up_line.get("length", 0)
        down_length = down_line.get("length", 0)
        avg_length = (up_length + down_length) / 2
        length_factor = min(avg_length / (max_dim * 0.3), 1.0)
        
        up_angle = abs(up_line.get("angle", 0))
        down_angle = abs(down_line.get("angle", 0))
        angle_factor = min((up_angle + down_angle) / 60, 1.0)
        
        confidence = 0.4 * length_factor + 0.4 * angle_factor + 0.2
        return min(max(confidence, 0.0), 0.9)
    
    def _calculate_apex(
        self,
        up_line: Dict[str, Any],
        down_line: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """Calculate apex point."""
        intersection = self._line_intersection(
            up_line["start"], up_line["end"],
            down_line["start"], down_line["end"]
        )
        if intersection:
            return {"x": intersection[0], "y": intersection[1]}
        return None

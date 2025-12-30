"""
Head and shoulders pattern detector.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from ...models.dataclasses import BoundingBox, FeatureMap, PatternDetection
from ...models.enums import PatternCategory, PatternType
from .base import BaseDetector


class HeadShouldersDetector(BaseDetector):
    """Detects head and shoulders patterns."""
    
    DEFAULT_MIN_CANDLES = 20
    DEFAULT_SHOULDER_TOLERANCE = 30
    
    def detect(
        self,
        features: FeatureMap,
        image: np.ndarray
    ) -> List[PatternDetection]:
        """Detect head and shoulders patterns."""
        detections: List[PatternDetection] = []
        candlesticks = features.candlestick_regions
        
        min_candles = self.config.get("min_candles_for_hs", self.DEFAULT_MIN_CANDLES)
        if len(candlesticks) < min_candles:
            return detections
        
        tolerance = self.config.get("hs_shoulder_tolerance", self.DEFAULT_SHOULDER_TOLERANCE)
        
        # Regular H&S (tops)
        peaks = self._find_local_peaks(candlesticks, is_maxima=True)
        hs_top = self._find_pattern(
            peaks, candlesticks, tolerance,
            PatternType.HEAD_SHOULDERS, is_inverted=False
        )
        detections.extend(hs_top)
        
        # Inverse H&S (bottoms)
        troughs = self._find_local_peaks(candlesticks, is_maxima=False)
        hs_bottom = self._find_pattern(
            troughs, candlesticks, tolerance,
            PatternType.INVERSE_HEAD_SHOULDERS, is_inverted=True
        )
        detections.extend(hs_bottom)
        
        return detections
    
    def _find_local_peaks(
        self,
        candlesticks: List[BoundingBox],
        is_maxima: bool,
        window: int = 2
    ) -> List[Tuple[int, BoundingBox]]:
        """Find local maxima or minima."""
        peaks: List[Tuple[int, BoundingBox]] = []
        n = len(candlesticks)
        
        for i in range(window, n - window):
            current = candlesticks[i]
            
            if is_maxima:
                current_y = current.y1
                is_peak = all(
                    current_y < candlesticks[j].y1
                    for j in range(i - window, i + window + 1)
                    if j != i
                )
            else:
                current_y = current.y2
                is_peak = all(
                    current_y > candlesticks[j].y2
                    for j in range(i - window, i + window + 1)
                    if j != i
                )
            
            if is_peak:
                peaks.append((i, current))
        
        return peaks
    
    def _find_pattern(
        self,
        peaks: List[Tuple[int, BoundingBox]],
        candlesticks: List[BoundingBox],
        tolerance: int,
        pattern_type: PatternType,
        is_inverted: bool
    ) -> List[PatternDetection]:
        """Find H&S pattern from peaks."""
        detections: List[PatternDetection] = []
        
        if len(peaks) < 3:
            return detections
        
        for i in range(len(peaks) - 2):
            left_idx, left = peaks[i]
            head_idx, head = peaks[i + 1]
            right_idx, right = peaks[i + 2]
            
            detection = self._validate_and_create(
                left, head, right, tolerance,
                pattern_type, is_inverted
            )
            if detection:
                detections.append(detection)
        
        return detections
    
    def _validate_and_create(
        self,
        left: BoundingBox,
        head: BoundingBox,
        right: BoundingBox,
        tolerance: int,
        pattern_type: PatternType,
        is_inverted: bool
    ) -> PatternDetection:
        """Validate pattern and create detection."""
        if is_inverted:
            left_y, head_y, right_y = left.y2, head.y2, right.y2
            head_is_extreme = head_y > left_y and head_y > right_y
        else:
            left_y, head_y, right_y = left.y1, head.y1, right.y1
            head_is_extreme = head_y < left_y and head_y < right_y
        
        if not head_is_extreme:
            return None
        
        shoulder_diff = abs(left_y - right_y)
        if shoulder_diff > tolerance:
            return None
        
        bbox = BoundingBox(
            left.x1,
            min(left.y1, head.y1, right.y1),
            right.x2,
            max(left.y2, head.y2, right.y2)
        )
        
        if not bbox.is_valid():
            return None
        
        confidence = self._calculate_confidence(
            left, head, right, shoulder_diff, tolerance, is_inverted
        )
        
        return PatternDetection(
            pattern_type=pattern_type,
            category=PatternCategory.REVERSAL,
            confidence=confidence,
            bounding_box=bbox,
            metadata={
                "left_shoulder": left.to_dict(),
                "head": head.to_dict(),
                "right_shoulder": right.to_dict(),
                "neckline_y": (left_y + right_y) // 2,
                "shoulder_difference": shoulder_diff
            },
            detector_id="rule_hs"
        )
    
    def _calculate_confidence(
        self,
        left: BoundingBox,
        head: BoundingBox,
        right: BoundingBox,
        shoulder_diff: int,
        tolerance: int,
        is_inverted: bool
    ) -> float:
        """Calculate confidence score."""
        symmetry_factor = 1.0 - (shoulder_diff / max(tolerance, 1))
        
        if is_inverted:
            left_y, head_y, right_y = left.y2, head.y2, right.y2
            avg_shoulder_y = (left_y + right_y) / 2
            head_prominence = head_y - avg_shoulder_y
        else:
            left_y, head_y, right_y = left.y1, head.y1, right.y1
            avg_shoulder_y = (left_y + right_y) / 2
            head_prominence = avg_shoulder_y - head_y
        
        prominence_factor = min(head_prominence / 50, 1.0) if head_prominence > 0 else 0.0
        confidence = 0.5 + 0.25 * symmetry_factor + 0.15 * prominence_factor
        
        return min(max(confidence, 0.0), 0.9)

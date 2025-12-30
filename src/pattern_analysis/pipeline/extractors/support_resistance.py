"""
Support and resistance zone detection.

Feature: chart-pattern-analysis-framework
Requirements: 2.3
"""

from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

from ...models.dataclasses import BoundingBox


class SupportResistanceDetector:
    """Detects horizontal support and resistance zones."""
    
    DEFAULT_PEAK_DISTANCE = 20
    DEFAULT_ZONE_HEIGHT = 10
    
    def detect(
        self,
        image: np.ndarray,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        """
        Find horizontal support and resistance zones.
        
        Args:
            image: Preprocessed image as numpy array (RGB)
            config: Optional configuration dictionary
                
        Returns:
            Tuple of (support_zones, resistance_zones) as lists of BoundingBox
        """
        if config is None:
            config = {}
        
        peak_distance = config.get("peak_distance", self.DEFAULT_PEAK_DISTANCE)
        zone_height = config.get("zone_height", self.DEFAULT_ZONE_HEIGHT)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = image.shape[:2]
        
        horizontal_proj = np.sum(gray, axis=1).astype(np.float64)
        
        if horizontal_proj.max() > 0:
            horizontal_proj = horizontal_proj / horizontal_proj.max()
        
        peaks = self._find_peaks(
            horizontal_proj,
            height=np.mean(horizontal_proj),
            distance=peak_distance
        )
        
        support = []
        resistance = []
        mid_y = h // 2
        half_zone = zone_height // 2
        
        for peak in peaks:
            y1 = max(0, peak - half_zone)
            y2 = min(h, peak + half_zone)
            
            if y1 < y2:
                zone = BoundingBox(0, y1, w, y2)
                
                if peak < mid_y:
                    resistance.append(zone)
                else:
                    support.append(zone)
        
        return support, resistance
    
    def _find_peaks(
        self,
        data: np.ndarray,
        height: float = 0.0,
        distance: int = 20
    ) -> List[int]:
        """Find peaks in 1D data array."""
        peaks = []
        n = len(data)
        
        for i in range(1, n - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                if data[i] >= height:
                    if not peaks or (i - peaks[-1]) >= distance:
                        peaks.append(i)
        
        return peaks

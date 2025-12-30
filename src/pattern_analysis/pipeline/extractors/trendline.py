"""
Trendline detection using Canny edge detection and Hough transforms.

Feature: chart-pattern-analysis-framework
Requirements: 2.2
"""

from typing import Dict, List, Any, Optional

import cv2
import numpy as np


class TrendlineDetector:
    """Detects trendlines using edge detection and Hough Line Transform."""
    
    DEFAULT_MIN_LENGTH = 100
    DEFAULT_GAP = 10
    DEFAULT_CANNY_LOW = 50
    DEFAULT_CANNY_HIGH = 150
    DEFAULT_HOUGH_THRESHOLD = 100
    DEFAULT_MIN_ANGLE = 5
    DEFAULT_MAX_ANGLE = 85
    
    def detect(
        self,
        image: np.ndarray,
        config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect trendlines using Canny edge detection and Hough Line Transform.
        
        Args:
            image: Preprocessed image as numpy array (RGB)
            config: Optional configuration dictionary
                
        Returns:
            List of trendline dictionaries with start, end, angle, direction, length
        """
        if config is None:
            config = {}
        
        min_length = config.get("min_trendline_length", self.DEFAULT_MIN_LENGTH)
        max_gap = config.get("trendline_gap", self.DEFAULT_GAP)
        canny_low = config.get("canny_low", self.DEFAULT_CANNY_LOW)
        canny_high = config.get("canny_high", self.DEFAULT_CANNY_HIGH)
        hough_threshold = config.get("hough_threshold", self.DEFAULT_HOUGH_THRESHOLD)
        min_angle = config.get("min_angle", self.DEFAULT_MIN_ANGLE)
        max_angle = config.get("max_angle", self.DEFAULT_MAX_ANGLE)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_length,
            maxLineGap=max_gap
        )
        
        trendlines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                if min_angle < abs(angle) < max_angle:
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    direction = "up" if angle < 0 else "down"
                    
                    trendlines.append({
                        "start": (int(x1), int(y1)),
                        "end": (int(x2), int(y2)),
                        "angle": float(angle),
                        "direction": direction,
                        "length": float(length)
                    })
        
        return sorted(trendlines, key=lambda t: t["length"], reverse=True)

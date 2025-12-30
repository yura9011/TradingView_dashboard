"""
Volume profile extraction.

Feature: chart-pattern-analysis-framework
Requirements: 2.4
"""

from typing import Dict, Any, Optional

import cv2
import numpy as np

from ...models.dataclasses import BoundingBox


class VolumeExtractor:
    """Extracts volume profile using adaptive region detection."""
    
    EDGE_THRESHOLD = 0.02
    
    def extract(
        self,
        image: np.ndarray,
        config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract volume profile using adaptive region detection.
        
        Args:
            image: Preprocessed image as numpy array (RGB)
            config: Configuration dictionary
                
        Returns:
            Dictionary with volume profile data or None if not found
        """
        if not config.get("extract_volume", True):
            return None
        
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx_abs = np.absolute(sobelx)
        sobelx_uint8 = np.uint8(sobelx_abs)
        
        def check_region(start_y, end_y):
            if start_y >= end_y:
                return None, 0
            region = sobelx_uint8[start_y:end_y, :]
            d = np.count_nonzero(region > 50) / region.size
            return region, d

        bottom_h = int(h * 0.25)
        bottom_region, bottom_density = check_region(h - bottom_h, h)
        
        top_h = int(h * 0.15)
        top_region, top_density = check_region(0, top_h)
        
        selected_region = None
        region_y_start = 0
        
        if bottom_density > self.EDGE_THRESHOLD and bottom_density > top_density:
            selected_region = image[h - bottom_h:h, :]
            region_y_start = h - bottom_h
        elif top_density > self.EDGE_THRESHOLD:
            selected_region = image[0:top_h, :]
            region_y_start = 0
        
        if selected_region is None:
            return None
            
        region_gray = cv2.cvtColor(selected_region, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(
            region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        if thresh[0, 0] > 127 and thresh[0, -1] > 127:
            thresh = cv2.bitwise_not(thresh)
            
        col_sums = np.sum(thresh, axis=0) / 255.0
        
        return {
            "region": BoundingBox(
                0, region_y_start, w, region_y_start + selected_region.shape[0]
            ),
            "distribution": col_sums.tolist(),
            "avg_volume": float(np.mean(col_sums)),
            "max_volume": float(np.max(col_sums))
        }

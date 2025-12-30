"""
Candlestick extraction using dynamic color clustering.

Feature: chart-pattern-analysis-framework
Requirements: 2.1
"""

from typing import Dict, List, Any, Optional

import cv2
import numpy as np

from ...models.dataclasses import BoundingBox


class CandlestickExtractor:
    """Extracts candlestick body regions using K-means color clustering."""
    
    def extract(
        self,
        image: np.ndarray,
        config: Optional[Dict[str, Any]] = None
    ) -> List[BoundingBox]:
        """
        Extract candlestick body regions using dynamic color clustering.
        
        Args:
            image: Preprocessed image as numpy array (RGB)
            config: Optional configuration dictionary
            
        Returns:
            List of BoundingBox objects for each detected candlestick
        """
        if config is None:
            config = {}
        
        try:
            colors, labels, centers = self._get_dominant_colors(image, k=3)
        except Exception:
            return []

        unique_labels, counts = np.unique(labels, return_counts=True)
        bg_label_idx = np.argmax(counts)
        bg_label = unique_labels[bg_label_idx]
        
        candle_labels = [l for l in unique_labels if l != bg_label]
        
        if len(candle_labels) < 2:
            return []
            
        h, w = image.shape[:2]
        labels_img = labels.reshape((h, w))
        
        c1 = centers[candle_labels[0]]
        c2 = centers[candle_labels[1]]
        
        c1_hsv = cv2.cvtColor(np.uint8([[c1]]), cv2.COLOR_RGB2HSV)[0][0]
        c2_hsv = cv2.cvtColor(np.uint8([[c2]]), cv2.COLOR_RGB2HSV)[0][0]
        
        def dist_to_green(h): return min(abs(h - 60), 180 - abs(h - 60))
        def dist_to_red(h): return min(abs(h - 0), abs(h - 180))
        
        is_c1_green = dist_to_green(c1_hsv[0]) < dist_to_red(c1_hsv[0])
        is_c2_green = dist_to_green(c2_hsv[0]) < dist_to_red(c2_hsv[0])
        
        if is_c1_green and not is_c2_green:
            bullish_label = candle_labels[0]
            bearish_label = candle_labels[1]
        elif not is_c1_green and is_c2_green:
            bullish_label = candle_labels[1]
            bearish_label = candle_labels[0]
        else:
            bullish_label = candle_labels[0]
            bearish_label = candle_labels[1]
            
        mask_bull = (labels_img == bullish_label).astype(np.uint8) * 255
        mask_bear = (labels_img == bearish_label).astype(np.uint8) * 255
        
        combined_mask = cv2.bitwise_or(mask_bull, mask_bear)
        
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        candles = []
        min_candle_height = config.get("min_candle_height", 5)
        min_candle_width = config.get("min_candle_width", 2)
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            is_vertical = ch > cw * 0.5
            meets_min_height = ch >= min_candle_height
            meets_min_width = cw >= min_candle_width
            not_too_wide = cw < w * 0.1
            
            if is_vertical and meets_min_height and meets_min_width and not_too_wide:
                x2 = x + cw
                y2 = y + ch
                if x < x2 and y < y2:
                    center_x, center_y = x + cw//2, y + ch//2
                    center_x = min(max(0, center_x), w-1)
                    center_y = min(max(0, center_y), h-1)
                    
                    is_bull = labels_img[center_y, center_x] == bullish_label
                    direction = "bullish" if is_bull else "bearish"
                    
                    bbox = BoundingBox(x, y, x2, y2)
                    bbox.metadata = {
                        "direction": direction,
                        "color_label": int(labels_img[center_y, center_x])
                    }
                    candles.append(bbox)
        
        return sorted(candles, key=lambda b: b.x1)

    def _get_dominant_colors(self, image: np.ndarray, k: int = 3):
        """Extract K dominant colors using K-means."""
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        centers = np.uint8(centers)
        return centers, labels, centers

"""
Edge-based feature extractor implementation for chart image analysis.

This module implements the FeatureExtractor interface using edge detection
and Hough transforms to extract visual features from preprocessed chart images.

Requirements:
- 2.1: Identify candlestick boundaries and extract OHLC visual regions
- 2.2: Detect and trace trendlines using edge detection algorithms
- 2.3: Identify support and resistance zones based on price clustering
- 2.4: Extract volume profile information
- 2.5: Output structured feature map with coordinates and confidence values
- 2.6: Return empty feature set with appropriate status if no features found
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..models.dataclasses import BoundingBox, FeatureMap, PreprocessResult
from .interfaces import FeatureExtractor


class EdgeBasedFeatureExtractor(FeatureExtractor):
    """
    Feature extractor using edge detection and Hough transforms.
    
    Extracts candlesticks, trendlines, support/resistance zones, and
    volume profile from preprocessed chart images.
    
    Requirements: 2.1-2.6
    """
    
    # Default configuration values
    DEFAULT_MIN_TRENDLINE_LENGTH = 100
    DEFAULT_TRENDLINE_GAP = 10
    DEFAULT_CANNY_LOW = 50
    DEFAULT_CANNY_HIGH = 150
    DEFAULT_HOUGH_THRESHOLD = 100
    DEFAULT_MIN_ANGLE = 5
    DEFAULT_MAX_ANGLE = 85
    DEFAULT_PEAK_DISTANCE = 20
    
    @property
    def stage_id(self) -> str:
        """Unique identifier for this feature extractor."""
        return "edge_feature_extractor_v1"
    
    def process(
        self, input_data: PreprocessResult, config: Dict[str, Any]
    ) -> FeatureMap:
        """
        Extract features from preprocessed image.
        
        Args:
            input_data: PreprocessResult from preprocessor stage
            config: Configuration dictionary with optional keys:
                - extract_volume: bool, whether to extract volume profile
                - min_trendline_length: int, minimum trendline length in pixels
                - trendline_gap: int, maximum gap between line segments
                - canny_low: int, Canny edge detection low threshold
                - canny_high: int, Canny edge detection high threshold
                - hough_threshold: int, Hough transform threshold
                - min_angle: float, minimum trendline angle in degrees
                - max_angle: float, maximum trendline angle in degrees
                - peak_distance: int, minimum distance between S/R peaks
                
        Returns:
            FeatureMap with extracted features
            
        Requirements: 2.5, 2.6
        """
        if not self.validate_input(input_data):
            return FeatureMap.empty()
        
        image = input_data.image
        
        # Extract all features
        candlesticks = self.extract_candlesticks(image, config)
        trendlines = self.detect_trendlines(image, config)
        support, resistance = self.find_support_resistance(image, config)
        volume = self._extract_volume_profile(image, config)
        
        return FeatureMap(
            candlestick_regions=candlesticks,
            trendlines=trendlines,
            support_zones=support,
            resistance_zones=resistance,
            volume_profile=volume,
            quality_score=input_data.quality_score
        )
    
    def extract_candlesticks(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> List[BoundingBox]:
        """
        Extract candlestick body regions using dynamic color clustering.
        
        Uses K-means clustering to identify dominant colors (background, bullish, bearish)
        and segments the image based on these learned colors, making it robust to 
        different chart themes.
        
        Args:
            image: Preprocessed image as numpy array (RGB)
            config: Optional configuration dictionary
            
        Returns:
            List of BoundingBox objects for each detected candlestick,
            sorted by x-coordinate (left to right)
            
        Requirements: 2.1
        """
        if config is None:
            config = {}
        
        # 1. Identify dominant colors using K-means
        # We assume 3 dominant colors: Background, Bullish, Bearish
        try:
            colors, labels, centers = self._get_dominant_colors(image, k=3)
        except Exception as e:
            # Fallback for very simple images (e.g. all black)
            return []

        # 2. Identify Background Color
        # The background is usually the most frequent color
        unique_labels, counts = np.unique(labels, return_counts=True)
        bg_label_idx = np.argmax(counts)
        bg_label = unique_labels[bg_label_idx]
        
        # The other two are bullish/bearish
        candle_labels = [l for l in unique_labels if l != bg_label]
        
        if len(candle_labels) < 2:
            # Not enough colors found to distinguish candles
            return []
            
        # 3. Create masks for each candle color
        # Reshape labels back to image dimensions
        h, w = image.shape[:2]
        labels_img = labels.reshape((h, w))
        
        # We need to distinguish Bullish vs Bearish.
        # Heuristic: 
        # - Standard: Green (Bullish), Red (Bearish)
        # - We can check Hue distance to typical Green/Red if present?
        # - Or just treat them as "Type A" and "Type B" for now, and rely on
        #   metadata to let downstream logic decide (e.g. by comparing Close > Open)
        #   BUT we don't have price data yet.
        #   Let's use a simple heuristic: Green-ish is Bullish, Red-ish/Darker is Bearish?
        #   Let's check the hue of the centers.
        
        c1 = centers[candle_labels[0]]
        c2 = centers[candle_labels[1]]
        
        # Convert centers to HSV to check Hue
        c1_hsv = cv2.cvtColor(np.uint8([[c1]]), cv2.COLOR_RGB2HSV)[0][0]
        c2_hsv = cv2.cvtColor(np.uint8([[c2]]), cv2.COLOR_RGB2HSV)[0][0]
        
        # Green is around Hue 60 (0-180 scale in OpenCV), Red is around 0 or 170
        def dist_to_green(h): return min(abs(h - 60), 180 - abs(h - 60))
        def dist_to_red(h): return min(abs(h - 0), abs(h - 180))
        
        # Assign labels based on proximity to Red/Green
        # This is a soft heuristic; if they are Blue/White, we fall back to brightness?
        # White usually Bullish in some dark themes? 
        # Let's rely on Red/Green preference if they are distinct enough.
        
        is_c1_green = dist_to_green(c1_hsv[0]) < dist_to_red(c1_hsv[0])
        is_c2_green = dist_to_green(c2_hsv[0]) < dist_to_red(c2_hsv[0])
        
        if is_c1_green and not is_c2_green:
            bullish_label = candle_labels[0]
            bearish_label = candle_labels[1]
        elif not is_c1_green and is_c2_green:
            bullish_label = candle_labels[1]
            bearish_label = candle_labels[0]
        else:
            # Ambiguous colors (e.g. Blue/White). 
            # In Blue/White themes, White is often Bearish (Hollow Red) or Bullish?
            # Standard "Hollow" candles: Black filled (Bear), White hollow (Bull).
            # "Blue/White": Blue usually Bull, White Bear? Or vice versa.
            # Let's just assign based on index for now and store color in metadata.
            bullish_label = candle_labels[0]
            bearish_label = candle_labels[1]
            
        mask_bull = (labels_img == bullish_label).astype(np.uint8) * 255
        mask_bear = (labels_img == bearish_label).astype(np.uint8) * 255
        
        # Combine masks for contour detection
        combined_mask = cv2.bitwise_or(mask_bull, mask_bear)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
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
                    # Determine direction by checking pixel center
                    # Sample center pixel of the bbox to see which mask it belongs to
                    center_x, center_y = x + cw//2, y + ch//2
                    # Clip to bounds
                    center_x = min(max(0, center_x), w-1)
                    center_y = min(max(0, center_y), h-1)
                    
                    is_bull = labels_img[center_y, center_x] == bullish_label
                    direction = "bullish" if is_bull else "bearish"
                    
                    bbox = BoundingBox(x, y, x2, y2)
                    bbox.metadata = {"direction": direction, "color_label": int(labels_img[center_y, center_x])}
                    candles.append(bbox)
        
        return sorted(candles, key=lambda b: b.x1)

    def _get_dominant_colors(self, image: np.ndarray, k: int = 3):
        """Helper to extract K dominant colors using K-means."""
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        return centers, labels, centers
    
    def detect_trendlines(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect trendlines using Canny edge detection and Hough Line Transform.
        
        Identifies diagonal lines that may represent trendlines by filtering
        for lines with angles between min_angle and max_angle degrees.
        
        Args:
            image: Preprocessed image as numpy array (RGB)
            config: Optional configuration dictionary with keys:
                - min_trendline_length: minimum line length
                - trendline_gap: maximum gap between segments
                - canny_low: Canny low threshold
                - canny_high: Canny high threshold
                - hough_threshold: Hough transform threshold
                - min_angle: minimum angle in degrees
                - max_angle: maximum angle in degrees
                
        Returns:
            List of trendline dictionaries with keys:
            - start: (x, y) tuple for line start
            - end: (x, y) tuple for line end
            - angle: Line angle in degrees (-90 to 90)
            - direction: "up" or "down"
            - length: Line length in pixels
            
        Requirements: 2.2
        """
        if config is None:
            config = {}
        
        # Get configuration values
        min_length = config.get("min_trendline_length", self.DEFAULT_MIN_TRENDLINE_LENGTH)
        max_gap = config.get("trendline_gap", self.DEFAULT_TRENDLINE_GAP)
        canny_low = config.get("canny_low", self.DEFAULT_CANNY_LOW)
        canny_high = config.get("canny_high", self.DEFAULT_CANNY_HIGH)
        hough_threshold = config.get("hough_threshold", self.DEFAULT_HOUGH_THRESHOLD)
        min_angle = config.get("min_angle", self.DEFAULT_MIN_ANGLE)
        max_angle = config.get("max_angle", self.DEFAULT_MAX_ANGLE)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
        
        # Apply Hough Line Transform (probabilistic version)
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
                
                # Calculate angle in degrees
                # atan2 returns angle in radians, convert to degrees
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Filter for diagonal lines (trendlines)
                # Exclude near-horizontal (< min_angle) and near-vertical (> max_angle)
                if min_angle < abs(angle) < max_angle:
                    # Calculate line length
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    
                    # Determine direction based on angle
                    # Negative angle = line goes up (from left to right)
                    # Positive angle = line goes down (from left to right)
                    direction = "up" if angle < 0 else "down"
                    
                    trendlines.append({
                        "start": (int(x1), int(y1)),
                        "end": (int(x2), int(y2)),
                        "angle": float(angle),
                        "direction": direction,
                        "length": float(length)
                    })
        
        # Sort by length (longest first)
        return sorted(trendlines, key=lambda t: t["length"], reverse=True)
    
    def find_support_resistance(
        self, image: np.ndarray, config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        """
        Find horizontal support and resistance zones.
        
        Identifies price levels with high activity by analyzing horizontal
        projections and finding peaks. Zones below the midpoint are classified
        as support, zones above as resistance.
        
        Args:
            image: Preprocessed image as numpy array (RGB)
            config: Optional configuration dictionary with keys:
                - peak_distance: minimum distance between peaks
                - zone_height: height of each zone in pixels
                
        Returns:
            Tuple of (support_zones, resistance_zones) as lists of BoundingBox
            
        Requirements: 2.3
        """
        if config is None:
            config = {}
        
        peak_distance = config.get("peak_distance", self.DEFAULT_PEAK_DISTANCE)
        zone_height = config.get("zone_height", 10)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        h, w = image.shape[:2]
        
        # Calculate horizontal projection (sum of pixel values per row)
        # This highlights rows with high activity (price levels)
        horizontal_proj = np.sum(gray, axis=1).astype(np.float64)
        
        # Normalize projection
        if horizontal_proj.max() > 0:
            horizontal_proj = horizontal_proj / horizontal_proj.max()
        
        # Find peaks in the projection
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
            # Create zone around the peak
            y1 = max(0, peak - half_zone)
            y2 = min(h, peak + half_zone)
            
            # Ensure valid bounding box
            if y1 < y2:
                zone = BoundingBox(0, y1, w, y2)
                
                # Classify based on position relative to midpoint
                # In chart images, y increases downward, so:
                # - Peaks above midpoint (lower y) are resistance
                # - Peaks below midpoint (higher y) are support
                if peak < mid_y:
                    resistance.append(zone)
                else:
                    support.append(zone)
        
        return support, resistance
    
    def _extract_volume_profile(
        self, image: np.ndarray, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract volume profile using adaptive region detection.
        
        Scans the image to find the region most likely containing volume bars,
        characterized by high vertical edge density typically at the bottom 
        or in a separate pane.
        
        Args:
            image: Preprocessed image as numpy array (RGB)
            config: Configuration dictionary with keys:
                - extract_volume: bool, whether to extract volume
                - volume_region_ratio: float, hints max height of volume pane
                
        Returns:
            Dictionary with volume profile data or None if disabled/not found.
        """
        if not config.get("extract_volume", True):
            return None
        
        h, w = image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use edge detection to find "bars"
        # Volume bars are vertical edges.
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate vertical projection (sum of edges per row) to find "busy" rows
        # Volume area has high density of vertical lines, so row sum should be high?
        # Actually, vertical bars create vertical edges.
        # Sobel Y would detect horizontal edges (tops of bars).
        # Sobel X would detect vertical edges (sides of bars).
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx_abs = np.absolute(sobelx)
        sobelx_uint8 = np.uint8(sobelx_abs)
        
        # Project horizontally (sum across width) to see finding rows with many vertical edges
        row_density = np.sum(sobelx_uint8 > 50, axis=1) # Count strong vertical edges per row
        
        # Smooth the density to find a region
        kernel_size = int(h * 0.05) # 5% of height window
        smoothed_density = np.convolve(row_density, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Look for the region with sustained high density, likely at the bottom.
        # Simple heuristic: Split image into N strips, find the strip with highest vertical edge density 
        # that is separated from main price action?
        # Or just stick to the bottom 30% but verify it actually HAS edges.
        
        # Let's revert to a slightly smarter heuristic: 
        # Check the bottom 25%. If edge density > threshold, use it.
        # If not, check the top. 
        # Dynamic scanning is complex because price candles ALSO have vertical edges.
        # Volume is distinguishable because it's usually monocolor or specific colors, and dense.
        
        # Improved approach:
        # 1. Look at bottom 25% (standard).
        # 2. If empty, look at top 15% (sometimes displayed there).
        # 3. If empty, return None.
        
        def check_region(start_y, end_y, threshold=0.05):
            if start_y >= end_y: return None, 0
            region = sobelx_uint8[start_y:end_y, :]
            # Density: fraction of pixels that are edges
            d = np.count_nonzero(region > 50) / region.size
            return region, d

        # Check bottom 25%
        bottom_h = int(h * 0.25)
        bottom_region, bottom_density = check_region(h - bottom_h, h)
        
        # Check top 15%
        top_h = int(h * 0.15)
        top_region, top_density = check_region(0, top_h)
        
        # Threshold for "presence of volume"
        EDGE_THRESHOLD = 0.02 # 2% of pixels are vertical edges
        
        selected_region = None
        region_y_start = 0
        
        if bottom_density > EDGE_THRESHOLD and bottom_density > top_density:
            selected_region = image[h - bottom_h:h, :]
            region_y_start = h - bottom_h
        elif top_density > EDGE_THRESHOLD:
            selected_region = image[0:top_h, :]
            region_y_start = 0
        
        if selected_region is None:
            return None
            
        # Refine extraction on the selected region
        # Convert to grayscale and threshold to get bars
        region_gray = cv2.cvtColor(selected_region, cv2.COLOR_RGB2GRAY)
        # Otsu's thresholding might work well for bars vs background
        _, thresh = cv2.threshold(region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if background is white (common in charts)?
        # Otsu usually separates FG/BG. 
        # If bars are dark on light BG, thresh might be inverted.
        # Check corners: if corners are white, then BG is white.
        if thresh[0, 0] > 127 and thresh[0, -1] > 127:
            thresh = cv2.bitwise_not(thresh)
            
        col_sums = np.sum(thresh, axis=0) / 255.0 # Count pixels
        
        return {
            "region": BoundingBox(0, region_y_start, w, region_y_start + selected_region.shape[0]),
            "distribution": col_sums.tolist(),
            "avg_volume": float(np.mean(col_sums)),
            "max_volume": float(np.max(col_sums))
        }
    
    def _find_peaks(
        self,
        data: np.ndarray,
        height: float = 0.0,
        distance: int = 20
    ) -> List[int]:
        """
        Find peaks in 1D data array.
        
        Simple peak detection that finds local maxima above a threshold.
        
        Args:
            data: 1D numpy array
            height: Minimum peak height
            distance: Minimum distance between peaks
            
        Returns:
            List of peak indices
        """
        peaks = []
        n = len(data)
        
        for i in range(1, n - 1):
            # Check if this is a local maximum
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                # Check if above height threshold
                if data[i] >= height:
                    # Check distance from previous peak
                    if not peaks or (i - peaks[-1]) >= distance:
                        peaks.append(i)
        
        return peaks
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate that input is a valid PreprocessResult.
        
        Args:
            input_data: Expected to be a PreprocessResult instance
            
        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(input_data, PreprocessResult):
            return False
        
        # Check that image is a valid numpy array
        if not isinstance(input_data.image, np.ndarray):
            return False
        
        # Check that image has 3 channels (RGB)
        if len(input_data.image.shape) != 3 or input_data.image.shape[2] != 3:
            return False
        
        return True

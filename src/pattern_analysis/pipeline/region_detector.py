"""
Chart Region Detector for identifying and separating chart regions.

This module implements the ChartRegionDetector class that identifies
different regions in a chart image (primary chart, volume panel, indicators).

Requirements:
- 1.1: Identify and separate each region when image contains multiple chart regions
- 1.2: Classify each region as Primary_Chart or Secondary_Chart
- 1.3: Analyze ONLY the Primary_Chart region for pattern detection
- 1.4: Exclude volume panel from pattern analysis
- 1.5: Use entire image if only one chart region detected
- 1.6: Output bounding coordinates of Primary_Chart region
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..models.dataclasses import BoundingBox, ChartRegion, RegionDetectionResult
from ..models.enums import RegionType


logger = logging.getLogger(__name__)


class ChartRegionDetector:
    """
    Detects and classifies different regions in a chart image.
    
    Uses horizontal projection analysis and color segmentation to identify:
    - Primary price chart (candlesticks)
    - Volume panel (typically at bottom)
    - Indicator panels (RSI, MACD, etc.)
    - UI elements (toolbars, legends)
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ChartRegionDetector.
        
        Args:
            config: Configuration dictionary with optional keys:
                - min_region_height_pct: Minimum region height as percentage (default: 0.1)
                - volume_position_threshold: Position threshold for volume detection (default: 0.7)
                - separator_threshold: Threshold for detecting separators (default: 0.3)
        """
        self.config = config
        self.min_region_height_pct = config.get("min_region_height_pct", 0.1)
        self.volume_position_threshold = config.get("volume_position_threshold", 0.7)
        self.separator_threshold = config.get("separator_threshold", 0.3)
    
    def detect_regions(self, image: np.ndarray) -> RegionDetectionResult:
        """
        Detect all chart regions in the image.
        
        Args:
            image: Input image as numpy array (RGB format, shape: H x W x C)
            
        Returns:
            RegionDetectionResult with all detected regions
            
        Requirements: 1.1, 1.6
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided to region detector")
            return self._create_empty_result((0, 0))
        
        h, w = image.shape[:2]
        original_size = (w, h)
        regions: List[ChartRegion] = []
        
        # 1. Find horizontal separators (gaps between charts)
        separators = self._find_horizontal_separators(image)
        logger.debug(f"Found {len(separators)} horizontal separators")
        
        # 2. Split image into regions based on separators
        region_bounds = self._split_by_separators(separators, h)
        logger.debug(f"Split into {len(region_bounds)} regions")
        
        # 3. Classify each region
        for bounds in region_bounds:
            y1, y2 = bounds
            region_image = image[y1:y2, :]
            region_type, confidence, characteristics = self._classify_region(
                region_image, y1, h
            )
            
            region = ChartRegion(
                region_type=region_type,
                bounding_box=BoundingBox(0, y1, w, y2),
                confidence=confidence,
                characteristics=characteristics
            )
            regions.append(region)
        
        # 4. Identify primary and secondary regions
        primary = self._find_primary_region(regions)
        secondary = [r for r in regions if r != primary]
        
        logger.info(
            f"Region detection complete: {len(regions)} regions found, "
            f"primary type: {primary.region_type.value if primary else 'None'}"
        )
        
        return RegionDetectionResult(
            regions=regions,
            primary_region=primary,
            secondary_regions=secondary,
            original_size=original_size,
            detection_method="horizontal_projection"
        )
    
    def _find_horizontal_separators(self, image: np.ndarray) -> List[int]:
        """
        Find horizontal lines that separate chart regions.
        
        Uses horizontal projection (sum of pixel values per row) to detect
        rows with low activity, which typically indicate separators between
        chart regions.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            List of y-coordinates where separators are detected
            
        Requirements: 1.1
        """
        # Convert to grayscale for projection analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # Calculate horizontal projection (sum of pixels per row)
        # Invert so that dark areas (separators) have high values
        inverted = 255 - gray
        projection = np.sum(inverted, axis=1).astype(np.float64)
        
        # Normalize projection to [0, 1]
        max_val = np.max(projection)
        if max_val > 0:
            projection = projection / max_val
        
        # Find valleys (rows with low activity = potential separators)
        # A separator is a region where the projection is below threshold
        separators: List[int] = []
        in_separator = False
        sep_start = 0
        min_separator_height = max(3, int(h * 0.005))  # At least 3 pixels or 0.5% of height
        
        for i, val in enumerate(projection):
            if val > (1.0 - self.separator_threshold) and not in_separator:
                # Entering a potential separator (high inverted value = dark row)
                in_separator = True
                sep_start = i
            elif val <= (1.0 - self.separator_threshold) and in_separator:
                # Exiting separator
                in_separator = False
                sep_height = i - sep_start
                
                # Only consider significant separators
                if sep_height >= min_separator_height:
                    sep_center = (sep_start + i) // 2
                    separators.append(sep_center)
        
        return separators
    
    def _split_by_separators(
        self, separators: List[int], height: int
    ) -> List[Tuple[int, int]]:
        """
        Split image height into regions based on separator positions.
        
        Args:
            separators: List of y-coordinates of separators
            height: Total image height
            
        Returns:
            List of (y1, y2) tuples defining region bounds
        """
        if not separators:
            # No separators found - entire image is one region
            return [(0, height)]
        
        min_region_height = int(height * self.min_region_height_pct)
        bounds: List[Tuple[int, int]] = []
        prev = 0
        
        for sep in sorted(separators):
            region_height = sep - prev
            if region_height >= min_region_height:
                bounds.append((prev, sep))
            prev = sep
        
        # Add final region
        final_height = height - prev
        if final_height >= min_region_height:
            bounds.append((prev, height))
        
        # If no valid regions found, return entire image
        if not bounds:
            return [(0, height)]
        
        return bounds

    def _classify_region(
        self, 
        region: np.ndarray, 
        y_start: int, 
        total_height: int
    ) -> Tuple[RegionType, float, Dict[str, Any]]:
        """
        Classify a region based on its visual characteristics.
        
        Uses position analysis and color distribution to determine
        the type of chart region.
        
        Args:
            region: Region image as numpy array (RGB)
            y_start: Starting y-coordinate of region in original image
            total_height: Total height of original image
            
        Returns:
            Tuple of (RegionType, confidence, characteristics dict)
            
        Requirements: 1.2, 1.4
        """
        h, w = region.shape[:2]
        
        # Calculate relative position (0 = top, 1 = bottom)
        relative_position = y_start / total_height if total_height > 0 else 0
        height_ratio = h / total_height if total_height > 0 else 0
        
        # Analyze color distribution for candlestick detection
        green_ratio, red_ratio = self._analyze_candlestick_colors(region)
        candle_ratio = green_ratio + red_ratio
        
        # Build characteristics dictionary
        characteristics = {
            "relative_position": relative_position,
            "height_ratio": height_ratio,
            "green_ratio": green_ratio,
            "red_ratio": red_ratio,
            "candle_ratio": candle_ratio,
            "width": w,
            "height": h
        }
        
        # Classification logic based on position and visual characteristics
        region_type, confidence = self._determine_region_type(
            relative_position, height_ratio, candle_ratio, characteristics
        )
        
        return region_type, confidence, characteristics
    
    def _analyze_candlestick_colors(self, region: np.ndarray) -> Tuple[float, float]:
        """
        Analyze region for candlestick colors (green/red).
        
        Args:
            region: Region image as numpy array (RGB)
            
        Returns:
            Tuple of (green_ratio, red_ratio) as percentages of total pixels
        """
        if region.size == 0:
            return 0.0, 0.0
        
        h, w = region.shape[:2]
        total_pixels = h * w
        
        if total_pixels == 0:
            return 0.0, 0.0
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        
        # Green mask (bullish candles) - HSV range for green
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Red mask (bearish candles) - HSV range for red (wraps around 0)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Calculate ratios
        green_pixels = np.sum(green_mask > 0)
        red_pixels = np.sum(red_mask > 0)
        
        green_ratio = green_pixels / total_pixels
        red_ratio = red_pixels / total_pixels
        
        return green_ratio, red_ratio
    
    def _determine_region_type(
        self,
        relative_position: float,
        height_ratio: float,
        candle_ratio: float,
        characteristics: Dict[str, Any]
    ) -> Tuple[RegionType, float]:
        """
        Determine the region type based on analyzed characteristics.
        
        Args:
            relative_position: Position of region (0=top, 1=bottom)
            height_ratio: Height of region relative to total image
            candle_ratio: Ratio of candlestick colors in region
            characteristics: Additional characteristics
            
        Returns:
            Tuple of (RegionType, confidence)
            
        Requirements: 1.2, 1.4
        """
        # Volume panel detection:
        # - Located at bottom of image (position > threshold)
        # - Relatively small height (< 30% of image)
        if (relative_position > self.volume_position_threshold and 
            height_ratio < 0.3):
            return RegionType.VOLUME_PANEL, 0.8
        
        # Primary chart detection:
        # - Has candlestick colors (green/red)
        # - Large region (> 40% of image height)
        if candle_ratio > 0.01 and height_ratio > 0.4:
            return RegionType.PRIMARY_CHART, 0.9
        
        # Toolbar detection:
        # - Located at top of image
        # - Very small height (< 15% of image)
        if relative_position < 0.1 and height_ratio < 0.15:
            return RegionType.TOOLBAR, 0.7
        
        # Indicator panel detection:
        # - Medium-sized region
        # - Not at extreme positions
        # - Low candlestick color ratio
        if height_ratio >= 0.1 and height_ratio <= 0.4:
            return RegionType.INDICATOR_PANEL, 0.6
        
        # If region is large but no candlestick colors detected,
        # it might still be the primary chart (different color scheme)
        if height_ratio > 0.4:
            return RegionType.PRIMARY_CHART, 0.5
        
        # Default to unknown
        return RegionType.UNKNOWN, 0.3

    def _find_primary_region(
        self, regions: List[ChartRegion]
    ) -> Optional[ChartRegion]:
        """
        Find the primary chart region from detected regions.
        
        Selection criteria:
        1. Prefer regions classified as PRIMARY_CHART
        2. Among PRIMARY_CHART regions, select the largest
        3. Fallback to largest region if no PRIMARY_CHART found
        
        Args:
            regions: List of detected ChartRegion objects
            
        Returns:
            The primary ChartRegion, or None if no regions detected
            
        Requirements: 1.3, 1.5
        """
        if not regions:
            return None
        
        # Filter for regions classified as PRIMARY_CHART
        primary_candidates = [
            r for r in regions 
            if r.region_type == RegionType.PRIMARY_CHART
        ]
        
        if primary_candidates:
            # Return the largest primary region by area
            return max(
                primary_candidates, 
                key=lambda r: r.bounding_box.area()
            )
        
        # Fallback: return largest region regardless of type
        # This handles cases where candlestick colors weren't detected
        # but there's still a main chart region
        logger.warning(
            "No PRIMARY_CHART region found, falling back to largest region"
        )
        return max(regions, key=lambda r: r.bounding_box.area())
    
    def _create_empty_result(
        self, original_size: Tuple[int, int]
    ) -> RegionDetectionResult:
        """
        Create an empty RegionDetectionResult for edge cases.
        
        Args:
            original_size: Original image size (width, height)
            
        Returns:
            Empty RegionDetectionResult
        """
        return RegionDetectionResult(
            regions=[],
            primary_region=None,
            secondary_regions=[],
            original_size=original_size,
            detection_method="horizontal_projection"
        )

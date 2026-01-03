"""
Region Extractor - Extracts pattern regions from charts.
"""

import logging
from typing import Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RegionExtractor:
    """
    Extracts pattern regions from chart images using bounding boxes.
    
    Handles padding, boundary clipping, and normalization for comparison.
    """
    
    def __init__(self, padding_percent: float = 0.15):
        """
        Initialize region extractor.
        
        Args:
            padding_percent: Percentage of padding to add around bbox (0.0 to 0.5)
        """
        self.padding_percent = max(0.0, min(0.5, padding_percent))
    
    def extract(self, image_path: str, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract pattern region from chart image.
        
        Args:
            image_path: Path to chart image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Cropped region as numpy array (BGR), or None if failed
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        return self.extract_from_array(image, bbox)
    
    def extract_from_array(self, image: np.ndarray, 
                           bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract pattern region from image array.
        
        Args:
            image: Image as numpy array (BGR)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Cropped region as numpy array (BGR), or None if failed
        """
        if image is None or len(image.shape) < 2:
            logger.error("Invalid image array")
            return None
        
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Validate bbox
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Invalid bbox dimensions: {bbox}")
            return None
        
        # Calculate padding
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * self.padding_percent)
        pad_y = int(box_height * self.padding_percent)
        
        # Apply padding with boundary clipping
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(width, x2 + pad_x)
        y2_padded = min(height, y2 + pad_y)
        
        # Extract region
        region = image[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Validate result
        if region.size == 0:
            logger.warning(f"Extracted region is empty for bbox: {bbox}")
            return None
        
        return region.copy()
    
    def normalize(self, region: np.ndarray, 
                  target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Resize region to standard size for comparison.
        
        Args:
            region: Image region as numpy array
            target_size: Target (width, height)
            
        Returns:
            Resized region
        """
        if region is None or region.size == 0:
            # Return blank image of target size
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        return cv2.resize(region, target_size, interpolation=cv2.INTER_AREA)
    
    def extract_and_normalize(self, image_path: str, 
                              bbox: Tuple[int, int, int, int],
                              target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Extract and normalize region in one step.
        
        Args:
            image_path: Path to chart image
            bbox: Bounding box (x1, y1, x2, y2)
            target_size: Target (width, height)
            
        Returns:
            Normalized region, or None if failed
        """
        region = self.extract(image_path, bbox)
        if region is None:
            return None
        
        return self.normalize(region, target_size)

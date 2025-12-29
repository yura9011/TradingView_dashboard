"""
Standard preprocessor implementation for chart image processing.

This module implements the Preprocessor interface using OpenCV for
image normalization, denoising, and ROI detection.

Requirements:
- 1.1: Normalize image dimensions preserving aspect ratio
- 1.2: Convert to consistent color space (RGB)
- 1.3: Apply filtering to reduce noise while preserving edges
- 1.4: Detect and mask overlay regions (toolbars, legends, watermarks)
- 1.5: Return descriptive error for corrupted/unreadable images
- 1.6: Output metadata including dimensions, transformations, quality score
"""

import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..models.dataclasses import BoundingBox, PreprocessResult
from .interfaces import Preprocessor


class ImageNotFoundError(Exception):
    """Raised when the image file does not exist."""
    pass


class ImageCorruptedError(Exception):
    """Raised when the image cannot be decoded."""
    pass


class ImageTooSmallError(Exception):
    """Raised when the image dimensions are below minimum requirements."""
    pass


class StandardPreprocessor(Preprocessor):
    """
    Default preprocessor implementation using OpenCV.
    
    Handles image normalization, denoising, and UI element detection
    for chart images before pattern analysis.
    
    Requirements: 1.1-1.6
    """
    
    DEFAULT_TARGET_SIZE = (1280, 720)
    MIN_IMAGE_SIZE = (100, 100)
    
    @property
    def stage_id(self) -> str:
        """Unique identifier for this preprocessor."""
        return "standard_preprocessor_v1"
    
    def process(self, input_data: str, config: Dict[str, Any]) -> PreprocessResult:
        """
        Process chart image file.
        
        Args:
            input_data: Path to image file
            config: Processing configuration with optional keys:
                - target_size: (width, height) tuple, default (1280, 720)
                - denoise: bool, whether to apply denoising, default True
                - denoise_d: int, bilateral filter diameter, default 9
                - denoise_sigma_color: int, color sigma, default 75
                - denoise_sigma_space: int, space sigma, default 75
                
        Returns:
            PreprocessResult with normalized image and metadata
            
        Raises:
            ImageNotFoundError: If image path does not exist
            ImageCorruptedError: If image cannot be decoded
            ImageTooSmallError: If image is below minimum size
        """
        # Validate input
        if not self.validate_input(input_data):
            raise ImageNotFoundError(f"Image file not found: {input_data}")
        
        # Load image
        image = cv2.imread(input_data)
        if image is None:
            raise ImageCorruptedError(
                f"Failed to decode image: {input_data}. "
                "The file may be corrupted or in an unsupported format."
            )
        
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Check minimum size
        if original_size[0] < self.MIN_IMAGE_SIZE[0] or original_size[1] < self.MIN_IMAGE_SIZE[1]:
            raise ImageTooSmallError(
                f"Image dimensions {original_size} are below minimum "
                f"requirements {self.MIN_IMAGE_SIZE}. "
                "Please provide a larger image."
            )
        
        # Track transformations applied
        transformations = []
        
        # 1. Color space conversion (BGR to RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformations.append("bgr_to_rgb")
        
        # 2. Normalize dimensions
        target = config.get("target_size", self.DEFAULT_TARGET_SIZE)
        image = self.normalize(image, target)
        transformations.append(f"resize_to_{target[0]}x{target[1]}")
        
        # 3. Denoise if configured
        if config.get("denoise", True):
            image = self.denoise(
                image,
                d=config.get("denoise_d", 9),
                sigma_color=config.get("denoise_sigma_color", 75),
                sigma_space=config.get("denoise_sigma_space", 75)
            )
            transformations.append("bilateral_denoise")
        
        # 4. Detect and mask UI elements
        masked_regions = self.detect_roi(image)
        if masked_regions:
            transformations.append(f"masked_{len(masked_regions)}_regions")
        
        # 5. Calculate quality score
        quality = self._calculate_quality(image)
        
        return PreprocessResult(
            image=image,
            original_size=original_size,
            processed_size=(image.shape[1], image.shape[0]),
            transformations=transformations,
            quality_score=quality,
            masked_regions=masked_regions
        )
    
    def normalize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image preserving aspect ratio with padding.
        
        The image is scaled to fit within target dimensions while maintaining
        its original aspect ratio. Black padding is added to reach exact
        target size.
        
        Args:
            image: Input image as numpy array (H, W, C)
            target_size: Target dimensions as (width, height)
            
        Returns:
            Normalized image with exact target dimensions
            
        Requirements: 1.1
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale factor to fit within target while preserving aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize using high-quality interpolation
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate padding to center the image
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        # Add black padding to reach exact target size
        padded = cv2.copyMakeBorder(
            resized,
            pad_h,                    # top
            target_h - new_h - pad_h, # bottom
            pad_w,                    # left
            target_w - new_w - pad_w, # right
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        return padded
    
    def denoise(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: int = 75,
        sigma_space: int = 75
    ) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving denoising.
        
        Bilateral filtering smooths images while keeping edges sharp by
        considering both spatial proximity and color similarity.
        
        Args:
            image: Input image as numpy array
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Denoised image with preserved edges
            
        Requirements: 1.3
        """
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    def detect_roi(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detect UI overlay regions to mask.
        
        Identifies non-chart elements like toolbars, legends, and watermarks
        by looking for high-contrast rectangular regions in typical UI
        locations (top/bottom edges).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of BoundingBox objects marking UI regions
            
        Requirements: 1.4
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect high-contrast rectangular regions (likely UI elements)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        masked = []
        h, w = image.shape[:2]
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # Filter for UI-like regions:
            # - Located at top or bottom 10% of image
            # - Width spans at least 10% of image width
            # - Has reasonable height (not too thin)
            is_at_edge = (y < h * 0.1) or (y + ch > h * 0.9)
            is_wide_enough = cw > w * 0.1
            has_reasonable_height = ch > 5
            
            if is_at_edge and is_wide_enough and has_reasonable_height:
                # Ensure valid bounding box (x1 < x2 and y1 < y2)
                x2 = x + cw
                y2 = y + ch
                if x < x2 and y < y2:
                    masked.append(BoundingBox(x, y, x2, y2))
        
        return masked
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate that image path exists and is readable.
        
        Args:
            input_data: Expected to be a string path to an image file
            
        Returns:
            True if input is a valid existing file path, False otherwise
        """
        if not isinstance(input_data, str):
            return False
        return os.path.exists(input_data) and os.path.isfile(input_data)
    
    def _calculate_quality(self, image: np.ndarray) -> float:
        """
        Calculate image quality score based on sharpness.
        
        Uses the variance of the Laplacian as a measure of image sharpness.
        Higher variance indicates sharper, higher quality images.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Quality score normalized to [0.0, 1.0]
            
        Requirements: 1.6
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range
        # Typical sharp images have variance > 500, blurry < 100
        return min(laplacian_var / 1000.0, 1.0)

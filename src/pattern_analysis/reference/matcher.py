"""
Pattern Matcher - Compares detected patterns with reference images.
"""

import logging
from typing import List, Optional
import cv2
import numpy as np

from .models import ReferenceImage, MatchResult, normalize_pattern_name
from .manager import ReferenceManager

logger = logging.getLogger(__name__)

# Try to import SSIM from scikit-image
try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    logger.warning("scikit-image not installed, using histogram-only comparison")


class PatternMatcher:
    """
    Compares detected pattern regions with reference images.
    
    Uses a combination of SSIM and histogram comparison to calculate
    similarity scores.
    """
    
    def __init__(self, reference_manager: ReferenceManager,
                 ssim_weight: float = 0.7,
                 hist_weight: float = 0.3):
        """
        Initialize pattern matcher.
        
        Args:
            reference_manager: ReferenceManager instance
            ssim_weight: Weight for SSIM score (0.0 to 1.0)
            hist_weight: Weight for histogram score (0.0 to 1.0)
        """
        self.reference_manager = reference_manager
        self.ssim_weight = ssim_weight
        self.hist_weight = hist_weight
        
        # Normalize weights
        total = ssim_weight + hist_weight
        if total > 0:
            self.ssim_weight = ssim_weight / total
            self.hist_weight = hist_weight / total
    
    def match(self, pattern_region: np.ndarray, 
              pattern_type: str) -> List[MatchResult]:
        """
        Find matching references for a pattern region.
        
        Args:
            pattern_region: Cropped pattern region (BGR)
            pattern_type: Pattern type name
            
        Returns:
            List of MatchResult sorted by similarity (highest first)
        """
        # Normalize pattern type
        normalized_type = normalize_pattern_name(pattern_type)
        
        # Get references for this pattern type
        references = self.reference_manager.get_references(normalized_type)
        
        if not references:
            logger.warning(f"No reference images for pattern type: {normalized_type}")
            return []
        
        # Calculate similarity for each reference
        results = []
        for ref in references:
            if ref.image_data is None:
                continue
            
            similarity = self.calculate_similarity(pattern_region, ref.image_data)
            
            result = MatchResult(
                reference=ref,
                similarity_score=similarity,
                pattern_type=normalized_type
            )
            results.append(result)
        
        # Sort by similarity (highest first)
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        
        return results
    
    def calculate_similarity(self, region: np.ndarray, 
                            reference: np.ndarray) -> float:
        """
        Calculate similarity score between two images.
        
        Uses SSIM (if available) and histogram comparison.
        
        Args:
            region: Pattern region image (BGR)
            reference: Reference image (BGR)
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Resize both to same dimensions
            target_size = (224, 224)
            region_resized = cv2.resize(region, target_size, interpolation=cv2.INTER_AREA)
            reference_resized = cv2.resize(reference, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to grayscale
            region_gray = cv2.cvtColor(region_resized, cv2.COLOR_BGR2GRAY)
            reference_gray = cv2.cvtColor(reference_resized, cv2.COLOR_BGR2GRAY)
            
            # Calculate scores
            ssim_score = self._calculate_ssim(region_gray, reference_gray)
            hist_score = self._calculate_histogram_similarity(region_gray, reference_gray)
            
            # Weighted average
            if SSIM_AVAILABLE:
                final_score = (ssim_score * self.ssim_weight) + (hist_score * self.hist_weight)
            else:
                # Use only histogram if SSIM not available
                final_score = hist_score
            
            # Clamp to [0, 1]
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        if not SSIM_AVAILABLE:
            return 0.0
        
        try:
            score = ssim(img1, img2)
            # SSIM returns -1 to 1, normalize to 0 to 1
            return (score + 1) / 2
        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
            return 0.0
    
    def _calculate_histogram_similarity(self, img1: np.ndarray, 
                                        img2: np.ndarray) -> float:
        """Calculate histogram correlation similarity."""
        try:
            # Calculate histograms
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Compare using correlation
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Correlation returns -1 to 1, normalize to 0 to 1
            return (score + 1) / 2
            
        except Exception as e:
            logger.warning(f"Histogram comparison failed: {e}")
            return 0.0
    
    def find_best_match(self, pattern_region: np.ndarray,
                        pattern_type: str) -> Optional[MatchResult]:
        """
        Find the best matching reference for a pattern.
        
        Args:
            pattern_region: Cropped pattern region (BGR)
            pattern_type: Pattern type name
            
        Returns:
            Best MatchResult or None if no matches
        """
        matches = self.match(pattern_region, pattern_type)
        return matches[0] if matches else None

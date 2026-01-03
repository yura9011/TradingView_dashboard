"""
Reference Manager - Loads and manages pattern reference images.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import cv2

from .models import ReferenceImage, normalize_pattern_name

logger = logging.getLogger(__name__)


class ReferenceManager:
    """
    Manages pattern reference images from trading books.
    
    Loads images from a directory structure and provides access
    by pattern type for comparison.
    """
    
    def __init__(self, references_dir: str = "data/pattern_references"):
        """
        Initialize reference manager.
        
        Args:
            references_dir: Path to directory containing reference images
        """
        self.references_dir = Path(references_dir)
        self._references: Dict[str, List[ReferenceImage]] = {}
        self._loaded = False
    
    def load_references(self) -> Dict[str, List[ReferenceImage]]:
        """
        Load all reference images from the directory.
        
        Supports two structures:
        1. Flat: All images in root with descriptive names
        2. Nested: Subdirectories per pattern type
        
        Returns:
            Dictionary mapping pattern types to lists of ReferenceImage
        """
        if not self.references_dir.exists():
            logger.warning(f"Reference directory not found: {self.references_dir}")
            self._create_directory_structure()
            return {}
        
        self._references = {}
        
        # Check for nested structure first
        subdirs = [d for d in self.references_dir.iterdir() if d.is_dir()]
        
        if subdirs:
            # Nested structure: subdirectories per pattern type
            self._load_nested_structure()
        else:
            # Flat structure: all images in root
            self._load_flat_structure()
        
        self._loaded = True
        
        # Log summary
        total_images = sum(len(refs) for refs in self._references.values())
        logger.info(f"Loaded {total_images} reference images for {len(self._references)} pattern types")
        
        return self._references
    
    def _load_flat_structure(self):
        """Load images from flat directory structure."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for file_path in self.references_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Extract pattern type from filename
                pattern_type = self._extract_pattern_type(file_path.stem)
                
                # Load image
                image_data = cv2.imread(str(file_path))
                if image_data is None:
                    logger.warning(f"Failed to load image: {file_path}")
                    continue
                
                # Create reference
                ref = ReferenceImage(
                    pattern_type=pattern_type,
                    image_path=str(file_path),
                    image_data=image_data,
                    metadata={"filename": file_path.name}
                )
                
                # Add to dictionary
                if pattern_type not in self._references:
                    self._references[pattern_type] = []
                self._references[pattern_type].append(ref)
                
                logger.debug(f"Loaded reference: {file_path.name} -> {pattern_type}")
    
    def _load_nested_structure(self):
        """Load images from nested directory structure."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        for subdir in self.references_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            pattern_type = normalize_pattern_name(subdir.name)
            
            # Load metadata if exists
            metadata_file = subdir / "metadata.json"
            base_metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        base_metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata: {e}")
            
            # Load images
            for file_path in subdir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_data = cv2.imread(str(file_path))
                    if image_data is None:
                        logger.warning(f"Failed to load image: {file_path}")
                        continue
                    
                    ref = ReferenceImage(
                        pattern_type=pattern_type,
                        image_path=str(file_path),
                        image_data=image_data,
                        metadata={**base_metadata, "filename": file_path.name}
                    )
                    
                    if pattern_type not in self._references:
                        self._references[pattern_type] = []
                    self._references[pattern_type].append(ref)
    
    def _extract_pattern_type(self, filename: str) -> str:
        """
        Extract pattern type from filename.
        
        Args:
            filename: Filename without extension
            
        Returns:
            Normalized pattern type
        """
        # Remove leading numbers (e.g., "001 ", "014 ")
        name = filename.lstrip("0123456789 ")
        
        # Normalize
        return normalize_pattern_name(name)
    
    def _create_directory_structure(self):
        """Create the reference directory structure."""
        self.references_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created reference directory: {self.references_dir}")
    
    def get_references(self, pattern_type: str) -> List[ReferenceImage]:
        """
        Get all reference images for a pattern type.
        
        Args:
            pattern_type: Pattern type name (will be normalized)
            
        Returns:
            List of ReferenceImage for that pattern type
        """
        if not self._loaded:
            self.load_references()
        
        # Normalize the pattern type
        normalized = normalize_pattern_name(pattern_type)
        
        return self._references.get(normalized, [])
    
    def get_all_pattern_types(self) -> List[str]:
        """Get list of all available pattern types."""
        if not self._loaded:
            self.load_references()
        
        return list(self._references.keys())
    
    def get_reference_count(self, pattern_type: str = None) -> int:
        """
        Get count of reference images.
        
        Args:
            pattern_type: Optional pattern type to count (None for total)
            
        Returns:
            Number of reference images
        """
        if not self._loaded:
            self.load_references()
        
        if pattern_type:
            normalized = normalize_pattern_name(pattern_type)
            return len(self._references.get(normalized, []))
        
        return sum(len(refs) for refs in self._references.values())
    
    def add_reference(self, pattern_type: str, image_path: str, 
                      metadata: dict = None) -> ReferenceImage:
        """
        Add a new reference image.
        
        Args:
            pattern_type: Pattern type name
            image_path: Path to image file
            metadata: Optional metadata dictionary
            
        Returns:
            Created ReferenceImage
        """
        normalized = normalize_pattern_name(pattern_type)
        
        image_data = cv2.imread(image_path)
        if image_data is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        ref = ReferenceImage(
            pattern_type=normalized,
            image_path=image_path,
            image_data=image_data,
            metadata=metadata or {}
        )
        
        if normalized not in self._references:
            self._references[normalized] = []
        self._references[normalized].append(ref)
        
        return ref

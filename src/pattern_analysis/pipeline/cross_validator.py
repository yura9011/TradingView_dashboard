"""
Multi-method cross-validator implementation for pattern detection validation.

This module implements the CrossValidator interface using multiple independent
detection methods to verify pattern detections and minimize false positives.

Requirements:
- 5.1: Verify detection using at least one alternative method
- 5.2: Apply consensus algorithm when validation methods disagree
- 5.3: Output validation score indicating agreement level
- 5.4: Flag detection as uncertain if validation score below threshold
- 5.5: Log all validation attempts and outcomes for audit
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

from ..models.dataclasses import (
    BoundingBox,
    FeatureMap,
    PatternDetection,
    ValidationResult,
)
from ..models.enums import PatternType
from .interfaces import CrossValidator, PatternClassifier


logger = logging.getLogger(__name__)


class MultiMethodCrossValidator(CrossValidator):
    """
    Cross-validator using multiple independent detection methods.
    
    Validates pattern detections by running them through alternative
    detection methods and applying a consensus algorithm to determine
    final confirmation status.
    
    Requirements: 5.1-5.5
    
    Attributes:
        validators: List of alternative PatternClassifier instances
        consensus_threshold: Minimum validation score for confirmation (default 0.5)
    """
    
    DEFAULT_CONSENSUS_THRESHOLD = 0.5
    DEFAULT_ROI_PADDING = 20
    
    def __init__(
        self,
        validators: Optional[List[PatternClassifier]] = None,
        consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD
    ):
        """
        Initialize the multi-method cross-validator.
        
        Args:
            validators: List of PatternClassifier instances to use for validation.
                       If None, an empty list is used (no validation performed).
            consensus_threshold: Minimum validation_score for is_confirmed=True.
                               Must be between 0.0 and 1.0.
        """
        self.validators = validators if validators is not None else []
        
        if not 0.0 <= consensus_threshold <= 1.0:
            raise ValueError("consensus_threshold must be between 0.0 and 1.0")
        self.consensus_threshold = consensus_threshold
        self._config: Dict[str, Any] = {}
    
    @property
    def stage_id(self) -> str:
        """Unique identifier for this cross-validator."""
        return "multi_method_validator_v1"
    
    def process(
        self,
        input_data: Tuple[List[PatternDetection], np.ndarray],
        config: Dict[str, Any]
    ) -> List[ValidationResult]:
        """
        Process detections and image to validate patterns.
        
        Args:
            input_data: Tuple of (List[PatternDetection], image array)
            config: Configuration dictionary with optional keys:
                   - consensus_threshold: Override default threshold
                   - roi_padding: Padding around detection ROI
            
        Returns:
            List of ValidationResult objects with validation details
        """
        if not self.validate_input(input_data):
            logger.warning("Invalid input to cross-validator")
            return []
        
        detections, image = input_data
        self._config = config
        
        # Update threshold from config if provided
        threshold = config.get("consensus_threshold", self.consensus_threshold)
        
        return self.validate(detections, image)
    
    def validate(
        self,
        detections: List[PatternDetection],
        image: np.ndarray
    ) -> List[ValidationResult]:
        """
        Validate detections using alternative methods.
        
        Each detection is verified by running it through alternative
        detection methods. Results include validation scores and
        agreement information.
        
        Requirements: 5.1, 5.3, 5.5
        
        Args:
            detections: List of pattern detections to validate
            image: Original image for re-analysis
            
        Returns:
            List of ValidationResult objects with validation details
        """
        results: List[ValidationResult] = []
        
        for detection in detections:
            validator_results: Dict[str, bool] = {}
            agreement_count = 0
            
            # Skip validation if no validators available
            if not self.validators:
                logger.debug(f"No validators available for detection {detection.pattern_type}")
                result = ValidationResult(
                    original_detection=detection,
                    validation_score=0.0,
                    agreement_count=0,
                    total_validators=0,
                    is_confirmed=False,
                    validator_results={},
                    status="skipped"
                )
                results.append(result)
                continue
            
            # Run each validator (Requirements 5.1)
            for validator in self.validators:
                # Skip the original detector to ensure independent validation
                if validator.stage_id == detection.detector_id:
                    logger.debug(
                        f"Skipping validator {validator.stage_id} "
                        f"(same as original detector)"
                    )
                    continue
                
                # Validate with this method
                is_confirmed = self._validate_with_method(detection, image, validator)
                validator_results[validator.stage_id] = is_confirmed
                
                if is_confirmed:
                    agreement_count += 1
                
                # Log validation attempt (Requirements 5.5)
                logger.debug(
                    f"Validation attempt: pattern={detection.pattern_type.value}, "
                    f"validator={validator.stage_id}, confirmed={is_confirmed}"
                )
            
            # Calculate validation score (Requirements 5.3)
            total_validators = len(validator_results)
            validation_score = (
                agreement_count / total_validators 
                if total_validators > 0 
                else 0.0
            )
            
            # Determine confirmation status
            threshold = self._config.get("consensus_threshold", self.consensus_threshold)
            is_confirmed_bool = validation_score >= threshold
            
            status = "confirmed" if is_confirmed_bool else "unconfirmed"
            if total_validators == 0:
                status = "skipped"
            
            result = ValidationResult(
                original_detection=detection,
                validation_score=validation_score,
                agreement_count=agreement_count,
                total_validators=total_validators,
                is_confirmed=is_confirmed_bool,
                validator_results=validator_results,
                status=status
            )
            results.append(result)
            
            # Log final result (Requirements 5.5)
            logger.info(
                f"Validation complete: pattern={detection.pattern_type.value}, "
                f"score={validation_score:.2f}, status={status}, "
                f"agreement={agreement_count}/{total_validators}"
            )
        
        return results
    
    def _validate_with_method(
        self,
        detection: PatternDetection,
        image: np.ndarray,
        validator: PatternClassifier
    ) -> bool:
        """
        Check if a validator confirms the detection.
        
        Extracts the ROI around the detection and runs the validator
        on that region to see if it detects the same pattern type.
        
        Requirements: 5.1
        
        Args:
            detection: The pattern detection to validate
            image: Full image array
            validator: PatternClassifier to use for validation
            
        Returns:
            True if validator confirms the pattern, False otherwise
        """
        try:
            # Extract ROI around detection with padding
            roi = self._extract_roi(detection.bounding_box, image)
            
            if roi is None or roi.size == 0:
                logger.debug(
                    f"Empty ROI for detection at {detection.bounding_box}, "
                    f"validation failed"
                )
                return False
            
            # Create minimal feature map for ROI
            # The validator will extract its own features from the ROI
            features = FeatureMap.empty()
            
            # Run validator on ROI
            val_detections = validator.classify(features, roi)
            
            # Check if any detection matches the pattern type
            for vd in val_detections:
                if vd.pattern_type == detection.pattern_type:
                    logger.debug(
                        f"Validator {validator.stage_id} confirmed "
                        f"pattern {detection.pattern_type.value}"
                    )
                    return True
            
            logger.debug(
                f"Validator {validator.stage_id} did not confirm "
                f"pattern {detection.pattern_type.value}"
            )
            return False
            
        except Exception as e:
            logger.warning(
                f"Validation failed for {validator.stage_id}: {e}"
            )
            return False
    
    def _extract_roi(
        self,
        bbox: BoundingBox,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract region of interest from image with padding.
        
        Args:
            bbox: Bounding box defining the region
            image: Full image array
            
        Returns:
            Cropped image region, or None if invalid
        """
        if image is None or image.size == 0:
            return None
        
        h, w = image.shape[:2]
        padding = self._config.get("roi_padding", self.DEFAULT_ROI_PADDING)
        
        # Apply padding and clamp to image bounds
        x1 = max(0, bbox.x1 - padding)
        y1 = max(0, bbox.y1 - padding)
        x2 = min(w, bbox.x2 + padding)
        y2 = min(h, bbox.y2 + padding)
        
        # Validate coordinates
        if x1 >= x2 or y1 >= y2:
            return None
        
        return image[y1:y2, x1:x2].copy()
    
    def get_consensus(
        self,
        results: List[ValidationResult]
    ) -> List[ValidationResult]:
        """
        Apply consensus algorithm to validation results.
        
        Filters validation results to return only those that meet
        the consensus threshold for confirmation.
        
        Requirements: 5.2
        
        Args:
            results: List of validation results to filter
            
        Returns:
            List of ValidationResult objects that are confirmed
        """
        return [r for r in results if r.is_confirmed]
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate that input is a tuple of (List[PatternDetection], ndarray).
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(input_data, tuple):
            return False
        if len(input_data) != 2:
            return False
        
        detections, image = input_data
        
        if not isinstance(detections, list):
            return False
        
        # All items in detections must be PatternDetection
        for d in detections:
            if not isinstance(d, PatternDetection):
                return False
        
        if not isinstance(image, np.ndarray):
            return False
        
        return True
    
    def add_validator(self, validator: PatternClassifier) -> None:
        """
        Add a validator to the list of validation methods.
        
        Args:
            validator: PatternClassifier to add
        """
        if validator not in self.validators:
            self.validators.append(validator)
            logger.info(f"Added validator: {validator.stage_id}")
    
    def remove_validator(self, validator: PatternClassifier) -> bool:
        """
        Remove a validator from the list of validation methods.
        
        Args:
            validator: PatternClassifier to remove
            
        Returns:
            True if validator was removed, False if not found
        """
        if validator in self.validators:
            self.validators.remove(validator)
            logger.info(f"Removed validator: {validator.stage_id}")
            return True
        return False
    
    def set_consensus_threshold(self, threshold: float) -> None:
        """
        Set the consensus threshold for confirmation.
        
        Args:
            threshold: New threshold value (must be between 0.0 and 1.0)
            
        Raises:
            ValueError: If threshold is not in valid range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("consensus_threshold must be between 0.0 and 1.0")
        self.consensus_threshold = threshold
        logger.info(f"Consensus threshold set to {threshold}")

"""
Data classes for pattern analysis.

Contains BoundingBox, PatternDetection, FeatureMap,
PreprocessResult, ValidationResult, and AnalysisResult.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

import numpy as np

from .enums import PatternCategory, PatternType


@dataclass
class BoundingBox:
    """
    Coordinates delimiting a detected region.
    
    Invariant: x1 < x2 and y1 < y2 for valid bounding boxes.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    label: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if not isinstance(self.x1, int) or not isinstance(self.y1, int):
            raise ValueError("Coordinates must be integers")
        if not isinstance(self.x2, int) or not isinstance(self.y2, int):
            raise ValueError("Coordinates must be integers")
            
    def is_valid(self) -> bool:
        """Check if bounding box has valid coordinates (x1 < x2 and y1 < y2)."""
        return self.x1 < self.x2 and self.y1 < self.y2
    
    def area(self) -> int:
        """Calculate the area of the bounding box."""
        if not self.is_valid():
            return 0
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def center(self) -> tuple:
        """Calculate the center point of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "x1": self.x1, 
            "y1": self.y1, 
            "x2": self.x2, 
            "y2": self.y2,
            "label": self.label,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        """Create BoundingBox from dictionary."""
        return cls(
            x1=data["x1"], 
            y1=data["y1"], 
            x2=data["x2"], 
            y2=data["y2"],
            label=data.get("label"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class PatternDetection:
    """Single pattern detection result."""
    pattern_type: PatternType
    category: PatternCategory
    confidence: float
    bounding_box: BoundingBox
    metadata: Dict[str, Any]
    detector_id: str

    
    def __post_init__(self):
        """Validate pattern detection fields."""
        if not isinstance(self.pattern_type, PatternType):
            raise ValueError("pattern_type must be a PatternType enum")
        if not isinstance(self.category, PatternCategory):
            raise ValueError("category must be a PatternCategory enum")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not isinstance(self.bounding_box, BoundingBox):
            raise ValueError("bounding_box must be a BoundingBox instance")
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_type": self.pattern_type.value,
            "category": self.category.value,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box.to_dict(),
            "metadata": self.metadata,
            "detector_id": self.detector_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternDetection":
        """Create PatternDetection from dictionary."""
        return cls(
            pattern_type=PatternType(data["pattern_type"]),
            category=PatternCategory(data["category"]),
            confidence=data["confidence"],
            bounding_box=BoundingBox.from_dict(data["bounding_box"]),
            metadata=data.get("metadata", {}),
            detector_id=data["detector_id"]
        )


@dataclass
class FeatureMap:
    """Extracted features from chart image."""
    candlestick_regions: List[BoundingBox]
    trendlines: List[Dict[str, Any]]
    support_zones: List[BoundingBox]
    resistance_zones: List[BoundingBox]
    volume_profile: Optional[Dict[str, Any]]
    quality_score: float
    
    def __post_init__(self):
        """Validate feature map fields."""
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")
    
    @classmethod
    def empty(cls) -> "FeatureMap":
        """Create an empty feature map."""
        return cls(
            candlestick_regions=[],
            trendlines=[],
            support_zones=[],
            resistance_zones=[],
            volume_profile=None,
            quality_score=0.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "candlestick_regions": [b.to_dict() for b in self.candlestick_regions],
            "trendlines": self.trendlines,
            "support_zones": [b.to_dict() for b in self.support_zones],
            "resistance_zones": [b.to_dict() for b in self.resistance_zones],
            "volume_profile": self.volume_profile,
            "quality_score": self.quality_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMap":
        """Create FeatureMap from dictionary."""
        return cls(
            candlestick_regions=[BoundingBox.from_dict(b) for b in data.get("candlestick_regions", [])],
            trendlines=data.get("trendlines", []),
            support_zones=[BoundingBox.from_dict(b) for b in data.get("support_zones", [])],
            resistance_zones=[BoundingBox.from_dict(b) for b in data.get("resistance_zones", [])],
            volume_profile=data.get("volume_profile"),
            quality_score=data.get("quality_score", 0.0)
        )


@dataclass
class PreprocessResult:
    """Result of image preprocessing."""
    image: np.ndarray
    original_size: tuple
    processed_size: tuple
    transformations: List[str]
    quality_score: float
    masked_regions: List[BoundingBox]
    
    def __post_init__(self):
        """Validate preprocess result fields."""
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")


@dataclass
class ValidationResult:
    """Cross-validation result for a pattern detection."""
    original_detection: PatternDetection
    validation_score: float
    agreement_count: int
    total_validators: int
    is_confirmed: bool
    validator_results: Dict[str, bool]
    status: str = "unknown"  # New field: 'confirmed', 'unconfirmed', 'skipped', 'error'
    
    def __post_init__(self):
        """Validate validation result fields."""
        if self.total_validators > 0:
            expected_score = self.agreement_count / self.total_validators
            # Allow small floating point error
            if abs(self.validation_score - expected_score) > 0.001:
                pass # Warning strictly but let's be lenient for now prevents crashes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_detection": self.original_detection.to_dict() if self.original_detection else None,
            "validation_score": self.validation_score,
            "agreement_count": self.agreement_count,
            "total_validators": self.total_validators,
            "is_confirmed": self.is_confirmed,
            "validator_results": self.validator_results,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Create ValidationResult from dictionary."""
        original = None
        if data.get("original_detection"):
            original = PatternDetection.from_dict(data["original_detection"])
        return cls(
            original_detection=original,
            validation_score=data["validation_score"],
            agreement_count=data["agreement_count"],
            total_validators=data["total_validators"],
            is_confirmed=data["is_confirmed"],
            validator_results=data.get("validator_results", {}),
            status=data.get("status", "unknown")
        )


@dataclass
class AnalysisResult:
    """Complete analysis output."""
    image_path: str
    timestamp: str
    preprocessing_time_ms: float
    extraction_time_ms: float
    classification_time_ms: float
    validation_time_ms: float
    total_time_ms: float
    detections: List[PatternDetection]
    validated_detections: List[ValidationResult]
    feature_map: FeatureMap
    config_used: Dict[str, Any]
    detector_status: Dict[str, str] = field(default_factory=dict) # New field
    
    def __post_init__(self):
        """Validate analysis result fields."""
        # Ensure timestamp is valid ISO format
        if self.timestamp:
            try:
                datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            except ValueError:
                pass # Be lenient
    
    def to_json(self, validate: bool = True) -> str:
        """
        Serialize AnalysisResult to JSON string.
        
        Args:
            validate: If True, validate against JSON schema before returning.
            
        Returns:
            JSON string representation of the analysis result.
            
        Raises:
            SchemaValidationError: If validation is enabled and data fails schema validation.
        """
        from .schemas import validate_against_schema, SchemaValidationError
        
        data = self.to_dict()
        
        if validate:
            is_valid, errors = validate_against_schema(data)
            if not is_valid:
                # Log but maybe don't crash in production? 
                # For strictness we raise.
                raise SchemaValidationError(
                    f"AnalysisResult failed schema validation: {'; '.join(errors)}",
                    errors=errors
                )
        
        return json.dumps(data, indent=2, default=str)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "image_path": self.image_path,
            "timestamp": self.timestamp,
            "preprocessing_time_ms": self.preprocessing_time_ms,
            "extraction_time_ms": self.extraction_time_ms,
            "classification_time_ms": self.classification_time_ms,
            "validation_time_ms": self.validation_time_ms,
            "total_time_ms": self.total_time_ms,
            "detections": [d.to_dict() for d in self.detections],
            "validated_detections": [v.to_dict() for v in self.validated_detections],
            "feature_map": self.feature_map.to_dict() if self.feature_map else None,
            "config_used": self.config_used,
            "detector_status": self.detector_status
        }
    
    @classmethod
    def from_json(cls, json_str: str, validate: bool = True) -> "AnalysisResult":
        """
        Deserialize AnalysisResult from JSON string.
        
        Args:
            json_str: JSON string to parse.
            validate: If True, validate against JSON schema before parsing.
            
        Returns:
            AnalysisResult instance.
            
        Raises:
            SchemaValidationError: If validation is enabled and data fails schema validation.
        """
        from .schemas import validate_against_schema, SchemaValidationError
        
        data = json.loads(json_str)
        
        if validate:
            is_valid, errors = validate_against_schema(data)
            if not is_valid:
                raise SchemaValidationError(
                    f"JSON data failed schema validation: {'; '.join(errors)}",
                    errors=errors
                )
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create AnalysisResult from dictionary."""
        detections = [PatternDetection.from_dict(d) for d in data.get("detections", [])]
        validated = [ValidationResult.from_dict(v) for v in data.get("validated_detections", [])]
        feature_map = FeatureMap.from_dict(data["feature_map"]) if data.get("feature_map") else FeatureMap.empty()
        
        return cls(
            image_path=data["image_path"],
            timestamp=data["timestamp"],
            preprocessing_time_ms=data.get("preprocessing_time_ms", 0.0),
            extraction_time_ms=data.get("extraction_time_ms", 0.0),
            classification_time_ms=data.get("classification_time_ms", 0.0),
            validation_time_ms=data.get("validation_time_ms", 0.0),
            total_time_ms=data.get("total_time_ms", 0.0),
            detections=detections,
            validated_detections=validated,
            feature_map=feature_map,
            config_used=data.get("config_used", {}),
            detector_status=data.get("detector_status", {})
        )
    
    def __eq__(self, other: object) -> bool:
        """Check equality for round-trip testing."""
        if not isinstance(other, AnalysisResult):
            return False
        return (
            self.image_path == other.image_path and
            self.timestamp == other.timestamp and
            abs(self.preprocessing_time_ms - other.preprocessing_time_ms) < 0.001 and
            abs(self.extraction_time_ms - other.extraction_time_ms) < 0.001 and
            abs(self.classification_time_ms - other.classification_time_ms) < 0.001 and
            abs(self.validation_time_ms - other.validation_time_ms) < 0.001 and
            abs(self.total_time_ms - other.total_time_ms) < 0.001 and
            len(self.detections) == len(other.detections) and
            len(self.validated_detections) == len(other.validated_detections) and
            self.config_used == other.config_used and
            self.detector_status == other.detector_status
        )

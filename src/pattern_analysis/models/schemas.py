"""
JSON schemas for pattern analysis data validation.

Contains schema definitions and validation functions for AnalysisResult
and related data structures.
"""

import json
from typing import Dict, Any, List, Tuple

# JSON Schema for AnalysisResult as defined in design document
ANALYSIS_RESULT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["image_path", "timestamp", "detections", "total_time_ms"],
    "properties": {
        "image_path": {"type": "string"},
        "timestamp": {"type": "string"},
        "total_time_ms": {"type": "number"},
        "preprocessing_time_ms": {"type": "number"},
        "extraction_time_ms": {"type": "number"},
        "classification_time_ms": {"type": "number"},
        "validation_time_ms": {"type": "number"},
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["pattern_type", "category", "confidence", "bounding_box"],
                "properties": {
                    "pattern_type": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["reversal", "continuation", "bilateral"]
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "bounding_box": {
                        "type": "object",
                        "required": ["x1", "y1", "x2", "y2"],
                        "properties": {
                            "x1": {"type": "integer"},
                            "y1": {"type": "integer"},
                            "x2": {"type": "integer"},
                            "y2": {"type": "integer"}
                        }
                    },
                    "metadata": {"type": "object"},
                    "detector_id": {"type": "string"},
                    "is_validated": {"type": "boolean"},
                    "validation_score": {"type": "number"}
                }
            }
        },
        "validated_detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original_detection": {"type": ["object", "null"]},
                    "validation_score": {"type": "number"},
                    "agreement_count": {"type": "integer"},
                    "total_validators": {"type": "integer"},
                    "is_confirmed": {"type": "boolean"},
                    "validator_results": {"type": "object"}
                }
            }
        },
        "feature_map": {
            "type": ["object", "null"],
            "properties": {
                "candlestick_regions": {"type": "array"},
                "trendlines": {"type": "array"},
                "support_zones": {"type": "array"},
                "resistance_zones": {"type": "array"},
                "volume_profile": {"type": ["object", "null"]},
                "quality_score": {"type": "number"}
            }
        },
        "config_used": {"type": "object"}
    }
}


class SchemaValidationError(Exception):
    """Raised when JSON data fails schema validation."""
    
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []


def validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
    """
    Validate data against JSON schema.
    
    This is a simplified validator that checks required fields and types.
    For production use, consider using jsonschema library.
    
    Args:
        data: Dictionary to validate
        schema: JSON schema to validate against (defaults to ANALYSIS_RESULT_SCHEMA)
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    if schema is None:
        schema = ANALYSIS_RESULT_SCHEMA
    
    errors = []
    
    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Check property types
    properties = schema.get("properties", {})
    for field, field_schema in properties.items():
        if field in data:
            value = data[field]
            field_type = field_schema.get("type")
            
            if field_type == "string" and not isinstance(value, str):
                errors.append(f"Field '{field}' must be a string, got {type(value).__name__}")
            elif field_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Field '{field}' must be a number, got {type(value).__name__}")
            elif field_type == "integer" and not isinstance(value, int):
                errors.append(f"Field '{field}' must be an integer, got {type(value).__name__}")
            elif field_type == "array" and not isinstance(value, list):
                errors.append(f"Field '{field}' must be an array, got {type(value).__name__}")
            elif field_type == "object" and not isinstance(value, dict):
                errors.append(f"Field '{field}' must be an object, got {type(value).__name__}")
            elif field_type == "boolean" and not isinstance(value, bool):
                errors.append(f"Field '{field}' must be a boolean, got {type(value).__name__}")
            
            # Check enum values
            if "enum" in field_schema and value not in field_schema["enum"]:
                errors.append(f"Field '{field}' must be one of {field_schema['enum']}, got '{value}'")
            
            # Check numeric bounds
            if isinstance(value, (int, float)):
                if "minimum" in field_schema and value < field_schema["minimum"]:
                    errors.append(f"Field '{field}' must be >= {field_schema['minimum']}, got {value}")
                if "maximum" in field_schema and value > field_schema["maximum"]:
                    errors.append(f"Field '{field}' must be <= {field_schema['maximum']}, got {value}")
    
    # Validate detections array items
    if "detections" in data and isinstance(data["detections"], list):
        detection_schema = properties.get("detections", {}).get("items", {})
        for i, detection in enumerate(data["detections"]):
            det_errors = _validate_detection(detection, detection_schema, i)
            errors.extend(det_errors)
    
    return len(errors) == 0, errors


def _validate_detection(detection: Dict[str, Any], schema: Dict[str, Any], index: int) -> List[str]:
    """Validate a single detection against schema."""
    errors = []
    
    required = schema.get("required", [])
    for field in required:
        if field not in detection:
            errors.append(f"Detection[{index}]: Missing required field '{field}'")
    
    properties = schema.get("properties", {})
    
    # Validate category enum
    if "category" in detection:
        category_schema = properties.get("category", {})
        if "enum" in category_schema and detection["category"] not in category_schema["enum"]:
            errors.append(f"Detection[{index}]: category must be one of {category_schema['enum']}")
    
    # Validate confidence bounds
    if "confidence" in detection:
        conf = detection["confidence"]
        if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
            errors.append(f"Detection[{index}]: confidence must be between 0 and 1")
    
    # Validate bounding_box
    if "bounding_box" in detection:
        bbox = detection["bounding_box"]
        if isinstance(bbox, dict):
            for coord in ["x1", "y1", "x2", "y2"]:
                if coord not in bbox:
                    errors.append(f"Detection[{index}]: bounding_box missing '{coord}'")
                elif not isinstance(bbox[coord], int):
                    errors.append(f"Detection[{index}]: bounding_box.{coord} must be integer")
    
    return errors

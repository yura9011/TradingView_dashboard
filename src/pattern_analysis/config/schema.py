"""
Configuration schema definitions.

Feature: chart-pattern-analysis-framework
Requirements: 10.1
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ConfigSchemaField:
    """Definition of a configuration field for validation."""
    name: str
    field_type: type
    required: bool = False
    default: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    nested_schema: Optional[Dict[str, 'ConfigSchemaField']] = None


# Define the configuration schema
CONFIG_SCHEMA: Dict[str, ConfigSchemaField] = {
    "preprocessing": ConfigSchemaField(
        name="preprocessing",
        field_type=dict,
        required=False,
        default={},
        nested_schema={
            "target_width": ConfigSchemaField("target_width", int, default=1280, min_value=100, max_value=4096),
            "target_height": ConfigSchemaField("target_height", int, default=720, min_value=100, max_value=4096),
            "color_space": ConfigSchemaField("color_space", str, default="RGB", allowed_values=["RGB", "BGR", "GRAY"]),
            "denoise": ConfigSchemaField("denoise", dict, default={"enabled": True}),
            "mask_ui_elements": ConfigSchemaField("mask_ui_elements", bool, default=True),
            "quality": ConfigSchemaField("quality", dict, default={"min_score": 0.3}),
        }
    ),
    "feature_extraction": ConfigSchemaField(
        name="feature_extraction",
        field_type=dict,
        required=False,
        default={},
    ),
    "classification": ConfigSchemaField(
        name="classification",
        field_type=dict,
        required=False,
        default={},
        nested_schema={
            "confidence_threshold": ConfigSchemaField("confidence_threshold", float, default=0.3, min_value=0.0, max_value=1.0),
            "high_confidence_threshold": ConfigSchemaField("high_confidence_threshold", float, default=0.7, min_value=0.0, max_value=1.0),
            "iou_threshold": ConfigSchemaField("iou_threshold", float, default=0.5, min_value=0.0, max_value=1.0),
        }
    ),
    "cross_validation": ConfigSchemaField(
        name="cross_validation",
        field_type=dict,
        required=False,
        default={},
        nested_schema={
            "enabled": ConfigSchemaField("enabled", bool, default=True),
            "consensus_threshold": ConfigSchemaField("consensus_threshold", float, default=0.5, min_value=0.0, max_value=1.0),
        }
    ),
    "output": ConfigSchemaField(
        name="output",
        field_type=dict,
        required=False,
        default={},
    ),
    "registry": ConfigSchemaField(
        name="registry",
        field_type=dict,
        required=False,
        default={},
    ),
    "metrics": ConfigSchemaField(
        name="metrics",
        field_type=dict,
        required=False,
        default={},
        nested_schema={
            "enabled": ConfigSchemaField("enabled", bool, default=True),
            "history_size": ConfigSchemaField("history_size", int, default=1000, min_value=1),
        }
    ),
    "logging": ConfigSchemaField(
        name="logging",
        field_type=dict,
        required=False,
        default={},
        nested_schema={
            "level": ConfigSchemaField("level", str, default="INFO", allowed_values=["DEBUG", "INFO", "WARNING", "ERROR"]),
            "format": ConfigSchemaField("format", str, default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            "file": ConfigSchemaField("file", str, default="logs/pattern_analysis.log"),
        }
    ),
}

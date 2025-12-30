"""
Configuration validation logic.

Feature: chart-pattern-analysis-framework
Requirements: 10.1
"""

from typing import Any, Dict, List

from .schema import ConfigSchemaField, CONFIG_SCHEMA
from .exceptions import ConfigValidationError


class ConfigValidator:
    """Validates configuration against schema."""
    
    def __init__(self, schema: Dict[str, ConfigSchemaField] = None):
        """Initialize with schema (defaults to CONFIG_SCHEMA)."""
        self.schema = schema or CONFIG_SCHEMA
    
    def validate(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        errors = []
        
        for section_name, schema_field in self.schema.items():
            section_errors = self._validate_field(
                config.get(section_name),
                schema_field,
                path=section_name
            )
            errors.extend(section_errors)
        
        if errors:
            raise ConfigValidationError(
                "Configuration validation failed",
                errors=errors
            )
    
    def _validate_field(
        self,
        value: Any,
        schema: ConfigSchemaField,
        path: str
    ) -> List[str]:
        """
        Validate a single configuration field.
        
        Args:
            value: Value to validate
            schema: Schema definition for the field
            path: Dot-separated path for error messages
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required
        if value is None:
            if schema.required:
                errors.append(f"Missing required field: {path}")
            return errors
        
        # Check type
        if not isinstance(value, schema.field_type):
            # Allow int for float fields
            if schema.field_type == float and isinstance(value, int):
                value = float(value)
            else:
                errors.append(
                    f"Invalid type for {path}: expected {schema.field_type.__name__}, "
                    f"got {type(value).__name__}"
                )
                return errors
        
        # Check numeric bounds
        if isinstance(value, (int, float)):
            if schema.min_value is not None and value < schema.min_value:
                errors.append(
                    f"Value for {path} is below minimum: {value} < {schema.min_value}"
                )
            if schema.max_value is not None and value > schema.max_value:
                errors.append(
                    f"Value for {path} exceeds maximum: {value} > {schema.max_value}"
                )
        
        # Check allowed values
        if schema.allowed_values is not None:
            if value not in schema.allowed_values:
                errors.append(
                    f"Invalid value for {path}: '{value}' not in {schema.allowed_values}"
                )
        
        # Validate nested schema
        if schema.nested_schema and isinstance(value, dict):
            for nested_name, nested_schema in schema.nested_schema.items():
                nested_errors = self._validate_field(
                    value.get(nested_name),
                    nested_schema,
                    path=f"{path}.{nested_name}"
                )
                errors.extend(nested_errors)
        
        return errors

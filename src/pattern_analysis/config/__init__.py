"""
Configuration management module.

Handles loading, validation, and access to configuration
settings from YAML files with environment variable overrides.

Feature: chart-pattern-analysis-framework
Requirements: 10.1, 10.3, 10.5
"""

from .manager import (
    ConfigurationManager,
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConfigParseError,
    ConfigSchemaField,
    CONFIG_SCHEMA,
    get_config_manager,
    get_config,
    get_logger,
)

__all__ = [
    # Main class
    "ConfigurationManager",
    
    # Exceptions
    "ConfigurationError",
    "ConfigFileNotFoundError",
    "ConfigValidationError",
    "ConfigParseError",
    
    # Schema
    "ConfigSchemaField",
    "CONFIG_SCHEMA",
    
    # Convenience functions
    "get_config_manager",
    "get_config",
    "get_logger",
]

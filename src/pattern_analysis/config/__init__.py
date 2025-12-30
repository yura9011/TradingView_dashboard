"""
Configuration management module.

Handles loading, validation, and access to configuration
settings from YAML files with environment variable overrides.

Feature: chart-pattern-analysis-framework
Requirements: 10.1, 10.3, 10.5
"""

# Import from modular structure
from .exceptions import (
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConfigParseError,
)
from .schema import ConfigSchemaField, CONFIG_SCHEMA
from .validators import ConfigValidator
from .env_parser import EnvConfigParser
from .manager_new import (
    ConfigurationManager,
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
    
    # Component classes
    "ConfigValidator",
    "EnvConfigParser",
    
    # Convenience functions
    "get_config_manager",
    "get_config",
    "get_logger",
]

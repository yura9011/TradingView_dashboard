"""
Configuration Manager for Chart Pattern Analysis Framework.

Handles loading, validation, and access to configuration settings
from YAML files with environment variable overrides.

Feature: chart-pattern-analysis-framework
Requirements: 10.1, 10.3, 10.5
"""

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from copy import deepcopy

import yaml


# =============================================================================
# Exceptions
# =============================================================================

class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []
    
    def __str__(self) -> str:
        if self.errors:
            error_list = "\n  - ".join(self.errors)
            return f"{self.message}\n  - {error_list}"
        return self.message


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when configuration file is not found."""
    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass


class ConfigParseError(ConfigurationError):
    """Raised when YAML parsing fails."""
    pass


# =============================================================================
# Configuration Schema Definition
# =============================================================================

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


# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigurationManager:
    """
    Manages configuration loading, validation, and access.
    
    Supports:
    - Loading from YAML files
    - Environment variable overrides (PATTERN_ANALYSIS_<SECTION>_<KEY>)
    - Schema validation at startup
    - Configurable logging
    
    Requirements: 10.1, 10.3, 10.5
    """
    
    ENV_PREFIX = "PATTERN_ANALYSIS"
    DEFAULT_CONFIG_PATH = "config/pattern_analysis.yaml"
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        auto_load: bool = True,
        validate: bool = True,
        apply_env_overrides: bool = True
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
            auto_load: Whether to load configuration on init
            validate: Whether to validate configuration against schema
            apply_env_overrides: Whether to apply environment variable overrides
        """
        self._config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Dict[str, Any] = {}
        self._validate = validate
        self._apply_env_overrides = apply_env_overrides
        self._logger: Optional[logging.Logger] = None
        
        if auto_load:
            self.load()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration dictionary."""
        return deepcopy(self._config)
    
    @property
    def config_path(self) -> str:
        """Get the configuration file path."""
        return self._config_path
    
    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment overrides.
        
        Args:
            config_path: Optional path to override the default config path
            
        Returns:
            Loaded and processed configuration dictionary
            
        Raises:
            ConfigFileNotFoundError: If config file doesn't exist
            ConfigParseError: If YAML parsing fails
            ConfigValidationError: If validation fails
        """
        path = config_path or self._config_path
        
        # Load YAML file
        self._config = self._load_yaml(path)
        
        # Apply environment variable overrides
        if self._apply_env_overrides:
            self._config = self._apply_environment_overrides(self._config)
        
        # Validate configuration
        if self._validate:
            self._validate_config(self._config)
        
        # Setup logging based on config
        self._setup_logging()
        
        return self.config
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Parsed configuration dictionary
            
        Raises:
            ConfigFileNotFoundError: If file doesn't exist
            ConfigParseError: If YAML parsing fails
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise ConfigFileNotFoundError(
                f"Configuration file not found: {path}",
                errors=[f"Expected file at: {file_path.absolute()}"]
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except yaml.YAMLError as e:
            raise ConfigParseError(
                f"Failed to parse YAML configuration: {path}",
                errors=[str(e)]
            )

    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables follow the pattern:
        PATTERN_ANALYSIS_<SECTION>_<KEY>=value
        
        For nested keys, use double underscore:
        PATTERN_ANALYSIS_PREPROCESSING_DENOISE__ENABLED=true
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        result = deepcopy(config)
        prefix = f"{self.ENV_PREFIX}_"
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            
            # Remove prefix and split into path components
            key_path = env_key[len(prefix):].lower()
            
            # Handle nested keys (double underscore for nesting)
            path_parts = self._parse_env_key_path(key_path)
            
            if not path_parts:
                continue
            
            # Convert value to appropriate type
            typed_value = self._convert_env_value(env_value)
            
            # Set the value in config
            self._set_nested_value(result, path_parts, typed_value)
        
        return result
    
    def _parse_env_key_path(self, key_path: str) -> List[str]:
        """
        Parse environment variable key path into nested path components.
        
        The first underscore separates section from key.
        Double underscore indicates nested key within a section.
        
        Examples:
            "preprocessing_target_width" -> ["preprocessing", "target_width"]
            "preprocessing_denoise__enabled" -> ["preprocessing", "denoise", "enabled"]
            "logging_level" -> ["logging", "level"]
            "feature_extraction_candlesticks__min_height_ratio" -> ["feature_extraction", "candlesticks", "min_height_ratio"]
        
        Args:
            key_path: Lowercase key path from environment variable
            
        Returns:
            List of path components
        """
        # Known section names (to handle multi-word sections like "feature_extraction")
        known_sections = [
            "preprocessing", "feature_extraction", "classification",
            "cross_validation", "output", "registry", "metrics", "logging"
        ]
        
        # First, try to match a known section
        section = None
        remaining = key_path
        
        for known in known_sections:
            if key_path.startswith(known + "_"):
                section = known
                remaining = key_path[len(known) + 1:]  # +1 for the underscore
                break
        
        if section is None:
            # No known section found, use first part as section
            first_underscore = key_path.find("_")
            if first_underscore == -1:
                return [key_path]
            section = key_path[:first_underscore]
            remaining = key_path[first_underscore + 1:]
        
        # Now parse the remaining part for nested keys (double underscore)
        # Double underscore indicates nesting
        result = [section]
        
        if remaining:
            # Split by double underscore for nesting
            nested_parts = remaining.split("__")
            result.extend(nested_parts)
        
        return result
    
    def _convert_env_value(self, value: str) -> Any:
        """
        Convert environment variable string value to appropriate Python type.
        
        Supports:
        - Booleans: "true", "false", "yes", "no", "1", "0"
        - Integers: "123"
        - Floats: "1.23"
        - Lists: "[1, 2, 3]" or "1,2,3"
        - Strings: everything else
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value
        """
        # Handle None/null
        if value.lower() in ("none", "null", ""):
            return None
        
        # Handle booleans
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        
        # Handle integers
        try:
            return int(value)
        except ValueError:
            pass
        
        # Handle floats
        try:
            return float(value)
        except ValueError:
            pass
        
        # Handle JSON-like lists
        if value.startswith("[") and value.endswith("]"):
            try:
                import json
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Handle comma-separated lists
        if "," in value and not value.startswith('"'):
            parts = [p.strip() for p in value.split(",")]
            # Try to convert each part
            return [self._convert_env_value(p) for p in parts]
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: List[str], value: Any) -> None:
        """
        Set a value in a nested dictionary using a path.
        
        Args:
            config: Configuration dictionary to modify
            path: List of keys forming the path
            value: Value to set
        """
        current = config
        
        for i, key in enumerate(path[:-1]):
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Convert non-dict to dict if we need to nest further
                current[key] = {}
            current = current[key]
        
        # Set the final value
        if path:
            current[path[-1]] = value
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        errors = []
        
        for section_name, schema_field in CONFIG_SCHEMA.items():
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
    
    def _setup_logging(self) -> None:
        """
        Setup logging based on configuration.
        
        Requirements: 10.3
        """
        log_config = self._config.get("logging", {})
        
        level_str = log_config.get("level", "INFO").upper()
        log_format = log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_file = log_config.get("file", "logs/pattern_analysis.log")
        
        # Get log level
        level = getattr(logging, level_str, logging.INFO)
        
        # Create logger for pattern analysis
        self._logger = logging.getLogger("pattern_analysis")
        self._logger.setLevel(level)
        
        # Clear existing handlers
        self._logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # File handler (if path is specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # Always capture debug to file
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        
        self._logger.debug(f"Logging configured: level={level_str}, file={log_file}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated key path.
        
        Args:
            key: Dot-separated key path (e.g., "preprocessing.target_width")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split(".")
        current = self._config
        
        for part in parts:
            if not isinstance(current, dict):
                return default
            if part not in current:
                return default
            current = current[part]
        
        return current
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name (e.g., "preprocessing")
            
        Returns:
            Section dictionary or empty dict if not found
        """
        return deepcopy(self._config.get(section, {}))
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by dot-separated key path.
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        parts = key.split(".")
        self._set_nested_value(self._config, parts, value)
    
    def reload(self) -> Dict[str, Any]:
        """
        Reload configuration from file.
        
        Returns:
            Reloaded configuration dictionary
        """
        return self.load(self._config_path)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (appended to "pattern_analysis")
            
        Returns:
            Logger instance
        """
        if name:
            return logging.getLogger(f"pattern_analysis.{name}")
        return logging.getLogger("pattern_analysis")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.
        
        Returns:
            Copy of configuration dictionary
        """
        return self.config
    
    def __repr__(self) -> str:
        return f"ConfigurationManager(path={self._config_path}, sections={list(self._config.keys())})"


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_manager: Optional[ConfigurationManager] = None


def get_config_manager(
    config_path: Optional[str] = None,
    force_reload: bool = False
) -> ConfigurationManager:
    """
    Get the default configuration manager instance.
    
    Args:
        config_path: Optional path to configuration file
        force_reload: Whether to force reload configuration
        
    Returns:
        ConfigurationManager instance
    """
    global _default_manager
    
    if _default_manager is None or force_reload:
        _default_manager = ConfigurationManager(
            config_path=config_path,
            auto_load=True,
            validate=True,
            apply_env_overrides=True
        )
    
    return _default_manager


def get_config(key: str, default: Any = None) -> Any:
    """
    Get a configuration value using the default manager.
    
    Args:
        key: Dot-separated key path
        default: Default value if not found
        
    Returns:
        Configuration value
    """
    manager = get_config_manager()
    return manager.get(key, default)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance using the default manager.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    manager = get_config_manager()
    return manager.get_logger(name)

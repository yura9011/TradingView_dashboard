"""
Configuration Manager for Chart Pattern Analysis Framework.

Handles loading, validation, and access to configuration settings
from YAML files with environment variable overrides.

Feature: chart-pattern-analysis-framework
Requirements: 10.1, 10.3, 10.5
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from copy import deepcopy

import yaml

from .exceptions import ConfigFileNotFoundError, ConfigParseError
from .validators import ConfigValidator
from .env_parser import EnvConfigParser


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
        
        # Initialize components
        self._validator = ConfigValidator()
        self._env_parser = EnvConfigParser()
        
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
            self._config = self._env_parser.apply_overrides(self._config)
        
        # Validate configuration
        if self._validate:
            self._validator.validate(self._config)
        
        # Setup logging based on config
        self._setup_logging()
        
        return self.config
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
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
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self._config.get("logging", {})
        
        level_str = log_config.get("level", "INFO").upper()
        log_format = log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_file = log_config.get("file", "logs/pattern_analysis.log")
        
        level = getattr(logging, level_str, logging.INFO)
        
        self._logger = logging.getLogger("pattern_analysis")
        self._logger.setLevel(level)
        self._logger.handlers.clear()
        
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        
        self._logger.debug(f"Logging configured: level={level_str}, file={log_file}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated key path."""
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
        """Get an entire configuration section."""
        return deepcopy(self._config.get(section, {}))
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by dot-separated key path."""
        parts = key.split(".")
        EnvConfigParser._set_nested_value(self._config, parts, value)
    
    def reload(self) -> Dict[str, Any]:
        """Reload configuration from file."""
        return self.load(self._config_path)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance."""
        if name:
            return logging.getLogger(f"pattern_analysis.{name}")
        return logging.getLogger("pattern_analysis")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
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
    """Get the default configuration manager instance."""
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
    """Get a configuration value using the default manager."""
    manager = get_config_manager()
    return manager.get(key, default)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance using the default manager."""
    manager = get_config_manager()
    return manager.get_logger(name)

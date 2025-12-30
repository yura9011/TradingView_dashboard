"""
Configuration exceptions.

Feature: chart-pattern-analysis-framework
Requirements: 10.1
"""

from typing import List, Optional


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

"""
Pattern Registry module.

Provides centralized management of pattern definitions,
including loading from YAML, validation, and lookup methods.
"""

# Import from modular structure
from .models import (
    PatternDefinition,
    PatternComponent,
    RegistryValidationError,
    RegistryVersion,
)
from .validators import PatternSchemaValidator
from .registry import PatternRegistry

__all__ = [
    "PatternRegistry",
    "PatternDefinition",
    "PatternComponent",
    "RegistryValidationError",
    "RegistryVersion",
    "PatternSchemaValidator",
]

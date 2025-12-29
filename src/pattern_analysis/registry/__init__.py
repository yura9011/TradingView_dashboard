"""
Pattern Registry module.

Provides centralized management of pattern definitions,
including loading from YAML, validation, and lookup methods.
"""

from .pattern_registry import (
    PatternRegistry,
    PatternDefinition,
    PatternComponent,
    RegistryValidationError,
    RegistryVersion,
)

__all__ = [
    "PatternRegistry",
    "PatternDefinition",
    "PatternComponent",
    "RegistryValidationError",
    "RegistryVersion",
]

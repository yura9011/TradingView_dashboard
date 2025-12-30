"""
Pattern Registry data models.

Feature: chart-pattern-analysis-framework
Requirements: 6.1
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


class RegistryValidationError(Exception):
    """Raised when pattern definition fails schema validation."""
    
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


@dataclass
class PatternComponent:
    """A component of a pattern definition."""
    name: str
    type: str
    required: bool = True


@dataclass
class PatternDefinition:
    """
    Complete definition of a chart pattern.
    
    Contains all metadata, components, validation rules, and aliases
    needed to identify and validate a pattern.
    """
    id: str
    name: str
    category: str
    direction: str
    description: str
    aliases: List[str]
    components: List[PatternComponent]
    validation_rules: List[str]
    min_confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "direction": self.direction,
            "description": self.description,
            "aliases": self.aliases,
            "components": [
                {"name": c.name, "type": c.type, "required": c.required}
                for c in self.components
            ],
            "validation_rules": self.validation_rules,
            "min_confidence": self.min_confidence
        }
    
    @classmethod
    def from_dict(cls, pattern_id: str, data: Dict[str, Any]) -> "PatternDefinition":
        """Create PatternDefinition from dictionary."""
        components = [
            PatternComponent(
                name=c["name"],
                type=c["type"],
                required=c.get("required", True)
            )
            for c in data.get("components", [])
        ]
        
        return cls(
            id=pattern_id,
            name=data["name"],
            category=data["category"],
            direction=data["direction"],
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            components=components,
            validation_rules=data.get("validation_rules", []),
            min_confidence=data.get("min_confidence", 0.5)
        )


@dataclass
class RegistryVersion:
    """Represents a version of the registry with timestamp and changes."""
    version: int
    timestamp: str
    changes: List[str]
    patterns_snapshot: Dict[str, Dict[str, Any]]

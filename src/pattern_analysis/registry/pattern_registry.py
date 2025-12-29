"""
Pattern Registry implementation.

Provides centralized management of pattern definitions with:
- YAML loading and schema validation
- Lookup by name, category, or alias
- Version tracking and history

Feature: chart-pattern-analysis-framework
Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import os
import copy
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..models.enums import PatternCategory, PatternType


logger = logging.getLogger(__name__)


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
    id: str  # Internal identifier (e.g., "head_shoulders")
    name: str  # Display name (e.g., "Head and Shoulders")
    category: str  # reversal, continuation, bilateral
    direction: str  # bullish, bearish, neutral
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


class PatternRegistry:
    """
    Centralized registry for pattern definitions.
    
    Provides:
    - Loading pattern definitions from YAML files
    - Schema validation for pattern definitions
    - Lookup by name, category, or alias
    - Version tracking and history
    
    Requirements:
    - 6.1: Store pattern definitions in structured, human-readable format (YAML)
    - 6.2: Validate new pattern definitions against schema
    - 6.3: Provide lookup methods by name, category, or characteristics
    - 6.4: Version changes and maintain history
    - 6.5: Support pattern aliases and localized names
    """
    
    # Required fields for pattern definition schema
    REQUIRED_FIELDS = {"name", "category", "direction"}
    VALID_CATEGORIES = {"reversal", "continuation", "bilateral"}
    VALID_DIRECTIONS = {"bullish", "bearish", "neutral"}
    
    def __init__(self, definitions_path: Optional[str] = None):
        """
        Initialize the pattern registry.
        
        Args:
            definitions_path: Optional path to pattern definitions YAML file.
                            If not provided, uses default path.
        """
        self._patterns: Dict[str, PatternDefinition] = {}
        self._alias_map: Dict[str, str] = {}  # alias -> pattern_id
        self._category_index: Dict[str, Set[str]] = {}  # category -> set of pattern_ids
        self._version_history: List[RegistryVersion] = []
        self._current_version: int = 0
        self._schema_version: str = "1.0.0"
        
        if definitions_path:
            self.load_from_yaml(definitions_path)
    
    def load_from_yaml(self, path: str) -> None:
        """
        Load pattern definitions from a YAML file.
        
        Args:
            path: Path to the YAML file containing pattern definitions.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            RegistryValidationError: If the YAML content is invalid.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pattern definitions file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data:
            raise RegistryValidationError("Empty YAML file")
        
        # Extract schema version if present
        if "schema_version" in data:
            self._schema_version = data["schema_version"]
        
        # Load patterns
        patterns_data = data.get("patterns", {})
        if not patterns_data:
            raise RegistryValidationError("No patterns found in YAML file")
        
        changes = []
        for pattern_id, pattern_data in patterns_data.items():
            try:
                self.add_pattern(pattern_id, pattern_data, record_history=False)
                changes.append(f"Added pattern: {pattern_id}")
            except RegistryValidationError as e:
                logger.warning(f"Skipping invalid pattern '{pattern_id}': {e}")
                raise
        
        # Record version
        self._record_version(changes)
        logger.info(f"Loaded {len(self._patterns)} patterns from {path}")
    
    def add_pattern(
        self,
        pattern_id: str,
        data: Dict[str, Any],
        record_history: bool = True
    ) -> PatternDefinition:
        """
        Add a new pattern definition to the registry.
        
        Args:
            pattern_id: Unique identifier for the pattern.
            data: Dictionary containing pattern definition data.
            record_history: Whether to record this change in version history.
            
        Returns:
            The created PatternDefinition.
            
        Raises:
            RegistryValidationError: If the pattern definition is invalid.
        """
        # Validate against schema
        errors = self._validate_pattern_schema(pattern_id, data)
        if errors:
            raise RegistryValidationError(
                f"Pattern '{pattern_id}' failed schema validation",
                errors=errors
            )
        
        # Create pattern definition
        pattern = PatternDefinition.from_dict(pattern_id, data)
        
        # Store pattern
        self._patterns[pattern_id] = pattern
        
        # Update alias map
        for alias in pattern.aliases:
            alias_lower = alias.lower()
            self._alias_map[alias_lower] = pattern_id
        # Also map the pattern_id itself
        self._alias_map[pattern_id.lower()] = pattern_id
        
        # Update category index
        category = pattern.category.lower()
        if category not in self._category_index:
            self._category_index[category] = set()
        self._category_index[category].add(pattern_id)
        
        # Record version if requested
        if record_history:
            self._record_version([f"Added pattern: {pattern_id}"])
        
        return pattern
    
    def _validate_pattern_schema(
        self,
        pattern_id: str,
        data: Dict[str, Any]
    ) -> List[str]:
        """
        Validate a pattern definition against the schema.
        
        Args:
            pattern_id: The pattern identifier.
            data: The pattern definition data.
            
        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return errors  # Return early if required fields missing
        
        # Validate category
        category = data.get("category", "").lower()
        if category not in self.VALID_CATEGORIES:
            errors.append(
                f"Invalid category '{category}'. "
                f"Must be one of: {', '.join(self.VALID_CATEGORIES)}"
            )
        
        # Validate direction
        direction = data.get("direction", "").lower()
        if direction not in self.VALID_DIRECTIONS:
            errors.append(
                f"Invalid direction '{direction}'. "
                f"Must be one of: {', '.join(self.VALID_DIRECTIONS)}"
            )
        
        # Validate min_confidence if present
        min_conf = data.get("min_confidence")
        if min_conf is not None:
            if not isinstance(min_conf, (int, float)):
                errors.append("min_confidence must be a number")
            elif not 0.0 <= min_conf <= 1.0:
                errors.append("min_confidence must be between 0.0 and 1.0")
        
        # Validate aliases if present
        aliases = data.get("aliases", [])
        if not isinstance(aliases, list):
            errors.append("aliases must be a list")
        else:
            for alias in aliases:
                if not isinstance(alias, str):
                    errors.append(f"Alias must be a string, got {type(alias)}")
        
        # Validate components if present
        components = data.get("components", [])
        if not isinstance(components, list):
            errors.append("components must be a list")
        else:
            for i, comp in enumerate(components):
                if not isinstance(comp, dict):
                    errors.append(f"Component {i} must be a dictionary")
                elif "name" not in comp or "type" not in comp:
                    errors.append(f"Component {i} missing required fields (name, type)")
        
        # Validate validation_rules if present
        rules = data.get("validation_rules", [])
        if not isinstance(rules, list):
            errors.append("validation_rules must be a list")
        
        return errors
    
    def get(self, identifier: str) -> Optional[PatternDefinition]:
        """
        Get a pattern definition by name or alias.
        
        Args:
            identifier: Pattern name, ID, or alias.
            
        Returns:
            PatternDefinition if found, None otherwise.
        """
        # Try direct lookup first
        if identifier in self._patterns:
            return self._patterns[identifier]
        
        # Try alias lookup (case-insensitive)
        identifier_lower = identifier.lower()
        if identifier_lower in self._alias_map:
            pattern_id = self._alias_map[identifier_lower]
            return self._patterns.get(pattern_id)
        
        return None
    
    def get_by_category(self, category: str) -> List[PatternDefinition]:
        """
        Get all patterns in a specific category.
        
        Args:
            category: Category name (reversal, continuation, bilateral).
            
        Returns:
            List of PatternDefinition objects in the category.
        """
        category_lower = category.lower()
        pattern_ids = self._category_index.get(category_lower, set())
        return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]
    
    def get_by_direction(self, direction: str) -> List[PatternDefinition]:
        """
        Get all patterns with a specific direction.
        
        Args:
            direction: Direction (bullish, bearish, neutral).
            
        Returns:
            List of PatternDefinition objects with the direction.
        """
        direction_lower = direction.lower()
        return [
            p for p in self._patterns.values()
            if p.direction.lower() == direction_lower
        ]
    
    def search(
        self,
        name: Optional[str] = None,
        category: Optional[str] = None,
        direction: Optional[str] = None
    ) -> List[PatternDefinition]:
        """
        Search patterns by multiple criteria.
        
        Args:
            name: Partial name match (case-insensitive).
            category: Category filter.
            direction: Direction filter.
            
        Returns:
            List of matching PatternDefinition objects.
        """
        results = list(self._patterns.values())
        
        if name:
            name_lower = name.lower()
            results = [
                p for p in results
                if name_lower in p.name.lower() or
                   name_lower in p.id.lower() or
                   any(name_lower in alias.lower() for alias in p.aliases)
            ]
        
        if category:
            category_lower = category.lower()
            results = [p for p in results if p.category.lower() == category_lower]
        
        if direction:
            direction_lower = direction.lower()
            results = [p for p in results if p.direction.lower() == direction_lower]
        
        return results
    
    def list_all(self) -> List[PatternDefinition]:
        """
        Get all registered patterns.
        
        Returns:
            List of all PatternDefinition objects.
        """
        return list(self._patterns.values())
    
    def list_aliases(self, pattern_id: str) -> List[str]:
        """
        Get all aliases for a pattern.
        
        Args:
            pattern_id: The pattern identifier.
            
        Returns:
            List of aliases for the pattern.
        """
        pattern = self._patterns.get(pattern_id)
        if pattern:
            return pattern.aliases.copy()
        return []
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """
        Remove a pattern from the registry.
        
        Args:
            pattern_id: The pattern identifier to remove.
            
        Returns:
            True if pattern was removed, False if not found.
        """
        if pattern_id not in self._patterns:
            return False
        
        pattern = self._patterns[pattern_id]
        
        # Remove from alias map
        for alias in pattern.aliases:
            alias_lower = alias.lower()
            if alias_lower in self._alias_map:
                del self._alias_map[alias_lower]
        if pattern_id.lower() in self._alias_map:
            del self._alias_map[pattern_id.lower()]
        
        # Remove from category index
        category = pattern.category.lower()
        if category in self._category_index:
            self._category_index[category].discard(pattern_id)
        
        # Remove pattern
        del self._patterns[pattern_id]
        
        # Record version
        self._record_version([f"Removed pattern: {pattern_id}"])
        
        return True
    
    def update_pattern(
        self,
        pattern_id: str,
        data: Dict[str, Any]
    ) -> PatternDefinition:
        """
        Update an existing pattern definition.
        
        Args:
            pattern_id: The pattern identifier to update.
            data: New pattern definition data.
            
        Returns:
            The updated PatternDefinition.
            
        Raises:
            KeyError: If pattern not found.
            RegistryValidationError: If new data is invalid.
        """
        if pattern_id not in self._patterns:
            raise KeyError(f"Pattern not found: {pattern_id}")
        
        # Remove old pattern
        self.remove_pattern(pattern_id)
        
        # Add updated pattern
        return self.add_pattern(pattern_id, data)
    
    def _record_version(self, changes: List[str]) -> None:
        """Record a new version in history."""
        self._current_version += 1
        
        # Create snapshot of current patterns
        snapshot = {
            pid: pattern.to_dict()
            for pid, pattern in self._patterns.items()
        }
        
        version = RegistryVersion(
            version=self._current_version,
            timestamp=datetime.now().isoformat(),
            changes=changes,
            patterns_snapshot=snapshot
        )
        
        self._version_history.append(version)
    
    def get_version(self) -> int:
        """Get current registry version number."""
        return self._current_version
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get version history.
        
        Returns:
            List of version records with timestamp and changes.
        """
        return [
            {
                "version": v.version,
                "timestamp": v.timestamp,
                "changes": v.changes
            }
            for v in self._version_history
        ]
    
    def get_version_snapshot(self, version: int) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Get pattern snapshot at a specific version.
        
        Args:
            version: Version number to retrieve.
            
        Returns:
            Dictionary of pattern definitions at that version, or None if not found.
        """
        for v in self._version_history:
            if v.version == version:
                return copy.deepcopy(v.patterns_snapshot)
        return None
    
    def rollback_to_version(self, version: int) -> bool:
        """
        Rollback registry to a previous version.
        
        Args:
            version: Version number to rollback to.
            
        Returns:
            True if rollback successful, False if version not found.
        """
        snapshot = self.get_version_snapshot(version)
        if snapshot is None:
            return False
        
        # Clear current state
        self._patterns.clear()
        self._alias_map.clear()
        self._category_index.clear()
        
        # Restore from snapshot
        for pattern_id, data in snapshot.items():
            # Convert back to the format expected by add_pattern
            pattern_data = {
                "name": data["name"],
                "category": data["category"],
                "direction": data["direction"],
                "description": data.get("description", ""),
                "aliases": data.get("aliases", []),
                "components": data.get("components", []),
                "validation_rules": data.get("validation_rules", []),
                "min_confidence": data.get("min_confidence", 0.5)
            }
            self.add_pattern(pattern_id, pattern_data, record_history=False)
        
        # Record rollback
        self._record_version([f"Rollback to version {version}"])
        
        return True
    
    def export_to_yaml(self, path: str) -> None:
        """
        Export current registry to a YAML file.
        
        Args:
            path: Path to write the YAML file.
        """
        data = {
            "schema_version": self._schema_version,
            "patterns": {
                pid: {
                    "name": p.name,
                    "category": p.category,
                    "direction": p.direction,
                    "description": p.description,
                    "aliases": p.aliases,
                    "components": [
                        {"name": c.name, "type": c.type, "required": c.required}
                        for c in p.components
                    ],
                    "validation_rules": p.validation_rules,
                    "min_confidence": p.min_confidence
                }
                for pid, p in self._patterns.items()
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def __len__(self) -> int:
        """Return number of registered patterns."""
        return len(self._patterns)
    
    def __contains__(self, identifier: str) -> bool:
        """Check if a pattern exists by name or alias."""
        return self.get(identifier) is not None
    
    def __iter__(self):
        """Iterate over pattern definitions."""
        return iter(self._patterns.values())

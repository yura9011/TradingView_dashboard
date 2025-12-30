"""
Pattern Registry implementation.

Provides centralized management of pattern definitions.

Feature: chart-pattern-analysis-framework
Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import os
import copy
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

import yaml

from .models import (
    PatternDefinition,
    RegistryVersion,
    RegistryValidationError,
)
from .validators import PatternSchemaValidator


logger = logging.getLogger(__name__)


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
    
    def __init__(self, definitions_path: Optional[str] = None):
        """Initialize the pattern registry."""
        self._patterns: Dict[str, PatternDefinition] = {}
        self._alias_map: Dict[str, str] = {}
        self._category_index: Dict[str, Set[str]] = {}
        self._version_history: List[RegistryVersion] = []
        self._current_version: int = 0
        self._schema_version: str = "1.0.0"
        self._validator = PatternSchemaValidator()
        
        if definitions_path:
            self.load_from_yaml(definitions_path)
    
    def load_from_yaml(self, path: str) -> None:
        """Load pattern definitions from a YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pattern definitions file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data:
            raise RegistryValidationError("Empty YAML file")
        
        if "schema_version" in data:
            self._schema_version = data["schema_version"]
        
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
        
        self._record_version(changes)
        logger.info(f"Loaded {len(self._patterns)} patterns from {path}")
    
    def add_pattern(
        self,
        pattern_id: str,
        data: Dict[str, Any],
        record_history: bool = True
    ) -> PatternDefinition:
        """Add a new pattern definition to the registry."""
        errors = self._validator.validate(pattern_id, data)
        if errors:
            raise RegistryValidationError(
                f"Pattern '{pattern_id}' failed schema validation",
                errors=errors
            )
        
        pattern = PatternDefinition.from_dict(pattern_id, data)
        self._patterns[pattern_id] = pattern
        
        # Update alias map
        for alias in pattern.aliases:
            self._alias_map[alias.lower()] = pattern_id
        self._alias_map[pattern_id.lower()] = pattern_id
        
        # Update category index
        category = pattern.category.lower()
        if category not in self._category_index:
            self._category_index[category] = set()
        self._category_index[category].add(pattern_id)
        
        if record_history:
            self._record_version([f"Added pattern: {pattern_id}"])
        
        return pattern
    
    def get(self, identifier: str) -> Optional[PatternDefinition]:
        """Get a pattern definition by name or alias."""
        if identifier in self._patterns:
            return self._patterns[identifier]
        
        identifier_lower = identifier.lower()
        if identifier_lower in self._alias_map:
            pattern_id = self._alias_map[identifier_lower]
            return self._patterns.get(pattern_id)
        
        return None
    
    def get_by_category(self, category: str) -> List[PatternDefinition]:
        """Get all patterns in a specific category."""
        pattern_ids = self._category_index.get(category.lower(), set())
        return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]
    
    def get_by_direction(self, direction: str) -> List[PatternDefinition]:
        """Get all patterns with a specific direction."""
        direction_lower = direction.lower()
        return [p for p in self._patterns.values() if p.direction.lower() == direction_lower]
    
    def search(
        self,
        name: Optional[str] = None,
        category: Optional[str] = None,
        direction: Optional[str] = None
    ) -> List[PatternDefinition]:
        """Search patterns by multiple criteria."""
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
            results = [p for p in results if p.category.lower() == category.lower()]
        
        if direction:
            results = [p for p in results if p.direction.lower() == direction.lower()]
        
        return results
    
    def list_all(self) -> List[PatternDefinition]:
        """Get all registered patterns."""
        return list(self._patterns.values())
    
    def list_aliases(self, pattern_id: str) -> List[str]:
        """Get all aliases for a pattern."""
        pattern = self._patterns.get(pattern_id)
        return pattern.aliases.copy() if pattern else []
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove a pattern from the registry."""
        if pattern_id not in self._patterns:
            return False
        
        pattern = self._patterns[pattern_id]
        
        for alias in pattern.aliases:
            self._alias_map.pop(alias.lower(), None)
        self._alias_map.pop(pattern_id.lower(), None)
        
        category = pattern.category.lower()
        if category in self._category_index:
            self._category_index[category].discard(pattern_id)
        
        del self._patterns[pattern_id]
        self._record_version([f"Removed pattern: {pattern_id}"])
        
        return True
    
    def update_pattern(self, pattern_id: str, data: Dict[str, Any]) -> PatternDefinition:
        """Update an existing pattern definition."""
        if pattern_id not in self._patterns:
            raise KeyError(f"Pattern not found: {pattern_id}")
        
        self.remove_pattern(pattern_id)
        return self.add_pattern(pattern_id, data)
    
    def _record_version(self, changes: List[str]) -> None:
        """Record a new version in history."""
        self._current_version += 1
        
        snapshot = {pid: pattern.to_dict() for pid, pattern in self._patterns.items()}
        
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
        """Get version history."""
        return [
            {"version": v.version, "timestamp": v.timestamp, "changes": v.changes}
            for v in self._version_history
        ]
    
    def get_version_snapshot(self, version: int) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get pattern snapshot at a specific version."""
        for v in self._version_history:
            if v.version == version:
                return copy.deepcopy(v.patterns_snapshot)
        return None
    
    def rollback_to_version(self, version: int) -> bool:
        """Rollback registry to a previous version."""
        snapshot = self.get_version_snapshot(version)
        if snapshot is None:
            return False
        
        self._patterns.clear()
        self._alias_map.clear()
        self._category_index.clear()
        
        for pattern_id, data in snapshot.items():
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
        
        self._record_version([f"Rollback to version {version}"])
        return True
    
    def export_to_yaml(self, path: str) -> None:
        """Export current registry to a YAML file."""
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
        return len(self._patterns)
    
    def __contains__(self, identifier: str) -> bool:
        return self.get(identifier) is not None
    
    def __iter__(self):
        return iter(self._patterns.values())

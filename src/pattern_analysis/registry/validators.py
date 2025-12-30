"""
Pattern Registry schema validation.

Feature: chart-pattern-analysis-framework
Requirements: 6.2
"""

from typing import Dict, List, Any


class PatternSchemaValidator:
    """Validates pattern definitions against schema."""
    
    REQUIRED_FIELDS = {"name", "category", "direction"}
    VALID_CATEGORIES = {"reversal", "continuation", "bilateral"}
    VALID_DIRECTIONS = {"bullish", "bearish", "neutral"}
    
    def validate(self, pattern_id: str, data: Dict[str, Any]) -> List[str]:
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
            return errors
        
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

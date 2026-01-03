"""
Progress Logger - Detailed console output for analysis progress.
"""

import sys
from datetime import datetime
from typing import Optional


class ProgressLogger:
    """
    Provides detailed progress messages during analysis.
    
    Outputs timestamped messages with emojis for visual feedback.
    """
    
    def __init__(self, symbol: str = "", verbose: bool = True):
        """
        Initialize progress logger.
        
        Args:
            symbol: Trading symbol being analyzed
            verbose: Whether to print messages
        """
        self.symbol = symbol
        self.verbose = verbose
        self._start_time: Optional[datetime] = None
    
    def _timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%H:%M:%S")
    
    def _print(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message, flush=True)
    
    def start_analysis(self, symbol: str = None):
        """Log analysis start."""
        if symbol:
            self.symbol = symbol
        self._start_time = datetime.now()
        self._print(f"[{self._timestamp()}] 🚀 Iniciando análisis de {self.symbol}...")
    
    def log_step(self, message: str, emoji: str = "📌"):
        """
        Log a progress step.
        
        Args:
            message: Step description
            emoji: Emoji to prefix message
        """
        self._print(f"[{self._timestamp()}] {emoji} {message}")
    
    def log_model_loading(self):
        """Log model loading step."""
        self.log_step("Cargando modelo de detección YOLO...", "📦")
    
    def log_pattern_detection(self):
        """Log pattern detection step."""
        self.log_step("Detectando patrones en el chart...", "🔍")
    
    def log_pattern_found(self, pattern: str, confidence: float):
        """
        Log a detected pattern.
        
        Args:
            pattern: Pattern name
            confidence: Detection confidence (0.0 to 1.0)
        """
        conf_percent = int(confidence * 100)
        self._print(f"[{self._timestamp()}] ✅ Patrón encontrado: {pattern.upper()} ({conf_percent}% confianza)")
    
    def log_reference_comparison(self):
        """Log reference comparison step."""
        self.log_step("Comparando con imágenes de referencia...", "📚")
    
    def log_match_result(self, pattern: str, similarity: float):
        """
        Log a pattern match result.
        
        Args:
            pattern: Pattern name
            similarity: Similarity score (0.0 to 1.0)
        """
        sim_percent = int(similarity * 100)
        self._print(f"[{self._timestamp()}] 📊 {pattern.upper()} → Mejor match: {sim_percent}% similitud")
    
    def log_no_references(self, pattern: str):
        """Log when no references are available for a pattern."""
        self._print(f"[{self._timestamp()}] ⚠️ {pattern.upper()} → Sin imágenes de referencia")
    
    def log_summary(self, patterns_found: int, matches_found: int):
        """
        Log analysis summary.
        
        Args:
            patterns_found: Number of patterns detected
            matches_found: Number of successful matches
        """
        elapsed = 0.0
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
        
        self._print(f"[{self._timestamp()}] ════════════════════════════════════════════════")
        self._print(f"[{self._timestamp()}] ✅ Análisis completado en {elapsed:.1f} segundos")
        self._print(f"[{self._timestamp()}]    Patrones detectados: {patterns_found}")
        self._print(f"[{self._timestamp()}]    Matches encontrados: {matches_found}")
        self._print(f"[{self._timestamp()}] ════════════════════════════════════════════════")
    
    def log_error(self, message: str):
        """Log an error message."""
        self._print(f"[{self._timestamp()}] ❌ Error: {message}")
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self._print(f"[{self._timestamp()}] ⚠️ {message}")

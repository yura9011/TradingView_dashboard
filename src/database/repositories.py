"""
Repository classes for database operations.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

from src.models import Signal, Asset, SignalType, PatternType, Market
from .connection import get_database, Database

logger = logging.getLogger(__name__)


class SignalRepository:
    """Repository for Signal CRUD operations."""
    
    def __init__(self, db: Database = None):
        self.db = db or get_database()
    
    def create(self, signal: Signal) -> int:
        """Create a new signal.
        
        Args:
            signal: Signal model to save
            
        Returns:
            ID of created signal
        """
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO signals (
                    symbol, signal_type, pattern_detected, pattern_confidence,
                    trend, trend_strength, market_phase, elliott_wave,
                    support_level, resistance_level, fibonacci_level,
                    sentiment_score, analysis_summary, detailed_reasoning,
                    chart_image_path, report_path, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol,
                signal.signal_type,
                signal.pattern_detected,
                signal.pattern_confidence,
                signal.trend,
                signal.trend_strength,
                signal.market_phase,
                signal.elliott_wave,
                signal.support_level,
                signal.resistance_level,
                signal.fibonacci_level,
                signal.sentiment_score,
                signal.analysis_summary,
                signal.detailed_reasoning,
                signal.chart_image_path,
                signal.report_path,
                signal.notes,
            ))
            signal_id = cursor.lastrowid
            logger.info(f"Created signal {signal_id} for {signal.symbol}")
            return signal_id
    
    def get_by_id(self, signal_id: int) -> Optional[Signal]:
        """Get signal by ID.
        
        Args:
            signal_id: Signal database ID
            
        Returns:
            Signal if found, None otherwise
        """
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM signals WHERE id = ?", (signal_id,)
            ).fetchone()
            
            if row:
                return self._row_to_signal(row)
            return None
    
    def get_recent(self, limit: int = 20) -> List[Signal]:
        """Get recent signals.
        
        Args:
            limit: Maximum number of signals
            
        Returns:
            List of recent signals
        """
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            
            return [self._row_to_signal(row) for row in rows]
    
    def get_by_symbol(self, symbol: str) -> List[Signal]:
        """Get signals for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            List of signals for symbol
        """
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM signals WHERE symbol = ? ORDER BY timestamp DESC",
                (symbol,)
            ).fetchall()
            
            return [self._row_to_signal(row) for row in rows]
    
    def get_pending(self) -> List[Signal]:
        """Get all pending signals for review.
        
        Returns:
            List of pending signals
        """
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM signals WHERE signal_type = 'pending' AND reviewed = 0 ORDER BY timestamp DESC"
            ).fetchall()
            
            return [self._row_to_signal(row) for row in rows]
    
    def update_review(self, signal_id: int, signal_type: SignalType, notes: str = None) -> None:
        """Update signal after human review.
        
        Args:
            signal_id: Signal ID
            signal_type: Final classification
            notes: Review notes
        """
        with self.db.get_connection() as conn:
            conn.execute("""
                UPDATE signals 
                SET signal_type = ?, reviewed = 1, notes = ?, updated_at = ?
                WHERE id = ?
            """, (signal_type, notes, datetime.utcnow(), signal_id))
            logger.info(f"Updated signal {signal_id} to {signal_type}")
    
    def update_chart_path(self, signal_id: int, chart_path: str) -> None:
        """Update chart image path for a signal.
        
        Args:
            signal_id: Signal ID
            chart_path: New chart image path
        """
        with self.db.get_connection() as conn:
            conn.execute("""
                UPDATE signals 
                SET chart_image_path = ?, updated_at = ?
                WHERE id = ?
            """, (chart_path, datetime.utcnow(), signal_id))
            logger.info(f"Updated chart path for signal {signal_id}")
    
    def _row_to_signal(self, row) -> Signal:
        """Convert database row to Signal model."""
        return Signal(
            id=row["id"],
            symbol=row["symbol"],
            signal_type=SignalType(row["signal_type"]) if row["signal_type"] else SignalType.PENDING,
            timestamp=row["timestamp"],
            pattern_detected=PatternType(row["pattern_detected"]) if row["pattern_detected"] else PatternType.NONE,
            pattern_confidence=row["pattern_confidence"] or 0.0,
            trend=row["trend"],
            trend_strength=row["trend_strength"] if "trend_strength" in row.keys() else None,
            market_phase=row["market_phase"] if "market_phase" in row.keys() else None,
            elliott_wave=row["elliott_wave"] if "elliott_wave" in row.keys() else None,
            support_level=row["support_level"],
            resistance_level=row["resistance_level"],
            fibonacci_level=row["fibonacci_level"],
            sentiment_score=row["sentiment_score"],
            analysis_summary=row["analysis_summary"],
            detailed_reasoning=row["detailed_reasoning"] if "detailed_reasoning" in row.keys() else None,
            chart_image_path=row["chart_image_path"],
            report_path=row["report_path"],
            reviewed=bool(row["reviewed"]),
            notes=row["notes"],
        )


class ScreenerLogRepository:
    """Repository for logging screener queries."""
    
    def __init__(self, db: Database = None):
        self.db = db or get_database()
    
    def log_query(
        self,
        query_type: str,
        market: str,
        filters: dict,
        result_count: int
    ) -> None:
        """Log a screener query.
        
        Args:
            query_type: Type of query (e.g., 'top_volume')
            market: Market queried
            filters: Filter parameters used
            result_count: Number of results
        """
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO screener_logs (query_type, market, filters, result_count)
                VALUES (?, ?, ?, ?)
            """, (query_type, market, json.dumps(filters), result_count))


# Factory functions
def get_signal_repository() -> SignalRepository:
    """Get SignalRepository instance."""
    return SignalRepository()


def get_screener_log_repository() -> ScreenerLogRepository:
    """Get ScreenerLogRepository instance."""
    return ScreenerLogRepository()

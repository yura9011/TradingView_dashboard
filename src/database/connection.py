"""
Database connection and session management.
"""

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "signals.db"


class Database:
    """SQLite database connection manager."""
    
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        """Initialize database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.executescript("""
                -- Signals table
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT DEFAULT 'pending',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Pattern analysis
                    pattern_detected TEXT DEFAULT 'none',
                    pattern_confidence REAL DEFAULT 0.0,
                    
                    -- Technical context
                    trend TEXT,
                    trend_strength TEXT,
                    market_phase TEXT,
                    elliott_wave TEXT,
                    support_level REAL,
                    resistance_level REAL,
                    fibonacci_level TEXT,
                    
                    -- Sentiment
                    sentiment_score REAL,
                    
                    -- Analysis
                    analysis_summary TEXT,
                    detailed_reasoning TEXT,
                    
                    -- Files
                    chart_image_path TEXT,
                    report_path TEXT,
                    
                    -- Review status
                    reviewed INTEGER DEFAULT 0,
                    notes TEXT,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Assets cache table
                CREATE TABLE IF NOT EXISTS assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    name TEXT,
                    market TEXT DEFAULT 'america',
                    exchange TEXT,
                    sector TEXT,
                    last_price REAL,
                    last_volume REAL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Screener results log
                CREATE TABLE IF NOT EXISTS screener_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_type TEXT NOT NULL,
                    market TEXT,
                    filters TEXT,
                    result_count INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create indexes
                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
                CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type);
            """)
            logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection as context manager.
        
        Yields:
            SQLite connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()


# Global database instance
_db: Database = None


def get_database(db_path: Path = DEFAULT_DB_PATH) -> Database:
    """Get or create database instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Database instance
    """
    global _db
    if _db is None:
        _db = Database(db_path)
    return _db

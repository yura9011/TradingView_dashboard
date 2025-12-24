"""
Logging configuration for the AI Trading Analysis Agent.
Provides centralized, structured logging for all modules.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Log format
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure centralized logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name (default: agent_YYYYMMDD.log)
        console: Whether to also log to console
        
    Returns:
        Root logger configured
    """
    # Parse level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = f"agent_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_path = LOG_DIR / log_file
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always capture debug to file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    root_logger.info(f"Logging initialized: level={level}, file={file_path}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger
    """
    return logging.getLogger(name)


# Auto-configure on import if not already configured
if not logging.getLogger().handlers:
    setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))

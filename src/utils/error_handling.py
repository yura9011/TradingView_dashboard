"""
Error Handling Utilities - Centralized error management for the application.

Provides graceful degradation and user-friendly error messages.
"""

import os
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Any
from functools import wraps

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Base exception for analysis errors."""
    def __init__(self, message: str, user_message: str = None, recoverable: bool = True):
        super().__init__(message)
        self.user_message = user_message or message
        self.recoverable = recoverable


class ChartCaptureError(AnalysisError):
    """Error capturing chart from TradingView."""
    pass


class DatabaseError(AnalysisError):
    """Error with database operations."""
    pass


class ModelNotAvailableError(AnalysisError):
    """Error when required model is not available."""
    pass


def check_disk_space(path: str = ".", min_mb: int = 100) -> Tuple[bool, str]:
    """
    Check if there's enough disk space.
    
    Args:
        path: Path to check
        min_mb: Minimum required space in MB
        
    Returns:
        Tuple of (has_space, message)
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_mb = free // (1024 * 1024)
        
        if free_mb < min_mb:
            return False, f"Espacio en disco insuficiente: {free_mb}MB disponibles, se requieren {min_mb}MB"
        return True, f"Espacio disponible: {free_mb}MB"
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True, "No se pudo verificar espacio en disco"


def check_database_health(db_path: str = "data/signals.db") -> Tuple[bool, str]:
    """
    Check if database is accessible and not corrupted.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Tuple of (is_healthy, message)
    """
    try:
        db_file = Path(db_path)
        
        # Check if file exists
        if not db_file.exists():
            # Will be created on first use
            return True, "Base de datos será creada en primer uso"
        
        # Check if file is readable
        if not os.access(db_file, os.R_OK | os.W_OK):
            return False, f"No se puede acceder a la base de datos: {db_path}"
        
        # Try to open and run integrity check
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        conn.close()
        
        if result == "ok":
            return True, "Base de datos OK"
        else:
            return False, f"Base de datos corrupta: {result}"
            
    except sqlite3.DatabaseError as e:
        return False, f"Error de base de datos: {e}"
    except Exception as e:
        logger.warning(f"Could not check database health: {e}")
        return True, "No se pudo verificar base de datos"


def check_internet_connection(timeout: int = 5) -> Tuple[bool, str]:
    """
    Check if internet connection is available.
    
    Args:
        timeout: Connection timeout in seconds
        
    Returns:
        Tuple of (is_connected, message)
    """
    try:
        import socket
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True, "Conexión a internet OK"
    except socket.error:
        return False, "Sin conexión a internet - no se pueden capturar charts"
    except Exception as e:
        logger.warning(f"Could not check internet: {e}")
        return True, "No se pudo verificar conexión"


def check_yolo_available() -> Tuple[bool, str]:
    """
    Check if YOLO model is available.
    
    Returns:
        Tuple of (is_available, message)
    """
    try:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        return True, "YOLO disponible"
    except ImportError:
        return False, "YOLO no instalado - usando modelo VLM alternativo"


def check_references_available() -> Tuple[bool, str, int]:
    """
    Check if pattern reference images are available.
    
    Returns:
        Tuple of (is_available, message, count)
    """
    try:
        from src.pattern_analysis.reference import ReferenceManager
        rm = ReferenceManager()
        rm.load_references()
        count = rm.get_reference_count()
        
        if count > 0:
            return True, f"{count} imágenes de referencia cargadas", count
        else:
            return False, "No hay imágenes de referencia en data/pattern_references/", 0
    except ImportError:
        return False, "Módulo de referencias no disponible", 0
    except Exception as e:
        return False, f"Error cargando referencias: {e}", 0


def run_health_check() -> dict:
    """
    Run complete system health check.
    
    Returns:
        Dictionary with health check results
    """
    results = {
        "overall": True,
        "checks": {},
        "warnings": [],
        "errors": []
    }
    
    # Check disk space
    ok, msg = check_disk_space()
    results["checks"]["disk_space"] = {"ok": ok, "message": msg}
    if not ok:
        results["errors"].append(msg)
        results["overall"] = False
    
    # Check database
    ok, msg = check_database_health()
    results["checks"]["database"] = {"ok": ok, "message": msg}
    if not ok:
        results["errors"].append(msg)
        results["overall"] = False
    
    # Check internet
    ok, msg = check_internet_connection()
    results["checks"]["internet"] = {"ok": ok, "message": msg}
    if not ok:
        results["warnings"].append(msg)
    
    # Check YOLO
    ok, msg = check_yolo_available()
    results["checks"]["yolo"] = {"ok": ok, "message": msg}
    if not ok:
        results["warnings"].append(msg)
    
    # Check references
    ok, msg, count = check_references_available()
    results["checks"]["references"] = {"ok": ok, "message": msg, "count": count}
    if not ok:
        results["warnings"].append(msg)
    
    return results


def safe_analysis(func):
    """
    Decorator for safe analysis execution with error handling.
    
    Catches common errors and provides user-friendly messages.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Pre-flight checks
            ok, msg = check_disk_space()
            if not ok:
                raise AnalysisError(msg, "No hay espacio en disco suficiente", recoverable=False)
            
            ok, msg = check_database_health()
            if not ok:
                raise DatabaseError(msg, "Error en base de datos - intente reiniciar", recoverable=False)
            
            # Run analysis
            return await func(*args, **kwargs)
            
        except AnalysisError:
            raise
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise DatabaseError(str(e), "Error guardando resultados en base de datos")
        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise ChartCaptureError(str(e), "Error de conexión - verifique su internet")
        except PermissionError as e:
            logger.error(f"Permission error: {e}")
            raise AnalysisError(str(e), "Error de permisos - verifique acceso a carpetas")
        except Exception as e:
            logger.exception(f"Unexpected error in analysis: {e}")
            raise AnalysisError(str(e), f"Error inesperado: {type(e).__name__}")
    
    return wrapper


def print_health_report():
    """Print a formatted health check report to console."""
    print("\n" + "=" * 50)
    print("🏥 VERIFICACIÓN DEL SISTEMA")
    print("=" * 50)
    
    results = run_health_check()
    
    for check_name, check_result in results["checks"].items():
        icon = "✅" if check_result["ok"] else "❌"
        print(f"{icon} {check_name}: {check_result['message']}")
    
    if results["warnings"]:
        print("\n⚠️ ADVERTENCIAS:")
        for w in results["warnings"]:
            print(f"   - {w}")
    
    if results["errors"]:
        print("\n❌ ERRORES:")
        for e in results["errors"]:
            print(f"   - {e}")
    
    print("=" * 50)
    
    if results["overall"]:
        print("✅ Sistema listo para análisis")
    else:
        print("❌ Sistema tiene problemas - revisar errores arriba")
    
    print("=" * 50 + "\n")
    
    return results["overall"]

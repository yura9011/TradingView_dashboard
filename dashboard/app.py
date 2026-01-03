"""
Dashboard - Web interface for viewing trading analysis results.
Run: python -m dashboard.app
"""

import os
import sys
import json
import logging
import threading
import queue
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, jsonify, send_from_directory, request

# Filter out HTTP 200 logs from werkzeug
class NoSuccessFilter(logging.Filter):
    def filter(self, record):
        return '200 -' not in record.getMessage()

# Apply filter to werkzeug logger
logging.getLogger('werkzeug').addFilter(NoSuccessFilter())

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import get_database

app = Flask(__name__)

# Bulk analysis state
bulk_analysis_state = {
    "running": False,
    "current_symbol": None,
    "progress": 0,
    "total": 0,
    "completed": [],
    "errors": [],
    "queue": queue.Queue(),
}

# Custom Jinja2 filter for parsing JSON
@app.template_filter('fromjson')
def fromjson_filter(value):
    """Parse JSON string to dict."""
    if not value:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
CHARTS_DIR = DATA_DIR / "charts"
PATTERN_REFS_DIR = DATA_DIR / "pattern_references"
WATCHLISTS_DIR = DATA_DIR / "watchlists"


def get_signals_summary():
    """Get summary of all signals grouped by symbol."""
    db = get_database()
    
    with db.get_connection() as conn:
        # Get latest signal per symbol
        rows = conn.execute("""
            SELECT 
                s1.id,
                s1.symbol,
                s1.signal_type,
                s1.pattern_detected,
                s1.pattern_confidence,
                s1.trend,
                s1.trend_strength,
                s1.market_phase,
                s1.elliott_wave,
                s1.analysis_summary,
                s1.chart_image_path,
                s1.report_path,
                s1.timestamp,
                s1.reviewed
            FROM signals s1
            INNER JOIN (
                SELECT symbol, MAX(timestamp) as max_ts
                FROM signals
                GROUP BY symbol
            ) s2 ON s1.symbol = s2.symbol AND s1.timestamp = s2.max_ts
            ORDER BY s1.timestamp DESC
        """).fetchall()
        
        signals = []
        for row in rows:
            signals.append({
                "id": row["id"],
                "symbol": row["symbol"],
                "signal_type": row["signal_type"],
                "pattern": row["pattern_detected"],
                "confidence": row["pattern_confidence"],
                "trend": row["trend"],
                "trend_strength": row["trend_strength"],
                "market_phase": row["market_phase"],
                "elliott_wave": row["elliott_wave"],
                "summary": row["analysis_summary"],
                "chart_path": row["chart_image_path"],
                "report_path": row["report_path"],
                "timestamp": row["timestamp"],
                "reviewed": row["reviewed"],
            })
        
        return signals


def get_signal_detail(signal_id: int):
    """Get full detail for a specific signal."""
    db = get_database()
    
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM signals WHERE id = ?", (signal_id,)
        ).fetchone()
        
        if row:
            return dict(row)
        return None


def get_signal_history(symbol: str):
    """Get all signals for a symbol."""
    db = get_database()
    
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM signals WHERE symbol = ? ORDER BY timestamp DESC",
            (symbol,)
        ).fetchall()
        
        return [dict(row) for row in rows]


@app.route("/")
def index():
    """Main dashboard page with signals table."""
    signals = get_signals_summary()
    return render_template("index.html", signals=signals)


@app.route("/signal/<int:signal_id>")
def signal_detail(signal_id: int):
    """Detail view for a specific signal."""
    signal = get_signal_detail(signal_id)
    if not signal:
        return "Signal not found", 404
    
    # Get history for this symbol
    history = get_signal_history(signal["symbol"])
    
    return render_template("detail.html", signal=signal, history=history)


@app.route("/api/signals")
def api_signals():
    """API endpoint for signals data."""
    signals = get_signals_summary()
    return jsonify(signals)


@app.route("/api/signal/<int:signal_id>")
def api_signal_detail(signal_id: int):
    """API endpoint for signal detail."""
    signal = get_signal_detail(signal_id)
    if not signal:
        return jsonify({"error": "Not found"}), 404
    return jsonify(signal)


@app.route("/charts/<path:filename>")
def serve_chart(filename):
    """Serve chart images."""
    return send_from_directory(CHARTS_DIR, filename)


@app.route("/reports/<path:filename>")
def serve_report_asset(filename):
    """Serve report assets (annotated charts)."""
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/pattern_references/<path:filename>")
def serve_pattern_reference(filename):
    """Serve pattern reference images from books."""
    return send_from_directory(PATTERN_REFS_DIR, filename)


# ============================================================
# BULK ANALYSIS ENDPOINTS
# ============================================================

def run_bulk_analysis_worker(symbols: list, use_local_model: bool = True, model_name: str = None):
    """Worker thread for bulk analysis."""
    import asyncio
    import time
    
    bulk_analysis_state["running"] = True
    bulk_analysis_state["total"] = len(symbols)
    bulk_analysis_state["progress"] = 0
    bulk_analysis_state["completed"] = []
    bulk_analysis_state["errors"] = []
    
    # Create a single event loop for all analyses
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Use unified analysis module
        from src.analysis import run_analysis
        selected_model = model_name or "Qwen/Qwen2-VL-2B-Instruct"
        
        for i, symbol in enumerate(symbols):
            if not bulk_analysis_state["running"]:
                break  # Cancelled
            
            bulk_analysis_state["current_symbol"] = symbol
            bulk_analysis_state["progress"] = i
            
            try:
                loop.run_until_complete(run_analysis(
                    symbol=symbol.strip().upper(),
                    use_local=use_local_model,
                    model_name=selected_model,
                ))
                bulk_analysis_state["completed"].append(symbol)
            except Exception as e:
                bulk_analysis_state["errors"].append({
                    "symbol": symbol,
                    "error": str(e)
                })
            
            bulk_analysis_state["progress"] = i + 1
            
            # Small delay between analyses
            time.sleep(1)
    finally:
        loop.close()
        bulk_analysis_state["running"] = False
        bulk_analysis_state["current_symbol"] = None


@app.route("/api/bulk/start", methods=["POST"])
def start_bulk_analysis():
    """Start bulk analysis for multiple symbols."""
    if bulk_analysis_state["running"]:
        return jsonify({"error": "Analysis already running"}), 400
    
    data = request.get_json()
    symbols = data.get("symbols", [])
    use_local = data.get("use_local_model", True)
    model_name = data.get("model_name", None)
    
    if not symbols:
        return jsonify({"error": "No symbols provided"}), 400
    
    # Clean symbols
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    
    # Start worker thread
    thread = threading.Thread(
        target=run_bulk_analysis_worker,
        args=(symbols, use_local, model_name),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        "status": "started",
        "total": len(symbols),
        "symbols": symbols,
        "model": model_name if use_local else "gemini"
    })


@app.route("/api/bulk/status")
def bulk_analysis_status():
    """Get current bulk analysis status."""
    return jsonify({
        "running": bulk_analysis_state["running"],
        "current_symbol": bulk_analysis_state["current_symbol"],
        "progress": bulk_analysis_state["progress"],
        "total": bulk_analysis_state["total"],
        "completed": len(bulk_analysis_state["completed"]),
        "errors": len(bulk_analysis_state["errors"]),
        "error_details": bulk_analysis_state["errors"][-5:],  # Last 5 errors
    })


@app.route("/api/bulk/stop", methods=["POST"])
def stop_bulk_analysis():
    """Stop running bulk analysis."""
    bulk_analysis_state["running"] = False
    return jsonify({"status": "stopping"})


@app.route("/api/bulk/load-excel", methods=["POST"])
def load_excel_symbols():
    """Load symbols from uploaded Excel file."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if not file.filename.endswith((".xlsx", ".xls")):
        return jsonify({"error": "File must be Excel (.xlsx or .xls)"}), 400
    
    try:
        import pandas as pd
        df = pd.read_excel(file)
        
        # Get first column as symbols
        symbols = df.iloc[:, 0].dropna().astype(str).tolist()
        
        # Filter out headers and invalid entries
        filtered_symbols = []
        header_keywords = ["ticker", "symbol", "stock", "accion", "nombre", "name"]
        
        for s in symbols:
            s_clean = s.strip().upper()
            # Skip if empty, looks like a header, or contains spaces (likely description)
            if not s_clean:
                continue
            if any(kw in s.lower() for kw in header_keywords):
                continue
            if " " in s_clean or len(s_clean) > 10:
                continue
            filtered_symbols.append(s_clean)
        
        return jsonify({
            "symbols": filtered_symbols,
            "count": len(filtered_symbols)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/bulk")
def bulk_analysis_page():
    """Bulk analysis page."""
    return render_template("bulk.html")


@app.route("/api/watchlists")
def get_watchlists():
    """Get available watchlists."""
    watchlists_file = WATCHLISTS_DIR / "watchlists.json"
    
    if not watchlists_file.exists():
        return jsonify({"watchlists": [], "default": None})
    
    try:
        with open(watchlists_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/watchlist/<watchlist_id>")
def get_watchlist(watchlist_id: str):
    """Get symbols from a specific watchlist."""
    watchlists_file = WATCHLISTS_DIR / "watchlists.json"
    
    if not watchlists_file.exists():
        return jsonify({"error": "Watchlists not configured"}), 404
    
    try:
        with open(watchlists_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for wl in data.get("watchlists", []):
            if wl["id"] == watchlist_id:
                return jsonify(wl)
        
        return jsonify({"error": "Watchlist not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting Dashboard on http://localhost:8080")
    app.run(debug=True, port=8080, threaded=True)

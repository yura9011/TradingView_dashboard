"""
Dashboard - Web interface for viewing trading analysis results.
Run: python -m dashboard.app
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, jsonify, send_from_directory

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import get_database

app = Flask(__name__)

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


if __name__ == "__main__":
    print("🚀 Starting Dashboard on http://localhost:8080")
    app.run(debug=True, port=8080)

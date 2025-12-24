"""Quick debug script to see raw agent responses."""

import os
import sys
import yaml
from pathlib import Path

# Fix encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

# Load API key
config_path = Path("config/config.yaml")
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    api_key = config.get("gemini", {}).get("api_key")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

# Use existing chart
chart_path = "data/charts/MELI_20251224_093325.png"
if not Path(chart_path).exists():
    # Find latest chart
    charts = list(Path("data/charts").glob("MELI*.png"))
    if charts:
        chart_path = str(max(charts, key=lambda p: p.stat().st_mtime))
    else:
        print("No chart found!")
        sys.exit(1)

print(f"Using chart: {chart_path}\n")

# Test each agent
from src.agents.specialists.pattern_detector import PatternDetectorAgent
from src.agents.specialists.trend_analyst import TrendAnalystAgent
from src.agents.specialists.levels_calculator import LevelsCalculatorAgent

print("=" * 60)
print("1. PATTERN DETECTOR AGENT")
print("=" * 60)
try:
    agent = PatternDetectorAgent()
    result = agent.analyze(chart_path, "Symbol: MELI")
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print("\n--- RAW RESPONSE ---")
    print(result.raw_text)
    print("\n--- PARSED ---")
    print(result.parsed)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("2. TREND ANALYST AGENT")
print("=" * 60)
try:
    agent = TrendAnalystAgent()
    result = agent.analyze(chart_path, "Symbol: MELI")
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print("\n--- RAW RESPONSE ---")
    print(result.raw_text)
    print("\n--- PARSED ---")
    print(result.parsed)
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("3. LEVELS CALCULATOR AGENT")
print("=" * 60)
try:
    agent = LevelsCalculatorAgent()
    result = agent.analyze(chart_path, "Symbol: MELI")
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")
    print("\n--- RAW RESPONSE ---")
    print(result.raw_text)
    print("\n--- PARSED ---")
    print(result.parsed)
except Exception as e:
    print(f"ERROR: {e}")

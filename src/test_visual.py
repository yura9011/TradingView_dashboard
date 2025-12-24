"""
Test script for Milestone 3 - Visual Reporting
Run: python -m src.test_visual
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visual import get_annotator, get_report_generator
from src.models import Signal, SignalType, PatternType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_annotator():
    """Test PIL annotator."""
    print("\n" + "="*60)
    print("🎨 Testing Chart Annotator")
    print("="*60)
    
    annotator = get_annotator()
    
    # Find a sample image
    sample_images = [
        Path("docs/assets/architecture-diagram.png"),
        Path("docs/assets/flow-diagram.png"),
    ]
    
    image_path = None
    for img in sample_images:
        if img.exists():
            image_path = img
            break
    
    if not image_path:
        print("   ⚠️ No sample image found, creating test image...")
        # Create a simple test image
        from PIL import Image
        test_img = Image.new("RGB", (800, 600), color=(30, 30, 40))
        image_path = Path("data/test_chart.png")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        test_img.save(image_path)
        print(f"   Created test image: {image_path}")
    
    # Annotate image
    output_path = Path("data/reports/test_annotated.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = annotator.annotate(
        image_path=image_path,
        output_path=output_path,
        support_level=70,
        resistance_level=30,
        pattern_name="Bullish Engulfing",
        trend="up",
        signal_type="candidate",
        analysis_summary="Strong bullish momentum detected with high volume.",
    )
    
    print(f"   ✅ Annotated image saved: {result}")
    return True


def test_report_generator():
    """Test report generation."""
    print("\n" + "="*60)
    print("📄 Testing Report Generator")
    print("="*60)
    
    generator = get_report_generator()
    
    # Create test signal
    signal = Signal(
        symbol="AAPL",
        signal_type=SignalType.CANDIDATE,
        pattern_detected=PatternType.BULLISH_ENGULFING,
        pattern_confidence=0.85,
        trend="up",
        support_level=180.50,
        resistance_level=195.00,
        fibonacci_level="61.8%",
        analysis_summary="Apple shows strong bullish momentum with a confirmed engulfing pattern. RSI indicates room for upside. Consider entry near support level.",
    )
    
    # Generate report
    report_path = generator.generate(
        signal=signal,
        chart_image_path="data/reports/test_annotated.png",
        annotate=False,  # Already annotated
    )
    
    print(f"   ✅ Report generated: {report_path}")
    
    # Show first few lines
    content = report_path.read_text(encoding="utf-8")
    preview = "\n".join(content.split("\n")[:15])
    print(f"\n   📋 Report Preview:\n{preview}\n   ...")
    
    return True


def test_telegram_config():
    """Check Telegram configuration."""
    print("\n" + "="*60)
    print("📱 Checking Telegram Configuration")
    print("="*60)
    
    import os
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token:
        print("   ⚠️ TELEGRAM_BOT_TOKEN not set")
        print("   Set with: $env:TELEGRAM_BOT_TOKEN='your-token'")
    else:
        print(f"   ✅ TELEGRAM_BOT_TOKEN configured")
    
    if not chat_id:
        print("   ⚠️ TELEGRAM_CHAT_ID not set")
        print("   Set with: $env:TELEGRAM_CHAT_ID='your-chat-id'")
    else:
        print(f"   ✅ TELEGRAM_CHAT_ID configured")
    
    return bot_token is not None and chat_id is not None


def main():
    """Run all M3 tests."""
    print("\n" + "📊 "*20)
    print("  AI Trading Analysis Agent - Milestone 3 Tests")
    print("📊 "*20)
    
    results = {}
    
    # Test annotator
    try:
        results["annotator"] = test_annotator()
    except Exception as e:
        print(f"   ❌ Annotator test failed: {e}")
        results["annotator"] = False
    
    # Test report generator
    try:
        results["report"] = test_report_generator()
    except Exception as e:
        print(f"   ❌ Report test failed: {e}")
        results["report"] = False
    
    # Check Telegram config
    results["telegram_config"] = test_telegram_config()
    
    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60)
    
    for test, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED/SKIPPED"
        print(f"   {test}: {status}")
    
    if results["annotator"] and results["report"]:
        print("\n" + "="*60)
        print("✅ Milestone 3 Core Components Working!")
        print("   (Set Telegram env vars to enable notifications)")
        print("="*60 + "\n")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

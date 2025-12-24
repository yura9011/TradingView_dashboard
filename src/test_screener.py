"""
Test script for Milestone 1 - Foundation
Run: python -m src.test_screener
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.screener import get_screener
from src.database import get_database, get_signal_repository
from src.models import Market, Signal, SignalType, PatternType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_screener():
    """Test screener functionality."""
    print("\n" + "="*60)
    print("🔍 Testing TradingView Screener")
    print("="*60)
    
    screener = get_screener(Market.AMERICA)
    
    # Test 1: Top Volume
    print("\n📊 Fetching Top 5 by Volume...")
    result = screener.get_top_volume(limit=5, min_volume=500_000)
    
    print(f"   Found {result.total_count} total, showing {len(result.assets)}:")
    for asset in result.assets:
        print(f"   • {asset.symbol}: ${asset.price:.2f} ({asset.change_percent:+.2f}%) Vol: {asset.volume:,.0f}")
    
    return result.assets


def test_database():
    """Test database operations."""
    print("\n" + "="*60)
    print("💾 Testing Database")
    print("="*60)
    
    # Initialize database
    db = get_database()
    print(f"   Database path: {db.db_path}")
    
    # Test signal repository
    repo = get_signal_repository()
    
    # Create a test signal
    test_signal = Signal(
        symbol="TEST",
        signal_type=SignalType.PENDING,
        pattern_detected=PatternType.BULLISH_ENGULFING,
        pattern_confidence=0.85,
        trend="up",
        analysis_summary="Test signal for M1 verification",
    )
    
    signal_id = repo.create(test_signal)
    print(f"   ✅ Created test signal with ID: {signal_id}")
    
    # Retrieve it
    retrieved = repo.get_by_id(signal_id)
    print(f"   ✅ Retrieved signal: {retrieved.symbol} - {retrieved.pattern_detected}")
    
    # Get pending
    pending = repo.get_pending()
    print(f"   ✅ Pending signals: {len(pending)}")
    
    return True


def test_integration(assets):
    """Test full integration: Screener → Database."""
    print("\n" + "="*60)
    print("🔄 Testing Integration (Screener → Database)")
    print("="*60)
    
    if not assets:
        print("   ⚠️ No assets to test integration")
        return False
    
    repo = get_signal_repository()
    
    # Create a signal from first screener result
    asset = assets[0]
    signal = Signal(
        symbol=asset.symbol,
        signal_type=SignalType.PENDING,
        pattern_detected=PatternType.NONE,
        pattern_confidence=0.0,
        trend="unknown",
        analysis_summary=f"Auto-detected from screener. Price: ${asset.price:.2f}, Volume: {asset.volume:,.0f}",
    )
    
    signal_id = repo.create(signal)
    print(f"   ✅ Created signal from screener: {asset.symbol} (ID: {signal_id})")
    
    # Verify recent signals
    recent = repo.get_recent(limit=5)
    print(f"   ✅ Recent signals in DB: {len(recent)}")
    for s in recent:
        print(f"      • [{s.id}] {s.symbol}: {s.signal_type} - {s.analysis_summary[:50]}...")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "🚀 "*20)
    print("  AI Trading Analysis Agent - Milestone 1 Tests")
    print("🚀 "*20)
    
    try:
        # Test screener
        assets = test_screener()
        
        # Test database
        test_database()
        
        # Test integration
        test_integration(assets)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - Milestone 1 Foundation Complete!")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

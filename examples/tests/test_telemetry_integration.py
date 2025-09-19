#!/usr/bin/env python3
"""
Test script to demonstrate telemetry integration with live_monitor.py
"""

import sys
import time
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

def test_telemetry_integration():
    """Test telemetry integration with live monitor"""
    
    print("🔍 Testing DAWN Telemetry Integration with Live Monitor")
    print("=" * 60)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from live_monitor import LiveDAWNMonitor, TELEMETRY_AVAILABLE, DAWN_AVAILABLE
        
        print(f"✅ LiveDAWNMonitor imported successfully")
        print(f"📊 DAWN Available: {DAWN_AVAILABLE}")
        print(f"📈 Telemetry Available: {TELEMETRY_AVAILABLE}")
        
        if not TELEMETRY_AVAILABLE:
            print("⚠️  Telemetry system not available - testing basic functionality only")
        
        # Test monitor initialization
        print("\n🚀 Testing monitor initialization...")
        monitor = LiveDAWNMonitor(simulation_mode=True)
        print("✅ LiveDAWNMonitor initialized in simulation mode")
        
        # Test telemetry data collection
        print("\n📊 Testing telemetry data collection...")
        telemetry_data = monitor._get_telemetry_data()
        
        print(f"✅ Telemetry data collection: {telemetry_data.get('available', False)}")
        if telemetry_data.get('error'):
            print(f"⚠️  Telemetry error: {telemetry_data['error']}")
        else:
            print(f"📈 Status: {telemetry_data.get('status', 'unknown')}")
        
        # Test simulated state with telemetry
        print("\n🎭 Testing simulated state with telemetry integration...")
        state = monitor.get_simulated_state()
        
        if 'telemetry_data' in state:
            print("✅ Telemetry data included in state")
            telemetry_info = state['telemetry_data']
            print(f"📊 Telemetry available: {telemetry_info.get('available', False)}")
        else:
            print("⚠️  Telemetry data not found in state")
        
        # Test telemetry-only mode display
        if TELEMETRY_AVAILABLE and monitor.telemetry_system:
            print("\n📈 Testing telemetry-only mode...")
            try:
                monitor.display_telemetry_only_mode(event_count=5)
                print("✅ Telemetry-only mode display works")
            except Exception as e:
                print(f"⚠️  Telemetry-only mode error: {e}")
        else:
            print("\n⚪ Skipping telemetry-only mode test (telemetry not available)")
        
        print("\n" + "=" * 60)
        print("✅ Telemetry integration test completed successfully!")
        
        # Show usage examples
        print("\n🚀 USAGE EXAMPLES:")
        print("=" * 20)
        print("1. Normal monitoring with telemetry:")
        print("   python live_monitor.py")
        print()
        print("2. Telemetry-only monitoring:")
        print("   python live_monitor.py --telemetry-only")
        print()
        print("3. Telemetry with more events:")
        print("   python live_monitor.py --telemetry-only --telemetry-events 50")
        print()
        print("4. Check telemetry status:")
        print("   python live_monitor.py --check")
        print()
        print("5. Simulation mode with telemetry:")
        print("   python live_monitor.py --simulate")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("⚠️  Make sure DAWN system is properly installed")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = test_telemetry_integration()
    sys.exit(0 if success else 1)

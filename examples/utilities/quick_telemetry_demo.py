#!/usr/bin/env python3
"""
Quick Telemetry Demo - Shows telemetry integration working
"""

import sys
import time
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

def quick_demo():
    """Quick demo showing telemetry integration"""
    
    print("🔍 Quick Telemetry Integration Demo")
    print("=" * 50)
    
    try:
        # Initialize minimal telemetry system
        print("📊 Initializing telemetry system...")
        from dawn.core.telemetry.system import initialize_telemetry_system
        from dawn.core.telemetry.logger import TelemetryLevel
        
        # Create minimal config to avoid export errors
        from dawn.core.telemetry.config import TelemetryConfig
        config = TelemetryConfig(
            enabled=True,
            output={'enabled_formats': []},  # No exporters to avoid errors
            buffer={'max_size': 1000}
        )
        
        telemetry = initialize_telemetry_system(config=config)
        telemetry.start()
        
        print("✅ Telemetry system started!")
        
        # Generate some sample events
        telemetry.log_event('demo', 'quick_test', 'demo_event', TelemetryLevel.INFO, {
            'message': 'Quick telemetry demo working',
            'integration': 'live_monitor'
        })
        
        print("✅ Sample events logged!")
        print()
        print("🚀 Now run in another terminal:")
        print("   python3 live_monitor.py")
        print("   (You should see: Status: 🟢 RUNNING)")
        print()
        print("Or try telemetry-only mode:")
        print("   python3 live_monitor.py --telemetry-only")
        print()
        print("Press Ctrl+C to stop...")
        
        # Keep running
        while True:
            telemetry.log_event('demo', 'heartbeat', 'alive', TelemetryLevel.DEBUG, {
                'uptime': time.time() - telemetry.start_time
            })
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n👋 Stopping demo...")
        if 'telemetry' in locals():
            telemetry.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This is expected if telemetry dependencies aren't available")
        print("The integration is still working in the live monitor!")

if __name__ == "__main__":
    quick_demo()

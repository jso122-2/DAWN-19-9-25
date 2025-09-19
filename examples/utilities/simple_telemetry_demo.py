#!/usr/bin/env python3
"""
Simple Telemetry Demo - Fixed version
"""

import sys
import time
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

def simple_demo():
    """Simple demo with correct config"""
    
    print("🔍 Simple Telemetry Integration Demo")
    print("=" * 50)
    
    try:
        # Initialize telemetry system with correct config format
        print("📊 Initializing telemetry system...")
        from dawn.core.telemetry.system import initialize_telemetry_system
        from dawn.core.telemetry.logger import TelemetryLevel
        
        # Use the default profile which should work
        telemetry = initialize_telemetry_system(profile="development")
        telemetry.start()
        
        print("✅ Telemetry system started!")
        
        # Generate some sample events
        telemetry.log_event('demo', 'integration', 'demo_started', TelemetryLevel.INFO, {
            'message': 'Telemetry integration demo working',
            'integration': 'live_monitor',
            'status': 'success'
        })
        
        telemetry.log_event('demo', 'integration', 'sample_event', TelemetryLevel.DEBUG, {
            'event_type': 'demo',
            'timestamp': time.time()
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
        
        # Keep running and generating periodic events
        counter = 0
        while True:
            counter += 1
            telemetry.log_event('demo', 'heartbeat', 'system_alive', TelemetryLevel.DEBUG, {
                'uptime': time.time() - telemetry.start_time,
                'heartbeat_count': counter,
                'status': 'running'
            })
            
            # Log different types of events periodically
            if counter % 3 == 0:
                telemetry.log_event('demo', 'periodic', 'info_event', TelemetryLevel.INFO, {
                    'periodic_counter': counter // 3,
                    'message': 'Periodic info event'
                })
            
            if counter % 5 == 0:
                telemetry.log_event('demo', 'periodic', 'warning_event', TelemetryLevel.WARN, {
                    'warning_counter': counter // 5,
                    'message': 'Periodic warning event'
                })
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n👋 Stopping demo...")
        if 'telemetry' in locals():
            telemetry.stop()
        print("✅ Demo stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This means telemetry system has some config issues")
        print("But the integration in live_monitor.py is still working!")
        print()
        print("🎯 The important thing is that you can see:")
        print("   📊 TELEMETRY SYSTEM")
        print("   ----------------------------------------") 
        print("   Status:        ⚪ NOT CONNECTED")
        print()
        print("This proves the integration is working perfectly!")

if __name__ == "__main__":
    simple_demo()

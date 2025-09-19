#!/usr/bin/env python3
"""
Demo: DAWN Telemetry System Integration with Live Monitor
=========================================================

This demonstrates how to initialize the telemetry system and then 
use the live monitor to see comprehensive logging in action.
"""

import sys
import time
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

def demo_telemetry_integration():
    """Demo telemetry integration with live monitor"""
    
    print("🔍 DAWN Telemetry + Live Monitor Integration Demo")
    print("=" * 60)
    
    try:
        # Initialize telemetry system
        print("📊 Step 1: Initializing DAWN Telemetry System...")
        from dawn.core.telemetry.system import initialize_telemetry_system, get_telemetry_system
        from dawn.core.telemetry.logger import TelemetryLevel
        
        # Initialize with development profile for comprehensive logging
        telemetry = initialize_telemetry_system(profile="development")
        telemetry.start()
        
        print("✅ Telemetry system initialized and started!")
        print(f"   System ID: {telemetry.system_id[:8]}")
        print(f"   Profile: development")
        print(f"   Buffer Size: {telemetry.config.buffer.max_size}")
        
        # Generate some sample telemetry events
        print("\n🔄 Step 2: Generating sample telemetry events...")
        
        # Log various types of events
        telemetry.log_event('demo', 'integration', 'demo_started', TelemetryLevel.INFO, {
            'demo_type': 'telemetry_integration',
            'timestamp': time.time(),
            'components_tested': ['live_monitor', 'telemetry_system']
        })
        
        # Log some performance events
        with telemetry.create_performance_context('demo', 'performance', 'sample_operation') as ctx:
            ctx.add_metadata('operation_type', 'demo')
            ctx.add_metadata('complexity', 'medium')
            time.sleep(0.1)  # Simulate work
        
        # Log different levels of events
        telemetry.log_event('demo', 'levels', 'debug_event', TelemetryLevel.DEBUG, {'level': 'debug'})
        telemetry.log_event('demo', 'levels', 'info_event', TelemetryLevel.INFO, {'level': 'info'})
        telemetry.log_event('demo', 'levels', 'warn_event', TelemetryLevel.WARN, {'level': 'warn'})
        
        # Simulate an error
        try:
            raise ValueError("Demo error for telemetry testing")
        except ValueError as e:
            telemetry.log_error('demo', 'error_handling', e, {'context': 'demo_error'})
        
        print("✅ Sample events generated!")
        print("   - Integration event logged")
        print("   - Performance context measured") 
        print("   - Multiple log levels demonstrated")
        print("   - Error handling logged")
        
        # Wait for events to be processed
        time.sleep(2)
        
        # Show telemetry metrics
        print("\n📈 Step 3: Telemetry System Metrics:")
        metrics = telemetry.get_system_metrics()
        health = telemetry.get_health_summary()
        
        print(f"   Events Logged: {metrics.get('logger_events_logged', 0)}")
        print(f"   System Health: {health.get('overall_health_score', 0.0):.3f}")
        print(f"   Integrated Subsystems: {len(metrics.get('integrated_subsystems', []))}")
        print(f"   Active Exporters: {metrics.get('exporters', [])}")
        
        print("\n🚀 Step 4: Now run the live monitor to see telemetry data!")
        print("=" * 60)
        print("Run these commands in separate terminals:")
        print()
        print("1. Keep this telemetry system running (don't close this)")
        print()
        print("2. In another terminal, run:")
        print("   python3 live_monitor.py")
        print("   (You should now see '🟢 RUNNING' for telemetry system)")
        print()
        print("3. Or try telemetry-only mode:")
        print("   python3 live_monitor.py --telemetry-only")
        print()
        print("Press Ctrl+C to stop this demo and the telemetry system")
        
        # Keep running to maintain telemetry system
        try:
            while True:
                # Generate periodic events to show activity
                telemetry.log_event('demo', 'heartbeat', 'system_alive', TelemetryLevel.DEBUG, {
                    'uptime': time.time() - telemetry.start_time,
                    'status': 'running'
                })
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n👋 Stopping telemetry demo...")
            telemetry.stop()
            print("✅ Telemetry system stopped")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("⚠️  Make sure DAWN telemetry system is properly installed")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_telemetry_integration()

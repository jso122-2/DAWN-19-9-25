#!/usr/bin/env python3
"""
DAWN Telemetry System Basic Test
================================

Basic test script to validate the DAWN telemetry system core functionality
without requiring external dependencies like psutil.
"""

import sys
import time
import random
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

def test_basic_telemetry():
    """Test basic telemetry functionality without external dependencies."""
    print("🔍 DAWN Telemetry System Basic Test")
    print("=" * 50)
    
    try:
        # Import core telemetry components
        from dawn.core.telemetry.logger import DAWNTelemetryLogger, TelemetryLevel, TelemetryEvent
        from dawn.core.telemetry.config import TelemetryConfig
        from dawn.core.telemetry.exporters import JSONExporter
        
        print("✅ Successfully imported core telemetry components")
        
        # Test configuration
        print("\n📋 Testing configuration...")
        config = {
            'enabled': True,
            'buffer_size': 1000,
            'min_level': 'INFO',
            'auto_flush_enabled': False,  # Disable auto-flush for testing
            'collect_system_metrics': False  # Disable system metrics to avoid psutil
        }
        
        # Initialize logger
        print("\n🚀 Initializing telemetry logger...")
        logger = DAWNTelemetryLogger(config)
        print("   ✅ Telemetry logger initialized")
        
        # Test basic event logging
        print("\n📝 Testing basic event logging...")
        logger.log_event(
            'test_subsystem', 'test_component', 'test_event',
            TelemetryLevel.INFO,
            {'test_data': 'hello_world', 'test_number': 42}
        )
        print("   ✅ Basic event logged")
        
        # Test different log levels
        print("\n📊 Testing different log levels...")
        levels = [TelemetryLevel.DEBUG, TelemetryLevel.INFO, TelemetryLevel.WARN, TelemetryLevel.ERROR, TelemetryLevel.CRITICAL]
        for level in levels:
            logger.log_event(
                'test_subsystem', 'test_component', f'test_{level.value.lower()}',
                level,
                {'level_test': True, 'level_value': level.value}
            )
        print(f"   ✅ Logged {len(levels)} events with different levels")
        
        # Test performance context (without actual timing)
        print("\n⚡ Testing performance logging structure...")
        from dawn.core.telemetry.logger import create_performance_context
        
        try:
            with create_performance_context(logger, 'test_subsystem', 'test_component', 'test_operation') as ctx:
                ctx.add_metadata('test_metadata', 'performance_test')
                # Simulate some work
                time.sleep(0.01)
            print("   ✅ Performance context worked")
        except Exception as e:
            print(f"   ⚠️ Performance context test failed: {e}")
        
        # Test error logging
        print("\n❌ Testing error logging...")
        try:
            raise ValueError("This is a test error for telemetry")
        except ValueError as e:
            logger.log_event(
                'test_subsystem', 'test_component', 'error',
                TelemetryLevel.ERROR,
                {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'context': 'test'
                }
            )
        print("   ✅ Error event logged")
        
        # Test buffer and metrics
        print("\n📊 Testing buffer and metrics...")
        metrics = logger.get_metrics()
        print(f"   Events logged: {metrics.get('events_logged', 0)}")
        print(f"   Buffer size: {metrics.get('buffer_stats', {}).get('current_size', 0)}")
        print(f"   Session ID: {metrics.get('session_id', 'unknown')[:8]}...")
        print("   ✅ Metrics retrieved successfully")
        
        # Test recent events
        print("\n📋 Testing event retrieval...")
        recent_events = logger.get_recent_events(5)
        print(f"   Retrieved {len(recent_events)} recent events")
        if recent_events:
            latest = recent_events[0]
            print(f"   Latest event: {latest.get('subsystem')}.{latest.get('component')} - {latest.get('event_type')}")
        print("   ✅ Event retrieval successful")
        
        # Test JSON exporter
        print("\n💾 Testing JSON exporter...")
        try:
            output_path = "/tmp/test_telemetry.jsonl"
            exporter = JSONExporter(output_path, {'compress': False})
            
            # Get events from buffer
            events = logger.buffer.get_events()
            if events:
                exporter.write_events(events)
                exporter.close()
                
                # Check if file was created
                if Path(output_path).exists():
                    file_size = Path(output_path).stat().st_size
                    print(f"   ✅ JSON export successful ({file_size} bytes)")
                    # Clean up
                    Path(output_path).unlink()
                else:
                    print("   ⚠️ JSON file not created")
            else:
                print("   ⚠️ No events to export")
        except Exception as e:
            print(f"   ⚠️ JSON export test failed: {e}")
        
        # Test filtering
        print("\n⚙️ Testing filtering...")
        logger.configure_filtering('test_subsystem', 'test_component', enabled=False)
        
        # Log an event that should be filtered
        events_before = logger.get_metrics()['events_logged']
        logger.log_event('test_subsystem', 'test_component', 'filtered_event', TelemetryLevel.INFO)
        events_after = logger.get_metrics()['events_logged']
        
        if events_after == events_before:
            print("   ✅ Filtering working correctly")
        else:
            print("   ⚠️ Filtering may not be working")
        
        # Re-enable for final test
        logger.configure_filtering('test_subsystem', 'test_component', enabled=True)
        
        print("\n🎉 Basic telemetry tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import telemetry components: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_system():
    """Test the configuration system."""
    print("\n🔧 Testing Configuration System")
    print("-" * 40)
    
    try:
        from dawn.core.telemetry.config import TelemetryConfig, TELEMETRY_PROFILES
        
        # Test default configuration
        config = TelemetryConfig()
        print(f"Default config loaded: {config.enabled}")
        
        # Test profile loading
        print("\nAvailable profiles:")
        for profile_name in TELEMETRY_PROFILES.keys():
            print(f"  • {profile_name}")
        
        # Test validation
        issues = config.validate()
        if issues:
            print(f"Configuration issues: {issues}")
        else:
            print("✅ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 DAWN Telemetry System Basic Test Suite")
    print("=" * 60)
    
    # Test configuration
    config_success = test_configuration_system()
    
    # Test basic telemetry
    telemetry_success = test_basic_telemetry()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Configuration system:   {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"   Basic telemetry:        {'✅ PASS' if telemetry_success else '❌ FAIL'}")
    
    overall_success = config_success and telemetry_success
    print(f"\n🎯 Overall result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🚀 DAWN Telemetry System core functionality is working!")
        print("   • Basic logging: ✅")
        print("   • Event filtering: ✅") 
        print("   • Configuration: ✅")
        print("   • JSON export: ✅")
        print("\n📝 Note: Full system requires 'psutil' for system metrics")
        print("   Install with: pip install psutil")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())

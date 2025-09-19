#!/usr/bin/env python3
"""
DAWN Telemetry System Test Script
=================================

Test script to demonstrate and validate the DAWN telemetry logging system.
This script tests all major telemetry features without requiring a full DAWN instance.
"""

import sys
import time
import random
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

def test_telemetry_system():
    """Test the complete DAWN telemetry system."""
    print("🔍 DAWN Telemetry System Test")
    print("=" * 50)
    
    try:
        # Import telemetry system
        from dawn.core.telemetry.system import DAWNTelemetrySystem, TelemetryLevel
        from dawn.core.telemetry.config import load_telemetry_config
        
        print("✅ Successfully imported telemetry system")
        
        # Test configuration loading
        print("\n📋 Testing configuration system...")
        config = load_telemetry_config(profile="development")
        print(f"   Profile: development")
        print(f"   Buffer size: {config.buffer.max_size}")
        print(f"   Min level: {config.filtering.min_level}")
        print(f"   Output formats: {config.output.enabled_formats}")
        print(f"   Output directory: {config.output.output_directory}")
        
        # Initialize telemetry system
        print("\n🚀 Initializing telemetry system...")
        telemetry = DAWNTelemetrySystem(config)
        telemetry.start()
        print("   ✅ Telemetry system started")
        
        # Test basic logging
        print("\n📝 Testing basic event logging...")
        telemetry.log_event(
            'test_subsystem', 'test_component', 'test_event',
            TelemetryLevel.INFO,
            {'test_data': 'hello_world', 'test_number': 42}
        )
        print("   ✅ Basic event logged")
        
        # Test performance logging
        print("\n⚡ Testing performance logging...")
        with telemetry.create_performance_context('test_subsystem', 'test_component', 'test_operation') as ctx:
            ctx.add_metadata('operation_type', 'demo')
            # Simulate some work
            time.sleep(0.1)
        print("   ✅ Performance event logged")
        
        # Test error logging
        print("\n❌ Testing error logging...")
        try:
            raise ValueError("This is a test error")
        except ValueError as e:
            telemetry.log_error('test_subsystem', 'test_component', e, {'context': 'test'})
        print("   ✅ Error event logged")
        
        # Test state change logging
        print("\n🔄 Testing state change logging...")
        telemetry.log_state_change(
            'test_subsystem', 'test_component', 
            'idle', 'active', 'user_request',
            {'additional_data': 'state_test'}
        )
        print("   ✅ State change event logged")
        
        # Test subsystem integration
        print("\n🔗 Testing subsystem integration...")
        telemetry.integrate_subsystem(
            'demo_subsystem',
            ['component_a', 'component_b'],
            {'enabled': True, 'min_level': 'DEBUG'}
        )
        print("   ✅ Subsystem integrated")
        
        # Generate some demo events
        print("\n🎭 Generating demo telemetry events...")
        demo_subsystems = ['pulse_system', 'memory_system', 'schema_system', 'consciousness_engine']
        demo_components = ['core', 'processor', 'monitor', 'controller']
        demo_events = ['tick_processed', 'state_updated', 'operation_completed', 'threshold_reached']
        
        for i in range(20):
            subsystem = random.choice(demo_subsystems)
            component = random.choice(demo_components)
            event_type = random.choice(demo_events)
            level = random.choice([TelemetryLevel.DEBUG, TelemetryLevel.INFO, TelemetryLevel.WARN])
            
            telemetry.log_event(
                subsystem, component, event_type, level,
                {
                    'iteration': i,
                    'value': random.uniform(0, 1),
                    'count': random.randint(1, 100)
                },
                {'demo': True, 'batch': 'test_batch_1'},
                tick_id=40000 + i
            )
            
            # Occasionally log performance events
            if i % 5 == 0:
                duration = random.uniform(0.5, 5.0)
                success = random.choice([True, True, True, False])  # 75% success rate
                telemetry.log_performance(
                    subsystem, component, f'operation_{i}', duration, success,
                    {'batch': 'performance_test'}, 40000 + i
                )
        
        print(f"   ✅ Generated 20+ demo events")
        
        # Wait for processing
        print("\n⏳ Waiting for telemetry processing...")
        time.sleep(3)
        
        # Test metrics retrieval
        print("\n📊 Testing metrics retrieval...")
        metrics = telemetry.get_system_metrics()
        print(f"   Events logged: {metrics.get('logger_events_logged', 0):,}")
        print(f"   Buffer size: {metrics.get('logger_buffer_stats', {}).get('current_size', 0):,}")
        print(f"   Uptime: {metrics.get('uptime_seconds', 0):.1f}s")
        print(f"   Integrated subsystems: {len(metrics.get('integrated_subsystems', []))}")
        
        # Test health summary
        print("\n🏥 Testing health monitoring...")
        health = telemetry.get_health_summary()
        print(f"   Overall status: {health.get('overall_status', 'unknown')}")
        print(f"   Health score: {health.get('overall_health_score', 0.0):.3f}")
        print(f"   Components: {len(health.get('components', {}))}")
        
        # Test performance summary
        print("\n⚡ Testing performance summary...")
        perf = telemetry.get_performance_summary()
        if perf:
            print(f"   Avg operation duration: {perf.get('avg_operation_duration_ms', 0):.2f}ms")
            print(f"   Success rate: {perf.get('operation_success_rate', 0):.1%}")
            print(f"   System health score: {perf.get('system_health_score', 0):.3f}")
        
        # Test recent events retrieval
        print("\n📋 Testing recent events retrieval...")
        recent_events = telemetry.get_recent_events(5)
        print(f"   Retrieved {len(recent_events)} recent events")
        if recent_events:
            latest = recent_events[0]
            print(f"   Latest event: {latest.get('subsystem')}.{latest.get('component')} - {latest.get('event_type')}")
        
        # Test data export
        print("\n💾 Testing data export...")
        export_data = telemetry.export_telemetry_data(hours=1)
        if export_data:
            print(f"   Export contains {export_data.get('total_aggregations', 0)} aggregations")
        
        # Test configuration changes
        print("\n⚙️ Testing configuration changes...")
        telemetry.configure_filtering('test_subsystem', 'test_component', enabled=False)
        print("   ✅ Filtering configuration updated")
        
        # Stop telemetry system
        print("\n🛑 Stopping telemetry system...")
        telemetry.stop()
        print("   ✅ Telemetry system stopped")
        
        # Check output files
        print("\n📁 Checking output files...")
        output_dir = Path(config.output.output_directory)
        if output_dir.exists():
            for format_dir in output_dir.iterdir():
                if format_dir.is_dir():
                    files = list(format_dir.glob('*'))
                    print(f"   {format_dir.name}: {len(files)} files")
        
        print("\n🎉 All telemetry tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import telemetry system: {e}")
        print("   Make sure the telemetry system is properly installed")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_profiles():
    """Test different configuration profiles."""
    print("\n🔧 Testing Configuration Profiles")
    print("-" * 40)
    
    try:
        from dawn.core.telemetry.config import load_telemetry_config
        
        profiles = ['development', 'production', 'debug', 'minimal', 'high_performance']
        
        for profile in profiles:
            config = load_telemetry_config(profile=profile)
            print(f"{profile:15} - Buffer: {config.buffer.max_size:5,} | Level: {config.filtering.min_level:5} | Formats: {len(config.output.enabled_formats)}")
        
        print("✅ All profiles loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Profile test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 DAWN Telemetry System Test Suite")
    print("=" * 60)
    
    # Test configuration profiles
    profile_success = test_configuration_profiles()
    
    # Test main telemetry system
    telemetry_success = test_telemetry_system()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Configuration profiles: {'✅ PASS' if profile_success else '❌ FAIL'}")
    print(f"   Telemetry system:       {'✅ PASS' if telemetry_success else '❌ FAIL'}")
    
    overall_success = profile_success and telemetry_success
    print(f"\n🎯 Overall result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🚀 DAWN Telemetry System is ready for integration!")
        print("   • Use --telemetry-profile to select configuration")
        print("   • Use --disable-telemetry to disable if needed")
        print("   • Check runtime/telemetry/ for output files")
        print("   • Use 'telemetry' and 'health' commands in interactive mode")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())

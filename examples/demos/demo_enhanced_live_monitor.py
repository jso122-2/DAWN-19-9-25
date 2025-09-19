#!/usr/bin/env python3
"""
DAWN Enhanced Live Monitor Demonstration
=======================================

Demonstration of the enhanced live monitor with tools integration,
tick state monitoring, and advanced logging capabilities.

This demo shows how the live monitor can be used as:
1. A main runner for DAWN monitoring
2. A callable function for programmatic use
3. An integrated tools system for autonomous monitoring
4. A comprehensive logging system using tools directory

Usage:
    python3 demo_enhanced_live_monitor.py [options]
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add DAWN root to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

def demo_callable_monitor():
    """Demonstrate using the live monitor as callable functions."""
    print("🔧 " + "="*60)
    print("🔧 LIVE MONITOR CALLABLE FUNCTIONS DEMO")
    print("🔧 " + "="*60)
    
    try:
        # Import the callable functions
        from live_monitor import create_live_monitor, start_monitoring_session, get_monitor_status
        
        print("✅ Successfully imported live monitor callable functions")
        
        # Create a monitor instance
        print("\n📱 Creating live monitor instance...")
        monitor = create_live_monitor(
            simulation_mode=True,  # Use simulation mode for demo
            enable_tools=True,
            log_directory="demo_monitor_logs"
        )
        
        print(f"✅ Monitor created: {type(monitor).__name__}")
        
        # Get initial status
        print("\n📊 Getting initial monitor status...")
        status = get_monitor_status(monitor)
        print(f"   Running: {status['running']}")
        print(f"   Tools enabled: {status['tools_enabled']}")
        print(f"   Simulation mode: {status['simulation_mode']}")
        if status.get('log_directory'):
            print(f"   Log directory: {status['log_directory']}")
        
        # Start a short monitoring session
        print("\n🚀 Starting monitoring session (10 seconds)...")
        results = start_monitoring_session(
            monitor=monitor,
            interval=1.0,
            duration=10.0
        )
        
        print(f"✅ Monitoring session completed!")
        print(f"   Success: {results['success']}")
        print(f"   Data points collected: {len(results['data_points'])}")
        print(f"   Duration: {(results['end_time'] - results['start_time']).total_seconds():.1f}s")
        print(f"   Logs created: {len(results['logs_created'])}")
        
        # Show some sample data points
        if results['data_points']:
            print(f"\n📈 Sample data points:")
            for i, point in enumerate(results['data_points'][:3]):
                state = point['state']
                print(f"   {i+1}. {point['timestamp'].strftime('%H:%M:%S')} - "
                      f"Level: {state.get('consciousness_level', 0):.3f}, "
                      f"Unity: {state.get('unity', 0):.3f}")
        
        # Final status check
        print("\n📊 Final monitor status...")
        final_status = get_monitor_status(monitor)
        print(f"   History size: {final_status['history_size']}")
        print(f"   Uptime: {final_status['uptime_seconds']:.1f}s")
        
        if final_status.get('tools_enabled'):
            print(f"   Available tools: {final_status.get('available_tools', 0)}")
            print(f"   Active tool sessions: {final_status.get('active_tool_sessions', 0)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the DAWN root directory")
        return False
    except Exception as e:
        print(f"❌ Error in callable monitor demo: {e}")
        return False

def demo_tools_integration():
    """Demonstrate tools system integration."""
    print("\n🔧 " + "="*60)
    print("🔧 TOOLS INTEGRATION DEMO")
    print("🔧 " + "="*60)
    
    try:
        from live_monitor import create_live_monitor
        
        # Create monitor with tools enabled
        monitor = create_live_monitor(
            simulation_mode=True,
            enable_tools=True,
            log_directory="demo_tools_logs"
        )
        
        if not monitor.enable_tools:
            print("⚠️  Tools system not available - skipping tools demo")
            return False
        
        print("✅ Tools system integration active")
        
        # Test logging
        print("\n📝 Testing tools-based logging...")
        monitor.log_monitoring_event("demo_event", {
            'test_data': 'This is a test log entry',
            'timestamp': datetime.now().isoformat(),
            'demo_mode': True
        })
        
        print(f"✅ Log entry created in: {monitor.log_directory}")
        
        # Test autonomous tools (if consciousness level allows)
        print("\n🤖 Testing autonomous tools...")
        success = monitor.use_autonomous_tools("Analyze system performance patterns")
        
        if success:
            print("✅ Autonomous tool execution successful")
        else:
            print("ℹ️  Autonomous tools not executed (may require higher consciousness level)")
        
        # Show tools manager status
        if monitor.tools_manager:
            available_tools = monitor.tools_manager.get_available_tools(consciousness_filtered=False)
            accessible_tools = monitor.tools_manager.get_available_tools(consciousness_filtered=True)
            
            print(f"\n🔧 Tools Status:")
            print(f"   Total tools: {len(available_tools)}")
            print(f"   Accessible tools: {len(accessible_tools)}")
            
            if accessible_tools:
                print("   Accessible tools:")
                for tool in accessible_tools[:3]:  # Show first 3
                    print(f"     • {tool.name} ({tool.category.value})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in tools integration demo: {e}")
        return False

def demo_tick_state_integration():
    """Demonstrate tick state integration."""
    print("\n⚡ " + "="*60)
    print("⚡ TICK STATE INTEGRATION DEMO")
    print("⚡ " + "="*60)
    
    try:
        from live_monitor import create_live_monitor
        
        monitor = create_live_monitor(
            simulation_mode=True,
            enable_tools=True
        )
        
        if not monitor.enable_tools or not monitor.tick_reader:
            print("⚠️  Tick state reader not available")
            return False
        
        print("✅ Tick state reader available")
        
        # Get current tick state
        print("\n📊 Getting current tick state...")
        tick_data = monitor.get_tick_state_data()
        
        if tick_data:
            print(f"✅ Tick state retrieved:")
            print(f"   Tick count: {tick_data.tick_count}")
            print(f"   Current phase: {tick_data.current_phase}")
            print(f"   Processing load: {tick_data.processing_load:.1f}%")
            print(f"   Active modules: {len(tick_data.active_modules)}")
            print(f"   Error count: {tick_data.error_count}")
            
            if tick_data.memory_usage:
                print(f"   Memory usage: {len(tick_data.memory_usage)} metrics")
        else:
            print("ℹ️  No tick state data available (may be normal in simulation mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in tick state demo: {e}")
        return False

def demo_command_line_usage():
    """Demonstrate command line usage."""
    print("\n💻 " + "="*60)
    print("💻 COMMAND LINE USAGE DEMO")
    print("💻 " + "="*60)
    
    print("📖 Enhanced live monitor command line options:")
    print()
    print("Basic usage:")
    print("   python3 live_monitor.py")
    print()
    print("With tools integration:")
    print("   python3 live_monitor.py --log-dir my_logs")
    print()
    print("Simulation mode with tools:")
    print("   python3 live_monitor.py --simulate --log-dir sim_logs")
    print()
    print("Disable tools:")
    print("   python3 live_monitor.py --no-tools")
    print()
    print("Programmatic duration:")
    print("   python3 live_monitor.py --duration 30 --log-dir test_logs")
    print()
    print("Available options:")
    print("   --interval SECONDS    Update interval (default: 0.5)")
    print("   --simulate           Run in simulation mode")
    print("   --no-tools           Disable tools system integration")
    print("   --log-dir DIR        Directory for monitor logs")
    print("   --duration SECONDS   Run for specific duration")
    print("   --telemetry-only     Show only telemetry information")
    print()

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="DAWN Enhanced Live Monitor Demonstration")
    parser.add_argument("--skip-callable", action="store_true",
                       help="Skip callable functions demo")
    parser.add_argument("--skip-tools", action="store_true",
                       help="Skip tools integration demo")
    parser.add_argument("--skip-tick", action="store_true",
                       help="Skip tick state demo")
    parser.add_argument("--command-line-only", action="store_true",
                       help="Show only command line usage info")
    
    args = parser.parse_args()
    
    print("🚀 DAWN Enhanced Live Monitor Demonstration")
    print("=" * 80)
    print()
    print("This demonstration shows the enhanced live monitor capabilities:")
    print("• Callable functions for programmatic use")
    print("• Tools system integration for autonomous monitoring")
    print("• Tick state integration for detailed monitoring")
    print("• Advanced logging using the tools directory")
    print()
    
    if args.command_line_only:
        demo_command_line_usage()
        return 0
    
    success_count = 0
    total_demos = 0
    
    # Run demos
    if not args.skip_callable:
        total_demos += 1
        if demo_callable_monitor():
            success_count += 1
    
    if not args.skip_tools:
        total_demos += 1
        if demo_tools_integration():
            success_count += 1
    
    if not args.skip_tick:
        total_demos += 1
        if demo_tick_state_integration():
            success_count += 1
    
    # Always show command line usage
    demo_command_line_usage()
    
    # Summary
    print("\n✅ " + "="*60)
    print("✅ DEMONSTRATION SUMMARY")
    print("✅ " + "="*60)
    
    if total_demos > 0:
        print(f"📊 Demos completed: {success_count}/{total_demos}")
        
        if success_count == total_demos:
            print("🎉 All demonstrations completed successfully!")
            print()
            print("🔧 The enhanced live monitor is ready for use:")
            print("   • As a main runner: python3 live_monitor.py")
            print("   • As callable functions: from live_monitor import create_live_monitor")
            print("   • With tools integration for autonomous monitoring")
            print("   • With comprehensive logging and tick state monitoring")
        else:
            print("⚠️  Some demonstrations had issues")
            print("   This may be due to missing dependencies or configuration")
    else:
        print("ℹ️  No functional demos run - showing command line usage only")
    
    print()
    print("🌟 The live monitor now serves as DAWN's main monitoring runner")
    print("   with full tools integration and consciousness-aware capabilities!")
    
    return 0 if success_count == total_demos else 1

if __name__ == "__main__":
    sys.exit(main())

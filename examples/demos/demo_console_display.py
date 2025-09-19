#!/usr/bin/env python3
"""
DAWN Console Display Demonstration
==================================

Demonstration of the new real-time console display functionality
that shows what's happening in the DAWN system in the CLI.

Usage:
    python3 demo_console_display.py [options]
"""

import sys
import time
import argparse
from pathlib import Path

# Add DAWN root to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

def demo_console_display():
    """Demonstrate the console display functionality."""
    print("🖥️  " + "="*60)
    print("🖥️  DAWN CONSOLE DISPLAY DEMONSTRATION")
    print("🖥️  " + "="*60)
    
    try:
        # Import the console functions
        from live_monitor import create_live_monitor, start_console_display
        
        print("✅ Successfully imported console display functions")
        
        # Create a monitor instance in simulation mode
        print("\n📱 Creating live monitor for console display...")
        monitor = create_live_monitor(
            simulation_mode=True,  # Use simulation for demo
            enable_tools=True,
            log_directory="console_demo_logs"
        )
        
        print(f"✅ Monitor created: {type(monitor).__name__}")
        print(f"   Simulation mode: {monitor.simulation_mode}")
        print(f"   Tools enabled: {monitor.enable_tools}")
        
        print("\n🚀 Starting console display...")
        print("   This will show real-time DAWN system activity")
        print("   Press Ctrl+C to stop the display")
        print()
        
        # Give user a moment to read
        time.sleep(2)
        
        # Start the console display
        results = start_console_display(
            monitor=monitor,
            interval=1.0,  # Update every second
            compact=False  # Full screen clearing
        )
        
        print(f"\n✅ Console display session completed!")
        print(f"   Success: {results['success']}")
        print(f"   Duration: {results.get('duration', 0):.1f}s")
        
        if results.get('stopped_by'):
            print(f"   Stopped by: {results['stopped_by']}")
        
        if results.get('errors'):
            print(f"   Errors: {len(results['errors'])}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the DAWN root directory")
        return False
    except Exception as e:
        print(f"❌ Error in console display demo: {e}")
        return False

def demo_compact_console():
    """Demonstrate compact console display mode."""
    print("\n📱 " + "="*60)
    print("📱 COMPACT CONSOLE DISPLAY DEMONSTRATION")
    print("📱 " + "="*60)
    
    try:
        from live_monitor import create_live_monitor, start_console_display
        
        print("🔧 Creating monitor for compact console display...")
        monitor = create_live_monitor(
            simulation_mode=True,
            enable_tools=True
        )
        
        print("📺 Starting compact console display (no screen clearing)...")
        print("   This mode shows updates without clearing the screen")
        print("   Press Ctrl+C to stop")
        print()
        
        time.sleep(1)
        
        # Start compact console display
        results = start_console_display(
            monitor=monitor,
            interval=2.0,  # Slower updates for compact mode
            compact=True   # No screen clearing
        )
        
        print(f"\n✅ Compact console session completed!")
        print(f"   Duration: {results.get('duration', 0):.1f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in compact console demo: {e}")
        return False

def demo_command_line_usage():
    """Show command line usage examples."""
    print("\n💻 " + "="*60)
    print("💻 CONSOLE DISPLAY COMMAND LINE USAGE")
    print("💻 " + "="*60)
    
    print("📖 Console display command line options:")
    print()
    print("Basic console display:")
    print("   python3 live_monitor.py --console")
    print()
    print("Console with custom interval:")
    print("   python3 live_monitor.py --console --interval 2.0")
    print()
    print("Compact console (no screen clearing):")
    print("   python3 live_monitor.py --console --compact")
    print()
    print("Console in simulation mode:")
    print("   python3 live_monitor.py --console --simulate")
    print()
    print("Console with tools and logging:")
    print("   python3 live_monitor.py --console --log-dir console_logs")
    print()
    print("Console without tools:")
    print("   python3 live_monitor.py --console --no-tools")
    print()
    print("Features of the console display:")
    print("   • Real-time consciousness metrics with color coding")
    print("   • Live tick count and phase information")
    print("   • System activity and module status")
    print("   • Tools system integration status")
    print("   • Recent warnings and errors")
    print("   • Trend analysis (every 10 updates)")
    print("   • Mini progress bars for visual feedback")
    print("   • Logging activity status")
    print()

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="DAWN Console Display Demonstration")
    parser.add_argument("--skip-full", action="store_true",
                       help="Skip full console display demo")
    parser.add_argument("--skip-compact", action="store_true",
                       help="Skip compact console display demo")
    parser.add_argument("--command-line-only", action="store_true",
                       help="Show only command line usage examples")
    
    args = parser.parse_args()
    
    print("🖥️  DAWN Console Display Demonstration")
    print("=" * 80)
    print()
    print("This demonstration shows the new console display functionality:")
    print("• Real-time view of DAWN's consciousness system activity")
    print("• Color-coded metrics with progress bars")
    print("• Live tick monitoring and system status")
    print("• Tools integration and logging activity")
    print("• Both full-screen and compact display modes")
    print()
    
    if args.command_line_only:
        demo_command_line_usage()
        return 0
    
    success_count = 0
    total_demos = 0
    
    # Run demos
    if not args.skip_full:
        total_demos += 1
        if demo_console_display():
            success_count += 1
    
    if not args.skip_compact:
        total_demos += 1
        if demo_compact_console():
            success_count += 1
    
    # Always show command line usage
    demo_command_line_usage()
    
    # Summary
    print("\n✅ " + "="*60)
    print("✅ CONSOLE DISPLAY DEMONSTRATION SUMMARY")
    print("✅ " + "="*60)
    
    if total_demos > 0:
        print(f"📊 Demos completed: {success_count}/{total_demos}")
        
        if success_count == total_demos:
            print("🎉 All console display demonstrations completed successfully!")
            print()
            print("🖥️  The console display is ready for use:")
            print("   • Command line: python3 live_monitor.py --console")
            print("   • Programmatic: start_console_display(monitor)")
            print("   • Real-time view of all DAWN system activity")
            print("   • Color-coded metrics and visual progress bars")
        else:
            print("⚠️  Some demonstrations had issues")
            print("   This may be due to missing dependencies")
    else:
        print("ℹ️  No functional demos run - showing usage examples only")
    
    print()
    print("🌟 The console display provides a real-time window into")
    print("   DAWN's consciousness system with live updates and")
    print("   comprehensive activity monitoring!")
    
    return 0 if success_count == total_demos else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Live Monitor Signal Wrapper
===========================

Wrapper script that adds global signal handling to the existing live_monitor.py
without modifying the original file extensively.

This demonstrates how to integrate the global signal handling system
with existing DAWN components.

Usage:
    python3 live_monitor_signal_wrapper.py [live_monitor args...]
"""

import sys
import os
import subprocess
import signal
from pathlib import Path

# Add DAWN root to Python path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

# Import global signal handling system
try:
    from dawn.core.signal_config import (
        setup_global_signals, register_shutdown_callback, 
        is_shutdown_requested, configure_signals
    )
    SIGNAL_CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️  Global signal config not available")
    SIGNAL_CONFIG_AVAILABLE = False

# Import live monitor functions
try:
    from live_monitor import (
        create_live_monitor, start_monitoring_session, 
        start_console_display, get_monitor_status
    )
    LIVE_MONITOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import live_monitor: {e}")
    LIVE_MONITOR_AVAILABLE = False

def cleanup_wrapper():
    """Cleanup function for the wrapper."""
    print("🔄 Live monitor wrapper shutting down...")
    # Any additional cleanup can go here

def main():
    """Main wrapper function."""
    print("🖥️  DAWN Live Monitor with Global Signal Handling")
    print("=" * 80)
    
    if not LIVE_MONITOR_AVAILABLE:
        print("❌ Live monitor not available")
        sys.exit(1)
    
    # Set up global signal handling
    if SIGNAL_CONFIG_AVAILABLE:
        print("🛡️  Setting up global signal handling...")
        setup_global_signals(
            timeout=15.0,
            verbose=True,
            emergency_timeout=5.0
        )
        register_shutdown_callback("live_monitor_wrapper", cleanup_wrapper)
        print("✅ Global signal handling configured")
    else:
        print("⚠️  Using basic signal handling")
    
    # Parse arguments for live monitor
    import argparse
    parser = argparse.ArgumentParser(description="Live Monitor with Signal Handling")
    parser.add_argument("--console", action="store_true", help="Start console display")
    parser.add_argument("--simulate", action="store_true", help="Use simulation mode")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval")
    parser.add_argument("--compact", action="store_true", help="Compact display")
    parser.add_argument("--no-tools", action="store_true", help="Disable tools integration")
    parser.add_argument("--log-dir", type=str, help="Log directory")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting live monitor...")
    print(f"   Console mode: {'✅' if args.console else '❌'}")
    print(f"   Simulation: {'✅' if args.simulate else '❌'}")
    print(f"   Tools integration: {'❌' if args.no_tools else '✅'}")
    print()
    
    try:
        # Create monitor instance
        monitor = create_live_monitor(
            simulation_mode=args.simulate,
            enable_tools=not args.no_tools,
            log_directory=args.log_dir
        )
        
        if args.console:
            # Start console display
            print("🖥️  Starting console display...")
            results = start_console_display(
                monitor, 
                interval=args.interval, 
                compact=args.compact
            )
            print(f"Console display results: {results}")
        else:
            # Start regular monitoring session
            print("📊 Starting monitoring session...")
            results = start_monitoring_session(
                monitor,
                duration=None,  # Run indefinitely
                interval=args.interval
            )
            print(f"Monitoring session results: {results}")
            
    except Exception as e:
        print(f"❌ Error running live monitor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("👋 Live monitor session completed")

if __name__ == "__main__":
    main()

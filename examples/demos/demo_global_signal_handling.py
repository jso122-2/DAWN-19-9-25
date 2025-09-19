#!/usr/bin/env python3
"""
DAWN Global Signal Handling Demonstration
=========================================

Comprehensive demonstration of the new global signal handling system.
Shows how consistent Ctrl-C behavior works across all DAWN components.

This script demonstrates:
- Global signal configuration
- Component registration and cleanup
- Graceful shutdown patterns
- Thread-safe signal handling
- Emergency timeout handling
- Configuration options

Usage:
    python3 demo_global_signal_handling.py [--mode MODE] [--timeout SECONDS]
    
Modes:
    basic       - Basic signal handling demo
    components  - Multiple component demo
    console     - Console display demo
    stress      - Stress test with many components
"""

import sys
import os
import time
import threading
import argparse
from pathlib import Path
from datetime import datetime

# Add DAWN root to Python path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

# Import global signal handling system
try:
    from dawn.core.signal_config import (
        setup_global_signals, register_shutdown_callback, unregister_shutdown_callback,
        is_shutdown_requested, configure_signals, get_shutdown_status,
        signal_protection, GracefulShutdownMixin, wait_for_shutdown
    )
    SIGNAL_CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"❌ Global signal config not available: {e}")
    SIGNAL_CONFIG_AVAILABLE = False
    sys.exit(1)

class DemoComponent(GracefulShutdownMixin):
    """Example component that demonstrates graceful shutdown."""
    
    def __init__(self, name: str, work_duration: float = 0.5):
        super().__init__()
        self.name = name
        self.work_duration = work_duration
        self.running = False
        self.work_thread = None
        self.work_count = 0
        
        # Set up graceful shutdown
        self.setup_graceful_shutdown(f"component_{name}")
        print(f"📦 Component '{name}' initialized")
    
    def start(self):
        """Start the component's work thread."""
        if self.running:
            return
            
        self.running = True
        self.work_thread = threading.Thread(target=self._work_loop, name=f"{self.name}_worker")
        self.work_thread.start()
        print(f"▶️  Component '{self.name}' started")
    
    def _work_loop(self):
        """Main work loop for the component."""
        while self.running and not is_shutdown_requested():
            # Simulate work
            print(f"⚙️  {self.name}: doing work #{self.work_count}")
            self.work_count += 1
            
            # Sleep in small chunks to be responsive to shutdown
            sleep_time = self.work_duration
            while sleep_time > 0 and self.running and not is_shutdown_requested():
                chunk = min(0.1, sleep_time)
                time.sleep(chunk)
                sleep_time -= chunk
        
        print(f"🛑 Component '{self.name}' work loop ended")
    
    def _cleanup(self):
        """Cleanup method called during graceful shutdown."""
        print(f"🔄 Cleaning up component '{self.name}'...")
        self.running = False
        
        if self.work_thread and self.work_thread.is_alive():
            print(f"⏳ Waiting for '{self.name}' thread to finish...")
            self.work_thread.join(timeout=2.0)
            if self.work_thread.is_alive():
                print(f"⚠️  '{self.name}' thread didn't finish in time")
            else:
                print(f"✅ '{self.name}' thread finished cleanly")
        
        print(f"✅ Component '{self.name}' cleanup completed (processed {self.work_count} items)")

def demo_basic_signal_handling():
    """Demonstrate basic signal handling."""
    print("🎯 BASIC SIGNAL HANDLING DEMO")
    print("=" * 50)
    print("This demo shows basic signal setup and shutdown detection.")
    print("Press Ctrl+C to trigger graceful shutdown.")
    print()
    
    # Simple cleanup function
    def basic_cleanup():
        print("🧹 Basic cleanup function called")
    
    # Register cleanup
    register_shutdown_callback("basic_demo", basic_cleanup)
    
    # Main loop
    counter = 0
    try:
        while not is_shutdown_requested():
            print(f"⚡ Working... iteration {counter}")
            counter += 1
            time.sleep(1.0)
            
            # Show status every 5 iterations
            if counter % 5 == 0:
                status = get_shutdown_status()
                print(f"📊 Status: {status['registered_callbacks']} callbacks registered")
    
    except Exception as e:
        print(f"❌ Error in basic demo: {e}")
    
    print(f"🏁 Basic demo completed after {counter} iterations")

def demo_multiple_components():
    """Demonstrate multiple components with signal handling."""
    print("🎯 MULTIPLE COMPONENTS DEMO")
    print("=" * 50)
    print("This demo shows multiple components with independent cleanup.")
    print("Press Ctrl+C to trigger graceful shutdown of all components.")
    print()
    
    # Create multiple components
    components = []
    for i in range(3):
        name = f"Worker{i+1}"
        duration = 0.5 + (i * 0.3)  # Different work durations
        component = DemoComponent(name, duration)
        components.append(component)
    
    # Start all components
    for component in components:
        component.start()
    
    print("🚀 All components started. Monitoring status...")
    print("   Press Ctrl+C to initiate graceful shutdown")
    print()
    
    # Monitor components
    monitor_counter = 0
    try:
        while not is_shutdown_requested():
            # Show status
            active_count = sum(1 for c in components if c.running)
            total_work = sum(c.work_count for c in components)
            
            print(f"📊 Monitor #{monitor_counter}: {active_count}/3 components active, {total_work} total work units")
            
            # Show detailed status every 10 iterations
            if monitor_counter % 10 == 0:
                status = get_shutdown_status()
                print(f"🔍 Signal status: {len(status['registered_callbacks'])} callbacks registered")
                for component in components:
                    print(f"   • {component.name}: {component.work_count} work units, running={component.running}")
            
            monitor_counter += 1
            time.sleep(1.0)
    
    except Exception as e:
        print(f"❌ Error in components demo: {e}")
    
    print("🏁 Multiple components demo completed")

def demo_console_with_signals():
    """Demonstrate console display with signal handling."""
    print("🎯 CONSOLE WITH SIGNALS DEMO")
    print("=" * 50)
    print("This demo shows a console display with graceful shutdown.")
    print("Press Ctrl+C to trigger graceful shutdown.")
    print()
    
    # Console cleanup function
    def console_cleanup():
        print("\n🖥️  Console display shutting down...")
        print("✅ Console cleanup completed")
    
    register_shutdown_callback("console_demo", console_cleanup)
    
    # Console display loop
    display_counter = 0
    start_time = time.time()
    
    try:
        while not is_shutdown_requested():
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Header
            current_time = datetime.now().strftime("%H:%M:%S")
            elapsed = time.time() - start_time
            
            print(f"🖥️  DAWN Signal Handling Console [{current_time}]")
            print("=" * 60)
            print(f"Update #{display_counter} | Elapsed: {elapsed:.1f}s")
            print()
            
            # Signal status
            status = get_shutdown_status()
            print("🛡️  SIGNAL STATUS")
            print(f"   Handlers installed: {'✅' if status['handlers_installed'] else '❌'}")
            print(f"   Shutdown requested: {'🔴 YES' if status['shutdown_requested'] else '🟢 NO'}")
            print(f"   Registered callbacks: {len(status['registered_callbacks'])}")
            for callback in status['registered_callbacks']:
                print(f"      • {callback}")
            print()
            
            # Configuration
            config = status['config']
            print("⚙️  CONFIGURATION")
            print(f"   Timeout: {config['timeout']}s")
            print(f"   Emergency timeout: {config['emergency_timeout']}s")
            print(f"   Verbose shutdown: {'✅' if config['verbose_shutdown'] else '❌'}")
            print()
            
            # Instructions
            print("🎮 CONTROLS")
            print("   Ctrl+C: Trigger graceful shutdown")
            print("   The system will cleanup and exit gracefully")
            print()
            
            # Footer
            print("=" * 60)
            print("Global Signal Handling Demo | Running...")
            
            display_counter += 1
            time.sleep(2.0)
    
    except Exception as e:
        print(f"❌ Error in console demo: {e}")
    
    print("🏁 Console demo completed")

def demo_stress_test():
    """Stress test with many components."""
    print("🎯 STRESS TEST DEMO")
    print("=" * 50)
    print("This demo creates many components to test signal handling under load.")
    print("Press Ctrl+C to trigger graceful shutdown of all components.")
    print()
    
    # Create many components
    components = []
    for i in range(10):
        name = f"StressWorker{i+1:02d}"
        duration = 0.1 + (i % 3) * 0.1  # Varied work durations
        component = DemoComponent(name, duration)
        components.append(component)
    
    # Start all components
    print(f"🚀 Starting {len(components)} components...")
    for component in components:
        component.start()
    
    print("📊 Monitoring stress test...")
    print("   This will generate a lot of output - press Ctrl+C to stop")
    print()
    
    # Monitor with high frequency
    monitor_counter = 0
    try:
        while not is_shutdown_requested():
            if monitor_counter % 20 == 0:  # Status every 20 iterations
                active_count = sum(1 for c in components if c.running)
                total_work = sum(c.work_count for c in components)
                print(f"📊 Stress #{monitor_counter//20}: {active_count}/{len(components)} active, {total_work} total work")
            
            monitor_counter += 1
            time.sleep(0.1)  # High frequency monitoring
    
    except Exception as e:
        print(f"❌ Error in stress test: {e}")
    
    print("🏁 Stress test completed")

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="DAWN Global Signal Handling Demo")
    parser.add_argument("--mode", choices=["basic", "components", "console", "stress"], 
                       default="basic", help="Demo mode to run")
    parser.add_argument("--timeout", type=float, default=10.0, 
                       help="Graceful shutdown timeout")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose shutdown messages")
    
    args = parser.parse_args()
    
    print("🚀 DAWN Global Signal Handling Demonstration")
    print("=" * 80)
    print("This script demonstrates the new global signal handling system")
    print("that provides consistent Ctrl-C behavior across all DAWN components.")
    print()
    
    # Set up global signal handling
    print("🛡️  Setting up global signal handling...")
    setup_global_signals(
        timeout=args.timeout,
        verbose=args.verbose,
        emergency_timeout=3.0
    )
    print("✅ Global signal handling configured")
    print()
    
    # Run the selected demo
    demos = {
        "basic": demo_basic_signal_handling,
        "components": demo_multiple_components,
        "console": demo_console_with_signals,
        "stress": demo_stress_test,
    }
    
    demo_func = demos[args.mode]
    print(f"🎯 Running demo mode: {args.mode}")
    print()
    
    try:
        demo_func()
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("👋 Global signal handling demonstration completed")
    print("   All components should have shut down gracefully")

if __name__ == "__main__":
    main()

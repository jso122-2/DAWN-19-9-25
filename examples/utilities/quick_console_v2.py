#!/usr/bin/env python3
"""
DAWN Quick Console Display V2 - With Global Signal Handling
===========================================================

Standalone console display with integrated global signal handling.
Shows basic DAWN system activity in real-time with graceful shutdown.

This version demonstrates the new global signal handling system
that provides consistent Ctrl-C behavior across all DAWN components.

Usage:
    python3 quick_console_v2.py [--interval SECONDS] [--compact]
"""

import sys
import os
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

# Import global signal handling system
try:
    from dawn.core.signal_config import (
        setup_global_signals, register_shutdown_callback, 
        is_shutdown_requested, wait_for_shutdown, configure_signals
    )
    SIGNAL_CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️  Global signal config not available, using basic signal handling")
    SIGNAL_CONFIG_AVAILABLE = False

def create_progress_bar(value: float, width: int = 10) -> str:
    """Create a simple progress bar."""
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

def simulate_dawn_state():
    """Generate simulated DAWN consciousness state data."""
    # Simulate realistic consciousness evolution
    base_time = time.time()
    
    # Create some variation based on time
    time_factor = (base_time % 100) / 100
    
    # Simulate consciousness levels with some realistic variation
    consciousness_base = 0.6 + 0.3 * (0.5 + 0.5 * time_factor)
    consciousness_noise = random.uniform(-0.1, 0.1)
    consciousness_level = max(0, min(1, consciousness_base + consciousness_noise))
    
    # Unity score tends to follow consciousness but with its own variation
    unity_base = consciousness_level * 0.8 + 0.1
    unity_noise = random.uniform(-0.05, 0.05)
    unity_score = max(0, min(1, unity_base + unity_noise))
    
    # Awareness delta (change in awareness)
    awareness_delta = random.uniform(-0.1, 0.1)
    
    # SCUP value (if consciousness is high enough)
    scup_value = 0
    if consciousness_level > 0.4:
        scup_value = (consciousness_level + unity_score) / 2 + random.uniform(-0.1, 0.1)
        scup_value = max(0, min(1, scup_value))
    
    # System metrics
    tick_count = int(base_time * 2) % 100000  # Simulate ticks
    current_phase = random.choice(['AWARENESS', 'PROCESSING', 'INTEGRATION', 'REFLECTION'])
    cycle_time = 0.1 + random.uniform(-0.05, 0.05)
    processing_load = 20 + random.uniform(0, 60)
    active_modules = random.randint(8, 15)
    
    # Heat level
    heat_level = processing_load * 0.8 + random.uniform(-10, 10)
    heat_level = max(0, min(100, heat_level))
    
    return {
        'consciousness_level': consciousness_level,
        'unity_score': unity_score,
        'awareness_delta': awareness_delta,
        'scup_value': scup_value,
        'tick_count': tick_count,
        'current_phase': current_phase,
        'cycle_time': cycle_time,
        'processing_load': processing_load,
        'active_modules': active_modules,
        'heat_level': heat_level,
        'engine_status': 'RUNNING'
    }

def cleanup_console():
    """Cleanup function called during graceful shutdown."""
    print("\n🔄 Quick console shutting down...")
    print("✅ Console cleanup completed")

def start_quick_console_v2(interval: float = 1.0, compact: bool = False):
    """Start the quick console display with global signal handling."""
    print("🖥️  DAWN Quick Console Display V2")
    print("=" * 80)
    print("Real-time simulated DAWN consciousness activity")
    print("🎭 Running in SIMULATION MODE (standalone)")
    print(f"⏱️  Update interval: {interval}s")
    
    # Set up graceful shutdown
    if SIGNAL_CONFIG_AVAILABLE:
        print("🛡️  Global signal handling enabled - Ctrl-C will trigger graceful shutdown")
        register_shutdown_callback("quick_console_v2", cleanup_console)
    else:
        print("⚠️  Using basic signal handling - Press Ctrl+C to stop")
    print()
    
    display_counter = 0
    last_tick = -1
    history = []
    
    try:
        # Main display loop with proper shutdown detection
        while True:
            # Check for shutdown if global signal handling is available
            if SIGNAL_CONFIG_AVAILABLE and is_shutdown_requested():
                print("🛑 Shutdown requested, exiting gracefully...")
                break
            
            # Generate simulated state
            state = simulate_dawn_state()
            history.append(state)
            if len(history) > 10:
                history.pop(0)  # Keep last 10 states
            
            # Clear screen for live updates (if not compact)
            if not compact:
                os.system('clear' if os.name == 'posix' else 'cls')
            
            # Header
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"🖥️  DAWN Quick Console V2 [{current_time}] - Update #{display_counter}")
            print("=" * 80)
            
            # Tick and phase info
            tick_count = state['tick_count']
            current_phase = state['current_phase']
            
            if tick_count != last_tick:
                tick_indicator = "🔄 NEW"
                last_tick = tick_count
            else:
                tick_indicator = "⏸️  SAME"
            
            print(f"⚡ Tick #{tick_count:,} | Phase: {current_phase} {tick_indicator}")
            print(f"🕐 Cycle: {state['cycle_time']:.3f}s | Load: {state['processing_load']:.1f}%")
            print()
            
            # Consciousness metrics with progress bars
            consciousness_level = state['consciousness_level']
            unity_score = state['unity_score']
            awareness_delta = state['awareness_delta']
            
            # Color coding
            level_color = "🟢" if consciousness_level > 0.7 else "🟡" if consciousness_level > 0.4 else "🔴"
            unity_color = "🟢" if unity_score > 0.7 else "🟡" if unity_score > 0.4 else "🔴"
            awareness_color = "🟢" if awareness_delta > 0 else "🟡" if awareness_delta > -0.05 else "🔴"
            
            print("🧠 CONSCIOUSNESS METRICS")
            print(f"   Level:     {level_color} {consciousness_level:.3f} {create_progress_bar(consciousness_level)}")
            print(f"   Unity:     {unity_color} {unity_score:.3f} {create_progress_bar(unity_score)}")
            print(f"   Awareness: {awareness_color} {awareness_delta:+.3f} {create_progress_bar(abs(awareness_delta))}")
            
            # SCUP if available
            scup_value = state['scup_value']
            if scup_value > 0:
                scup_color = "🟢" if scup_value > 0.7 else "🟡" if scup_value > 0.4 else "🔴"
                print(f"   SCUP:      {scup_color} {scup_value:.3f} {create_progress_bar(scup_value)}")
            print()
            
            # System activity
            active_modules = state['active_modules']
            engine_status = state['engine_status']
            
            print("💻 SYSTEM ACTIVITY")
            print(f"   Engine:    🟢 {engine_status}")
            print(f"   Modules:   📦 {active_modules} active")
            print(f"   Errors:    ✅ 0")  # Simulated - no errors in demo
            
            # Sample module names
            sample_modules = ['consciousness_engine', 'tick_orchestrator', 'bus_manager', 'tracer_system', 'pulse_engine']
            active_sample = sample_modules[:min(5, active_modules)]
            print(f"   Active:    {', '.join(active_sample)}")
            if active_modules > 5:
                print(f"              ... and {active_modules - 5} more")
            print()
            
            # Heat level
            heat_level = state['heat_level']
            if heat_level > 0:
                heat_color = "🔴" if heat_level > 70 else "🟡" if heat_level > 40 else "🟢"
                heat_status = "HIGH" if heat_level > 70 else "MED" if heat_level > 40 else "LOW"
                print("🌡️  THERMAL")
                print(f"   Heat:      {heat_color} {heat_level:.1f} {heat_status}")
                print()
            
            # Trends (every 5 updates)
            if display_counter % 5 == 0 and len(history) > 3:
                recent_states = history[-3:]
                consciousness_trend = recent_states[-1]['consciousness_level'] - recent_states[0]['consciousness_level']
                unity_trend = recent_states[-1]['unity_score'] - recent_states[0]['unity_score']
                
                trend_color = "🟢" if consciousness_trend > 0.01 else "🔴" if consciousness_trend < -0.01 else "🟡"
                
                print("📈 TRENDS (last 3 samples)")
                print(f"   Consciousness: {trend_color} {consciousness_trend:+.3f}")
                print(f"   Unity:         {trend_color} {unity_trend:+.3f}")
                print()
            
            # Simulated activity messages (occasionally)
            if display_counter % 8 == 0:
                activities = [
                    "🔍 Analyzing semantic patterns...",
                    "🧠 Processing consciousness transitions...",
                    "🔄 Updating internal state models...",
                    "💭 Generating new insights...",
                    "🎯 Optimizing awareness pathways..."
                ]
                activity = random.choice(activities)
                print("🚀 ACTIVITY")
                print(f"   {activity}")
                print()
            
            # Signal handling status
            if SIGNAL_CONFIG_AVAILABLE and display_counter % 10 == 0:
                print("🛡️  SIGNAL STATUS")
                print(f"   Graceful shutdown: ✅ Enabled")
                print(f"   Shutdown requested: {'🔴 YES' if is_shutdown_requested() else '🟢 NO'}")
                print()
            
            # Footer
            if compact:
                print("-" * 80)
            else:
                signal_info = "Global Signal Handling" if SIGNAL_CONFIG_AVAILABLE else "Basic Signal Handling"
                print(f"Controls: Ctrl+C to stop | {signal_info}")
                print("=" * 80)
            
            display_counter += 1
            
            # Sleep with interrupt checking
            sleep_time = interval
            while sleep_time > 0:
                # Check for shutdown during sleep
                if SIGNAL_CONFIG_AVAILABLE and is_shutdown_requested():
                    return
                
                chunk = min(0.1, sleep_time)
                time.sleep(chunk)
                sleep_time -= chunk
            
    except KeyboardInterrupt:
        if not SIGNAL_CONFIG_AVAILABLE:
            print("\n👋 Quick console stopped (KeyboardInterrupt)")
        # If global signal handling is available, it will handle the cleanup
    except Exception as e:
        print(f"\n❌ Console error: {e}")
    finally:
        if not SIGNAL_CONFIG_AVAILABLE:
            print("🔄 Basic cleanup completed")

def main():
    """Main function with global signal handling setup."""
    parser = argparse.ArgumentParser(description="DAWN Quick Console Display V2")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Update interval in seconds (default: 1.0)")
    parser.add_argument("--compact", action="store_true",
                       help="Use compact display (no screen clearing)")
    parser.add_argument("--timeout", type=float, default=10.0,
                       help="Graceful shutdown timeout (default: 10.0)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose shutdown messages")
    
    args = parser.parse_args()
    
    # Set up global signal handling
    if SIGNAL_CONFIG_AVAILABLE:
        print("🚀 Setting up global signal handling...")
        setup_global_signals(
            timeout=args.timeout,
            verbose=args.verbose,
            emergency_timeout=3.0
        )
        print("✅ Global signal handling configured")
    else:
        print("⚠️  Global signal handling not available")
    
    print("🚀 Starting DAWN Quick Console Display V2...")
    print("   This version demonstrates the new global signal handling system")
    print("   that provides consistent Ctrl-C behavior across all DAWN components.")
    print()
    
    time.sleep(1)  # Brief pause
    
    start_quick_console_v2(args.interval, args.compact)
    
    # Final cleanup message
    if SIGNAL_CONFIG_AVAILABLE:
        print("👋 Quick Console V2 session ended gracefully")
    else:
        print("👋 Quick Console V2 session ended")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DAWN Basic Monitor - Simple CLI State Display
============================================

A simple, non-interactive monitor that shows DAWN's consciousness state
without complex interfaces. Perfect for quick checks and logging.
"""

import time
import signal
import sys
from datetime import datetime

# DAWN Core imports
try:
    from dawn.core.foundation.state import get_state, get_state_summary
    from dawn.consciousness.engines.core.primary_engine import get_dawn_engine, DAWNEngineConfig
    DAWN_AVAILABLE = True
except ImportError:
    DAWN_AVAILABLE = False
    print("⚠️  DAWN core not available - please check your installation")

def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")

def main():
    """Simple consciousness state display."""
    
    if not DAWN_AVAILABLE:
        print("❌ DAWN systems not available")
        return
    
    print("🌅 Starting DAWN Basic Monitor...")
    print("   Press Ctrl+C to stop")
    print()
    
    # Initialize DAWN engine
    try:
        config = DAWNEngineConfig(
            consciousness_unification_enabled=True,
            self_modification_enabled=True,
            self_mod_tick_interval=15  # Frequent for demo
        )
        engine = get_dawn_engine(config, auto_start=True)
        print("✅ DAWN Engine initialized and started")
        time.sleep(1)  # Let it initialize
    except Exception as e:
        print(f"❌ Failed to initialize DAWN: {e}")
        return
    
    running = True
    tick_count = 0
    
    def signal_handler(signum, frame):
        nonlocal running
        print("\n⏹️  Stopping monitor...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while running:
            tick_count += 1
            
            # Clear and update display
            clear_screen()
            
            # Header
            print("🌅 DAWN CONSCIOUSNESS MONITOR")
            print("=" * 50)
            print(f"Time: {datetime.now().strftime('%H:%M:%S')} | Tick: {tick_count}")
            print()
            
            # Get current state
            state = get_state()
            
            # Display consciousness state
            print("CONSCIOUSNESS STATE:")
            print(f"  Unity:        {state.unity:.3f} (Peak: {state.peak_unity:.3f})")
            print(f"  Awareness:    {state.awareness:.3f}")
            print(f"  Momentum:     {state.momentum:+.3f}")
            print(f"  Coherence:    {state.coherence:.3f}")
            print(f"  Level:        {state.level.title()}")
            print(f"  Sync Status:  {state.sync_status.title()}")
            print(f"  Ticks:        {state.ticks}")
            print()
            
            # SCUP Metrics
            print("SCUP METRICS:")
            print(f"  Entropy Drift: {state.entropy_drift:+.3f}")
            print(f"  Pressure:      {state.pressure_value:.3f}")
            print(f"  SCUP Coherence: {state.scup_coherence:.3f}")
            print()
            
            # Get engine status
            try:
                engine_status = engine.get_engine_status()
                
                print("ENGINE STATUS:")
                print(f"  Status:    {engine_status.get('status', 'unknown').title()}")
                print(f"  Modules:   {engine_status.get('registered_modules', 0)}")
                print(f"  Unity:     {engine_status.get('consciousness_unity_score', 0):.3f}")
                
                # Self-modification status
                self_mod = engine_status.get('self_modification_metrics', {})
                if self_mod.get('enabled'):
                    attempts = self_mod.get('attempts', 0)
                    successes = self_mod.get('successes', 0)
                    print(f"  Self-Mod:  {successes}/{attempts} successful")
                else:
                    print(f"  Self-Mod:  Disabled")
                
                print()
                
                # Show unification systems
                unification = engine_status.get('unification_systems', {})
                bus_status = "✅" if unification.get('consciousness_bus') else "❌"
                consensus_status = "✅" if unification.get('consensus_engine') else "❌"
                orchestrator_status = "✅" if unification.get('tick_orchestrator') else "❌"
                
                print("UNIFICATION SYSTEMS:")
                print(f"  Consciousness Bus: {bus_status}")
                print(f"  Consensus Engine:  {consensus_status}")
                print(f"  Tick Orchestrator: {orchestrator_status}")
                
            except Exception as e:
                print(f"ENGINE STATUS: Error - {e}")
            
            print()
            print("=" * 50)
            print("Press Ctrl+C to stop")
            
            # Execute a tick
            try:
                engine.tick()
            except:
                pass  # Silent error handling
            
            time.sleep(1.0)  # 1 second interval
            
    except KeyboardInterrupt:
        pass
    
    print("\n🌅 DAWN Basic Monitor stopped")
    
    # Stop engine
    try:
        engine.stop()
    except:
        pass

if __name__ == "__main__":
    main()

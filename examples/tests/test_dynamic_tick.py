#!/usr/bin/env python3
"""
Test script to demonstrate dynamic tick monitoring by creating a live DAWN system
"""

import sys
import os
import time
import threading
import signal
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

from dawn.consciousness.engines.core.primary_engine import get_dawn_engine, DAWNEngineConfig
from dawn.core.communication.bus import get_consciousness_bus
from dawn.core.foundation.state import set_state, get_state

class DAWNSimulator:
    """Simulates a running DAWN system with dynamic updates"""
    
    def __init__(self):
        self.running = False
        self.tick_count = 0
        self.engine = None
        self.bus = None
        self.thread = None
        
    def start(self):
        """Start the DAWN simulation"""
        print("🌅 Starting DAWN simulation...")
        
        # Initialize DAWN components
        engine_config = DAWNEngineConfig(
            consciousness_unification_enabled=True,
            self_modification_enabled=False,
            auto_synchronization=True,
            adaptive_timing=True,
            target_unity_threshold=0.85
        )
        
        self.engine = get_dawn_engine(config=engine_config, auto_start=True)
        self.bus = get_consciousness_bus(auto_start=True)
        
        # Start simulation thread
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.thread.start()
        
        print("✅ DAWN simulation started")
        
    def stop(self):
        """Stop the DAWN simulation"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("🛑 DAWN simulation stopped")
        
    def _simulation_loop(self):
        """Main simulation loop with dynamic updates"""
        start_time = time.time()
        
        while self.running:
            try:
                # Update tick count
                self.tick_count += 1
                
                # Calculate dynamic values
                elapsed = time.time() - start_time
                import math
                unity = 0.3 + 0.6 * (1 + abs(math.sin(elapsed * 0.5))) / 2
                awareness = 0.2 + 0.7 * (1 + abs(math.cos(elapsed * 0.3))) / 2
                momentum = abs(math.sin(elapsed * 0.2)) * 0.8
                
                # Update global state with dynamic values
                set_state(
                    tick_count=self.tick_count,
                    ticks=self.tick_count,
                    current_tick=self.tick_count,
                    unity=unity,
                    awareness=awareness,
                    momentum=momentum,
                    processing_load=abs(math.sin(elapsed * 0.1)) * 50,
                    current_tick_phase='PROCESSING' if self.tick_count % 3 == 0 else 'INTEGRATION' if self.tick_count % 3 == 1 else 'PERCEPTION',
                    phase_start_time=time.time() - (elapsed % 2.0),
                    last_cycle_time=0.5 + 0.3 * abs(math.sin(elapsed * 0.4)),
                    system_pressure=0.1 + 0.4 * abs(math.cos(elapsed * 0.15))
                )
                
                # Update engine tick count if available
                if hasattr(self.engine, 'tick_count'):
                    self.engine.tick_count = self.tick_count
                
                # Update bus metrics if available
                if hasattr(self.bus, 'metrics'):
                    self.bus.metrics['bus_coherence_score'] = unity
                    self.bus.metrics['module_integration_quality'] = awareness
                    self.bus.metrics['synchronization_cycles'] = self.tick_count
                    self.bus.metrics['average_response_time'] = 0.1 + 0.2 * abs(math.sin(elapsed * 0.3))
                
                print(f"⚡ Tick #{self.tick_count} | Unity: {unity:.3f} | Awareness: {awareness:.3f} | Phase: {get_state().current_tick_phase}")
                
                time.sleep(1.0)  # 1 second tick rate
                
            except Exception as e:
                print(f"❌ Simulation error: {e}")
                time.sleep(1.0)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n👋 Received interrupt signal, stopping simulation...")
    if 'simulator' in globals():
        simulator.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start simulator
    simulator = DAWNSimulator()
    simulator.start()
    
    print("\n🚀 DAWN simulation running!")
    print("💡 In another terminal, run: python tick_reader.py --mode live")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

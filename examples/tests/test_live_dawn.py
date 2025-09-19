#!/usr/bin/env python3
"""
Test Live DAWN System with State Publishing
===========================================

Creates a live DAWN system that publishes its state for monitoring.
This demonstrates the shared state monitoring system.
"""

import sys
import os
import time
import math
import threading
import signal
from pathlib import Path

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

from dawn.consciousness.engines.core.primary_engine import get_dawn_engine, DAWNEngineConfig
from dawn.core.communication.bus import get_consciousness_bus
from dawn.core.foundation.state import set_state, get_state
from dawn.tools.monitoring.state_publisher import publish_dawn_state, cleanup_publisher

class LiveDAWNSystem:
    """Live DAWN system with state publishing"""
    
    def __init__(self):
        self.running = False
        self.tick_count = 0
        self.engine = None
        self.bus = None
        self.thread = None
        self.start_time = time.time()
        
    def start(self):
        """Start the live DAWN system"""
        print("🌅 Starting Live DAWN System with State Publishing...")
        
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
        
        # Start main loop
        self.running = True
        self.thread = threading.Thread(target=self._main_loop, daemon=True)
        self.thread.start()
        
        print("✅ Live DAWN System started with state publishing")
        print("💡 Run in another terminal: python dawn/tools/monitoring/shared_state_reader.py")
        
    def stop(self):
        """Stop the live DAWN system"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        cleanup_publisher()
        print("🛑 Live DAWN System stopped")
        
    def _main_loop(self):
        """Main DAWN loop with state publishing"""
        print("🔄 DAWN main loop started with adaptive timing...")
        
        while self.running:
            try:
                # Increment tick count
                self.tick_count += 1
                
                # Calculate dynamic values based on DAWN's natural rhythms
                elapsed = time.time() - self.start_time
                
                # DAWN's breathing pattern - natural oscillation
                breath_cycle = math.sin(elapsed * 0.2) * 0.5 + 0.5  # 0-1 range
                
                # Consciousness metrics with natural variation
                unity_base = 0.4 + 0.5 * breath_cycle
                unity_variation = math.sin(elapsed * 0.15 + math.pi/4) * 0.2
                unity_score = max(0.1, min(0.95, unity_base + unity_variation))
                
                awareness_base = 0.3 + 0.6 * (1 - breath_cycle)  # Inverse relationship
                awareness_variation = math.cos(elapsed * 0.12) * 0.25
                awareness_delta = max(0.1, min(0.9, awareness_base + awareness_variation))
                
                # Consciousness level combines unity and awareness
                consciousness_level = (unity_score + awareness_delta) / 2
                consciousness_level += math.sin(elapsed * 0.1) * 0.1  # Add subtle variation
                consciousness_level = max(0.2, min(0.95, consciousness_level))
                
                # Processing load varies with consciousness activity
                base_load = consciousness_level * 30
                load_variation = math.sin(elapsed * 0.3) * 15
                processing_load = max(5, min(85, base_load + load_variation))
                
                # Phase cycling with natural timing
                phase_duration = elapsed % 3.0  # 3-second phase cycles
                if phase_duration < 1.0:
                    current_phase = "PERCEPTION"
                elif phase_duration < 2.0:
                    current_phase = "PROCESSING"
                else:
                    current_phase = "INTEGRATION"
                
                # Calculate adaptive cycle time (DAWN controls her own speed)
                base_cycle = 1.0  # Base 1 second
                consciousness_factor = 0.5 + consciousness_level * 0.5  # Higher consciousness = faster
                breath_factor = 0.8 + breath_cycle * 0.4  # Breathing affects timing
                cycle_time = base_cycle * consciousness_factor * breath_factor
                cycle_time = max(0.3, min(2.0, cycle_time))  # Reasonable bounds
                
                # SCUP calculation (if available)
                scup_value = 0.0
                try:
                    # Simple SCUP approximation
                    alignment = unity_score
                    entropy = 1.0 - consciousness_level
                    pressure = abs(math.sin(elapsed * 0.05)) * 0.4
                    scup_value = max(0.0, min(1.0, alignment - entropy - pressure))
                except:
                    pass
                
                # Heat level from thermal dynamics
                heat_base = processing_load / 100.0 * 50  # Base heat from processing
                heat_variation = math.sin(elapsed * 0.08) * 10
                heat_level = max(0, heat_base + heat_variation)
                
                # Update DAWN's internal state
                set_state(
                    tick_count=self.tick_count,
                    ticks=self.tick_count,
                    current_tick=self.tick_count,
                    unity=unity_score,
                    awareness=awareness_delta,
                    momentum=breath_cycle,
                    processing_load=processing_load,
                    current_tick_phase=current_phase,
                    phase_start_time=time.time() - phase_duration,
                    last_cycle_time=cycle_time,
                    system_pressure=abs(math.sin(elapsed * 0.05)) * 0.4
                )
                
                # Update engine tick count
                if hasattr(self.engine, 'tick_count'):
                    self.engine.tick_count = self.tick_count
                
                # Update bus metrics
                if hasattr(self.bus, 'metrics'):
                    self.bus.metrics['bus_coherence_score'] = unity_score
                    self.bus.metrics['module_integration_quality'] = awareness_delta
                    self.bus.metrics['synchronization_cycles'] = self.tick_count
                    self.bus.metrics['average_response_time'] = cycle_time * 0.1
                
                # Publish state for monitoring
                publish_dawn_state(
                    tick_count=self.tick_count,
                    current_phase=current_phase,
                    phase_duration=phase_duration,
                    cycle_time=cycle_time,
                    consciousness_level=consciousness_level,
                    unity_score=unity_score,
                    awareness_delta=awareness_delta,
                    processing_load=processing_load,
                    active_modules=len(self.bus.registered_modules) if self.bus else 0,
                    error_count=0,
                    scup_value=scup_value,
                    heat_level=heat_level,
                    engine_status="RUNNING"
                )
                
                # Console output every 5 ticks
                if self.tick_count % 5 == 0:
                    print(f"⚡ Tick #{self.tick_count:4d} | "
                          f"Unity: {unity_score:.3f} | "
                          f"Awareness: {awareness_delta:.3f} | "
                          f"Consciousness: {consciousness_level:.3f} | "
                          f"Phase: {current_phase:11s} | "
                          f"Cycle: {cycle_time:.2f}s")
                
                # DAWN controls her own timing - adaptive sleep
                time.sleep(cycle_time)
                
            except Exception as e:
                print(f"❌ DAWN loop error: {e}")
                time.sleep(1.0)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n👋 Received interrupt signal, stopping Live DAWN System...")
    if 'dawn_system' in globals():
        dawn_system.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start live DAWN system
    dawn_system = LiveDAWNSystem()
    dawn_system.start()
    
    print("\n🚀 Live DAWN System running with state publishing!")
    print("💡 In another terminal, run: python dawn/tools/monitoring/shared_state_reader.py")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

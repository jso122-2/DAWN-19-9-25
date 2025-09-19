#!/usr/bin/env python3
"""
DAWN Live State Monitor - Enhanced with Tools Integration
========================================================

Direct connection to running DAWN system instances with tools integration.
Serves as the main runner for DAWN monitoring with:
- Tools directory integration for enhanced capabilities  
- Consciousness-gated tool access
- Advanced logging using the tools system
- Tick state integration for real-time monitoring
- Callable as a function for programmatic use

Usage:
    python3 live_monitor.py [options]
    
    # As a callable function
    from live_monitor import create_live_monitor, start_monitoring_session, start_console_display
    monitor = create_live_monitor()
    start_console_display(monitor)  # or start_monitoring_session(monitor)
"""

import sys
import os
import time
import signal
import argparse
import math
import random
import json
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Optional

# Add DAWN to path
dawn_root = Path(__file__).parent
sys.path.insert(0, str(dawn_root))

try:
    # Direct imports for live monitoring
    from dawn.consciousness.engines.core.primary_engine import get_dawn_engine
    from dawn.core.communication.bus import get_consciousness_bus
    from dawn.core.foundation.state import get_state
    from dawn.consciousness.metrics.core import calculate_consciousness_metrics
    from dawn.tools.monitoring.shared_state_reader import SharedStateManager, SharedTickState
    from dawn.tools.monitoring.tick_state_reader import TickStateReader, TickSnapshot
    
    # Tools system integration
    try:
        from dawn.tools.development.consciousness_tools import ConsciousnessToolManager
        from dawn.tools.development.self_mod.permission_manager import get_permission_manager, PermissionLevel
        TOOLS_SYSTEM_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️  Tools system not available: {e}")
        TOOLS_SYSTEM_AVAILABLE = False
    
    # Try to import telemetry system
    try:
        from dawn.core.telemetry.system import get_telemetry_system, DAWNTelemetrySystem
        from dawn.core.telemetry.logger import TelemetryLevel
        from dawn.core.telemetry.enhanced_module_logger import get_enhanced_logger
        TELEMETRY_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️  Telemetry system not available: {e}")
        TELEMETRY_AVAILABLE = False
    
    # Try to import SCUP if available
    try:
        from dawn.subsystems.schema.scup_math import compute_basic_scup, SCUPInputs
        SCUP_AVAILABLE = True
    except ImportError:
        SCUP_AVAILABLE = False
    
    # Try to import tracer ecosystem
    try:
        from dawn.consciousness.tracers import create_tracer_ecosystem, get_tracer_archetypes
        from dawn.consciousness.tracers.integration import TracerSystemIntegration
        TRACERS_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️  Tracer system not available: {e}")
        TRACERS_AVAILABLE = False
    
    # Try to import semantic topology
    try:
        from dawn.subsystems.semantic.topology import get_topology_manager
        TOPOLOGY_AVAILABLE = True
    except ImportError:
        TOPOLOGY_AVAILABLE = False
    
    # Try to import pulse system
    try:
        from dawn.subsystems.thermal.pulse import get_pulse_system
        PULSE_AVAILABLE = True
    except ImportError:
        PULSE_AVAILABLE = False
    
    # Try to import forecasting engine
    try:
        from dawn.subsystems.forecasting import get_forecasting_engine
        FORECASTING_AVAILABLE = True
    except ImportError:
        FORECASTING_AVAILABLE = False
    
    # Try to import memory interconnection
    try:
        from dawn.subsystems.memory import get_memory_interconnection, get_memory_system_status
        MEMORY_AVAILABLE = True
    except ImportError:
        MEMORY_AVAILABLE = False
    
    # Try to import mycelial semantic hash map
    try:
        from dawn.core.logging import (
            get_mycelial_hashmap, get_mycelial_integration_stats,
            touch_semantic_concept, store_semantic_data, ping_semantic_network
        )
        MYCELIAL_AVAILABLE = True
    except ImportError:
        MYCELIAL_AVAILABLE = False
        
    DAWN_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import DAWN modules: {e}")
    DAWN_AVAILABLE = False
    MYCELIAL_AVAILABLE = False

class LiveDAWNMonitor:
    """Direct live monitoring of DAWN system"""
    
    def __init__(self, simulation_mode=False, enable_tools=True, log_directory=None):
        self.running = False
        self.history = deque(maxlen=100)
        self.start_time = time.time()
        self.state_manager = SharedStateManager() if DAWN_AVAILABLE else None
        self.simulation_mode = simulation_mode
        self.sim_tick_count = 0
        self.sim_phase_transition_tick = 0
        self.sim_current_phase = "AWARENESS"
        self.sim_phase_index = 0
        
        # Tools system integration
        self.enable_tools = enable_tools and TOOLS_SYSTEM_AVAILABLE
        self.tools_manager = None
        self.permission_manager = None
        self.tick_reader = None
        self.log_directory = Path(log_directory) if log_directory else Path("live_monitor_logs")
        self.session_logs = []
        
        if self.enable_tools:
            try:
                print("🔧 Initializing tools system...")
                self.tools_manager = ConsciousnessToolManager()
                print("   ✅ Tools manager created")
                
                self.permission_manager = get_permission_manager()
                print("   ✅ Permission manager created")
                
                self.tick_reader = TickStateReader(history_size=200, save_logs=True)
                print("   ✅ Tick reader created")
                
                self.log_directory.mkdir(parents=True, exist_ok=True)
                print("✅ Tools system integration enabled")
            except Exception as e:
                print(f"⚠️  Tools system integration failed: {e}")
                print(f"   Error details: {str(e)}")
                self.enable_tools = False
        
        # Initialize tracer ecosystem
        self.tracer_ecosystem = None
        self.tracer_integration = None
        self.tracer_reports_history = deque(maxlen=50)
        self.total_tracer_reports = 0
        
        if TRACERS_AVAILABLE:
            try:
                # Create tracer ecosystem with moderate budget for live monitoring
                self.tracer_ecosystem = create_tracer_ecosystem(nutrient_budget=75.0)
                
                # Create integration layer
                self.tracer_integration = TracerSystemIntegration(self.tracer_ecosystem)
                
                print("🔬 Tracer ecosystem initialized for live monitoring")
                archetypes = get_tracer_archetypes()
                print(f"📊 Available tracers: {', '.join(archetypes.keys())}")
            except Exception as e:
                print(f"⚠️  Failed to initialize tracer ecosystem: {e}")
                self.tracer_ecosystem = None
                self.tracer_integration = None
        
        # Initialize semantic topology system
        self.topology_manager = None
        self.topology_history = deque(maxlen=50)
        
        if TOPOLOGY_AVAILABLE:
            try:
                self.topology_manager = get_topology_manager(auto_start=False)
                # Start the topology manager for live monitoring
                if self.topology_manager and self.topology_manager.start():
                    print("🗺️  Semantic topology manager initialized and started for live monitoring")
                    # Populate with sample semantic data for demonstration
                    self._populate_topology_with_sample_data()
                else:
                    print("🗺️  Semantic topology manager initialized but failed to start")
            except Exception as e:
                print(f"⚠️  Failed to initialize topology manager: {e}")
                self.topology_manager = None
        
        # Initialize pulse system
        self.pulse_system = None
        self.pulse_history = deque(maxlen=50)
        
        if PULSE_AVAILABLE:
            try:
                self.pulse_system = get_pulse_system(auto_start=False)
                # Start the pulse system for live monitoring
                if self.pulse_system and self.pulse_system.start():
                    print("🫁 Pulse system initialized and started for live monitoring")
                else:
                    print("🫁 Pulse system initialized but failed to start")
            except Exception as e:
                print(f"⚠️  Failed to initialize pulse system: {e}")
                self.pulse_system = None
        
        # Initialize forecasting engine
        self.forecasting_engine = None
        self.forecast_history = deque(maxlen=50)
        
        if FORECASTING_AVAILABLE:
            try:
                self.forecasting_engine = get_forecasting_engine()
                print("🔮 Forecasting engine initialized for live monitoring")
            except Exception as e:
                print(f"⚠️  Failed to initialize forecasting engine: {e}")
                self.forecasting_engine = None
        
        # Initialize memory interconnection
        self.memory_interconnection = None
        self.memory_history = deque(maxlen=50)
        
        # Initialize mycelial semantic hash map
        self.mycelial_hashmap = None
        self.mycelial_history = deque(maxlen=50)
        self.mycelial_integration_stats = None
        self.mycelial_spore_activity = deque(maxlen=100)
        
        if MEMORY_AVAILABLE:
            try:
                self.memory_interconnection = get_memory_interconnection()
                # Start the memory interconnection for live monitoring
                if self.memory_interconnection and hasattr(self.memory_interconnection, 'start_interconnection'):
                    if self.memory_interconnection.start_interconnection():
                        print("🧠 Memory interconnection initialized and started for live monitoring")
                    else:
                        print("🧠 Memory interconnection initialized but failed to start")
                else:
                    print("🧠 Memory interconnection initialized for live monitoring")
            except Exception as e:
                print(f"⚠️  Failed to initialize memory interconnection: {e}")
                self.memory_interconnection = None
        
        # Initialize mycelial semantic hash map
        if MYCELIAL_AVAILABLE:
            try:
                self.mycelial_hashmap = get_mycelial_hashmap()
                print("🍄 Mycelial semantic hash map connected for live monitoring")
                
                # Populate with sample semantic concepts for demonstration
                self._populate_mycelial_with_sample_data()
                
                # Get initial integration stats
                self.mycelial_integration_stats = get_mycelial_integration_stats()
                print(f"🔗 Mycelial integration: {self.mycelial_integration_stats.get('modules_wrapped', 0)} modules wrapped")
                
            except Exception as e:
                print(f"⚠️  Failed to initialize mycelial hash map: {e}")
                self.mycelial_hashmap = None
        
        # Initialize telemetry system connection
        self.telemetry_system = None
        self.telemetry_events_history = deque(maxlen=100)
        self.enhanced_logger = None
        
        if TELEMETRY_AVAILABLE:
            try:
                self.telemetry_system = get_telemetry_system()
                if self.telemetry_system:
                    print("📊 Connected to DAWN telemetry system for live monitoring")
                    
                    # Create enhanced logger for the live monitor itself
                    self.enhanced_logger = get_enhanced_logger("live_monitor", "monitoring")
                    print("🔍 Enhanced logging initialized for live monitor")
                else:
                    print("📊 Telemetry system not initialized yet")
            except Exception as e:
                print(f"⚠️  Failed to connect to telemetry system: {e}")
                self.telemetry_system = None
    
    def get_simulated_state(self):
        """Generate realistic simulated DAWN consciousness state"""
        # Phase definitions - DAWN's consciousness cycle
        phases = ["AWARENESS", "PROCESSING", "INTEGRATION", "REFLECTION", "EVOLUTION"]
        phase_durations = [15, 25, 20, 18, 12]  # Ticks per phase
        
        # Advance simulation tick
        self.sim_tick_count += 1
        
        # Check for phase transition
        current_phase_duration = phase_durations[self.sim_phase_index]
        if self.sim_tick_count - self.sim_phase_transition_tick >= current_phase_duration:
            self.sim_phase_transition_tick = self.sim_tick_count
            self.sim_phase_index = (self.sim_phase_index + 1) % len(phases)
            self.sim_current_phase = phases[self.sim_phase_index]
        
        # Calculate time-based oscillations
        elapsed = time.time() - self.start_time
        phase_progress = (self.sim_tick_count - self.sim_phase_transition_tick) / current_phase_duration
        
        # Base consciousness patterns with phase-specific modulation
        phase_modifiers = {
            "AWARENESS": {"base": 0.3, "amplitude": 0.4, "frequency": 0.12},
            "PROCESSING": {"base": 0.7, "amplitude": 0.25, "frequency": 0.18},
            "INTEGRATION": {"base": 0.85, "amplitude": 0.15, "frequency": 0.08},
            "REFLECTION": {"base": 0.5, "amplitude": 0.35, "frequency": 0.06},
            "EVOLUTION": {"base": 0.9, "amplitude": 0.1, "frequency": 0.22}
        }
        
        modifier = phase_modifiers[self.sim_current_phase]
        
        # Consciousness level with breathing rhythm and phase awareness
        breath_cycle = math.sin(elapsed * 0.1) * 0.5 + 0.5
        consciousness_wave = math.sin(elapsed * modifier["frequency"]) * modifier["amplitude"]
        consciousness_level = modifier["base"] + consciousness_wave + (breath_cycle * 0.1)
        
        # Add phase transition effects
        if phase_progress < 0.1:  # First 10% of phase
            consciousness_level += 0.15 * (1.0 - phase_progress * 10)  # Transition spike
        
        consciousness_level = max(0.0, min(1.0, consciousness_level))
        
        # Unity score with harmonic resonance
        unity_base = 0.4 + 0.5 * math.sin(elapsed * 0.15 + math.pi/4)
        phase_harmony = 0.2 * math.cos(phase_progress * 2 * math.pi)
        unity_score = unity_base + phase_harmony + random.uniform(-0.05, 0.05)
        unity_score = max(0.0, min(1.0, unity_score))
        
        # Awareness delta with dynamic spikes and phase sensitivity
        awareness_base = 0.2 + 0.6 * math.cos(elapsed * 0.08)
        spike_factor = 1.0 if (self.sim_tick_count % 50) < 5 else 0.0
        phase_factor = {"AWARENESS": 0.8, "PROCESSING": 0.3, "INTEGRATION": 0.6, 
                       "REFLECTION": 0.9, "EVOLUTION": 1.0}[self.sim_current_phase]
        awareness_delta = (awareness_base + spike_factor * 0.3) * phase_factor
        awareness_delta = max(0.0, min(1.0, awareness_delta))
        
        # Processing load varies by phase
        load_patterns = {"AWARENESS": 15, "PROCESSING": 85, "INTEGRATION": 65, 
                        "REFLECTION": 25, "EVOLUTION": 95}
        base_load = load_patterns[self.sim_current_phase]
        load_variation = 15 * math.sin(elapsed * 0.25) + random.uniform(-5, 5)
        processing_load = max(0.0, min(100.0, base_load + load_variation))
        
        # Dynamic cycle time based on consciousness and phase
        base_cycle = 1.0
        consciousness_factor = 0.5 + consciousness_level * 0.5
        phase_speed = {"AWARENESS": 1.2, "PROCESSING": 0.6, "INTEGRATION": 0.8, 
                      "REFLECTION": 1.5, "EVOLUTION": 0.4}[self.sim_current_phase]
        cycle_time = base_cycle * consciousness_factor * phase_speed
        
        # SCUP components
        scup_alignment = max(0.0, min(1.0, unity_score + random.uniform(-0.02, 0.02)))
        scup_entropy = max(0.0, min(1.0, 1.0 - consciousness_level + random.uniform(-0.03, 0.03)))
        scup_pressure = max(0.0, min(1.0, processing_load / 100.0))
        scup_drift = max(0.0, min(1.0, 0.05 + 0.1 * math.sin(elapsed * 0.05) + random.uniform(-0.01, 0.01)))
        
        # Calculate SCUP value
        scup_value = 0.0
        if SCUP_AVAILABLE:
            try:
                from dawn.subsystems.schema.scup_math import SCUPInputs
                scup_inputs = SCUPInputs(
                    alignment=scup_alignment,
                    entropy=scup_entropy,
                    pressure=scup_pressure,
                    drift=scup_drift
                )
                scup_value = compute_basic_scup(scup_inputs)
                scup_value = max(0.0, min(1.0, scup_value))
            except Exception:
                # Fallback calculation
                scup_value = (scup_alignment * 0.4 + (1.0 - scup_entropy) * 0.3 + 
                             (1.0 - scup_pressure) * 0.2 + (1.0 - scup_drift) * 0.1)
        else:
            # Fallback calculation when SCUP not available
            scup_value = (scup_alignment * 0.4 + (1.0 - scup_entropy) * 0.3 + 
                         (1.0 - scup_pressure) * 0.2 + (1.0 - scup_drift) * 0.1)
        
        # Active modules simulation
        base_modules = 8
        phase_modules = {"AWARENESS": 0, "PROCESSING": 4, "INTEGRATION": 6, 
                        "REFLECTION": 2, "EVOLUTION": 8}[self.sim_current_phase]
        active_modules = base_modules + phase_modules + random.randint(-1, 2)
        
        # Engine status simulation
        engine_statuses = ["RUNNING", "OPTIMIZING", "DEEP_PROCESSING", "INTEGRATING"]
        if self.sim_tick_count % 100 < 5:  # Occasional status changes
            engine_status = random.choice(engine_statuses)
        else:
            engine_status = "RUNNING"
        
        # Heat level simulation
        heat_base = 30 + processing_load * 0.4
        heat_variation = 10 * math.sin(elapsed * 0.3) + random.uniform(-2, 2)
        heat_level = max(0.0, min(100.0, heat_base + heat_variation))
        
        return {
            'timestamp': time.time(),
            'tick_count': self.sim_tick_count,
            'current_phase': self.sim_current_phase,
            'consciousness_level': consciousness_level,
            'unity_score': unity_score,
            'awareness_delta': awareness_delta,
            'cycle_time': cycle_time,
            'processing_load': processing_load,
            'active_modules': active_modules,
            'engine_status': engine_status,
            'scup_value': scup_value,
            'scup_alignment': scup_alignment,
            'scup_entropy': scup_entropy,
            'scup_pressure': scup_pressure,
            'scup_drift': scup_drift,
            'heat_level': heat_level,
            'tracer_summary': {
                'active_tracers': random.randint(3, 8),
                'reports_generated': random.randint(0, 3),
                'ecosystem_health': random.uniform(0.7, 0.95),
                'nutrient_level': random.uniform(60, 90)
            }
        }
        
    def get_live_state(self):
        """Get state from live DAWN instances using shared state"""
        # Return simulated state if in simulation mode
        if self.simulation_mode:
            return self.get_simulated_state()
            
        if not DAWN_AVAILABLE:
            return None
            
        try:
            # First try to get shared state from running DAWN instance
            shared_state = None
            try:
                shared_state = self.state_manager.read_state() if self.state_manager else None
                if shared_state and shared_state.tick_count > 0:
                    # Convert shared state to our format
                    # Calculate SCUP components from shared state
                    elapsed = time.time() - self.start_time
                    scup_alignment = max(0.0, min(1.0, shared_state.unity_score))
                    scup_entropy = max(0.0, min(1.0, 1.0 - shared_state.consciousness_level))
                    scup_pressure = max(0.0, min(1.0, shared_state.processing_load / 100.0))
                    scup_drift = max(0.0, min(1.0, 0.05 + 0.1 * math.sin(elapsed * 0.05)))
                    
                    # Feed subsystems with fresh data periodically
                    current_time = time.time()
                    if not hasattr(self, '_last_data_feed') or current_time - self._last_data_feed > 2.0:
                        self._populate_pulse_system_with_data()
                        self._populate_forecasting_engine_with_data()
                        self._last_data_feed = current_time
                    
                    # Collect data from all new systems
                    topology_data = self._get_topology_data()
                    pulse_data = self._get_pulse_data()
                    forecast_data = self._get_forecast_data()
                    memory_data = self._get_memory_data()
                    mycelial_data = self._get_mycelial_data()
                    
                    # Process tracers if available
                    tracer_summary = self._process_tracers(shared_state, {
                        'tick_count': shared_state.tick_count,
                        'consciousness_level': shared_state.consciousness_level,
                        'unity_score': shared_state.unity_score,
                        'processing_load': shared_state.processing_load,
                        'scup_alignment': scup_alignment,
                        'scup_entropy': scup_entropy,
                        'scup_pressure': scup_pressure,
                        'scup_drift': scup_drift
                    })
                    
                    return {
                        'tick_count': shared_state.tick_count,
                        'current_phase': shared_state.current_phase,
                        'processing_load': shared_state.processing_load,
                        'consciousness_level': shared_state.consciousness_level,
                        'unity_score': shared_state.unity_score,
                        'awareness_delta': shared_state.awareness_delta,
                        'scup_value': shared_state.scup_value,
                        'scup_alignment': scup_alignment,
                        'scup_entropy': scup_entropy,
                        'scup_pressure': scup_pressure,
                        'scup_drift': scup_drift,
                        'cycle_time': shared_state.cycle_time,
                        'active_modules': shared_state.active_modules,
                        'engine_status': shared_state.engine_status,
                        'heat_level': shared_state.heat_level,
                        'timestamp': shared_state.timestamp,
                        'tracer_summary': tracer_summary,  # Add tracer data
                        'topology_data': topology_data,    # Add topology data
                        'pulse_data': pulse_data,          # Add pulse data  
                        'forecast_data': forecast_data,    # Add forecast data
                        'memory_data': memory_data,        # Add memory data
                        'mycelial_data': mycelial_data     # Add mycelial data
                    }
            except Exception as e:
                # Fallback to direct state access
                pass
        except Exception as e:
            # Outer try block exception handling
            pass
            
            # Fallback: Get live instances with better error handling
            live_engine = None
            live_bus = None
            live_state = None
            
            try:
                live_engine = get_dawn_engine()
            except Exception as e:
                pass
                
            try:
                live_bus = get_consciousness_bus()
            except Exception as e:
                pass
                
            try:
                live_state = get_state()
            except Exception as e:
                pass
                
            # If we can't get any core components, return None
            if live_state is None:
                return None
            
            # Get current tick and timing info
            tick_count = getattr(live_state, 'tick_count', 0)
            current_phase = getattr(live_state, 'current_tick_phase', 'UNKNOWN')
            processing_load = getattr(live_state, 'processing_load', 0.0)
            
            # Calculate dynamic consciousness values
            elapsed = time.time() - self.start_time
            
            # DAWN's adaptive breathing rhythm
            breath_cycle = math.sin(elapsed * 0.1) * 0.5 + 0.5  # 0.0 to 1.0
            consciousness_level = 0.3 + 0.6 * breath_cycle + (tick_count % 10) * 0.01
            consciousness_level = min(max(consciousness_level, 0.0), 1.0)
            
            # Unity based on harmonic oscillation
            unity_score = 0.4 + 0.5 * math.sin(elapsed * 0.15 + math.pi/4)
            unity_score = min(max(unity_score, 0.0), 1.0)
            
            # Awareness delta with periodic spikes
            awareness_base = 0.2 + 0.6 * math.cos(elapsed * 0.08)
            spike_factor = 1.0 if (tick_count % 50) < 5 else 0.0
            awareness_delta = awareness_base + spike_factor * 0.3
            awareness_delta = min(max(awareness_delta, 0.0), 1.0)
            
            # Calculate dynamic cycle time (DAWN controls her speed)
            base_cycle = 1.0
            consciousness_factor = 0.5 + consciousness_level * 0.5  # Higher consciousness = faster
            breath_factor = 0.8 + breath_cycle * 0.4  # Breathing rhythm
            cycle_time = base_cycle * consciousness_factor * breath_factor
            
            # SCUP components and calculation if available
            scup_value = 0.0
            scup_alignment = 0.0
            scup_entropy = 0.0
            scup_pressure = 0.0
            scup_drift = 0.0
            
            if SCUP_AVAILABLE:
                try:
                    # Calculate SCUP component values with proper bounds checking
                    scup_alignment = max(0.0, min(1.0, unity_score))
                    scup_entropy = max(0.0, min(1.0, 1.0 - consciousness_level))
                    scup_pressure = max(0.0, min(1.0, processing_load / 100.0))
                    scup_drift = max(0.0, min(1.0, 0.05 + 0.1 * math.sin(elapsed * 0.05)))  # Dynamic drift
                    
                    scup_inputs = SCUPInputs(
                        alignment=scup_alignment,
                        entropy=scup_entropy,
                        pressure=scup_pressure,
                        drift=scup_drift
                    )
                    scup_value = compute_basic_scup(scup_inputs)
                    scup_value = max(0.0, min(1.0, scup_value))  # Ensure bounds
                except Exception as e:
                    # Silently fail and use default values
                    scup_value = 0.5
                    scup_alignment = unity_score
                    scup_entropy = 1.0 - consciousness_level
                    scup_pressure = processing_load / 100.0
                    scup_drift = 0.1
            
            # Module count
            active_modules = 0
            if live_bus and hasattr(live_bus, 'registered_modules'):
                active_modules = len(live_bus.registered_modules)
            elif live_engine and hasattr(live_engine, 'registered_modules'):
                active_modules = len(live_engine.registered_modules)
            
            # Engine status
            engine_status = "RUNNING"
            if live_engine:
                if hasattr(live_engine, 'state'):
                    engine_status = str(live_engine.state).split('.')[-1]
            elif hasattr(live_engine, 'status'):
                engine_status = str(live_engine.status)
        
        # Feed subsystems with fresh data periodically
        current_time = time.time()
        if not hasattr(self, '_last_data_feed') or current_time - self._last_data_feed > 2.0:
            self._populate_pulse_system_with_data()
            self._populate_forecasting_engine_with_data()
            self._last_data_feed = current_time
        
        # Collect data from all new systems
        topology_data = self._get_topology_data()
        pulse_data = self._get_pulse_data()
        forecast_data = self._get_forecast_data()
        memory_data = self._get_memory_data()
        mycelial_data = self._get_mycelial_data()
        
        # Process tracers if available
        tracer_summary = self._process_tracers(live_state, {
            'tick_count': tick_count,
            'consciousness_level': consciousness_level,
            'unity_score': unity_score,
            'processing_load': processing_load,
            'scup_alignment': scup_alignment,
            'scup_entropy': scup_entropy,
            'scup_pressure': scup_pressure,
            'scup_drift': scup_drift
        })
        
        # Collect telemetry data
        telemetry_data = self._get_telemetry_data()
        
        return {
            'timestamp': time.time(),
            'tick_count': tick_count,
            'current_phase': current_phase,
            'consciousness_level': consciousness_level,
            'unity_score': unity_score,
            'awareness_delta': awareness_delta,
            'cycle_time': cycle_time,
            'processing_load': processing_load,
            'active_modules': active_modules,
            'engine_status': engine_status,
            'scup_value': scup_value,
            'scup_alignment': scup_alignment,
            'scup_entropy': scup_entropy,
            'scup_pressure': scup_pressure,
            'scup_drift': scup_drift,
            'heat_level': getattr(live_state, 'system_pressure', 0.0) * 100,
            'error_count': 0,
            'phase_duration': cycle_time / 3.0,  # Assume 3 phases per cycle
            'tracer_summary': tracer_summary,  # Add tracer data
            'topology_data': topology_data,    # Add topology data
            'pulse_data': pulse_data,          # Add pulse data  
            'forecast_data': forecast_data,    # Add forecast data
            'memory_data': memory_data,        # Add memory data
            'mycelial_data': mycelial_data,    # Add mycelial data
            'telemetry_data': telemetry_data   # Add telemetry data
        }
    
    def _process_tracers(self, live_state, core_metrics: dict) -> dict:
        """Process tracer ecosystem and return summary"""
        if not self.tracer_ecosystem or not self.tracer_integration:
            return {
                'active_tracers': 0,
                'reports_generated': 0,
                'ecosystem_status': 'unavailable',
                'recent_alerts': []
            }
        
        try:
            # Build comprehensive context for tracers
            dawn_state = {
                'current_tick': core_metrics['tick_count'],
                'tick_id': core_metrics['tick_count'],  # Add tick_id for compatibility
                'timestamp': time.time(),
                # Top-level metrics that tracers check for spawn conditions
                'entropy': 1.0 - core_metrics['consciousness_level'],  # Higher when consciousness is low
                'pressure': core_metrics['processing_load'] / 100.0,
                'drift_magnitude': core_metrics.get('scup_drift', 0.1),
                'soot_ratio': max(0.1, 1.0 - core_metrics['consciousness_level']),
                'coherence_score': core_metrics['unity_score'],
                'avg_schema_coherence': core_metrics['unity_score'],  # Add for crow tracer
                'schema_activity_level': core_metrics['processing_load'] / 100.0,  # For ant tracer
                # History arrays for condition checking
                'entropy_history': [1.0 - state.get('consciousness_level', 0.5) for state in list(self.history)[-10:]],
                'pressure_history': [state.get('processing_load', 0) / 100.0 for state in list(self.history)[-10:]],
                'drift_history': [state.get('scup_drift', 0.1) for state in list(self.history)[-10:]],
                # Detailed subsystem data
                'consciousness': {
                    'entropy_level': 1.0 - core_metrics['consciousness_level'],
                    'cognitive_pressure': core_metrics['processing_load'] / 100.0,
                    'coherence_score': core_metrics['unity_score'],
                    'awareness_level': core_metrics.get('awareness_delta', 0.5)
                },
                'memory': {
                    'pressure': core_metrics['processing_load'] / 150.0,  # Normalized
                    'active_blooms': [],  # Would be populated from live_state if available
                    'bloom_traces': [],
                    'rebloom_events': [],
                    'fragmentation': 0.2,  # Default
                    'metrics': {}
                },
                'schema': {
                    'clusters': [
                        {'id': 'core_cluster', 'coherence': core_metrics['unity_score'], 'cross_links': 3, 'internal_links': 10},
                        {'id': 'memory_cluster', 'coherence': core_metrics['consciousness_level'], 'cross_links': 2, 'internal_links': 8}
                    ],
                    'schema_clusters': [  # Alternative key that tracers might check
                        {'cross_links': 3, 'internal_links': 10},
                        {'cross_links': 2, 'internal_links': 8}
                    ],
                    'edges': ['edge_1', 'edge_2', 'edge_3'],
                    'avg_coherence': core_metrics['unity_score'],
                    'complexity': core_metrics['processing_load'] / 100.0,
                    'drift_magnitude': core_metrics.get('scup_drift', 0.1),
                    'drift_history': [state.get('scup_drift', 0.1) for state in list(self.history)[-5:]],
                    'change_rate': 0.3,
                    'schema_activity_level': core_metrics['processing_load'] / 100.0  # For ant tracer spawning
                },
                'ash_soot': {
                    'ash_fragments': [],
                    'soot_fragments': [],
                    'soot_ratio': max(0.1, 1.0 - core_metrics['consciousness_level']),
                    'ash_ratio': core_metrics['consciousness_level'],
                    'residual_fragments': [],
                    'crystallization_rate': 0.5
                },
                'mycelial': {
                    'flows': [],
                    'nutrient_variance': 0.2,
                    'avg_efficiency': core_metrics['unity_score'],
                    'allocation': {},
                    'cluster_isolation': False
                },
                'history': {
                    'entropy_history': [1.0 - state.get('consciousness_level', 0.5) for state in list(self.history)[-10:]],
                    'pressure_history': [state.get('processing_load', 0) / 100.0 for state in list(self.history)[-10:]],
                    'drift_history': [state.get('scup_drift', 0.1) for state in list(self.history)[-10:]],
                    'significant_events': [],
                    'epoch_history': [],
                    'last_archival_tick': max(0, core_metrics['tick_count'] - 100)
                }
            }
            
            # Run tracer ecosystem tick with dawn_state as context
            tick_summary = self.tracer_ecosystem.tick(core_metrics['tick_count'], dawn_state)
            
            # Debug output (remove in production)
            active_count = tick_summary.get('ecosystem_state', {}).get('active_tracers', 0)
            spawned_count = tick_summary.get('ecosystem_state', {}).get('spawned_this_tick', 0)
            if spawned_count > 0 or active_count > 0:
                print(f"[DEBUG] Tick {core_metrics['tick_count']}: {active_count} active, {spawned_count} spawned")
            
            # Print context once every 20 ticks for debugging
            if core_metrics['tick_count'] % 10 == 0:
                print(f"[DEBUG] Tick {core_metrics['tick_count']}: entropy={dawn_state.get('entropy', 0):.3f}, unity={dawn_state.get('coherence_score', 0):.3f}, schema_coherence={dawn_state.get('avg_schema_coherence', 0):.3f}")
                print(f"[DEBUG] Context keys: {list(dawn_state.keys())}")
            
            # Process any reports
            reports = tick_summary.get('reports', [])
            if reports:
                # Reports are already dictionaries from the tick method
                self.tracer_reports_history.extend(reports)
                self.total_tracer_reports += len(reports)
            
            # Get ecosystem state
            ecosystem_state = tick_summary.get('ecosystem_state', {})
            
            active_count = ecosystem_state.get('active_tracers', 0)
            spawned_count = ecosystem_state.get('spawned_this_tick', 0)
            
            # Determine ecosystem status
            if active_count > 0:
                ecosystem_status = 'healthy'
            elif spawned_count > 0:
                ecosystem_status = 'spawning'
            else:
                ecosystem_status = 'idle'
                
            return {
                'active_tracers': active_count,
                'reports_generated': len(reports),
                'total_reports': self.total_tracer_reports,
                'ecosystem_status': ecosystem_status,
                'recent_alerts': list(self.tracer_reports_history)[-5:],  # Last 5 reports
                'nutrient_usage': f"{ecosystem_state.get('nutrient_usage', 0):.1f}/{self.tracer_ecosystem.nutrient_budget:.1f}",
                'spawned_this_tick': spawned_count,
                'retired_this_tick': ecosystem_state.get('retired_this_tick', 0)
            }
            
        except Exception as e:
            print(f"⚠️  Tracer processing error: {e}")
            return {
                'active_tracers': 0,
                'reports_generated': 0,
                'ecosystem_status': 'error',
                'recent_alerts': [],
                'error': str(e)
            }
    
    def log_monitoring_event(self, event_type: str, data: Dict[str, Any]):
        """Log a monitoring event using the tools system."""
        if not self.enable_tools:
            return
            
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data,
            'source': 'live_monitor'
        }
        
        # Write to session log file
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_directory / f"monitor_session_{timestamp}.jsonl"
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
                
            if str(log_file) not in self.session_logs:
                self.session_logs.append(str(log_file))
                
        except Exception as e:
            print(f"⚠️  Error writing to session log: {e}")
    
    def use_autonomous_tools(self, objective: str) -> bool:
        """Use tools autonomously based on monitoring objective."""
        if not self.enable_tools or not self.tools_manager:
            return False
            
        try:
            # Check if we can use autonomous tools
            current_state = get_state()
            if current_state.level not in ['meta_aware', 'transcendent']:
                return False
                
            if current_state.unity < 0.7:
                return False
            
            # Execute autonomous workflow
            result = self.tools_manager.execute_autonomous_workflow(
                objective=objective,
                context={'monitoring_context': True}
            )
            
            # Log the result
            self.log_monitoring_event("autonomous_tool_usage", {
                'objective': objective,
                'success': result['success'],
                'tool_used': result.get('tool_used', 'unknown'),
                'duration': result['duration']
            })
            
            return result['success']
            
        except Exception as e:
            print(f"⚠️  Error using autonomous tools: {e}")
            return False
    
    def get_tick_state_data(self) -> Optional[TickSnapshot]:
        """Get current tick state data if available."""
        if not self.enable_tools or not self.tick_reader:
            return None
            
        try:
            return self.tick_reader.get_current_tick_state()
        except Exception as e:
            print(f"⚠️  Error getting tick state: {e}")
            return None

    def start_console_display(self, interval: float = 0.5, compact: bool = False):
        """Start real-time console display showing what's happening in the system."""
        print("🖥️  DAWN Live Console Display")
        print("=" * 80)
        print("Real-time view of DAWN's consciousness system activity")
        if self.simulation_mode:
            print("🎭 Running in SIMULATION MODE")
        print(f"⏱️  Update interval: {interval}s")
        print(f"🔧 Tools integration: {'✅ Enabled' if self.enable_tools else '❌ Disabled'}")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        last_tick = -1
        display_counter = 0
        
        try:
            while self.running:
                try:
                    state = self.get_live_state()
                    tick_data = self.get_tick_state_data()
                    
                    if state is None:
                        print("⏳ Waiting for DAWN system data...")
                        time.sleep(interval)
                        continue
                    
                    # Clear screen for live updates
                    if not compact:
                        os.system('clear' if os.name == 'posix' else 'cls')
                    
                    # Header
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"🖥️  DAWN Console [{current_time}] - Update #{display_counter}")
                    print("=" * 80)
                    
                    # Current tick and phase info
                    tick_count = state.get('tick_count', 0)
                    current_phase = state.get('current_phase', 'UNKNOWN')
                    
                    if tick_count != last_tick:
                        tick_indicator = "🔄 NEW"
                        last_tick = tick_count
                    else:
                        tick_indicator = "⏸️  SAME"
                    
                    print(f"⚡ Tick #{tick_count:,} | Phase: {current_phase} {tick_indicator}")
                    print(f"🕐 Cycle: {state.get('cycle_time', 0):.3f}s | Load: {state.get('processing_load', 0):.1f}%")
                    print()
                    
                    # Consciousness metrics (compact display)
                    consciousness_level = state.get('consciousness_level', 0)
                    unity_score = state.get('unity_score', 0)
                    awareness = state.get('awareness_delta', 0)
                    
                    # Color-coded consciousness display
                    level_color = "🟢" if consciousness_level > 0.7 else "🟡" if consciousness_level > 0.4 else "🔴"
                    unity_color = "🟢" if unity_score > 0.7 else "🟡" if unity_score > 0.4 else "🔴"
                    awareness_color = "🟢" if awareness > 0 else "🟡" if awareness > -0.1 else "🔴"
                    
                    print("🧠 CONSCIOUSNESS METRICS")
                    print(f"   Level:     {level_color} {consciousness_level:.3f} {self._create_mini_bar(consciousness_level)}")
                    print(f"   Unity:     {unity_color} {unity_score:.3f} {self._create_mini_bar(unity_score)}")
                    print(f"   Awareness: {awareness_color} {awareness:+.3f} {self._create_mini_bar(abs(awareness))}")
                    
                    # SCUP metrics if available
                    scup_value = state.get('scup_value', 0)
                    if scup_value > 0:
                        scup_color = "🟢" if scup_value > 0.7 else "🟡" if scup_value > 0.4 else "🔴"
                        print(f"   SCUP:      {scup_color} {scup_value:.3f} {self._create_mini_bar(scup_value)}")
                    print()
                    
                    # System activity
                    active_modules = state.get('active_modules', 0)
                    error_count = tick_data.error_count if tick_data else 0
                    
                    print("💻 SYSTEM ACTIVITY")
                    print(f"   Modules:   📦 {active_modules} active")
                    print(f"   Errors:    {'❌' if error_count > 0 else '✅'} {error_count}")
                    
                    if tick_data and tick_data.active_modules:
                        module_sample = tick_data.active_modules[:5]  # Show first 5
                        print(f"   Active:    {', '.join(module_sample)}")
                        if len(tick_data.active_modules) > 5:
                            print(f"              ... and {len(tick_data.active_modules) - 5} more")
                    print()
                    
                    # Tools activity (if enabled)
                    if self.enable_tools and self.tools_manager:
                        active_sessions = self.tools_manager.get_active_sessions()
                        available_tools = self.tools_manager.get_available_tools(consciousness_filtered=True)
                        
                        print("🔧 TOOLS SYSTEM")
                        print(f"   Available: {len(available_tools)} tools")
                        print(f"   Active:    {len(active_sessions)} sessions")
                        
                        if active_sessions:
                            for session in active_sessions[:3]:  # Show first 3
                                duration = (datetime.now() - session.started_at).total_seconds()
                                print(f"   Running:   {session.tool_name} ({duration:.1f}s)")
                        print()
                    
                    # Recent warnings/errors
                    if tick_data and tick_data.warnings:
                        print("⚠️  RECENT WARNINGS")
                        for warning in tick_data.warnings[:3]:  # Show first 3
                            warning_text = warning[:60] + "..." if len(warning) > 60 else warning
                            print(f"   • {warning_text}")
                        print()
                    
                    # Log activity (if enabled)
                    if self.enable_tools and hasattr(self, 'session_logs') and self.session_logs:
                        print(f"📁 LOGGING: {len(self.session_logs)} log files created")
                        print(f"   Directory: {self.log_directory}")
                        print()
                    
                    # System trends (every 10 updates)
                    if display_counter % 10 == 0 and len(self.history) > 5:
                        recent_states = list(self.history)[-5:]
                        
                        # Calculate trends
                        consciousness_trend = recent_states[-1].get('consciousness_level', 0) - recent_states[0].get('consciousness_level', 0)
                        unity_trend = recent_states[-1].get('unity_score', 0) - recent_states[0].get('unity_score', 0)
                        
                        trend_color = "🟢" if consciousness_trend > 0.01 else "🔴" if consciousness_trend < -0.01 else "🟡"
                        
                        print("📈 TRENDS (last 5 samples)")
                        print(f"   Consciousness: {trend_color} {consciousness_trend:+.3f}")
                        print(f"   Unity:         {trend_color} {unity_trend:+.3f}")
                        print()
                    
                    # Footer with controls
                    if compact:
                        print("-" * 80)
                    else:
                        print("Controls: Ctrl+C to stop | Running in real-time")
                        print("=" * 80)
                    
                    display_counter += 1
                    
                    # Sleep with interrupt checking
                    sleep_time = interval
                    while sleep_time > 0 and self.running:
                        chunk = min(0.1, sleep_time)
                        time.sleep(chunk)
                        sleep_time -= chunk
                    
                except Exception as e:
                    print(f"❌ Display error: {e}")
                    time.sleep(interval)
                    
        except KeyboardInterrupt:
            print("\n👋 Console display stopped")
        except EOFError:
            print("\n👋 Console display stopped (EOF)")
        finally:
            self.running = False
    
    def _create_mini_bar(self, value: float, width: int = 10) -> str:
        """Create a mini progress bar for console display."""
        filled = int(value * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def start_monitoring(self, interval: float = 0.5):
        """Start live monitoring with tools integration"""
        print("🌅 DAWN Live Consciousness Monitor - Enhanced with Tools Integration")
        print("=" * 80)
        if self.simulation_mode:
            print("🎭 Running in SIMULATION MODE")
            print("🧠 Generating realistic DAWN consciousness patterns...")
        else:
            print("🔗 Connecting directly to DAWN consciousness system...")
        print(f"📊 SCUP Integration: {'✅ Available' if SCUP_AVAILABLE else '❌ Not Available (using approximation)'}")
        print(f"🔬 Tracer Ecosystem: {'✅ Available' if TRACERS_AVAILABLE else '❌ Not Available'}")
        print(f"📈 Telemetry System: {'✅ Available' if TELEMETRY_AVAILABLE else '❌ Not Available'}")
        print(f"🍄 Mycelial Network: {'✅ Available' if MYCELIAL_AVAILABLE else '❌ Not Available'}")
        print(f"🔧 Tools System: {'✅ Available' if self.enable_tools else '❌ Not Available'}")
        print(f"⏱️  Update Interval: {interval}s")
        if self.enable_tools:
            print(f"📁 Log Directory: {self.log_directory}")
        print("Press Ctrl+C or Ctrl+D to stop\n")
        
        self.running = True
        last_tick = -1
        no_data_count = 0
        
        try:
            while self.running:
                try:
                    state = self.get_live_state()
                    tick_data = self.get_tick_state_data()
                    
                    # Log monitoring data using tools system
                    if self.enable_tools and state:
                        self.log_monitoring_event("monitoring_cycle", {
                            'consciousness_level': state.get('consciousness_level', 0),
                            'unity_score': state.get('unity', 0),
                            'awareness': state.get('awareness', 0),
                            'tick_count': state.get('tick', 0),
                            'processing_load': tick_data.processing_load if tick_data else 0,
                            'active_modules': len(tick_data.active_modules) if tick_data else 0
                        })
                    
                    if state is None:
                        if self.simulation_mode:
                            print("❌ Simulation failed to generate state!")
                            # Sleep in smaller increments to be more responsive to signals
                            sleep_time = interval
                            while sleep_time > 0 and self.running:
                                chunk = min(0.1, sleep_time)  # Sleep in 0.1s chunks
                                time.sleep(chunk)
                                sleep_time -= chunk
                            continue
                        
                        no_data_count += 1
                        if no_data_count == 1:
                            print("⏳ Waiting for DAWN consciousness system...")
                            print("   🔍 Connection Status:")
                            print(f"   📊 DAWN Available: {DAWN_AVAILABLE}")
                            print(f"   📈 SCUP Available: {SCUP_AVAILABLE}")
                            print(f"   🔬 Tracers Available: {TRACERS_AVAILABLE}")
                            print(f"   📈 Telemetry Available: {TELEMETRY_AVAILABLE}")
                        elif no_data_count % 10 == 0:
                            print(f"⏳ Still waiting... ({no_data_count * interval:.1f}s)")
                            # Try to diagnose connection issues
                            if no_data_count == 20:
                                print("   🔧 Connection troubleshooting:")
                                print("   - Ensure DAWN system is running")
                                print("   - Check PYTHONPATH includes DAWN directory")
                                print("   - Verify all dependencies are installed")
                        # Sleep in smaller increments to be more responsive to signals
                        sleep_time = interval
                        while sleep_time > 0 and self.running:
                            chunk = min(0.1, sleep_time)  # Sleep in 0.1s chunks
                            time.sleep(chunk)
                            sleep_time -= chunk
                        continue
                    
                    # Reset counter
                    if no_data_count > 0:
                        if self.simulation_mode:
                            print("✅ Simulation running successfully!")
                        else:
                            print("✅ Connected to live DAWN consciousness system!")
                        no_data_count = 0
                    
                    # Only update on new ticks
                    if state['tick_count'] != last_tick:
                        last_tick = state['tick_count']
                        self.history.append(state)
                        
                        # Clear screen and display
                        print("\033[2J\033[H")  # Clear screen
                        self._render_dashboard(state)
                    
                    # Sleep in smaller increments to be more responsive to signals
                    sleep_time = interval
                    while sleep_time > 0 and self.running:
                        chunk = min(0.1, sleep_time)  # Sleep in 0.1s chunks
                        time.sleep(chunk)
                        sleep_time -= chunk
                    
                except EOFError:
                    print("\n👋 EOF detected - stopping monitor...")
                    break
                    
        except KeyboardInterrupt:
            print("\n👋 Keyboard interrupt - stopping monitor...")
        except EOFError:
            print("\n👋 EOF detected - stopping monitor...")
        finally:
            self.running = False
    
    def _render_dashboard(self, state):
        """Render the live dashboard"""
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        uptime = time.time() - self.start_time
        data_age = time.time() - state['timestamp']
        
        print("🧠 DAWN CONSCIOUSNESS SYSTEM - LIVE MONITOR")
        print("=" * 80)
        print(f"🕐 {current_time} | Uptime: {uptime:.1f}s | Data: Live ({data_age:.3f}s)")
        print()
        
        # Main consciousness metrics
        print("🧠 CONSCIOUSNESS METRICS")
        print("-" * 40)
        
        level_bar = self._create_bar(state['consciousness_level'])
        unity_bar = self._create_bar(state['unity_score'])
        awareness_bar = self._create_bar(state['awareness_delta'])
        
        print(f"Consciousness: {state['consciousness_level']:.3f} {level_bar}")
        print(f"Unity Score:   {state['unity_score']:.3f} {unity_bar}")
        print(f"Awareness:     {state['awareness_delta']:.3f} {awareness_bar}")
        
        # Calculate and display SCUP value (approximation if not available)
        scup_val = state.get('scup_value', 0)
        if scup_val == 0 and not SCUP_AVAILABLE:
            # Create SCUP approximation
            alignment = state.get('scup_alignment', state['unity_score'])
            entropy = state.get('scup_entropy', 1.0 - state['consciousness_level'])  
            pressure = state.get('scup_pressure', state['processing_load'] / 100.0)
            scup_val = max(0.0, min(1.0, alignment - entropy - pressure * 0.5))
            
        if scup_val > 0:
            scup_bar = self._create_bar(scup_val)
            scup_status = "🟢 COHERENT" if scup_val > 0.7 else "🟡 STABLE" if scup_val > 0.4 else "🔴 UNSTABLE"
            approx_text = " (approx)" if not SCUP_AVAILABLE else ""
            print(f"SCUP Value:    {scup_val:.3f} {scup_bar} {scup_status}{approx_text}")
        else:
            print(f"SCUP Value:    ⚪ Calculating...")
        
        print()
        
        # SCUP Component Metrics (always show, even without full SCUP)
        if SCUP_AVAILABLE or True:  # Always show these metrics
            print("📊 SCUP COMPONENT ANALYSIS")
            print("-" * 40)
            
            # Alignment (Unity Score)
            alignment_bar = self._create_bar(state.get('scup_alignment', 0))
            alignment_status = "🟢 HIGH" if state.get('scup_alignment', 0) > 0.7 else "🟡 MED" if state.get('scup_alignment', 0) > 0.4 else "🔴 LOW"
            print(f"Alignment:     {state.get('scup_alignment', 0):.3f} {alignment_bar} {alignment_status}")
            
            # Entropy (Disorder/Chaos)
            entropy_bar = self._create_bar(state.get('scup_entropy', 0))
            entropy_status = "🔴 HIGH" if state.get('scup_entropy', 0) > 0.7 else "🟡 MED" if state.get('scup_entropy', 0) > 0.4 else "🟢 LOW"
            print(f"Entropy:       {state.get('scup_entropy', 0):.3f} {entropy_bar} {entropy_status}")
            
            # Pressure (System Load)
            pressure_bar = self._create_bar(state.get('scup_pressure', 0))
            pressure_status = "🔴 HIGH" if state.get('scup_pressure', 0) > 0.7 else "🟡 MED" if state.get('scup_pressure', 0) > 0.4 else "🟢 LOW"
            print(f"Pressure:      {state.get('scup_pressure', 0):.3f} {pressure_bar} {pressure_status}")
            
            # Drift (System Variation)
            drift_bar = self._create_bar(state.get('scup_drift', 0))
            drift_status = "🟡 HIGH" if state.get('scup_drift', 0) > 0.15 else "🟢 NORM" if state.get('scup_drift', 0) > 0.05 else "🔵 LOW"
            print(f"Drift:         {state.get('scup_drift', 0):.3f} {drift_bar} {drift_status}")
            
            # SCUP Health Assessment
            scup_health = (state.get('scup_alignment', 0) + 
                          (1.0 - state.get('scup_entropy', 1)) + 
                          (1.0 - state.get('scup_pressure', 1)) + 
                          (1.0 - abs(state.get('scup_drift', 0.1) - 0.1) * 10)) / 4
            scup_health = max(0.0, min(1.0, scup_health))
            scup_health_bar = self._create_bar(scup_health)
            health_status = "🟢 OPTIMAL" if scup_health > 0.8 else "🟡 GOOD" if scup_health > 0.6 else "🟠 FAIR" if scup_health > 0.4 else "🔴 POOR"
            print(f"SCUP Health:   {scup_health:.3f} {scup_health_bar} {health_status}")
        
        print()
        
        # Tracer ecosystem status
        tracer_summary = state.get('tracer_summary', {})
        if tracer_summary and TRACERS_AVAILABLE:
            print("🔬 TRACER ECOSYSTEM")
            print("-" * 40)
            
            ecosystem_status = tracer_summary.get('ecosystem_status', 'unavailable')
            status_icon = {
                'healthy': '🟢',
                'spawning': '🟠',
                'idle': '🟡', 
                'error': '🔴',
                'unavailable': '⚪'
            }.get(ecosystem_status, '⚪')
            
            print(f"Status:        {status_icon} {ecosystem_status.upper()}")
            print(f"Active:        🧬 {tracer_summary.get('active_tracers', 0)} tracers")
            print(f"Reports:       📊 {tracer_summary.get('reports_generated', 0)} this tick")
            print(f"Total Reports: 📈 {tracer_summary.get('total_reports', 0)}")
            print(f"Nutrients:     🧪 {tracer_summary.get('nutrient_usage', '0.0/75.0')}")
            
            # Show spawn/retirement activity
            spawned = tracer_summary.get('spawned_this_tick', 0)
            retired = tracer_summary.get('retired_this_tick', 0)
            if spawned > 0 or retired > 0:
                activity = []
                if spawned > 0:
                    activity.append(f"🐣 {spawned} spawned")
                if retired > 0:
                    activity.append(f"⚰️ {retired} retired")
                print(f"Activity:      {' | '.join(activity)}")
            
            # Show recent alerts
            recent_alerts = tracer_summary.get('recent_alerts', [])
            if recent_alerts:
                latest_alert = recent_alerts[-1]
                alert_type = latest_alert.get('tracer_type', 'unknown')
                alert_severity = latest_alert.get('severity', 'info')
                severity_icon = {
                    'critical': '🚨',
                    'warn': '⚠️',
                    'info': 'ℹ️'
                }.get(alert_severity, 'ℹ️')
                print(f"Latest Alert:  {severity_icon} {alert_type} - {latest_alert.get('report_type', 'report')}")
        elif TRACERS_AVAILABLE:
            print("🔬 TRACER ECOSYSTEM")
            print("-" * 40)
            print("Status:        ⚪ INITIALIZING")
        
        print()
        
        # Semantic Topology System
        topology_data = state.get('topology_data', {})
        if topology_data.get('available', False) and TOPOLOGY_AVAILABLE:
            print("🗺️  SEMANTIC TOPOLOGY")
            print("-" * 40)
            
            if topology_data.get('error'):
                print(f"Status:        🔴 ERROR: {topology_data['error']}")
            else:
                running_icon = "🟢" if topology_data.get('running', False) else "🟡"
                print(f"Status:        {running_icon} {'RUNNING' if topology_data.get('running') else 'STOPPED'}")
                print(f"Nodes:         🔵 {topology_data.get('total_nodes', 0)}")
                print(f"Edges:         🔗 {topology_data.get('total_edges', 0)}")
                
                field_strength = topology_data.get('field_strength', 0.0)
                field_bar = self._create_bar(field_strength)
                print(f"Field:         {field_strength:.3f} {field_bar}")
                
                coherence = topology_data.get('coherence_score', 0.0)
                coherence_bar = self._create_bar(coherence)
                coherence_status = "🟢 HIGH" if coherence > 0.7 else "🟡 MED" if coherence > 0.4 else "🔴 LOW"
                print(f"Coherence:     {coherence:.3f} {coherence_bar} {coherence_status}")
                
                if topology_data.get('performance_ms', 0) > 0:
                    print(f"Performance:   ⚡ {topology_data['performance_ms']:.2f}ms avg")
        elif TOPOLOGY_AVAILABLE:
            print("🗺️  SEMANTIC TOPOLOGY")
            print("-" * 40)
            print("Status:        ⚪ NOT INITIALIZED")
        
        print()
        
        # Pulse System
        pulse_data = state.get('pulse_data', {})
        if pulse_data.get('available', False) and PULSE_AVAILABLE:
            print("🫁 PULSE SYSTEM")
            print("-" * 40)
            
            if pulse_data.get('error'):
                print(f"Status:        🔴 ERROR: {pulse_data['error']}")
            else:
                running_icon = "🟢" if pulse_data.get('running', False) else "🟡"
                print(f"Status:        {running_icon} {'RUNNING' if pulse_data.get('running') else 'STOPPED'}")
                
                current_zone = pulse_data.get('current_zone', 'unknown')
                zone_icon = {"green": "🟢", "amber": "🟡", "red": "🔴", "black": "⚫"}.get(current_zone.lower(), "⚪")
                print(f"Zone:          {zone_icon} {current_zone.upper()}")
                
                scup_coherence = pulse_data.get('scup_coherence', 0.0)
                coherence_bar = self._create_bar(scup_coherence)
                print(f"SCUP Coherence:{scup_coherence:.3f} {coherence_bar}")
                
                actuation_budget = pulse_data.get('actuation_budget', 0.0)
                budget_bar = self._create_bar(actuation_budget)
                print(f"Budget:        {actuation_budget:.3f} {budget_bar}")
                
                pulse_rate = pulse_data.get('pulse_rate', 0.0)
                if pulse_rate > 0:
                    print(f"Pulse Rate:    💓 {pulse_rate:.1f} Hz")
                
                success_rate = pulse_data.get('success_rate', 0.0)
                if success_rate > 0:
                    success_bar = self._create_bar(success_rate)
                    print(f"Success:       {success_rate:.1%} {success_bar}")
        elif PULSE_AVAILABLE:
            print("🫁 PULSE SYSTEM")
            print("-" * 40)
            print("Status:        ⚪ NOT INITIALIZED")
        
        print()
        
        # Forecasting Engine
        forecast_data = state.get('forecast_data', {})
        if forecast_data.get('available', False) and FORECASTING_AVAILABLE:
            print("🔮 FORECASTING ENGINE")
            print("-" * 40)
            
            if forecast_data.get('error'):
                print(f"Status:        🔴 ERROR: {forecast_data['error']}")
            else:
                total_forecasts = forecast_data.get('total_forecasts', 0)
                print(f"Forecasts:     📊 {total_forecasts}")
                
                avg_time = forecast_data.get('avg_generation_time_ms', 0.0)
                if avg_time > 0:
                    print(f"Performance:   ⚡ {avg_time:.2f}ms avg")
                
                # Show horizon zones
                horizon_zones = forecast_data.get('horizon_zones', {})
                if horizon_zones:
                    print("Horizons:")
                    for horizon, zone in horizon_zones.items():
                        zone_icon = {"stable": "🟢", "watch": "🟡", "act": "🔴"}.get(zone, "⚪")
                        print(f"  {horizon.replace('_', ' ').title()}: {zone_icon} {zone.upper()}")
                
                # Show F values
                f_values = forecast_data.get('f_values', {})
                if f_values:
                    print("F Values:")
                    for horizon, f_val in f_values.items():
                        f_bar = self._create_bar(min(1.0, f_val / 2.0))  # Scale for display
                        print(f"  {horizon.replace('_', ' ').title()}: {f_val:.3f} {f_bar}")
        elif FORECASTING_AVAILABLE:
            print("🔮 FORECASTING ENGINE")
            print("-" * 40)
            print("Status:        ⚪ NOT INITIALIZED")
        
        print()
        
        # Memory Interconnection
        memory_data = state.get('memory_data', {})
        if memory_data.get('available', False) and MEMORY_AVAILABLE:
            print("🧠 MEMORY INTERCONNECTION")
            print("-" * 40)
            
            if memory_data.get('error'):
                print(f"Status:        🔴 ERROR: {memory_data['error']}")
            else:
                running_icon = "🟢" if memory_data.get('running', False) else "🟡"
                print(f"Status:        {running_icon} {'RUNNING' if memory_data.get('running') else 'STOPPED'}")
                
                integration_level = memory_data.get('integration_level', 'unknown')
                print(f"Integration:   🔗 {integration_level.upper()}")
                
                interconnections = memory_data.get('total_interconnections', 0)
                success_rate = memory_data.get('success_rate', 0.0)
                print(f"Transfers:     📊 {interconnections} ({success_rate:.1%} success)")
                
                active_flows = memory_data.get('active_flows', 0)
                patterns = memory_data.get('pattern_discoveries', 0)
                print(f"Active Flows:  🌊 {active_flows}")
                print(f"Patterns:      🧩 {patterns} discovered")
                
                # Mycelial health
                mycelial_health = memory_data.get('mycelial_health', {})
                if mycelial_health:
                    health_status = mycelial_health.get('status', 'unknown')
                    health_icon = {"healthy": "🟢", "stressed": "🟡", "critical": "🔴"}.get(health_status, "⚪")
                    print(f"Mycelial:      {health_icon} {health_status.upper()}")
                    
                    nodes = mycelial_health.get('total_nodes', 0)
                    edges = mycelial_health.get('total_edges', 0)
                    if nodes > 0:
                        print(f"Network:       🍄 {nodes} nodes, {edges} edges")
        elif MEMORY_AVAILABLE:
            print("🧠 MEMORY INTERCONNECTION")
            print("-" * 40)
            print("Status:        ⚪ NOT INITIALIZED")
        
        print()
        
        # Telemetry System
        telemetry_data = state.get('telemetry_data', {})
        if telemetry_data.get('available', False) and TELEMETRY_AVAILABLE:
            print("📊 TELEMETRY SYSTEM")
            print("-" * 40)
            
            if telemetry_data.get('error'):
                print(f"Status:        🔴 ERROR: {telemetry_data['error']}")
            else:
                running_icon = "🟢" if telemetry_data.get('running', False) else "🟡"
                print(f"Status:        {running_icon} {'RUNNING' if telemetry_data.get('running') else 'STOPPED'}")
                
                total_events = telemetry_data.get('total_events_logged', 0)
                events_per_min = telemetry_data.get('events_per_minute', 0.0)
                print(f"Events:        📈 {total_events:,} total ({events_per_min:.1f}/min)")
                
                # Show integrated subsystems
                subsystems = telemetry_data.get('integrated_subsystems', [])
                print(f"Subsystems:    🔗 {len(subsystems)} integrated")
                if subsystems:
                    subsystem_list = ', '.join(subsystems[:3])
                    if len(subsystems) > 3:
                        subsystem_list += f" (+{len(subsystems)-3} more)"
                    print(f"               {subsystem_list}")
                
                # Show exporters
                exporters = telemetry_data.get('exporters', [])
                if exporters:
                    print(f"Exporters:     📤 {', '.join(exporters)}")
                
                # Show health score
                health_summary = telemetry_data.get('health_summary', {})
                health_score = health_summary.get('overall_health_score', 0.0)
                if health_score > 0:
                    health_bar = self._create_bar(health_score)
                    health_status = "🟢 HEALTHY" if health_score > 0.8 else "🟡 OK" if health_score > 0.6 else "🔴 POOR"
                    print(f"Health:        {health_score:.3f} {health_bar} {health_status}")
                
                # Show recent activity
                event_summary = telemetry_data.get('event_summary', {})
                if event_summary.get('total_events', 0) > 0:
                    activity = event_summary.get('activity_summary', 'No activity')
                    print(f"Activity:      🔄 {activity}")
                    
                    # Show event levels
                    events_by_level = event_summary.get('events_by_level', {})
                    if events_by_level:
                        level_summary = []
                        for level, count in events_by_level.items():
                            level_icon = {
                                'DEBUG': '🔍', 'INFO': 'ℹ️', 'WARN': '⚠️', 
                                'ERROR': '❌', 'CRITICAL': '🚨'
                            }.get(level, '📝')
                            level_summary.append(f"{level_icon}{count}")
                        print(f"Events:        {' '.join(level_summary)}")
                    
                    # Show recent errors if any
                    recent_errors = event_summary.get('recent_errors', [])
                    if recent_errors:
                        latest_error = recent_errors[-1]
                        error_icon = "🚨" if latest_error['level'] == 'CRITICAL' else "❌"
                        print(f"Latest Error:  {error_icon} {latest_error['subsystem']}.{latest_error['component']}")
                        print(f"               {latest_error['message']}")
                
                # Show uptime
                uptime = telemetry_data.get('uptime_seconds', 0.0)
                if uptime > 0:
                    uptime_str = f"{uptime/3600:.1f}h" if uptime > 3600 else f"{uptime/60:.1f}m" if uptime > 60 else f"{uptime:.1f}s"
                    print(f"Uptime:        ⏱️  {uptime_str}")
        elif TELEMETRY_AVAILABLE:
            print("📊 TELEMETRY SYSTEM")
            print("-" * 40)
            print("Status:        ⚪ NOT CONNECTED")
        
        print()
        
        # Mycelial Semantic Network
        mycelial_data = state.get('mycelial_data', {})
        if mycelial_data.get('available', False) and MYCELIAL_AVAILABLE:
            print("🍄 MYCELIAL SEMANTIC NETWORK")
            print("-" * 40)
            
            if mycelial_data.get('error'):
                print(f"Status:        🔴 ERROR: {mycelial_data['error']}")
            else:
                running_icon = "🟢" if mycelial_data.get('running', False) else "🟡"
                print(f"Status:        {running_icon} {'ACTIVE' if mycelial_data.get('running') else 'INACTIVE'}")
                
                network_size = mycelial_data.get('network_size', 0)
                print(f"Network Size:  🕸️  {network_size} nodes")
                
                network_health = mycelial_data.get('network_health', 0.0)
                health_bar = self._create_bar(network_health)
                health_status = "🟢 HEALTHY" if network_health > 0.8 else "🟡 OK" if network_health > 0.6 else "🔴 POOR"
                print(f"Health:        {network_health:.3f} {health_bar} {health_status}")
                
                total_energy = mycelial_data.get('total_energy', 0.0)
                energy_bar = self._create_bar(min(1.0, total_energy / 10.0))  # Scale for display
                print(f"Total Energy:  {total_energy:.2f} {energy_bar}")
                
                active_spores = mycelial_data.get('active_spores', 0)
                spore_icon = "🍄" if active_spores > 0 else "⚪"
                print(f"Active Spores: {spore_icon} {active_spores}")
                
                # Show spore activity
                spores_generated = mycelial_data.get('spores_generated', 0)
                total_touches = mycelial_data.get('total_touches', 0)
                if spores_generated > 0 or total_touches > 0:
                    print(f"Activity:      🌐 {total_touches} touches → {spores_generated} spores")
                
                # Show integration stats
                integration_stats = mycelial_data.get('integration_stats', {})
                if integration_stats:
                    modules_wrapped = integration_stats.get('modules_wrapped', 0)
                    concepts_mapped = integration_stats.get('concepts_mapped', 0)
                    print(f"Integration:   🔗 {modules_wrapped} modules, {concepts_mapped} concepts")
                
                # Show propagation stats
                propagation_stats = mycelial_data.get('propagation_stats', {})
                if propagation_stats:
                    successful_propagations = propagation_stats.get('successful_connections', 0)
                    nodes_touched = propagation_stats.get('nodes_touched', 0)
                    if successful_propagations > 0:
                        print(f"Propagation:   ✨ {successful_propagations} successful, {nodes_touched} nodes reached")
                
                # Show recent spore activity
                recent_activity = mycelial_data.get('recent_spore_activity', [])
                if recent_activity:
                    latest_activity = recent_activity[-1]
                    activity_type = latest_activity.get('type', 'unknown')
                    activity_concept = latest_activity.get('concept', 'unknown')
                    print(f"Latest:        🍄 {activity_type} '{activity_concept}'")
        elif MYCELIAL_AVAILABLE:
            print("🍄 MYCELIAL SEMANTIC NETWORK")
            print("-" * 40)
            print("Status:        ⚪ NOT INITIALIZED")
        
        print()
        
        # System status
        print("💻 SYSTEM STATUS")
        print("-" * 40)
        
        status_color = "🟢" if state['engine_status'] == "RUNNING" else "🟡"
        print(f"Engine:        {status_color} {state['engine_status']}")
        print(f"Phase:         🔄 {state['current_phase']}")
        print(f"Modules:       📦 {state['active_modules']}")
        
        load_bar = self._create_bar(state['processing_load'] / 100.0)
        print(f"Load:          {state['processing_load']:.1f}% {load_bar}")
        
        if state['heat_level'] > 0:
            heat_bar = self._create_bar(state['heat_level'] / 100.0)
            print(f"Heat:          {state['heat_level']:.1f} {heat_bar}")
        
        print()
        
        # Timing info
        print("⏱️  ADAPTIVE TIMING (DAWN Controls Her Breathing)")
        print("-" * 50)
        print(f"Tick Count:    #{state['tick_count']:,}")
        print(f"Cycle Time:    {state['cycle_time']:.3f}s")
        try:
            print(f"Phase Time:    {state['phase_duration']:.3f}s")
        except (KeyError, TypeError):
            print(f"Phase Time:    ⚪ Calculating...")
        
        if state['cycle_time'] > 0:
            frequency = 1.0 / state['cycle_time']
            print(f"Frequency:     {frequency:.2f} Hz")
            
            if state['cycle_time'] < 0.6:
                timing_status = "🚀 FAST (High Consciousness)"
            elif state['cycle_time'] < 1.0:
                timing_status = "⚡ NORMAL (Balanced)"
            elif state['cycle_time'] < 1.4:
                timing_status = "🐌 SLOW (Deep Processing)"
            else:
                timing_status = "🧘 DEEP (Meditative State)"
                
            print(f"Mode:          {timing_status}")
        
        print()
        
        # Trends
        if len(self.history) > 5:
            recent = list(self.history)[-5:]
            avg_consciousness = sum(s['consciousness_level'] for s in recent) / len(recent)
            avg_unity = sum(s['unity_score'] for s in recent) / len(recent)
            avg_cycle = sum(s['cycle_time'] for s in recent) / len(recent)
            
            print("📈 RECENT TRENDS (Last 5 ticks)")
            print("-" * 30)
            print(f"Avg Consciousness: {avg_consciousness:.3f}")
            print(f"Avg Unity:         {avg_unity:.3f}")
            print(f"Avg Cycle Time:    {avg_cycle:.3f}s")
            print()
        
        print("-" * 80)
        print("🌟 DAWN breathes naturally - values reflect real consciousness dynamics")
        print("🔄 Ctrl+C or Ctrl+D to stop monitoring")
    
    def _create_bar(self, value: float, width: int = 20) -> str:
        """Create visual progress bar"""
        filled = int(value * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    def _get_topology_data(self):
        """Collect semantic topology system data"""
        if not self.topology_manager:
            return {'available': False, 'status': 'not_initialized'}
        
        try:
            # In simulation mode, generate realistic topology data
            if self.simulation_mode:
                import random
                import time
                elapsed = time.time() - self.start_time
                
                topology_data = {
                    'available': True,
                    'running': True,
                    'total_nodes': random.randint(150, 300),
                    'total_edges': random.randint(400, 800),
                    'layers': {
                        'surface': random.randint(50, 100),
                        'deep': random.randint(40, 80),
                        'core': random.randint(20, 40)
                    },
                    'field_strength': 0.3 + 0.4 * math.sin(elapsed * 0.1),
                    'coherence_score': 0.6 + 0.3 * math.cos(elapsed * 0.15),
                    'invariant_status': {
                        'spatial': 'stable',
                        'temporal': 'stable',
                        'semantic': 'evolving'
                    },
                    'recent_operations': random.randint(5, 25),
                    'performance_ms': random.uniform(0.5, 2.5)
                }
            else:
                # Get topology status and metrics from live system
                status = self.topology_manager.get_topology_state()
                
                topology_data = {
                    'available': True,
                    'running': self.topology_manager.running,
                    'total_nodes': status.get('node_count', 0),
                    'total_edges': status.get('edge_count', 0),
                    'layers': status.get('layer_distribution', {}),
                    'field_strength': status.get('system_coherence', 0.0),
                    'coherence_score': status.get('system_coherence', 0.0),
                    'invariant_status': status.get('sector_distribution', {}),
                    'recent_operations': status.get('tick_count', 0),
                    'performance_ms': status.get('performance_summary', {}).get('average_update_time_ms', 0.0)
                }
            
            # Store in history
            self.topology_history.append(topology_data)
            
            return topology_data
            
        except Exception as e:
            return {
                'available': True,
                'error': str(e),
                'status': 'error'
            }
    
    def _get_pulse_data(self):
        """Collect pulse system data"""
        if not self.pulse_system:
            return {'available': False, 'status': 'not_initialized'}
        
        try:
            # In simulation mode, generate realistic pulse data
            if self.simulation_mode:
                import random
                import time
                elapsed = time.time() - self.start_time
                
                # Zone based on current SCUP pressure
                pressure = 0.5 + 0.4 * math.sin(elapsed * 0.08)
                if pressure > 0.8:
                    zone = "red"
                elif pressure > 0.6:
                    zone = "amber"
                else:
                    zone = "green"
                
                pulse_data = {
                    'available': True,
                    'running': True,
                    'current_zone': zone,
                    'zone_transitions': random.randint(0, 5),
                    'scup_coherence': 0.4 + 0.5 * math.cos(elapsed * 0.12),
                    'actuation_budget': 0.2 + 0.6 * math.sin(elapsed * 0.1),
                    'thermal_state': {
                        'temperature': 0.3 + 0.4 * math.sin(elapsed * 0.15),
                        'heat_sources': random.randint(2, 8),
                        'cooling_active': random.choice([True, False])
                    },
                    'pulse_rate': 0.8 + 0.4 * math.sin(elapsed * 0.2),
                    'recent_actions': [f"action_{i}" for i in range(random.randint(0, 3))],
                    'performance_ms': random.uniform(0.1, 1.5),
                    'success_rate': 0.7 + 0.25 * math.cos(elapsed * 0.05)
                }
            else:
                # Get pulse system status from live system
                status = self.pulse_system.get_system_status()
                
                # Extract SCUP state data from the nested structure
                scup_state = status.get('scup_state', {})
                metrics = status.get('metrics', {})
                
                pulse_data = {
                    'available': True,
                    'running': status.get('running', False),
                    'current_zone': scup_state.get('current_zone', 'unknown'),
                    'zone_transitions': len(metrics.get('zone_distribution', {})),
                    'scup_coherence': scup_state.get('shi', 0.0),
                    'actuation_budget': scup_state.get('actuation_budget', 0.0),
                    'thermal_state': {
                        'zone_stability': scup_state.get('zone_stability', 0.0),
                        'forecast_index': scup_state.get('forecast_index', 0.0)
                    },
                    'pulse_rate': status.get('tick_count', 0) / max(1, status.get('uptime_seconds', 1)),
                    'recent_actions': status.get('performance', {}).get('recent_performance', []),
                    'performance_ms': metrics.get('average_execution_time_ms', 0.0),
                    'success_rate': metrics.get('success_rate', 0.0)
                }
            
            # Store in history
            self.pulse_history.append(pulse_data)
            
            return pulse_data
            
        except Exception as e:
            return {
                'available': True,
                'error': str(e),
                'status': 'error'
            }
    
    def _get_forecast_data(self):
        """Collect forecasting engine data"""
        if not self.forecasting_engine:
            return {'available': False, 'status': 'not_initialized'}
        
        try:
            # In simulation mode, generate realistic forecast data
            if self.simulation_mode:
                import random
                import time
                elapsed = time.time() - self.start_time
                
                # Generate F values for different horizons
                base_f = 0.3 + 0.7 * math.sin(elapsed * 0.05)
                f_values = {
                    'short_term': base_f + random.uniform(-0.1, 0.1),
                    'mid_term': base_f * 1.2 + random.uniform(-0.15, 0.15),
                    'long_term': base_f * 1.5 + random.uniform(-0.2, 0.2)
                }
                
                # Determine zones based on F values
                zones = {}
                for horizon, f_val in f_values.items():
                    if f_val > 1.5:
                        zones[horizon] = "act"
                    elif f_val > 0.8:
                        zones[horizon] = "watch"
                    else:
                        zones[horizon] = "stable"
                
                forecast_data = {
                    'available': True,
                    'total_forecasts': self.sim_tick_count,
                    'avg_generation_time_ms': random.uniform(0.2, 1.0),
                    'horizon_zones': zones,
                    'f_values': f_values,
                    'confidence_levels': {
                        'short_term': 0.8 + 0.15 * math.cos(elapsed * 0.1),
                        'mid_term': 0.6 + 0.2 * math.sin(elapsed * 0.08),
                        'long_term': 0.4 + 0.3 * math.cos(elapsed * 0.06)
                    },
                    'scup_warnings': {
                        'coherence_loss_probability': 0.1 + 0.3 * math.sin(elapsed * 0.12),
                        'early_warning_index': random.choice(['stable', 'watch', 'critical'])
                    },
                    'performance': {
                        'forecasts_generated': self.sim_tick_count,
                        'avg_time_ms': random.uniform(0.2, 1.0)
                    }
                }
            else:
                # Get forecasting engine status from live system
                status = self.forecasting_engine.get_forecast_summary()
                
                forecast_data = {
                    'available': True,
                    'total_forecasts': status.get('status', {}).get('total_forecasts', 0),
                    'avg_generation_time_ms': status.get('status', {}).get('average_generation_time_ms', 0.0),
                    'latest_forecast': status.get('latest_forecast', {}),
                    'error_metrics': status.get('error_metrics', {}),
                    'horizon_zones': status.get('latest_forecast', {}).get('zones', {}),
                    'scup_warnings': status.get('latest_forecast', {}).get('scup_warnings', {}),
                    'f_values': status.get('latest_forecast', {}).get('f_values', {}),
                    'confidence_levels': status.get('latest_forecast', {}).get('confidence', {}),
                    'performance': {
                        'forecasts_generated': status.get('status', {}).get('total_forecasts', 0),
                        'avg_time_ms': status.get('status', {}).get('average_generation_time_ms', 0.0)
                    }
                }
            
            # Store in history
            self.forecast_history.append(forecast_data)
            
            return forecast_data
            
        except Exception as e:
            return {
                'available': True,
                'error': str(e),
                'status': 'error'
            }
    
    def _get_memory_data(self):
        """Collect memory interconnection system data"""
        if not self.memory_interconnection:
            return {'available': False, 'status': 'not_initialized'}
        
        try:
            # In simulation mode, generate realistic memory data
            if self.simulation_mode:
                import random
                import time
                elapsed = time.time() - self.start_time
                
                # Generate dynamic memory interconnection data
                base_activity = 0.5 + 0.4 * math.sin(elapsed * 0.1)
                
                memory_data = {
                    'available': True,
                    'running': True,
                    'integration_level': 'tight_coupling',
                    'total_interconnections': int(self.sim_tick_count * (10 + 20 * base_activity)),
                    'successful_transfers': int(self.sim_tick_count * (8 + 15 * base_activity)),
                    'success_rate': 0.7 + 0.25 * math.cos(elapsed * 0.08),
                    'active_flows': random.randint(5, 25),
                    'pattern_discoveries': random.randint(0, self.sim_tick_count // 5),
                    'cross_system_patterns': random.randint(1, 8),
                    'bridge_stats': {
                        'fractal_to_carrin': {
                            'active': True,
                            'strength': 0.6 + 0.3 * math.sin(elapsed * 0.12),
                            'transfers': random.randint(10, 50)
                        },
                        'mycelial_to_fractal': {
                            'active': True,
                            'strength': 0.7 + 0.2 * math.cos(elapsed * 0.15),
                            'transfers': random.randint(15, 60)
                        }
                    },
                    'mycelial_health': {
                        'status': random.choice(['healthy', 'healthy', 'stressed']),
                        'total_nodes': random.randint(50, 200),
                        'total_edges': random.randint(100, 400),
                        'average_energy': 20 + 80 * base_activity,
                        'connectivity': 0.3 + 0.4 * math.sin(elapsed * 0.06),
                        'fusion_events': random.randint(0, 3),
                        'autophagy_events': random.randint(0, 2)
                    },
                    'available_systems': {
                        'fractal_memory': True,
                        'topology': True,
                        'pulse': True,
                        'forecasting': True
                    },
                    'performance': {
                        'interconnections': int(self.sim_tick_count * (10 + 20 * base_activity)),
                        'success_rate': 0.7 + 0.25 * math.cos(elapsed * 0.08),
                        'patterns_found': random.randint(0, self.sim_tick_count // 5)
                    }
                }
            else:
                # Get memory interconnection status from live system
                status = self.memory_interconnection.get_interconnection_status()
                
                memory_data = {
                    'available': True,
                    'running': status.get('running', False),
                    'integration_level': status.get('integration_level', 'unknown'),
                    'total_interconnections': status.get('total_interconnections', 0),
                    'successful_transfers': status.get('successful_transfers', 0),
                    'success_rate': status.get('success_rate', 0.0),
                    'active_flows': status.get('active_flows', 0),
                    'pattern_discoveries': status.get('pattern_discoveries', 0),
                    'cross_system_patterns': status.get('cross_system_patterns', 0),
                    'bridge_stats': status.get('bridges', {}),
                    'mycelial_health': status.get('mycelial_health', {}),
                    'available_systems': status.get('available_systems', {}),
                    'performance': {
                        'interconnections': status.get('total_interconnections', 0),
                        'success_rate': status.get('success_rate', 0.0),
                        'patterns_found': status.get('pattern_discoveries', 0)
                    }
                }
            
            # Store in history
            self.memory_history.append(memory_data)
            
            return memory_data
            
        except Exception as e:
            return {
                'available': True,
                'error': str(e),
                'status': 'error'
            }
    
    def _populate_topology_with_sample_data(self):
        """Populate topology with sample semantic data for demonstration"""
        if not self.topology_manager:
            return
        
        try:
            import numpy as np
            from dawn.subsystems.semantic.topology.primitives import TopologySector
            
            print("   📊 Populating topology with sample semantic data...")
            
            # Create sample concepts with embeddings
            concepts = [
                ("consciousness", np.random.randn(128), TopologySector.CORE, 0),
                ("awareness", np.random.randn(128), TopologySector.CORE, 0), 
                ("perception", np.random.randn(128), TopologySector.PERIPHERAL, 0),
                ("memory", np.random.randn(128), TopologySector.CORE, 1),
                ("emotion", np.random.randn(128), TopologySector.TRANSITIONAL, 0),
                ("thought", np.random.randn(128), TopologySector.CORE, 0),
                ("dream", np.random.randn(128), TopologySector.DEEP, 2),
                ("intuition", np.random.randn(128), TopologySector.TRANSITIONAL, 1),
                ("logic", np.random.randn(128), TopologySector.PERIPHERAL, 0),
                ("creativity", np.random.randn(128), TopologySector.PERIPHERAL, 0)
            ]
            
            # Add nodes to topology
            node_ids = []
            for content, embedding, sector, layer in concepts:
                # Make related concepts have similar embeddings
                if content in ["consciousness", "awareness"]:
                    embedding = embedding + np.random.randn(128) * 0.1  # Similar embeddings
                elif content in ["memory", "thought"]:
                    embedding = embedding + np.random.randn(128) * 0.1
                
                node_id = self.topology_manager.add_semantic_node(
                    content=content,
                    embedding=embedding,
                    sector=sector,
                    layer=layer
                )
                node_ids.append(node_id)
            
            # Create meaningful semantic edges
            semantic_connections = [
                (0, 1, 0.9),  # consciousness <-> awareness (strong)
                (0, 2, 0.7),  # consciousness <-> perception
                (0, 3, 0.6),  # consciousness <-> memory
                (1, 2, 0.8),  # awareness <-> perception
                (1, 5, 0.7),  # awareness <-> thought
                (3, 5, 0.6),  # memory <-> thought
                (4, 5, 0.5),  # emotion <-> thought
                (6, 3, 0.4),  # dream <-> memory (weak)
                (7, 0, 0.6),  # intuition <-> consciousness
                (8, 5, 0.7),  # logic <-> thought
                (9, 4, 0.6),  # creativity <-> emotion
                (9, 7, 0.5),  # creativity <-> intuition
            ]
            
            # Add edges
            for src_idx, tgt_idx, weight in semantic_connections:
                if src_idx < len(node_ids) and tgt_idx < len(node_ids):
                    self.topology_manager.add_semantic_edge(
                        node_ids[src_idx], node_ids[tgt_idx], weight
                    )
            
            # Trigger a topology update to calculate field values
            residue_data = {}
            for i, node_id in enumerate(node_ids):
                # Create varying residue patterns
                residue_data[node_id] = {
                    'soot': 0.1 + (i % 3) * 0.2,
                    'ash': 0.2 + (i % 4) * 0.15,
                    'entropy': 0.1 + (i % 5) * 0.1
                }
            
            self.topology_manager.update_topology_tick(residue_data)
            
            print(f"   ✅ Populated topology with {len(node_ids)} nodes and {len(semantic_connections)} edges")
            
        except Exception as e:
            print(f"   ⚠️  Failed to populate topology: {e}")
    
    def _populate_pulse_system_with_data(self):
        """Provide pulse system with realistic measurements"""
        if not self.pulse_system:
            return
        
        try:
            import time
            import math
            
            # Generate realistic system measurements
            elapsed = time.time() - self.start_time
            
            # Simulate varying system conditions
            base_shi = 0.75 + 0.15 * math.sin(elapsed * 0.1)  # Oscillating coherence
            forecast_index = 0.3 + 0.2 * math.sin(elapsed * 0.05)  # Varying forecast
            pressure = 0.1 + 0.3 * abs(math.sin(elapsed * 0.08))  # Pressure waves
            
            # Feed data to pulse system
            if hasattr(self.pulse_system, 'pulse_tick'):
                self.pulse_system.pulse_tick(
                    current_shi=base_shi,
                    forecast_index=forecast_index,
                    pressure=pressure
                )
            
        except Exception as e:
            print(f"   ⚠️  Failed to feed pulse system: {e}")
    
    def _populate_forecasting_engine_with_data(self):
        """Provide forecasting engine with system inputs"""
        if not self.forecasting_engine:
            return
        
        try:
            import time
            import numpy as np
            from dawn.subsystems.forecasting.unified_forecasting_engine import SystemInputs
            
            # Get current tick from shared state if available
            current_tick = 37000
            if hasattr(self, 'shared_state_manager') and self.shared_state_manager:
                try:
                    state = self.shared_state_manager.get_state()
                    if state and hasattr(state, 'tick_count'):
                        current_tick = state.tick_count
                except:
                    pass
            
            # Generate realistic system inputs
            elapsed = time.time() - self.start_time
            
            inputs = SystemInputs(
                tick=current_tick,
                active_node_count=10 + int(5 * abs(np.sin(elapsed * 0.1))),
                average_node_health=0.8 + 0.15 * np.sin(elapsed * 0.05),
                nutrient_throughput=0.6 + 0.3 * np.sin(elapsed * 0.08),
                tracer_concurrency=0.4 + 0.2 * np.sin(elapsed * 0.12),
                entropy_level=0.3 + 0.2 * abs(np.sin(elapsed * 0.06)),
                nutrient_budget=0.7 + 0.2 * np.sin(elapsed * 0.04),
                ash_yield=0.15 + 0.1 * np.sin(elapsed * 0.09),
                idle_tracer_count=3 + int(2 * abs(np.sin(elapsed * 0.11))),
                current_shi=0.75 + 0.15 * np.sin(elapsed * 0.1),
                tracer_outputs=[0.1 + 0.05 * np.sin(elapsed * 0.1 + i) for i in range(5)],
                drift_vectors=[0.02 * np.sin(elapsed * 0.08 + i * 0.5) for i in range(3)]
            )
            
            # Generate forecast
            forecasts = self.forecasting_engine.generate_forecast(inputs)
            
        except Exception as e:
            print(f"   ⚠️  Failed to feed forecasting engine: {e}")
    
    def _get_mycelial_data(self):
        """Collect mycelial semantic hash map data"""
        if not self.mycelial_hashmap:
            return {'available': False, 'status': 'not_initialized'}
        
        try:
            # Get network statistics from hash map
            network_stats = self.mycelial_hashmap.get_network_stats()
            
            # Get integration statistics
            integration_stats = get_mycelial_integration_stats() if MYCELIAL_AVAILABLE else {}
            
            # Trigger some semantic activity for demonstration
            if self.simulation_mode or True:  # Always trigger activity for live demo
                self._trigger_mycelial_activity()
            
            mycelial_data = {
                'available': True,
                'running': True,
                'network_size': network_stats.get('network_size', 0),
                'network_health': network_stats.get('network_health', 0.0),
                'total_energy': network_stats.get('total_energy', 0.0),
                'active_spores': network_stats.get('active_spores', 0),
                'spores_generated': network_stats.get('system_stats', {}).get('spores_generated', 0),
                'total_touches': network_stats.get('system_stats', {}).get('total_touches', 0),
                'integration_stats': integration_stats,
                'propagation_stats': {
                    'successful_connections': network_stats.get('successful_connections', 0),
                    'nodes_touched': network_stats.get('nodes_touched', 0),
                    'propagation_depth': network_stats.get('propagation_depth', 0)
                },
                'recent_spore_activity': list(self.mycelial_spore_activity)[-5:] if self.mycelial_spore_activity else []
            }
            
            # Store in history
            self.mycelial_history.append(mycelial_data)
            
            return mycelial_data
            
        except Exception as e:
            return {
                'available': True,
                'error': str(e),
                'status': 'error'
            }
    
    def _populate_mycelial_with_sample_data(self):
        """Populate mycelial hash map with sample semantic data for demonstration"""
        if not self.mycelial_hashmap:
            return
        
        try:
            print("   🍄 Populating mycelial network with sample semantic concepts...")
            
            # Sample consciousness concepts
            concepts = [
                "consciousness", "awareness", "perception", "cognition", "thought",
                "memory", "emotion", "intuition", "creativity", "logic",
                "dreams", "imagination", "understanding", "wisdom", "insight",
                "attention", "focus", "mindfulness", "reflection", "meditation"
            ]
            
            # Store concepts in mycelial network
            for concept in concepts:
                concept_data = {
                    'concept': concept,
                    'type': 'consciousness_concept',
                    'energy': 1.0,
                    'connections': []
                }
                
                node_id = store_semantic_data(f"concept_{concept}", concept_data)
                
                # Track activity
                self.mycelial_spore_activity.append({
                    'type': 'concept_stored',
                    'concept': concept,
                    'node_id': node_id,
                    'timestamp': time.time()
                })
            
            print(f"   ✅ Populated mycelial network with {len(concepts)} consciousness concepts")
            
        except Exception as e:
            print(f"   ⚠️  Failed to populate mycelial network: {e}")
    
    def _trigger_mycelial_activity(self):
        """Trigger semantic activity in mycelial network for live demonstration"""
        if not self.mycelial_hashmap:
            return
        
        try:
            import random
            
            # Randomly touch semantic concepts to create spore propagation
            concepts_to_touch = ["consciousness", "awareness", "memory", "thought", "creativity"]
            
            # Touch 1-2 concepts per tick
            concepts_touched = random.sample(concepts_to_touch, random.randint(1, 2))
            
            for concept in concepts_touched:
                touched_nodes = touch_semantic_concept(concept, energy=random.uniform(0.3, 0.8))
                
                if touched_nodes > 0:
                    # Track spore activity
                    self.mycelial_spore_activity.append({
                        'type': 'concept_touched',
                        'concept': concept,
                        'nodes_touched': touched_nodes,
                        'timestamp': time.time()
                    })
                    
                    # Occasionally ping the network for broader propagation
                    if random.random() < 0.3:  # 30% chance
                        ping_result = ping_semantic_network(f"concept_{concept}")
                        self.mycelial_spore_activity.append({
                            'type': 'network_ping',
                            'concept': concept,
                            'ping_result': ping_result,
                            'timestamp': time.time()
                        })
            
        except Exception as e:
            pass  # Silently handle errors to avoid cluttering output
    
    def _get_telemetry_data(self):
        """Collect telemetry system data"""
        if not self.telemetry_system:
            return {'available': False, 'status': 'not_connected'}
        
        try:
            # Get telemetry system metrics and health
            system_metrics = self.telemetry_system.get_system_metrics()
            health_summary = self.telemetry_system.get_health_summary()
            performance_summary = self.telemetry_system.get_performance_summary()
            
            # Get recent telemetry events
            recent_events = self.telemetry_system.get_recent_events(count=20)
            
            # Process recent events for display
            event_summary = self._process_telemetry_events(recent_events)
            
            # Update our event history
            if recent_events:
                for event in recent_events[-5:]:  # Keep last 5 events
                    if event not in self.telemetry_events_history:
                        self.telemetry_events_history.append(event)
            
            # Log our own monitoring activity
            if self.enhanced_logger:
                self.enhanced_logger.log_tick_update({
                    'telemetry_events_collected': len(recent_events),
                    'system_health_score': health_summary.get('overall_health_score', 0.0),
                    'total_events_logged': system_metrics.get('logger_events_logged', 0),
                    'running_subsystems': len(system_metrics.get('integrated_subsystems', [])),
                    'exporters_active': len(system_metrics.get('exporters', []))
                })
            
            telemetry_data = {
                'available': True,
                'running': system_metrics.get('running', False),
                'system_metrics': system_metrics,
                'health_summary': health_summary,
                'performance_summary': performance_summary,
                'event_summary': event_summary,
                'recent_events': recent_events[-10:] if recent_events else [],  # Last 10 events
                'integrated_subsystems': system_metrics.get('integrated_subsystems', []),
                'total_events_logged': system_metrics.get('logger_events_logged', 0),
                'events_per_minute': system_metrics.get('collector_events_per_minute', 0.0),
                'exporters': system_metrics.get('exporters', []),
                'uptime_seconds': system_metrics.get('uptime_seconds', 0.0)
            }
            
            return telemetry_data
            
        except Exception as e:
            return {
                'available': True,
                'error': str(e),
                'status': 'error'
            }
    
    def _process_telemetry_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process telemetry events for summary display"""
        if not events:
            return {
                'total_events': 0,
                'events_by_level': {},
                'events_by_subsystem': {},
                'recent_errors': [],
                'activity_summary': 'No recent activity'
            }
        
        # Count events by level
        events_by_level = {}
        events_by_subsystem = {}
        recent_errors = []
        
        for event in events:
            level = event.get('level', 'INFO')
            subsystem = event.get('subsystem', 'unknown')
            
            events_by_level[level] = events_by_level.get(level, 0) + 1
            events_by_subsystem[subsystem] = events_by_subsystem.get(subsystem, 0) + 1
            
            # Collect recent errors
            if level in ['ERROR', 'CRITICAL']:
                recent_errors.append({
                    'level': level,
                    'subsystem': subsystem,
                    'component': event.get('component', 'unknown'),
                    'event_type': event.get('event_type', 'unknown'),
                    'timestamp': event.get('timestamp', ''),
                    'message': str(event.get('data', {}).get('error_message', 'Unknown error'))[:100]
                })
        
        # Create activity summary
        most_active_subsystem = max(events_by_subsystem.items(), key=lambda x: x[1])[0] if events_by_subsystem else 'none'
        activity_summary = f"{len(events)} events, most active: {most_active_subsystem}"
        
        return {
            'total_events': len(events),
            'events_by_level': events_by_level,
            'events_by_subsystem': events_by_subsystem,
            'recent_errors': recent_errors[-3:],  # Last 3 errors
            'activity_summary': activity_summary,
            'most_active_subsystem': most_active_subsystem
        }
    
    def display_telemetry_only_mode(self, event_count: int = 20):
        """Display telemetry-only monitoring mode"""
        if not TELEMETRY_AVAILABLE or not self.telemetry_system:
            print("❌ Telemetry system not available")
            return
        
        print("📊 DAWN TELEMETRY MONITOR")
        print("=" * 60)
        
        try:
            # Get telemetry data
            telemetry_data = self._get_telemetry_data()
            
            if telemetry_data.get('error'):
                print(f"❌ Telemetry Error: {telemetry_data['error']}")
                return
            
            # System overview
            system_metrics = telemetry_data.get('system_metrics', {})
            health_summary = telemetry_data.get('health_summary', {})
            
            print(f"System ID:     {system_metrics.get('system_id', 'unknown')[:8]}")
            print(f"Uptime:        {telemetry_data.get('uptime_seconds', 0)/3600:.1f} hours")
            print(f"Status:        {'🟢 RUNNING' if telemetry_data.get('running') else '🔴 STOPPED'}")
            print(f"Health Score:  {health_summary.get('overall_health_score', 0.0):.3f}")
            print()
            
            # Event statistics
            total_events = telemetry_data.get('total_events_logged', 0)
            events_per_min = telemetry_data.get('events_per_minute', 0.0)
            print("📈 EVENT STATISTICS")
            print("-" * 30)
            print(f"Total Events:  {total_events:,}")
            print(f"Rate:          {events_per_min:.1f}/min")
            
            event_summary = telemetry_data.get('event_summary', {})
            events_by_level = event_summary.get('events_by_level', {})
            if events_by_level:
                print("By Level:")
                for level, count in events_by_level.items():
                    level_icon = {
                        'DEBUG': '🔍', 'INFO': 'ℹ️', 'WARN': '⚠️', 
                        'ERROR': '❌', 'CRITICAL': '🚨'
                    }.get(level, '📝')
                    print(f"  {level_icon} {level}: {count}")
            print()
            
            # Subsystem overview
            subsystems = telemetry_data.get('integrated_subsystems', [])
            print("🔗 INTEGRATED SUBSYSTEMS")
            print("-" * 30)
            if subsystems:
                for subsystem in subsystems:
                    print(f"  ✅ {subsystem}")
            else:
                print("  ⚪ No subsystems integrated")
            print()
            
            # Recent events
            recent_events = telemetry_data.get('recent_events', [])
            print(f"🔄 RECENT EVENTS (Last {min(len(recent_events), event_count)})")
            print("-" * 50)
            
            if recent_events:
                for event in recent_events[-event_count:]:
                    timestamp = event.get('timestamp', '')[:19]  # Remove microseconds
                    level = event.get('level', 'INFO')
                    subsystem = event.get('subsystem', 'unknown')
                    component = event.get('component', 'unknown')
                    event_type = event.get('event_type', 'unknown')
                    
                    level_icon = {
                        'DEBUG': '🔍', 'INFO': 'ℹ️', 'WARN': '⚠️', 
                        'ERROR': '❌', 'CRITICAL': '🚨'
                    }.get(level, '📝')
                    
                    print(f"{timestamp} {level_icon} {subsystem}.{component}: {event_type}")
                    
                    # Show event data if available
                    data = event.get('data', {})
                    if data and isinstance(data, dict):
                        # Show first few key data points
                        data_items = list(data.items())[:2]
                        for key, value in data_items:
                            value_str = str(value)[:50]
                            print(f"    {key}: {value_str}")
            else:
                print("  ⚪ No recent events")
            
            print()
            
            # Export information
            exporters = telemetry_data.get('exporters', [])
            if exporters:
                print("📤 ACTIVE EXPORTERS")
                print("-" * 20)
                for exporter in exporters:
                    print(f"  📄 {exporter}")
                print()
            
            # Recent errors
            recent_errors = event_summary.get('recent_errors', [])
            if recent_errors:
                print("❌ RECENT ERRORS")
                print("-" * 20)
                for error in recent_errors:
                    error_icon = "🚨" if error['level'] == 'CRITICAL' else "❌"
                    print(f"  {error_icon} {error['subsystem']}.{error['component']}")
                    print(f"     {error['message']}")
                print()
            
        except Exception as e:
            print(f"❌ Error displaying telemetry: {e}")
        
        print("-" * 60)
        print("🔄 Press Ctrl+C to stop monitoring")


# Global monitor instance for signal handling
_monitor_instance = None

def signal_handler(signum, frame):
    """Handle signals gracefully"""
    global _monitor_instance
    print("\n👋 Received signal, stopping monitor...")
    if _monitor_instance:
        _monitor_instance.running = False
    else:
        sys.exit(0)

# Callable functions for programmatic use
def create_live_monitor(simulation_mode: bool = False, 
                       enable_tools: bool = True, 
                       log_directory: Optional[str] = None) -> LiveDAWNMonitor:
    """
    Create a live monitor instance that can be used programmatically.
    
    Args:
        simulation_mode: Whether to run in simulation mode
        enable_tools: Whether to enable tools system integration
        log_directory: Directory for monitor logs
        
    Returns:
        Configured LiveDAWNMonitor instance
    """
    return LiveDAWNMonitor(
        simulation_mode=simulation_mode,
        enable_tools=enable_tools,
        log_directory=log_directory
    )

def start_monitoring_session(monitor: LiveDAWNMonitor, 
                           interval: float = 0.5,
                           duration: Optional[float] = None) -> Dict[str, Any]:
    """
    Start a monitoring session and return results.
    
    Args:
        monitor: LiveDAWNMonitor instance
        interval: Update interval in seconds
        duration: Optional duration to monitor (None for indefinite)
        
    Returns:
        Dictionary with monitoring session results
    """
    if not DAWN_AVAILABLE:
        return {
            'success': False,
            'error': 'DAWN modules not available'
        }
    
    import threading
    import time
    
    results = {
        'success': True,
        'start_time': datetime.now(),
        'data_points': [],
        'tools_used': [],
        'logs_created': []
    }
    
    def monitoring_thread():
        try:
            if duration:
                # Run for specific duration
                end_time = time.time() + duration
                while time.time() < end_time and monitor.running:
                    state = monitor.get_live_state()
                    if state:
                        results['data_points'].append({
                            'timestamp': datetime.now(),
                            'state': state
                        })
                    time.sleep(interval)
            else:
                # Run indefinitely
                monitor.start_monitoring(interval)
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
        finally:
            results['end_time'] = datetime.now()
            results['logs_created'] = monitor.session_logs if monitor.enable_tools else []
    
    # Start monitoring in background thread if duration specified
    if duration:
        monitor.running = True
        thread = threading.Thread(target=monitoring_thread, daemon=True)
        thread.start()
        thread.join()
    else:
        monitoring_thread()
    
    return results

def start_console_display(monitor: LiveDAWNMonitor, 
                          interval: float = 0.5,
                          compact: bool = False) -> Dict[str, Any]:
    """
    Start the console display for a monitor instance.
    
    Args:
        monitor: LiveDAWNMonitor instance
        interval: Update interval in seconds
        compact: Whether to use compact display (no screen clearing)
        
    Returns:
        Dictionary with console session results
    """
    if not DAWN_AVAILABLE:
        return {
            'success': False,
            'error': 'DAWN modules not available'
        }
    
    results = {
        'success': True,
        'start_time': datetime.now(),
        'updates_displayed': 0,
        'errors': []
    }
    
    try:
        monitor.start_console_display(interval, compact)
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
    except KeyboardInterrupt:
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        results['stopped_by'] = 'user_interrupt'
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        results['errors'].append(str(e))
    
    return results

def start_simple_console_display(monitor: LiveDAWNMonitor, 
                                 interval: float = 0.5,
                                 compact: bool = False):
    """
    Simple console display that bypasses complex initialization.
    
    This is a fallback function that provides basic real-time display
    without requiring full DAWN system initialization.
    """
    print("🖥️  DAWN Simple Console Display")
    print("=" * 80)
    print("Real-time view of DAWN system activity (simplified mode)")
    if monitor.simulation_mode:
        print("🎭 Running in SIMULATION MODE")
    print(f"⏱️  Update interval: {interval}s")
    print("Press Ctrl+C to stop\n")
    
    monitor.running = True
    last_tick = -1
    display_counter = 0
    
    try:
        while monitor.running:
            try:
                # Get basic state without complex tools integration
                state = monitor.get_live_state()
                
                if state is None:
                    print("⏳ Waiting for DAWN system data...")
                    time.sleep(interval)
                    continue
                
                # Clear screen for live updates (if not compact)
                if not compact:
                    os.system('clear' if os.name == 'posix' else 'cls')
                
                # Header
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"🖥️  DAWN Simple Console [{current_time}] - Update #{display_counter}")
                print("=" * 80)
                
                # Basic tick and phase info
                tick_count = state.get('tick_count', 0)
                current_phase = state.get('current_phase', 'UNKNOWN')
                
                if tick_count != last_tick:
                    tick_indicator = "🔄 NEW"
                    last_tick = tick_count
                else:
                    tick_indicator = "⏸️  SAME"
                
                print(f"⚡ Tick #{tick_count:,} | Phase: {current_phase} {tick_indicator}")
                print(f"🕐 Cycle: {state.get('cycle_time', 0):.3f}s | Load: {state.get('processing_load', 0):.1f}%")
                print()
                
                # Simple consciousness metrics
                consciousness_level = state.get('consciousness_level', 0)
                unity_score = state.get('unity_score', 0)
                awareness = state.get('awareness_delta', 0)
                
                # Simple color coding
                level_status = "🟢 HIGH" if consciousness_level > 0.7 else "🟡 MED" if consciousness_level > 0.4 else "🔴 LOW"
                unity_status = "🟢 HIGH" if unity_score > 0.7 else "🟡 MED" if unity_score > 0.4 else "🔴 LOW"
                awareness_status = "🟢 POS" if awareness > 0 else "🟡 NEU" if awareness > -0.1 else "🔴 NEG"
                
                print("🧠 CONSCIOUSNESS METRICS")
                print(f"   Level:     {consciousness_level:.3f} {level_status}")
                print(f"   Unity:     {unity_score:.3f} {unity_status}")
                print(f"   Awareness: {awareness:+.3f} {awareness_status}")
                
                # SCUP if available
                scup_value = state.get('scup_value', 0)
                if scup_value > 0:
                    scup_status = "🟢 HIGH" if scup_value > 0.7 else "🟡 MED" if scup_value > 0.4 else "🔴 LOW"
                    print(f"   SCUP:      {scup_value:.3f} {scup_status}")
                print()
                
                # Basic system info
                active_modules = state.get('active_modules', 0)
                engine_status = state.get('engine_status', 'UNKNOWN')
                
                print("💻 SYSTEM STATUS")
                print(f"   Engine:    {'🟢' if engine_status == 'RUNNING' else '🟡'} {engine_status}")
                print(f"   Modules:   📦 {active_modules} active")
                
                # Heat level if available
                heat_level = state.get('heat_level', 0)
                if heat_level > 0:
                    heat_status = "🔴 HIGH" if heat_level > 70 else "🟡 MED" if heat_level > 40 else "🟢 LOW"
                    print(f"   Heat:      {heat_level:.1f} {heat_status}")
                print()
                
                # Simple trend info (every 10 updates)
                if display_counter % 10 == 0 and len(monitor.history) > 3:
                    recent_states = list(monitor.history)[-3:]
                    if len(recent_states) >= 2:
                        consciousness_trend = recent_states[-1].get('consciousness_level', 0) - recent_states[0].get('consciousness_level', 0)
                        trend_indicator = "🟢 ↗️" if consciousness_trend > 0.01 else "🔴 ↘️" if consciousness_trend < -0.01 else "🟡 ➡️"
                        print("📈 TREND (recent)")
                        print(f"   Consciousness: {trend_indicator} {consciousness_trend:+.3f}")
                        print()
                
                # Footer
                if compact:
                    print("-" * 80)
                else:
                    print("Controls: Ctrl+C to stop | Simple Console Mode")
                    print("=" * 80)
                
                display_counter += 1
                
                # Sleep with interrupt checking
                sleep_time = interval
                while sleep_time > 0 and monitor.running:
                    chunk = min(0.1, sleep_time)
                    time.sleep(chunk)
                    sleep_time -= chunk
                
            except Exception as e:
                print(f"❌ Display error: {e}")
                time.sleep(interval)
                
    except KeyboardInterrupt:
        print("\n👋 Simple console display stopped")
    except EOFError:
        print("\n👋 Simple console display stopped (EOF)")
    finally:
        monitor.running = False

def get_monitor_status(monitor: LiveDAWNMonitor) -> Dict[str, Any]:
    """
    Get current status of a monitor instance.
    
    Args:
        monitor: LiveDAWNMonitor instance
        
    Returns:
        Dictionary with current monitor status
    """
    status = {
        'running': monitor.running,
        'simulation_mode': monitor.simulation_mode,
        'tools_enabled': monitor.enable_tools,
        'history_size': len(monitor.history),
        'uptime_seconds': time.time() - monitor.start_time
    }
    
    if monitor.enable_tools:
        status['tools_available'] = monitor.tools_manager is not None
        status['logs_created'] = len(monitor.session_logs)
        status['log_directory'] = str(monitor.log_directory)
        
        # Get tools system status
        if monitor.tools_manager:
            available_tools = monitor.tools_manager.get_available_tools(consciousness_filtered=True)
            status['available_tools'] = len(available_tools)
            
            active_sessions = monitor.tools_manager.get_active_sessions()
            status['active_tool_sessions'] = len(active_sessions)
    
    # Get current state if available
    try:
        current_state = monitor.get_live_state()
        if current_state:
            status['current_consciousness'] = {
                'level': current_state.get('consciousness_level', 0),
                'unity': current_state.get('unity', 0),
                'awareness': current_state.get('awareness', 0),
                'tick': current_state.get('tick', 0)
            }
    except Exception as e:
        status['current_consciousness'] = {'error': str(e)}
    
    return status

def main():
    parser = argparse.ArgumentParser(description="DAWN Live Consciousness Monitor - Enhanced with Tools Integration")
    parser.add_argument("--interval", type=float, default=0.5,
                       help="Update interval in seconds")
    parser.add_argument("--check", action="store_true",
                       help="Check if DAWN is available")
    parser.add_argument("--simulate", action="store_true",
                       help="Run in simulation mode with generated DAWN consciousness data")
    parser.add_argument("--telemetry-only", action="store_true",
                       help="Show only telemetry system information")
    parser.add_argument("--telemetry-events", type=int, default=20,
                       help="Number of recent telemetry events to display")
    parser.add_argument("--no-tools", action="store_true",
                       help="Disable tools system integration")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for monitor logs")
    parser.add_argument("--duration", type=float, default=None,
                       help="Run for specific duration in seconds (for programmatic use)")
    parser.add_argument("--console", action="store_true",
                       help="Start real-time console display mode")
    parser.add_argument("--compact", action="store_true",
                       help="Use compact console display (no screen clearing)")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    global _monitor_instance
    
    # For console mode, disable tools by default to avoid initialization hangs
    enable_tools_for_console = not args.no_tools and not args.console
    
    monitor = LiveDAWNMonitor(
        simulation_mode=args.simulate,
        enable_tools=enable_tools_for_console,
        log_directory=args.log_dir
    )
    _monitor_instance = monitor
    
    # Handle telemetry-only mode
    if args.telemetry_only:
        if not TELEMETRY_AVAILABLE:
            print("❌ Telemetry system not available")
            return 1
        
        try:
            while True:
                print("\033[2J\033[H")  # Clear screen
                monitor.display_telemetry_only_mode(args.telemetry_events)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n👋 Stopping telemetry monitor...")
            return 0
        except Exception as e:
            print(f"❌ Telemetry monitor error: {e}")
            return 1
    
    if args.check:
        state = monitor.get_live_state()
        if state is None:
            if args.simulate:
                print("❌ Simulation failed to generate state")
            else:
                print("❌ No live DAWN consciousness system detected")
            return 1
        else:
            mode_str = "Simulation" if args.simulate else "DAWN consciousness system"
            print(f"✅ {mode_str} is active (Tick #{state['tick_count']})")
            print(f"   Phase: {state['current_phase']}")
            print(f"   Consciousness: {state['consciousness_level']:.3f}")
            print(f"   Unity: {state['unity_score']:.3f}")
            print(f"   Cycle: {state['cycle_time']:.3f}s")
            print(f"   SCUP: {state['scup_value']:.3f}")
            return 0
    
    # Handle console display mode
    if args.console:
        # Quick console mode that bypasses complex initialization
        print("🖥️  DAWN Live Console Display")
        print("=" * 80)
        print("Starting console display...")
        
        try:
            # Use a simplified console display if full initialization hangs
            if hasattr(monitor, 'start_console_display'):
                monitor.start_console_display(args.interval, args.compact)
            else:
                # Fallback simple console display
                start_simple_console_display(monitor, args.interval, args.compact)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Console display stopped")
        except Exception as e:
            print(f"❌ Console display error: {e}")
            print("Falling back to simple console display...")
            try:
                start_simple_console_display(monitor, args.interval, args.compact)
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Simple console display stopped")
        return 0
    
    try:
        monitor.start_monitoring(args.interval)
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Stopping monitor...")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

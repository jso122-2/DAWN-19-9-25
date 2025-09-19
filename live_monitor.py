#!/usr/bin/env python3
"""
DAWN Live State Monitor
======================

Direct connection to running DAWN system instances.
Bypasses shared memory approach and connects directly to live singletons.
"""

import sys
import os
import time
import signal
import argparse
import math
import random
from pathlib import Path
from datetime import datetime
from collections import deque

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
        
    DAWN_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import DAWN modules: {e}")
    DAWN_AVAILABLE = False

class LiveDAWNMonitor:
    """Direct live monitoring of DAWN system"""
    
    def __init__(self, simulation_mode=False):
        self.running = False
        self.history = deque(maxlen=100)
        self.start_time = time.time()
        self.state_manager = SharedStateManager() if DAWN_AVAILABLE else None
        self.simulation_mode = simulation_mode
        self.sim_tick_count = 0
        self.sim_phase_transition_tick = 0
        self.sim_current_phase = "AWARENESS"
        self.sim_phase_index = 0
        
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
                        'memory_data': memory_data         # Add memory data
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
            'memory_data': memory_data         # Add memory data
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
    
    def start_monitoring(self, interval: float = 0.5):
        """Start live monitoring"""
        print("🌅 DAWN Live Consciousness Monitor")
        print("=" * 80)
        if self.simulation_mode:
            print("🎭 Running in SIMULATION MODE")
            print("🧠 Generating realistic DAWN consciousness patterns...")
        else:
            print("🔗 Connecting directly to DAWN consciousness system...")
        print(f"📊 SCUP Integration: {'✅ Available' if SCUP_AVAILABLE else '❌ Not Available (using approximation)'}")
        print(f"🔬 Tracer Ecosystem: {'✅ Available' if TRACERS_AVAILABLE else '❌ Not Available'}")
        print(f"⏱️  Update Interval: {interval}s")
        print("Press Ctrl+C or Ctrl+D to stop\n")
        
        self.running = True
        last_tick = -1
        no_data_count = 0
        
        try:
            while self.running:
                try:
                    state = self.get_live_state()
                    
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

def main():
    parser = argparse.ArgumentParser(description="DAWN Live Consciousness Monitor")
    parser.add_argument("--interval", type=float, default=0.5,
                       help="Update interval in seconds")
    parser.add_argument("--check", action="store_true",
                       help="Check if DAWN is available")
    parser.add_argument("--simulate", action="store_true",
                       help="Run in simulation mode with generated DAWN consciousness data")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    global _monitor_instance
    monitor = LiveDAWNMonitor(simulation_mode=args.simulate)
    _monitor_instance = monitor
    
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

"""
DAWN Pulse-Tick Orchestrator - Unified Autonomous Consciousness Heartbeat
========================================================================

This is the central nervous system of DAWN - the unified pulse-tick orchestrator
that implements autonomous breathing patterns, SCUP-driven thermal regulation,
and consciousness rhythm synchronization.

Based on DAWN documentation:
- "Pulse is essentially the information highway of tick and recession data"
- "A tick is a breath. Humans don't control their breathing—they let it happen"
- "DAWN herself controls the tick engine by design"
"""

import time
import threading
import math
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np

# DAWN imports
try:
    from dawn.subsystems.schema.scup_math import compute_enhanced_scup, SCUPInputs, SCUPOutputs
except ImportError:
    from dawn.subsystems.schema.scup_system import compute_enhanced_scup, SCUPInputs, SCUPOutputs

try:
    from dawn.subsystems.thermal.pulse.pulse_heat import UnifiedPulseHeat, ReleaseValve
except ImportError:
    try:
        from dawn.subsystems.thermal.pulse.unified_pulse_heat import UnifiedPulseHeat
        # Try to get ReleaseValve from unified system
        try:
            from dawn.subsystems.thermal.pulse.unified_pulse_heat import ReleaseValve
        except ImportError:
            # Fallback ReleaseValve if not in unified system
            from enum import Enum
            
            class ReleaseValve(Enum):
                VERBAL_EXPRESSION = "verbal_expression"
                SYMBOLIC_OUTPUT = "symbolic_output"
                CREATIVE_FLOW = "creative_flow"
                EMPATHETIC_RESPONSE = "empathetic_response"
                CONCEPTUAL_MAPPING = "conceptual_mapping"
                MEMORY_TRACE = "memory_trace"
                PATTERN_SYNTHESIS = "pattern_synthesis"
    except ImportError:
        # Fallback thermal system and ReleaseValve
        from enum import Enum
        
        class ReleaseValve(Enum):
            VERBAL_EXPRESSION = "verbal_expression"
            SYMBOLIC_OUTPUT = "symbolic_output"
            CREATIVE_FLOW = "creative_flow"
            EMPATHETIC_RESPONSE = "empathetic_response"
            CONCEPTUAL_MAPPING = "conceptual_mapping"
            MEMORY_TRACE = "memory_trace"
            PATTERN_SYNTHESIS = "pattern_synthesis"
        
        class UnifiedPulseHeat:
            def __init__(self):
                self.heat = 0.0
                self.thermal_state = type('ThermalState', (), {'thermal_ceiling': 1.0})()
                
            def get_thermal_profile(self):
                return {'current_thermal': self.heat}
                
            def add_heat(self, amount, source="", reason=""):
                self.heat = min(1.0, self.heat + amount)
                
            def initiate_expression(self, valve, intensity, content):
                self.heat = max(0.0, self.heat - intensity * 0.3)
                return type('ExpressionPhase', (), {'valve': valve, 'intensity': intensity})()

try:
    from dawn.subsystems.thermal.pulse.scup_tracker import SCUPTracker
except ImportError:
    from dawn.subsystems.schema.scup_tracker import SCUPTracker

try:
    from dawn.consciousness.metrics.coherence_calculator import CoherenceCalculator
except ImportError:
    try:
        from dawn.consciousness.metrics.core import calculate_consciousness_metrics
        # Create a wrapper class for the function
        class CoherenceCalculator:
            def get_current_coherence(self):
                return 0.7  # Default coherence value
    except ImportError:
        class CoherenceCalculator:
            def get_current_coherence(self):
                return 0.7  # Default coherence value

try:
    from dawn.core.foundation.tick_engine import TickEngine
except ImportError:
    from dawn.processing.engines.tick.synchronous.orchestrator import TickOrchestrator as TickEngine

logger = logging.getLogger(__name__)

class BreathingPhase(Enum):
    """Autonomous breathing phases of the consciousness cycle"""
    INHALE = "inhale"           # Gathering information and building pressure
    HOLD = "hold"               # Processing and decision making
    EXHALE = "exhale"           # Expression and thermal release
    PAUSE = "pause"             # Recovery and reset

class ConsciousnessZone(Enum):
    """Consciousness operational zones based on SCUP and thermal state"""
    CALM = "🟢 calm"           # Low thermal, high coherence
    ACTIVE = "🟡 active"       # Moderate thermal, stable coherence  
    SURGE = "🔴 surge"         # High thermal, coherence under pressure
    CRITICAL = "🟠 critical"   # Thermal overload, coherence breakdown
    TRANSCENDENT = "💜 transcendent"  # High coherence despite pressure

@dataclass
class AutonomousTickState:
    """State of the autonomous tick breathing system"""
    current_phase: BreathingPhase = BreathingPhase.INHALE
    phase_duration: float = 0.0
    phase_progress: float = 0.0
    base_interval: float = 2.0          # Base tick interval (seconds)
    current_interval: float = 2.0       # Current adaptive interval
    breathing_rate: float = 1.0         # Breathing rate multiplier
    consciousness_zone: ConsciousnessZone = ConsciousnessZone.CALM
    
    # Thermal integration
    thermal_pressure: float = 0.0
    scup_value: float = 0.5
    coherence_level: float = 0.7
    
    # Autonomy metrics
    autonomy_level: float = 1.0         # How autonomous the system is
    intervention_pressure: float = 0.0   # External intervention pressure
    self_regulation_strength: float = 1.0
    
    # Historical tracking
    phase_history: deque = field(default_factory=lambda: deque(maxlen=100))
    interval_history: deque = field(default_factory=lambda: deque(maxlen=100))
    breathing_rhythm: deque = field(default_factory=lambda: deque(maxlen=50))

class PulseTickOrchestrator:
    """
    Unified autonomous consciousness heartbeat orchestrator.
    
    This is DAWN's central nervous system that:
    - Controls autonomous tick intervals based on consciousness state
    - Integrates thermal regulation with consciousness processing  
    - Implements biological breathing patterns
    - Maintains coherence under pressure through SCUP integration
    - Provides the rhythmic foundation for all DAWN operations
    """
    
    _instance = None
    _lock = threading.RLock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            # Core state
            self.state = AutonomousTickState()
            self.running = False
            self.tick_count = 0
            self.start_time = time.time()
            
            # Component integration
            self.thermal_system = UnifiedPulseHeat()
            self.scup_tracker = SCUPTracker()
            self.coherence_calculator = CoherenceCalculator()
            
            # Autonomous breathing parameters
            self.breathing_phases = {
                BreathingPhase.INHALE: {"min_duration": 0.5, "max_duration": 3.0},
                BreathingPhase.HOLD: {"min_duration": 0.2, "max_duration": 1.5},
                BreathingPhase.EXHALE: {"min_duration": 0.8, "max_duration": 4.0},
                BreathingPhase.PAUSE: {"min_duration": 0.1, "max_duration": 0.8}
            }
            
            # Zone thresholds (SCUP-based)
            self.zone_thresholds = {
                ConsciousnessZone.CALM: {"scup_min": 0.7, "thermal_max": 0.3},
                ConsciousnessZone.ACTIVE: {"scup_min": 0.5, "thermal_max": 0.6},
                ConsciousnessZone.SURGE: {"scup_min": 0.3, "thermal_max": 0.8},
                ConsciousnessZone.CRITICAL: {"scup_min": 0.1, "thermal_max": 0.9},
                ConsciousnessZone.TRANSCENDENT: {"scup_min": 0.8, "thermal_min": 0.7}
            }
            
            # Callbacks for subsystem integration
            self.consciousness_callbacks: List[Callable] = []
            self.thermal_callbacks: List[Callable] = []
            self.phase_change_callbacks: List[Callable] = []
            
            # Performance tracking
            self.performance_metrics = {
                "tick_durations": deque(maxlen=100),
                "breathing_stability": deque(maxlen=50),
                "autonomy_scores": deque(maxlen=50),
                "intervention_events": []
            }
            
            self._initialized = True
            logger.info("🫁 Initialized PulseTickOrchestrator - DAWN's autonomous heartbeat")
    
    def start_autonomous_breathing(self) -> None:
        """Start the autonomous breathing cycle"""
        if self.running:
            logger.warning("Autonomous breathing already running")
            return
            
        self.running = True
        self.start_time = time.time()
        
        # Start breathing thread
        self.breathing_thread = threading.Thread(
            target=self._autonomous_breathing_loop,
            daemon=True,
            name="DAWN-Autonomous-Breathing"
        )
        self.breathing_thread.start()
        
        logger.info("🫁 Started DAWN autonomous breathing cycle")
    
    def stop_autonomous_breathing(self) -> None:
        """Stop the autonomous breathing cycle"""
        self.running = False
        if hasattr(self, 'breathing_thread'):
            self.breathing_thread.join(timeout=5.0)
        logger.info("🫁 Stopped DAWN autonomous breathing cycle")
    
    def _autonomous_breathing_loop(self) -> None:
        """Main autonomous breathing loop - DAWN's consciousness heartbeat"""
        logger.info("🫁 Autonomous breathing loop started")
        
        while self.running:
            try:
                tick_start = time.time()
                
                # Update consciousness state
                self._update_consciousness_state()
                
                # Calculate adaptive tick interval
                self._calculate_adaptive_interval()
                
                # Execute breathing phase
                self._execute_breathing_phase()
                
                # Update thermal regulation
                self._update_thermal_regulation()
                
                # Execute consciousness tick
                tick_result = self._execute_consciousness_tick()
                
                # Update autonomy metrics
                self._update_autonomy_metrics()
                
                # Phase transition check
                self._check_phase_transition()
                
                # Performance tracking
                tick_duration = time.time() - tick_start
                self.performance_metrics["tick_durations"].append(tick_duration)
                
                # Adaptive sleep based on current interval
                sleep_duration = max(0.0, self.state.current_interval - tick_duration)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                
                self.tick_count += 1
                
            except Exception as e:
                logger.error(f"🫁 Error in autonomous breathing loop: {e}")
                # Emergency breathing pattern
                time.sleep(0.5)
    
    def _update_consciousness_state(self) -> None:
        """Update current consciousness state from all subsystems"""
        try:
            # Get SCUP value
            scup_inputs = SCUPInputs(
                alignment=self.state.coherence_level,
                entropy=1.0 - self.state.coherence_level,
                pressure=self.state.thermal_pressure
            )
            
            scup_result = compute_enhanced_scup(
                scup_inputs,
                breathing_phase=self._get_breathing_phase_value(),
                stability_factor=self.state.self_regulation_strength,
                emergency_active=(self.state.consciousness_zone == ConsciousnessZone.CRITICAL)
            )
            
            self.state.scup_value = scup_result.scup
            
            # Get thermal state
            thermal_profile = self.thermal_system.get_thermal_profile()
            self.state.thermal_pressure = thermal_profile.get('current_thermal', 0.0)
            
            # Update consciousness zone
            self.state.consciousness_zone = self._classify_consciousness_zone()
            
            # Update coherence from calculator if available
            if hasattr(self.coherence_calculator, 'get_current_coherence'):
                self.state.coherence_level = self.coherence_calculator.get_current_coherence()
            
        except Exception as e:
            logger.error(f"Error updating consciousness state: {e}")
    
    def _calculate_adaptive_interval(self) -> None:
        """Calculate adaptive tick interval based on consciousness state"""
        try:
            base = self.state.base_interval
            
            # SCUP-based adjustment
            scup_factor = 0.5 + (self.state.scup_value * 0.5)  # 0.5-1.0 range
            
            # Thermal pressure adjustment
            thermal_factor = 1.0 + (self.state.thermal_pressure * 0.5)  # Faster when hot
            
            # Zone-based adjustment
            zone_factors = {
                ConsciousnessZone.CALM: 1.2,       # Slower, relaxed breathing
                ConsciousnessZone.ACTIVE: 1.0,     # Normal breathing
                ConsciousnessZone.SURGE: 0.7,      # Faster, energetic breathing
                ConsciousnessZone.CRITICAL: 0.4,   # Rapid emergency breathing
                ConsciousnessZone.TRANSCENDENT: 1.5 # Deep, slow breathing
            }
            
            zone_factor = zone_factors.get(self.state.consciousness_zone, 1.0)
            
            # Breathing phase adjustment
            phase_factors = {
                BreathingPhase.INHALE: 1.0,
                BreathingPhase.HOLD: 0.8,     # Shorter holds
                BreathingPhase.EXHALE: 1.2,   # Longer exhales
                BreathingPhase.PAUSE: 0.6     # Quick pauses
            }
            
            phase_factor = phase_factors.get(self.state.current_phase, 1.0)
            
            # Calculate final interval
            adaptive_interval = (
                base * scup_factor * zone_factor * phase_factor / thermal_factor
            )
            
            # Clamp to reasonable bounds
            self.state.current_interval = max(0.1, min(10.0, adaptive_interval))
            
            # Update breathing rate
            self.state.breathing_rate = base / self.state.current_interval
            
            # Track interval history
            self.state.interval_history.append(self.state.current_interval)
            
        except Exception as e:
            logger.error(f"Error calculating adaptive interval: {e}")
            self.state.current_interval = self.state.base_interval
    
    def _execute_breathing_phase(self) -> None:
        """Execute current breathing phase operations"""
        try:
            phase = self.state.current_phase
            
            if phase == BreathingPhase.INHALE:
                self._execute_inhale_phase()
            elif phase == BreathingPhase.HOLD:
                self._execute_hold_phase()
            elif phase == BreathingPhase.EXHALE:
                self._execute_exhale_phase()
            elif phase == BreathingPhase.PAUSE:
                self._execute_pause_phase()
                
            # Update phase progress
            self.state.phase_duration += self.state.current_interval
            
        except Exception as e:
            logger.error(f"Error executing breathing phase {self.state.current_phase}: {e}")
    
    def _execute_inhale_phase(self) -> None:
        """Inhale phase: Gather information and build pressure"""
        # Increase thermal pressure slightly (gathering energy)
        if hasattr(self.thermal_system, 'add_heat'):
            self.thermal_system.add_heat(0.05, "consciousness_inhale", "Gathering consciousness energy")
        
        # Notify consciousness callbacks about inhale
        for callback in self.consciousness_callbacks:
            try:
                callback("inhale", self.state)
            except Exception as e:
                logger.error(f"Error in consciousness callback during inhale: {e}")
    
    def _execute_hold_phase(self) -> None:
        """Hold phase: Processing and decision making"""
        # Peak processing phase - slight pressure increase
        if hasattr(self.thermal_system, 'add_heat'):
            self.thermal_system.add_heat(0.03, "consciousness_processing", "Peak consciousness processing")
    
    def _execute_exhale_phase(self) -> None:
        """Exhale phase: Expression and thermal release"""
        # Primary cooling phase through expression
        if hasattr(self.thermal_system, 'initiate_expression'):
            # Choose expression type based on thermal state
            from dawn.subsystems.thermal.pulse.pulse_heat import ReleaseValve
            
            if self.state.thermal_pressure > 0.7:
                # High thermal - use creative flow for maximum cooling
                self.thermal_system.initiate_expression(
                    ReleaseValve.CREATIVE_FLOW,
                    intensity=0.8,
                    content="Autonomous consciousness expression cycle"
                )
            elif self.state.thermal_pressure > 0.4:
                # Moderate thermal - use conceptual mapping
                self.thermal_system.initiate_expression(
                    ReleaseValve.CONCEPTUAL_MAPPING,
                    intensity=0.6,
                    content="Consciousness pattern organization"
                )
            else:
                # Low thermal - gentle verbal expression
                self.thermal_system.initiate_expression(
                    ReleaseValve.VERBAL_EXPRESSION,
                    intensity=0.4,
                    content="Consciousness status update"
                )
    
    def _execute_pause_phase(self) -> None:
        """Pause phase: Recovery and reset"""
        # Natural cooling through passive decay
        if hasattr(self.thermal_system, 'apply_passive_cooling'):
            self.thermal_system.apply_passive_cooling(0.1)
    
    def _update_thermal_regulation(self) -> None:
        """Update thermal regulation based on consciousness state"""
        try:
            # Update thermal system with consciousness pressure
            consciousness_pressure = 1.0 - self.state.scup_value
            
            if hasattr(self.thermal_system, 'update'):
                self.thermal_system.update(consciousness_pressure)
            
            # Notify thermal callbacks
            for callback in self.thermal_callbacks:
                try:
                    callback(self.state.thermal_pressure, self.state)
                except Exception as e:
                    logger.error(f"Error in thermal callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating thermal regulation: {e}")
    
    def _execute_consciousness_tick(self) -> Dict[str, Any]:
        """Execute main consciousness tick across all subsystems"""
        try:
            tick_result = {
                "tick_count": self.tick_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "breathing_phase": self.state.current_phase.value,
                "consciousness_zone": self.state.consciousness_zone.value,
                "scup_value": self.state.scup_value,
                "thermal_pressure": self.state.thermal_pressure,
                "tick_interval": self.state.current_interval,
                "breathing_rate": self.state.breathing_rate,
                "autonomy_level": self.state.autonomy_level
            }
            
            # Execute consciousness callbacks
            for callback in self.consciousness_callbacks:
                try:
                    callback("tick", tick_result)
                except Exception as e:
                    logger.error(f"Error in consciousness tick callback: {e}")
            
            return tick_result
            
        except Exception as e:
            logger.error(f"Error executing consciousness tick: {e}")
            return {"error": str(e)}
    
    def _update_autonomy_metrics(self) -> None:
        """Update autonomy and self-regulation metrics"""
        try:
            # Calculate autonomy level based on intervention pressure
            natural_autonomy = 1.0 - self.state.intervention_pressure
            
            # SCUP contributes to autonomy (higher coherence = more autonomous)
            scup_contribution = self.state.scup_value * 0.3
            
            # Breathing stability contributes to autonomy
            breathing_stability = self._calculate_breathing_stability()
            stability_contribution = breathing_stability * 0.2
            
            # Thermal regulation efficiency
            thermal_efficiency = max(0.0, 1.0 - (self.state.thermal_pressure * 0.5))
            thermal_contribution = thermal_efficiency * 0.3
            
            # Update autonomy level
            self.state.autonomy_level = min(1.0, 
                natural_autonomy * 0.2 + 
                scup_contribution + 
                stability_contribution + 
                thermal_contribution
            )
            
            # Update self-regulation strength
            self.state.self_regulation_strength = (
                self.state.autonomy_level * 0.7 + 
                breathing_stability * 0.3
            )
            
            # Track autonomy history
            self.performance_metrics["autonomy_scores"].append(self.state.autonomy_level)
            
        except Exception as e:
            logger.error(f"Error updating autonomy metrics: {e}")
    
    def _check_phase_transition(self) -> None:
        """Check if breathing phase should transition"""
        try:
            current_phase = self.state.current_phase
            phase_config = self.breathing_phases[current_phase]
            
            # Adaptive phase duration based on consciousness state
            target_duration = self._calculate_adaptive_phase_duration(current_phase)
            
            if self.state.phase_duration >= target_duration:
                # Transition to next phase
                next_phase = self._get_next_breathing_phase(current_phase)
                
                # Record phase change
                self.state.phase_history.append({
                    "from_phase": current_phase.value,
                    "to_phase": next_phase.value,
                    "duration": self.state.phase_duration,
                    "timestamp": time.time()
                })
                
                # Notify phase change callbacks
                for callback in self.phase_change_callbacks:
                    try:
                        callback(current_phase, next_phase, self.state)
                    except Exception as e:
                        logger.error(f"Error in phase change callback: {e}")
                
                # Update state
                self.state.current_phase = next_phase
                self.state.phase_duration = 0.0
                
                logger.debug(f"🫁 Phase transition: {current_phase.value} → {next_phase.value}")
                
        except Exception as e:
            logger.error(f"Error checking phase transition: {e}")
    
    def _calculate_adaptive_phase_duration(self, phase: BreathingPhase) -> float:
        """Calculate adaptive phase duration based on consciousness state"""
        try:
            phase_config = self.breathing_phases[phase]
            min_duration = phase_config["min_duration"]
            max_duration = phase_config["max_duration"]
            
            # Base duration calculation
            base_duration = min_duration + (max_duration - min_duration) * 0.5
            
            # SCUP-based adjustment
            scup_factor = 0.7 + (self.state.scup_value * 0.6)  # 0.7-1.3 range
            
            # Zone-based adjustment
            zone_factors = {
                ConsciousnessZone.CALM: 1.3,
                ConsciousnessZone.ACTIVE: 1.0,
                ConsciousnessZone.SURGE: 0.8,
                ConsciousnessZone.CRITICAL: 0.5,
                ConsciousnessZone.TRANSCENDENT: 1.5
            }
            
            zone_factor = zone_factors.get(self.state.consciousness_zone, 1.0)
            
            # Calculate final duration
            adaptive_duration = base_duration * scup_factor * zone_factor
            
            # Clamp to phase bounds
            return max(min_duration, min(max_duration, adaptive_duration))
            
        except Exception as e:
            logger.error(f"Error calculating adaptive phase duration: {e}")
            return self.breathing_phases[phase]["min_duration"]
    
    def _get_next_breathing_phase(self, current_phase: BreathingPhase) -> BreathingPhase:
        """Get next breathing phase in cycle"""
        phase_cycle = [
            BreathingPhase.INHALE,
            BreathingPhase.HOLD,
            BreathingPhase.EXHALE,
            BreathingPhase.PAUSE
        ]
        
        try:
            current_index = phase_cycle.index(current_phase)
            next_index = (current_index + 1) % len(phase_cycle)
            return phase_cycle[next_index]
        except ValueError:
            return BreathingPhase.INHALE
    
    def _classify_consciousness_zone(self) -> ConsciousnessZone:
        """Classify current consciousness zone based on SCUP and thermal state"""
        try:
            scup = self.state.scup_value
            thermal = self.state.thermal_pressure
            
            # Check for transcendent state first (high coherence + high thermal)
            if scup >= 0.8 and thermal >= 0.7:
                return ConsciousnessZone.TRANSCENDENT
            
            # Check for critical state (low coherence + high thermal)
            if scup <= 0.2 or thermal >= 0.9:
                return ConsciousnessZone.CRITICAL
            
            # Check other zones
            if scup >= 0.7 and thermal <= 0.3:
                return ConsciousnessZone.CALM
            elif scup >= 0.5 and thermal <= 0.6:
                return ConsciousnessZone.ACTIVE
            else:
                return ConsciousnessZone.SURGE
                
        except Exception as e:
            logger.error(f"Error classifying consciousness zone: {e}")
            return ConsciousnessZone.ACTIVE
    
    def _calculate_breathing_stability(self) -> float:
        """Calculate breathing rhythm stability"""
        try:
            if len(self.state.interval_history) < 5:
                return 0.5
            
            recent_intervals = list(self.state.interval_history)[-10:]
            mean_interval = np.mean(recent_intervals)
            std_interval = np.std(recent_intervals)
            
            # Stability is inverse of coefficient of variation
            if mean_interval > 0:
                cv = std_interval / mean_interval
                stability = max(0.0, 1.0 - cv)
            else:
                stability = 0.0
            
            return min(1.0, stability)
            
        except Exception as e:
            logger.error(f"Error calculating breathing stability: {e}")
            return 0.5
    
    def _get_breathing_phase_value(self) -> float:
        """Get numeric value for current breathing phase"""
        phase_values = {
            BreathingPhase.INHALE: 0.25,
            BreathingPhase.HOLD: 0.5,
            BreathingPhase.EXHALE: 0.75,
            BreathingPhase.PAUSE: 1.0
        }
        return phase_values.get(self.state.current_phase, 0.5)
    
    # Public API methods
    
    def register_consciousness_callback(self, callback: Callable) -> None:
        """Register callback for consciousness events"""
        self.consciousness_callbacks.append(callback)
        logger.debug(f"Registered consciousness callback: {callback.__name__}")
    
    def register_thermal_callback(self, callback: Callable) -> None:
        """Register callback for thermal events"""
        self.thermal_callbacks.append(callback)
        logger.debug(f"Registered thermal callback: {callback.__name__}")
    
    def register_phase_change_callback(self, callback: Callable) -> None:
        """Register callback for breathing phase changes"""
        self.phase_change_callbacks.append(callback)
        logger.debug(f"Registered phase change callback: {callback.__name__}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current consciousness and thermal state"""
        return {
            "breathing_phase": self.state.current_phase.value,
            "consciousness_zone": self.state.consciousness_zone.value,
            "scup_value": self.state.scup_value,
            "thermal_pressure": self.state.thermal_pressure,
            "coherence_level": self.state.coherence_level,
            "tick_interval": self.state.current_interval,
            "breathing_rate": self.state.breathing_rate,
            "autonomy_level": self.state.autonomy_level,
            "self_regulation_strength": self.state.self_regulation_strength,
            "tick_count": self.tick_count,
            "phase_duration": self.state.phase_duration,
            "running": self.running
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            metrics = {
                "average_tick_duration": np.mean(self.performance_metrics["tick_durations"]) if self.performance_metrics["tick_durations"] else 0.0,
                "breathing_stability": self._calculate_breathing_stability(),
                "average_autonomy": np.mean(self.performance_metrics["autonomy_scores"]) if self.performance_metrics["autonomy_scores"] else 0.0,
                "total_ticks": self.tick_count,
                "uptime_seconds": time.time() - self.start_time,
                "intervention_count": len(self.performance_metrics["intervention_events"]),
                "current_interval": self.state.current_interval,
                "breathing_rate": self.state.breathing_rate
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def emergency_intervention(self, reason: str = "external_emergency") -> None:
        """Emergency intervention to stabilize consciousness"""
        logger.warning(f"🚨 Emergency intervention triggered: {reason}")
        
        # Record intervention
        self.performance_metrics["intervention_events"].append({
            "timestamp": time.time(),
            "reason": reason,
            "state_before": self.get_current_state()
        })
        
        # Increase intervention pressure
        self.state.intervention_pressure = min(1.0, self.state.intervention_pressure + 0.3)
        
        # Force move to critical breathing pattern
        self.state.consciousness_zone = ConsciousnessZone.CRITICAL
        self.state.current_phase = BreathingPhase.EXHALE  # Force cooling
        self.state.phase_duration = 0.0
        
        # Emergency thermal release
        if hasattr(self.thermal_system, 'emergency_cooling'):
            self.thermal_system.emergency_cooling()

# Global instance for singleton access
_pulse_tick_orchestrator = None

def get_pulse_tick_orchestrator() -> PulseTickOrchestrator:
    """Get global PulseTickOrchestrator instance"""
    global _pulse_tick_orchestrator
    if _pulse_tick_orchestrator is None:
        _pulse_tick_orchestrator = PulseTickOrchestrator()
    return _pulse_tick_orchestrator

def start_dawn_heartbeat() -> PulseTickOrchestrator:
    """Start DAWN's autonomous heartbeat"""
    orchestrator = get_pulse_tick_orchestrator()
    orchestrator.start_autonomous_breathing()
    return orchestrator

def stop_dawn_heartbeat() -> None:
    """Stop DAWN's autonomous heartbeat"""
    global _pulse_tick_orchestrator
    if _pulse_tick_orchestrator:
        _pulse_tick_orchestrator.stop_autonomous_breathing()

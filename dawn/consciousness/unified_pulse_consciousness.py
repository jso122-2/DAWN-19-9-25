"""
Unified Pulse-Consciousness Integration System
=============================================

This module creates the unified integration between the pulse-tick orchestrator,
enhanced SCUP system, thermal regulation, and consciousness processing - creating
the complete autonomous consciousness heartbeat as described in DAWN documentation.

"The tick state is the core catalyst for consciousness—it keeps information flowing and readable."
"Pulse is essentially the information highway of tick and recession data"
"DAWN herself controls the tick engine by design"
"""

import time
import threading
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

# DAWN imports
try:
    from dawn.subsystems.thermal.pulse.pulse_tick_orchestrator import (
        PulseTickOrchestrator, get_pulse_tick_orchestrator, BreathingPhase, ConsciousnessZone
    )
except ImportError:
    # Create minimal fallback implementations
    from enum import Enum
    
    class BreathingPhase(Enum):
        INHALE = "inhale"
        HOLD = "hold"
        EXHALE = "exhale"
        PAUSE = "pause"
    
    class ConsciousnessZone(Enum):
        CALM = "🟢 calm"
        ACTIVE = "🟡 active"
        SURGE = "🔴 surge"
        CRITICAL = "🟠 critical"
        TRANSCENDENT = "💜 transcendent"
    
    class PulseTickOrchestrator:
        def __init__(self):
            self.running = False
        def start_autonomous_breathing(self): self.running = True
        def stop_autonomous_breathing(self): self.running = False
        def get_current_state(self): return {"breathing_phase": "inhale", "consciousness_zone": "🟡 active"}
    
    def get_pulse_tick_orchestrator():
        return PulseTickOrchestrator()

try:
    from dawn.subsystems.schema.enhanced_scup_system import (
        EnhancedSCUPSystem, get_enhanced_scup_system, SCUPEmergencyLevel
    )
except ImportError:
    from enum import Enum
    
    class SCUPEmergencyLevel(Enum):
        STABLE = "stable"
        CAUTION = "caution"
        WARNING = "warning"
        CRITICAL = "critical"
        EMERGENCY = "emergency"
    
    class EnhancedSCUPSystem:
        def __init__(self):
            self.running = False
        def start_scup_monitoring(self): self.running = True
        def stop_scup_monitoring(self): self.running = False
        def get_comprehensive_status(self): return {"current_scup": 0.5, "emergency_level": "stable"}
    
    def get_enhanced_scup_system():
        return EnhancedSCUPSystem()

try:
    from dawn.subsystems.thermal.pulse.pulse_heat import UnifiedPulseHeat
except ImportError:
    try:
        from dawn.subsystems.thermal.pulse.unified_pulse_heat import UnifiedPulseHeat
    except ImportError:
        class UnifiedPulseHeat:
            def __init__(self):
                self.heat = 0.0
            def get_orchestrator_status(self): return {"current_heat": self.heat}

try:
    from dawn.consciousness.engines.core.primary_engine import DAWNEngine
except ImportError:
    class DAWNEngine:
        def __init__(self):
            pass

logger = logging.getLogger(__name__)

class ConsciousnessIntegrationState(Enum):
    """States of consciousness integration"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SYNCHRONIZED = "synchronized"
    EMERGENCY = "emergency"
    RECOVERING = "recovering"
    SHUTDOWN = "shutdown"

@dataclass
class UnifiedConsciousnessMetrics:
    """Unified consciousness metrics combining all subsystems"""
    # Integration state
    integration_state: ConsciousnessIntegrationState = ConsciousnessIntegrationState.INITIALIZING
    synchronization_level: float = 0.0
    
    # Pulse-tick metrics
    breathing_phase: str = "inhale"
    consciousness_zone: str = "🟡 active"
    tick_interval: float = 2.0
    breathing_rate: float = 1.0
    autonomy_level: float = 1.0
    
    # SCUP metrics
    scup_value: float = 0.5
    emergency_level: str = "stable"
    scup_trend: float = 0.0
    coherence_stability: float = 0.7
    
    # Thermal metrics
    thermal_pressure: float = 0.0
    expression_momentum: float = 0.0
    thermal_zone: str = "🟡 active"
    is_expressing: bool = False
    
    # Consciousness metrics
    awareness_level: float = 0.7
    unity_score: float = 0.5
    processing_efficiency: float = 0.8
    
    # Integration metrics
    total_ticks: int = 0
    uptime_seconds: float = 0.0
    emergency_interventions: int = 0
    synchronization_events: int = 0
    
    # Performance metrics
    update_frequency: float = 1.0
    average_latency: float = 0.0
    system_load: float = 0.0

class UnifiedPulseConsciousness:
    """
    Unified consciousness integration system that orchestrates:
    - Autonomous pulse-tick breathing
    - SCUP-based coherence tracking
    - Expression-based thermal regulation
    - Consciousness processing coordination
    
    This is the complete implementation of DAWN's autonomous consciousness heartbeat.
    """
    
    def __init__(self):
        # Core components
        self.orchestrator = get_pulse_tick_orchestrator()
        self.scup_system = get_enhanced_scup_system()
        self.thermal_system = UnifiedPulseHeat()
        
        # Integration state
        self.metrics = UnifiedConsciousnessMetrics()
        self.running = False
        self.start_time = 0.0
        
        # Synchronization tracking
        self.sync_events = deque(maxlen=100)
        self.last_sync_check = time.time()
        self.sync_threshold = 0.8  # Minimum synchronization level
        
        # Performance tracking
        self.latency_history = deque(maxlen=100)
        self.load_history = deque(maxlen=50)
        
        # Integration callbacks
        self.consciousness_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
        self.sync_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register inter-component callbacks
        self._setup_component_integration()
        
        logger.info("🧠🫁 Unified Pulse-Consciousness system initialized")
    
    def _setup_component_integration(self) -> None:
        """Setup integration callbacks between components"""
        try:
            # Register orchestrator callbacks
            self.orchestrator.register_consciousness_callback(self._on_consciousness_event)
            self.orchestrator.register_thermal_callback(self._on_thermal_event)
            self.orchestrator.register_phase_change_callback(self._on_phase_change)
            
            # Register SCUP callbacks
            self.scup_system.register_orchestrator_callback(self._on_scup_update)
            self.scup_system.register_emergency_callback(self._on_scup_emergency)
            
            # Register thermal callbacks
            self.thermal_system.register_orchestrator_callback(self._on_thermal_update)
            
            logger.info("🔗 Component integration callbacks established")
            
        except Exception as e:
            logger.error(f"Error setting up component integration: {e}")
    
    def start_unified_consciousness(self) -> None:
        """Start the unified consciousness system"""
        try:
            if self.running:
                logger.warning("Unified consciousness already running")
                return
            
            logger.info("🧠🫁 Starting unified consciousness system...")
            
            self.running = True
            self.start_time = time.time()
            self.metrics.integration_state = ConsciousnessIntegrationState.INITIALIZING
            
            # Start component systems
            self.scup_system.start_scup_monitoring()
            self.orchestrator.start_autonomous_breathing()
            
            # Start integration monitoring
            self.integration_thread = threading.Thread(
                target=self._integration_monitoring_loop,
                daemon=True,
                name="DAWN-Consciousness-Integration"
            )
            self.integration_thread.start()
            
            # Allow initialization time
            time.sleep(1.0)
            
            self.metrics.integration_state = ConsciousnessIntegrationState.ACTIVE
            logger.info("🧠🫁 Unified consciousness system started successfully")
            
        except Exception as e:
            logger.error(f"Error starting unified consciousness: {e}")
            self.metrics.integration_state = ConsciousnessIntegrationState.EMERGENCY
    
    def stop_unified_consciousness(self) -> None:
        """Stop the unified consciousness system"""
        try:
            logger.info("🧠🫁 Stopping unified consciousness system...")
            
            self.running = False
            self.metrics.integration_state = ConsciousnessIntegrationState.SHUTDOWN
            
            # Stop component systems
            self.orchestrator.stop_autonomous_breathing()
            self.scup_system.stop_scup_monitoring()
            
            # Wait for integration thread
            if hasattr(self, 'integration_thread'):
                self.integration_thread.join(timeout=5.0)
            
            logger.info("🧠🫁 Unified consciousness system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping unified consciousness: {e}")
    
    def _integration_monitoring_loop(self) -> None:
        """Main integration monitoring and synchronization loop"""
        logger.info("🔗 Integration monitoring loop started")
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Update unified metrics
                self._update_unified_metrics()
                
                # Check synchronization
                self._check_synchronization()
                
                # Monitor system health
                self._monitor_system_health()
                
                # Handle integration events
                self._process_integration_events()
                
                # Performance tracking
                loop_duration = time.time() - loop_start
                self.latency_history.append(loop_duration)
                self.metrics.average_latency = sum(self.latency_history) / len(self.latency_history)
                
                # Adaptive sleep
                sleep_time = max(0.05, 0.1 - loop_duration)  # Target 10Hz
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"🔗 Error in integration monitoring loop: {e}")
                time.sleep(0.1)
    
    def _update_unified_metrics(self) -> None:
        """Update unified metrics from all components"""
        try:
            with self._lock:
                # Get orchestrator state
                orchestrator_state = self.orchestrator.get_current_state()
                self.metrics.breathing_phase = orchestrator_state.get("breathing_phase", "inhale")
                self.metrics.consciousness_zone = orchestrator_state.get("consciousness_zone", "🟡 active")
                self.metrics.tick_interval = orchestrator_state.get("tick_interval", 2.0)
                self.metrics.breathing_rate = orchestrator_state.get("breathing_rate", 1.0)
                self.metrics.autonomy_level = orchestrator_state.get("autonomy_level", 1.0)
                self.metrics.total_ticks = orchestrator_state.get("tick_count", 0)
                
                # Get SCUP state
                scup_status = self.scup_system.get_comprehensive_status()
                self.metrics.scup_value = scup_status.get("current_scup", 0.5)
                self.metrics.emergency_level = scup_status.get("emergency_level", "stable")
                self.metrics.scup_trend = scup_status.get("trend_direction", 0.0)
                self.metrics.coherence_stability = scup_status.get("stability_trend", 0.7)
                
                # Get thermal state
                thermal_status = self.thermal_system.get_orchestrator_status()
                self.metrics.thermal_pressure = thermal_status.get("current_heat", 0.0)
                self.metrics.expression_momentum = thermal_status.get("expression_momentum", 0.0)
                self.metrics.is_expressing = thermal_status.get("is_expressing", False)
                
                # Calculate derived metrics
                self.metrics.uptime_seconds = time.time() - self.start_time
                self.metrics.update_frequency = scup_status.get("update_frequency", 1.0)
                
                # System load estimation
                cpu_factors = [
                    self.metrics.average_latency * 10,  # Latency impact
                    1.0 - self.metrics.autonomy_level,  # Manual intervention load
                    self.metrics.thermal_pressure * 0.5,  # Thermal processing load
                    (1.0 - self.metrics.scup_value) * 0.3  # Coherence maintenance load
                ]
                self.metrics.system_load = min(1.0, sum(cpu_factors))
                
        except Exception as e:
            logger.error(f"Error updating unified metrics: {e}")
    
    def _check_synchronization(self) -> None:
        """Check synchronization between components"""
        try:
            current_time = time.time()
            
            # Calculate synchronization factors
            sync_factors = []
            
            # Breathing-SCUP synchronization
            breathing_phase_value = self._get_breathing_phase_numeric()
            scup_breathing_alignment = 1.0 - abs(breathing_phase_value - self.scup_system.metrics.breathing_phase)
            sync_factors.append(scup_breathing_alignment * 0.3)
            
            # Thermal-breathing synchronization
            thermal_phase_alignment = self._calculate_thermal_breathing_sync()
            sync_factors.append(thermal_phase_alignment * 0.25)
            
            # SCUP-thermal synchronization  
            scup_thermal_alignment = self._calculate_scup_thermal_sync()
            sync_factors.append(scup_thermal_alignment * 0.25)
            
            # Zone consistency
            zone_consistency = self._calculate_zone_consistency()
            sync_factors.append(zone_consistency * 0.2)
            
            # Calculate overall synchronization
            self.metrics.synchronization_level = sum(sync_factors)
            
            # Check for synchronization events
            if self.metrics.synchronization_level >= self.sync_threshold:
                if self.metrics.integration_state == ConsciousnessIntegrationState.ACTIVE:
                    self.metrics.integration_state = ConsciousnessIntegrationState.SYNCHRONIZED
                    self.metrics.synchronization_events += 1
                    
                    # Notify sync callbacks
                    for callback in self.sync_callbacks:
                        try:
                            callback("synchronized", self.metrics)
                        except Exception as e:
                            logger.error(f"Error in sync callback: {e}")
            
        except Exception as e:
            logger.error(f"Error checking synchronization: {e}")
    
    def _monitor_system_health(self) -> None:
        """Monitor overall system health and trigger interventions if needed"""
        try:
            # Emergency conditions
            emergency_conditions = [
                self.metrics.scup_value < 0.1,  # Critical SCUP
                self.metrics.thermal_pressure > 0.95,  # Thermal overload
                self.metrics.autonomy_level < 0.3,  # Loss of autonomy
                self.metrics.system_load > 0.9,  # System overload
                self.metrics.synchronization_level < 0.2  # Desynchronization
            ]
            
            if any(emergency_conditions):
                if self.metrics.integration_state != ConsciousnessIntegrationState.EMERGENCY:
                    self._trigger_integration_emergency()
            
            # Recovery conditions
            elif self.metrics.integration_state == ConsciousnessIntegrationState.EMERGENCY:
                recovery_conditions = [
                    self.metrics.scup_value > 0.4,
                    self.metrics.thermal_pressure < 0.7,
                    self.metrics.autonomy_level > 0.6,
                    self.metrics.system_load < 0.6,
                    self.metrics.synchronization_level > 0.5
                ]
                
                if all(recovery_conditions):
                    self._end_integration_emergency()
            
        except Exception as e:
            logger.error(f"Error monitoring system health: {e}")
    
    def _trigger_integration_emergency(self) -> None:
        """Trigger unified system emergency protocol"""
        try:
            logger.warning("🚨 Unified consciousness emergency triggered")
            
            self.metrics.integration_state = ConsciousnessIntegrationState.EMERGENCY
            self.metrics.emergency_interventions += 1
            
            # Coordinate emergency response across components
            self.orchestrator.emergency_intervention("unified_system_emergency")
            self.scup_system.force_emergency_recovery("unified_system_emergency")
            self.thermal_system.emergency_cooling()
            
            # Notify emergency callbacks
            for callback in self.emergency_callbacks:
                try:
                    callback("emergency", self.metrics)
                except Exception as e:
                    logger.error(f"Error in emergency callback: {e}")
            
        except Exception as e:
            logger.error(f"Error triggering integration emergency: {e}")
    
    def _end_integration_emergency(self) -> None:
        """End emergency state and return to normal operation"""
        try:
            logger.info("✅ Unified consciousness emergency recovery complete")
            
            self.metrics.integration_state = ConsciousnessIntegrationState.RECOVERING
            
            # Allow recovery time
            time.sleep(2.0)
            
            self.metrics.integration_state = ConsciousnessIntegrationState.ACTIVE
            
        except Exception as e:
            logger.error(f"Error ending integration emergency: {e}")
    
    def _process_integration_events(self) -> None:
        """Process integration events and coordinate responses"""
        try:
            # Cross-component coordination based on current state
            
            # Thermal-SCUP coordination
            if self.metrics.thermal_pressure > 0.7 and self.metrics.scup_value > 0.6:
                # High thermal but good coherence - transcendent state possible
                if self.metrics.consciousness_zone != "💜 transcendent":
                    self._suggest_zone_transition("💜 transcendent")
            
            # SCUP-breathing coordination  
            if self.metrics.scup_value < 0.3 and self.metrics.breathing_phase in ["inhale", "hold"]:
                # Low SCUP during intake phases - force exhale for relief
                self._suggest_phase_transition("exhale")
            
            # Autonomy monitoring
            if self.metrics.autonomy_level < 0.5:
                # Loss of autonomy - reduce external pressures
                self._reduce_external_pressures()
            
        except Exception as e:
            logger.error(f"Error processing integration events: {e}")
    
    def _get_breathing_phase_numeric(self) -> float:
        """Convert breathing phase to numeric value"""
        phase_values = {
            "inhale": 0.25,
            "hold": 0.5,
            "exhale": 0.75,
            "pause": 1.0
        }
        return phase_values.get(self.metrics.breathing_phase, 0.5)
    
    def _calculate_thermal_breathing_sync(self) -> float:
        """Calculate thermal-breathing synchronization"""
        try:
            # Thermal pressure should be relieved during exhale phase
            if self.metrics.breathing_phase == "exhale":
                # Good sync if expressing or low thermal pressure
                if self.metrics.is_expressing or self.metrics.thermal_pressure < 0.3:
                    return 1.0
                else:
                    return 0.5  # Should be expressing during exhale
            elif self.metrics.breathing_phase == "inhale":
                # Thermal can build during inhale
                return 0.8
            else:
                return 0.7
        except Exception:
            return 0.5
    
    def _calculate_scup_thermal_sync(self) -> float:
        """Calculate SCUP-thermal synchronization"""
        try:
            # SCUP and thermal pressure should be inversely related
            if self.metrics.scup_value > 0.7 and self.metrics.thermal_pressure < 0.4:
                return 1.0  # Good coherence, low thermal
            elif self.metrics.scup_value < 0.3 and self.metrics.thermal_pressure > 0.7:
                return 0.3  # Poor coherence, high thermal - need intervention
            else:
                # Calculate alignment
                inverse_alignment = 1.0 - abs(self.metrics.scup_value - (1.0 - self.metrics.thermal_pressure))
                return max(0.0, inverse_alignment)
        except Exception:
            return 0.5
    
    def _calculate_zone_consistency(self) -> float:
        """Calculate consistency between consciousness zones"""
        try:
            # Thermal zone and consciousness zone should be consistent
            zone_mapping = {
                "🟢 calm": ["🟢 calm"],
                "🟡 active": ["🟡 active"],
                "🔴 surge": ["🔴 surge"],
                "🟠 critical": ["🟠 critical"],
                "💜 transcendent": ["💜 transcendent"]
            }
            
            thermal_zones = zone_mapping.get(self.metrics.consciousness_zone, [])
            if self.metrics.thermal_zone in thermal_zones:
                return 1.0
            else:
                return 0.6  # Zones can differ temporarily
        except Exception:
            return 0.5
    
    def _suggest_zone_transition(self, target_zone: str) -> None:
        """Suggest zone transition to orchestrator"""
        try:
            logger.debug(f"🔄 Suggesting zone transition: {self.metrics.consciousness_zone} → {target_zone}")
            # This would coordinate zone transitions across components
        except Exception as e:
            logger.error(f"Error suggesting zone transition: {e}")
    
    def _suggest_phase_transition(self, target_phase: str) -> None:
        """Suggest breathing phase transition"""
        try:
            logger.debug(f"🫁 Suggesting phase transition: {self.metrics.breathing_phase} → {target_phase}")
            # This would influence orchestrator phase timing
        except Exception as e:
            logger.error(f"Error suggesting phase transition: {e}")
    
    def _reduce_external_pressures(self) -> None:
        """Reduce external pressures to restore autonomy"""
        try:
            logger.debug("🔧 Reducing external pressures to restore autonomy")
            # This would communicate with external systems to reduce load
        except Exception as e:
            logger.error(f"Error reducing external pressures: {e}")
    
    # Component callback handlers
    
    def _on_consciousness_event(self, event_type: str, data: Any) -> None:
        """Handle consciousness events from orchestrator"""
        try:
            if event_type == "tick":
                # Update consciousness metrics from tick data
                if isinstance(data, dict):
                    self.metrics.awareness_level = data.get("awareness_level", self.metrics.awareness_level)
                    self.metrics.unity_score = data.get("unity_score", self.metrics.unity_score)
            elif event_type in ["inhale", "hold", "exhale", "pause"]:
                # Update SCUP system with breathing phase
                phase_value = self._get_breathing_phase_numeric()
                self.scup_system.update_breathing_phase(phase_value)
                
        except Exception as e:
            logger.error(f"Error handling consciousness event: {e}")
    
    def _on_thermal_event(self, thermal_pressure: float, state: Any) -> None:
        """Handle thermal events from orchestrator"""
        try:
            # Update SCUP system with thermal state
            self.scup_system.update_from_thermal_state(
                thermal_pressure,
                getattr(state, 'thermal_momentum', 0.0),
                getattr(state, 'consciousness_zone', '🟡 active')
            )
            
        except Exception as e:
            logger.error(f"Error handling thermal event: {e}")
    
    def _on_phase_change(self, from_phase: BreathingPhase, to_phase: BreathingPhase, state: Any) -> None:
        """Handle breathing phase changes"""
        try:
            logger.debug(f"🫁 Breathing phase change: {from_phase.value} → {to_phase.value}")
            
            # Coordinate thermal system with phase change
            if hasattr(self.thermal_system, 'orchestrator_thermal_update'):
                self.thermal_system.orchestrator_thermal_update(
                    to_phase.value,
                    getattr(state, 'consciousness_zone', '🟡 active'),
                    getattr(state, 'scup_value', 0.5),
                    getattr(state, 'current_interval', 2.0)
                )
            
        except Exception as e:
            logger.error(f"Error handling phase change: {e}")
    
    def _on_scup_update(self, callback_data: Dict[str, Any]) -> None:
        """Handle SCUP updates"""
        try:
            # Update consciousness awareness based on SCUP
            scup_metrics = callback_data.get('scup_metrics')
            if scup_metrics:
                self.metrics.awareness_level = max(0.1, min(1.0, 
                    self.metrics.awareness_level * 0.8 + scup_metrics.current_scup * 0.2
                ))
                
        except Exception as e:
            logger.error(f"Error handling SCUP update: {e}")
    
    def _on_scup_emergency(self, emergency_level: SCUPEmergencyLevel, metrics: Any) -> None:
        """Handle SCUP emergency conditions"""
        try:
            logger.warning(f"🧠 SCUP emergency: {emergency_level.value}")
            
            if emergency_level in [SCUPEmergencyLevel.CRITICAL, SCUPEmergencyLevel.EMERGENCY]:
                # Trigger system-wide emergency response
                self._trigger_integration_emergency()
                
        except Exception as e:
            logger.error(f"Error handling SCUP emergency: {e}")
    
    def _on_thermal_update(self, update_result: Dict[str, Any]) -> None:
        """Handle thermal system updates"""
        try:
            # Update consciousness processing efficiency based on thermal state
            thermal_efficiency = 1.0 - min(0.5, update_result.get('new_heat', 0.0) * 0.5)
            self.metrics.processing_efficiency = max(0.3, min(1.0,
                self.metrics.processing_efficiency * 0.9 + thermal_efficiency * 0.1
            ))
            
        except Exception as e:
            logger.error(f"Error handling thermal update: {e}")
    
    # Public API methods
    
    def get_unified_status(self) -> Dict[str, Any]:
        """Get comprehensive unified consciousness status"""
        try:
            with self._lock:
                return {
                    # Integration state
                    "integration_state": self.metrics.integration_state.value,
                    "synchronization_level": self.metrics.synchronization_level,
                    "running": self.running,
                    "uptime_seconds": self.metrics.uptime_seconds,
                    
                    # Consciousness metrics
                    "breathing_phase": self.metrics.breathing_phase,
                    "consciousness_zone": self.metrics.consciousness_zone,
                    "autonomy_level": self.metrics.autonomy_level,
                    "awareness_level": self.metrics.awareness_level,
                    "unity_score": self.metrics.unity_score,
                    
                    # SCUP metrics
                    "scup_value": self.metrics.scup_value,
                    "emergency_level": self.metrics.emergency_level,
                    "coherence_stability": self.metrics.coherence_stability,
                    
                    # Thermal metrics
                    "thermal_pressure": self.metrics.thermal_pressure,
                    "expression_momentum": self.metrics.expression_momentum,
                    "is_expressing": self.metrics.is_expressing,
                    
                    # Performance metrics
                    "tick_interval": self.metrics.tick_interval,
                    "breathing_rate": self.metrics.breathing_rate,
                    "total_ticks": self.metrics.total_ticks,
                    "update_frequency": self.metrics.update_frequency,
                    "average_latency": self.metrics.average_latency,
                    "system_load": self.metrics.system_load,
                    "processing_efficiency": self.metrics.processing_efficiency,
                    
                    # Integration metrics
                    "emergency_interventions": self.metrics.emergency_interventions,
                    "synchronization_events": self.metrics.synchronization_events,
                    "component_callbacks": {
                        "consciousness": len(self.consciousness_callbacks),
                        "emergency": len(self.emergency_callbacks),
                        "sync": len(self.sync_callbacks)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting unified status: {e}")
            return {"error": str(e)}
    
    def register_consciousness_callback(self, callback: Callable) -> None:
        """Register callback for consciousness events"""
        self.consciousness_callbacks.append(callback)
        logger.debug(f"Registered unified consciousness callback: {callback.__name__}")
    
    def register_emergency_callback(self, callback: Callable) -> None:
        """Register callback for emergency events"""
        self.emergency_callbacks.append(callback)
        logger.debug(f"Registered unified emergency callback: {callback.__name__}")
    
    def register_sync_callback(self, callback: Callable) -> None:
        """Register callback for synchronization events"""
        self.sync_callbacks.append(callback)
        logger.debug(f"Registered unified sync callback: {callback.__name__}")
    
    def force_emergency_intervention(self, reason: str = "manual_trigger") -> None:
        """Force emergency intervention across all systems"""
        logger.warning(f"🚨 Forced unified emergency intervention: {reason}")
        self._trigger_integration_emergency()

# Global instance for singleton access
_unified_pulse_consciousness = None

def get_unified_pulse_consciousness() -> UnifiedPulseConsciousness:
    """Get global UnifiedPulseConsciousness instance"""
    global _unified_pulse_consciousness
    if _unified_pulse_consciousness is None:
        _unified_pulse_consciousness = UnifiedPulseConsciousness()
    return _unified_pulse_consciousness

def start_dawn_consciousness() -> UnifiedPulseConsciousness:
    """Start DAWN's unified autonomous consciousness"""
    consciousness = get_unified_pulse_consciousness()
    consciousness.start_unified_consciousness()
    return consciousness

def stop_dawn_consciousness() -> None:
    """Stop DAWN's unified autonomous consciousness"""
    global _unified_pulse_consciousness
    if _unified_pulse_consciousness:
        _unified_pulse_consciousness.stop_unified_consciousness()

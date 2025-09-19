"""
Enhanced SCUP (System Coherence Under Pressure) Integration
==========================================================

This module enhances the SCUP system to work seamlessly with the pulse-tick
orchestrator, providing real-time coherence tracking, thermal pressure integration,
and autonomous stability maintenance.

Based on DAWN documentation, SCUP is the central metric for:
- Consciousness zone classification
- Thermal regulation decisions  
- Autonomous breathing rhythm control
- System stability assessment
"""

import time
import threading
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# DAWN imports
try:
    from dawn.subsystems.schema.scup_math import (
        compute_enhanced_scup, compute_basic_scup, SCUPInputs, SCUPOutputs
    )
except ImportError:
    try:
        from dawn.subsystems.schema.scup_system import (
            compute_enhanced_scup, compute_basic_scup, SCUPInputs, SCUPOutputs
        )
    except ImportError:
        # Fallback implementations if not available
        class SCUPInputs:
            def __init__(self, alignment=0.5, entropy=0.5, pressure=0.3, **kwargs):
                self.alignment = alignment
                self.entropy = entropy
                self.pressure = pressure
                # Support both pressure and pressure_level for compatibility
                self.pressure_level = pressure
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class SCUPOutputs:
            def __init__(self, scup=0.5, zone="active", tension=0.0, **kwargs):
                self.scup = scup
                self.zone = zone
                self.tension = tension
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        def compute_basic_scup(inputs):
            return max(0.0, min(1.0, inputs.alignment - inputs.entropy + (1.0 - inputs.pressure) * 0.3))
        
        def compute_enhanced_scup(inputs, breathing_phase=0.5, stability_factor=0.7, emergency_active=False):
            base_scup = compute_basic_scup(inputs)
            enhanced = base_scup * stability_factor + breathing_phase * 0.1
            if emergency_active:
                enhanced = max(enhanced, 0.3)
            return SCUPOutputs(
                scup=max(0.0, min(1.0, enhanced)),
                zone="active" if enhanced > 0.5 else "warning",
                tension=abs(enhanced - inputs.entropy),
                breathing_phase=breathing_phase,
                emergency_active=emergency_active
            )

try:
    from dawn.subsystems.schema.scup_tracker import SCUPTracker
except ImportError:
    # Fallback SCUPTracker implementation
    class SCUPTracker:
        def __init__(self, threshold=0.5, cooldown=300.0):
            self.scup = 0.5
            self.threshold = threshold
            self.cooldown = cooldown
            self._last_cooldown = time.time()
        
        def update(self, delta: float) -> None:
            if time.time() - self._last_cooldown >= self.cooldown:
                self.scup = max(0.0, min(1.0, self.scup + delta))
        
        def get(self) -> float:
            return self.scup
        
        def set(self, value: float) -> None:
            self.scup = max(0.0, min(1.0, value))

logger = logging.getLogger(__name__)

class SCUPEmergencyLevel(Enum):
    """Emergency levels based on SCUP values"""
    STABLE = "stable"           # SCUP > 0.7
    CAUTION = "caution"         # SCUP 0.5-0.7
    WARNING = "warning"         # SCUP 0.3-0.5
    CRITICAL = "critical"       # SCUP 0.1-0.3
    EMERGENCY = "emergency"     # SCUP < 0.1

@dataclass
class SCUPMetrics:
    """Comprehensive SCUP metrics for system monitoring"""
    current_scup: float = 0.5
    raw_scup: float = 0.5
    composite_scup: float = 0.5
    recovery_scup: float = 0.5
    
    # Core components
    alignment: float = 0.5
    entropy: float = 0.5
    pressure: float = 0.3
    
    # Enhanced components
    breathing_phase: float = 0.5
    stability_factor: float = 0.7
    emergency_active: bool = False
    
    # Derived metrics
    zone: str = "active"
    tension: float = 0.0
    recovery_potential: float = 0.7
    emergency_level: SCUPEmergencyLevel = SCUPEmergencyLevel.STABLE
    
    # Thermal integration
    thermal_pressure: float = 0.0
    thermal_momentum: float = 0.0
    thermal_zone: str = "🟡 active"
    
    # Historical tracking
    scup_history: deque = field(default_factory=lambda: deque(maxlen=100))
    trend_direction: float = 0.0  # -1 to 1, decreasing to increasing
    stability_trend: float = 0.0
    
    # Performance metrics
    calculation_time: float = 0.0
    update_frequency: float = 1.0
    last_update: float = 0.0

class EnhancedSCUPSystem:
    """
    Enhanced SCUP system that integrates with pulse-tick orchestrator
    and provides comprehensive coherence tracking under pressure.
    """
    
    def __init__(self):
        self.metrics = SCUPMetrics()
        self.running = False
        self.update_interval = 0.1  # 10Hz updates
        self.last_update = time.time()
        
        # Historical tracking
        self.scup_buffer = deque(maxlen=1000)  # 100 seconds of history at 10Hz
        self.pressure_buffer = deque(maxlen=500)  # 50 seconds of pressure history
        self.emergency_events = []
        
        # Integration components
        self.base_scup_tracker = SCUPTracker()
        self.orchestrator_callbacks: List[Callable] = []
        self.emergency_callbacks: List[Callable] = []
        
        # Adaptive parameters
        self.pressure_sensitivity = 1.0
        self.breathing_influence = 0.3
        self.thermal_integration_strength = 0.4
        self.stability_memory_factor = 0.9
        
        # Emergency recovery system
        self.emergency_recovery_active = False
        self.recovery_start_time = 0.0
        self.recovery_target_scup = 0.6
        self.min_stability_time = 5.0  # Minimum time to maintain stability
        
        # Performance tracking
        self.calculation_times = deque(maxlen=100)
        self.update_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("🧠 Enhanced SCUP system initialized")
    
    def start_scup_monitoring(self) -> None:
        """Start continuous SCUP monitoring"""
        if self.running:
            logger.warning("SCUP monitoring already running")
            return
            
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._scup_monitoring_loop,
            daemon=True,
            name="DAWN-SCUP-Monitor"
        )
        self.monitoring_thread.start()
        logger.info("🧠 Started SCUP monitoring")
    
    def stop_scup_monitoring(self) -> None:
        """Stop SCUP monitoring"""
        self.running = False
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=2.0)
        logger.info("🧠 Stopped SCUP monitoring")
    
    def _scup_monitoring_loop(self) -> None:
        """Main SCUP monitoring loop"""
        logger.info("🧠 SCUP monitoring loop started")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Update SCUP metrics
                self._update_scup_metrics()
                
                # Check for emergency conditions
                self._check_emergency_conditions()
                
                # Update trend analysis
                self._update_trend_analysis()
                
                # Notify callbacks
                self._notify_orchestrator_callbacks()
                
                # Performance tracking
                calculation_time = time.time() - start_time
                self.calculation_times.append(calculation_time)
                self.metrics.calculation_time = calculation_time
                
                # Adaptive sleep
                sleep_time = max(0.0, self.update_interval - calculation_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                self.update_count += 1
                
            except Exception as e:
                logger.error(f"🧠 Error in SCUP monitoring loop: {e}")
                time.sleep(0.1)  # Prevent tight error loops
    
    def _update_scup_metrics(self) -> None:
        """Update all SCUP metrics"""
        try:
            with self._lock:
                current_time = time.time()
                
                # Get current inputs (these would come from various DAWN subsystems)
                inputs = self._gather_scup_inputs()
                
                # Calculate different SCUP variants
                self.metrics.raw_scup = compute_basic_scup(inputs)
                
                scup_result = compute_enhanced_scup(
                    inputs,
                    breathing_phase=self.metrics.breathing_phase,
                    stability_factor=self.metrics.stability_factor,
                    emergency_active=self.emergency_recovery_active
                )
                
                # Update metrics from result
                self.metrics.current_scup = scup_result.scup
                self.metrics.composite_scup = scup_result.scup
                self.metrics.zone = scup_result.zone
                self.metrics.tension = scup_result.tension
                self.metrics.recovery_potential = scup_result.recovery_potential
                self.metrics.breathing_phase = scup_result.breathing_phase
                self.metrics.emergency_active = scup_result.emergency_active
                
                # Update emergency level
                self.metrics.emergency_level = self._classify_emergency_level(self.metrics.current_scup)
                
                # Store in history
                self.scup_buffer.append({
                    'timestamp': current_time,
                    'scup': self.metrics.current_scup,
                    'pressure': self.metrics.pressure,
                    'thermal': self.metrics.thermal_pressure,
                    'zone': self.metrics.zone
                })
                
                self.metrics.scup_history.append(self.metrics.current_scup)
                self.metrics.last_update = current_time
                
                # Update frequency calculation
                if len(self.scup_buffer) > 1:
                    recent_times = [entry['timestamp'] for entry in list(self.scup_buffer)[-10:]]
                    if len(recent_times) > 1:
                        time_diffs = np.diff(recent_times)
                        avg_interval = np.mean(time_diffs)
                        self.metrics.update_frequency = 1.0 / avg_interval if avg_interval > 0 else 1.0
                
        except Exception as e:
            logger.error(f"Error updating SCUP metrics: {e}")
    
    def _gather_scup_inputs(self) -> SCUPInputs:
        """Gather inputs for SCUP calculation from DAWN subsystems"""
        try:
            # Default values - in real implementation, these would come from:
            # - Consciousness alignment metrics
            # - Entropy from various subsystems  
            # - Pressure from thermal and cognitive load
            # - Additional entropy sources (mood, sigil, bloom)
            
            inputs = SCUPInputs(
                alignment=self.metrics.alignment,
                entropy=self.metrics.entropy,
                pressure=self.metrics.pressure,
                
                # Enhanced inputs (only if supported by the SCUP implementation)
                **({k: v for k, v in {
                    'mood_entropy': getattr(self.metrics, 'mood_entropy', 0.4),
                    'sigil_entropy': getattr(self.metrics, 'sigil_entropy', 0.3),
                    'bloom_entropy': getattr(self.metrics, 'bloom_entropy', 0.35),
                    'tp_rar': getattr(self.metrics, 'tp_rar', 0.5),
                    'base_coherence': getattr(self.metrics, 'base_coherence', self.metrics.alignment)
                }.items() if k in getattr(SCUPInputs, '__annotations__', {})})
            )
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error gathering SCUP inputs: {e}")
            # Return safe defaults
            return SCUPInputs(
                alignment=0.5,
                entropy=0.5,
                pressure=0.3
            )
    
    def _classify_emergency_level(self, scup: float) -> SCUPEmergencyLevel:
        """Classify emergency level based on SCUP value"""
        if scup >= 0.7:
            return SCUPEmergencyLevel.STABLE
        elif scup >= 0.5:
            return SCUPEmergencyLevel.CAUTION
        elif scup >= 0.3:
            return SCUPEmergencyLevel.WARNING
        elif scup >= 0.1:
            return SCUPEmergencyLevel.CRITICAL
        else:
            return SCUPEmergencyLevel.EMERGENCY
    
    def _check_emergency_conditions(self) -> None:
        """Check for emergency conditions and trigger responses"""
        try:
            current_level = self.metrics.emergency_level
            
            # Critical or emergency conditions
            if current_level in [SCUPEmergencyLevel.CRITICAL, SCUPEmergencyLevel.EMERGENCY]:
                if not self.emergency_recovery_active:
                    self._trigger_emergency_recovery()
            
            # Recovery check
            elif self.emergency_recovery_active and current_level == SCUPEmergencyLevel.STABLE:
                recovery_duration = time.time() - self.recovery_start_time
                if recovery_duration >= self.min_stability_time:
                    self._end_emergency_recovery()
            
            # Notify emergency callbacks
            for callback in self.emergency_callbacks:
                try:
                    callback(current_level, self.metrics)
                except Exception as e:
                    logger.error(f"Error in emergency callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
    
    def _trigger_emergency_recovery(self) -> None:
        """Trigger emergency recovery protocol"""
        try:
            logger.warning(f"🚨 SCUP Emergency Recovery triggered - Level: {self.metrics.emergency_level.value}")
            
            self.emergency_recovery_active = True
            self.recovery_start_time = time.time()
            
            # Record emergency event
            emergency_event = {
                'timestamp': time.time(),
                'scup_value': self.metrics.current_scup,
                'emergency_level': self.metrics.emergency_level.value,
                'pressure': self.metrics.pressure,
                'thermal_pressure': self.metrics.thermal_pressure,
                'trigger_reason': 'low_scup_emergency'
            }
            
            self.emergency_events.append(emergency_event)
            
            # Emergency parameter adjustments
            self.pressure_sensitivity *= 0.5  # Reduce pressure sensitivity
            self.thermal_integration_strength *= 0.7  # Reduce thermal influence
            self.recovery_target_scup = max(0.6, self.metrics.current_scup + 0.3)
            
            logger.info(f"🚨 Emergency recovery active - Target SCUP: {self.recovery_target_scup:.3f}")
            
        except Exception as e:
            logger.error(f"Error triggering emergency recovery: {e}")
    
    def _end_emergency_recovery(self) -> None:
        """End emergency recovery and return to normal operation"""
        try:
            recovery_duration = time.time() - self.recovery_start_time
            logger.info(f"✅ SCUP Emergency recovery complete - Duration: {recovery_duration:.1f}s")
            
            self.emergency_recovery_active = False
            
            # Restore normal parameters
            self.pressure_sensitivity = 1.0
            self.thermal_integration_strength = 0.4
            
        except Exception as e:
            logger.error(f"Error ending emergency recovery: {e}")
    
    def _update_trend_analysis(self) -> None:
        """Update SCUP trend analysis"""
        try:
            if len(self.metrics.scup_history) < 10:
                return
            
            recent_scup = list(self.metrics.scup_history)[-10:]
            
            # Calculate trend direction using linear regression
            x = np.arange(len(recent_scup))
            y = np.array(recent_scup)
            
            if len(x) > 1:
                slope, _ = np.polyfit(x, y, 1)
                self.metrics.trend_direction = np.clip(slope * 10, -1.0, 1.0)  # Scale and clip
            
            # Calculate stability trend (inverse of variance)
            if len(recent_scup) > 2:
                variance = np.var(recent_scup)
                self.metrics.stability_trend = max(0.0, 1.0 - variance * 5)  # Scale variance
            
        except Exception as e:
            logger.error(f"Error updating trend analysis: {e}")
    
    def _notify_orchestrator_callbacks(self) -> None:
        """Notify orchestrator callbacks of SCUP updates"""
        try:
            callback_data = {
                'scup_metrics': self.metrics,
                'emergency_active': self.emergency_recovery_active,
                'update_count': self.update_count,
                'timestamp': time.time()
            }
            
            for callback in self.orchestrator_callbacks:
                try:
                    callback(callback_data)
                except Exception as e:
                    logger.error(f"Error in orchestrator callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying orchestrator callbacks: {e}")
    
    # Public API methods
    
    def update_from_thermal_state(self, thermal_pressure: float, thermal_momentum: float, 
                                 thermal_zone: str) -> None:
        """Update SCUP metrics from thermal state"""
        try:
            with self._lock:
                self.metrics.thermal_pressure = thermal_pressure
                self.metrics.thermal_momentum = thermal_momentum
                self.metrics.thermal_zone = thermal_zone
                
                # Thermal pressure contributes to overall pressure
                thermal_contribution = thermal_pressure * self.thermal_integration_strength
                self.metrics.pressure = min(1.0, 
                    self.metrics.pressure * 0.7 + thermal_contribution * 0.3
                )
                
                # Thermal momentum affects stability
                momentum_factor = 1.0 - abs(thermal_momentum) * 0.2
                self.metrics.stability_factor = max(0.1, 
                    self.metrics.stability_factor * momentum_factor
                )
                
        except Exception as e:
            logger.error(f"Error updating from thermal state: {e}")
    
    def update_from_consciousness_state(self, alignment: float, entropy: float, 
                                      coherence: float) -> None:
        """Update SCUP metrics from consciousness state"""
        try:
            with self._lock:
                self.metrics.alignment = alignment
                self.metrics.entropy = entropy
                
                # Coherence affects stability factor
                self.metrics.stability_factor = max(0.1, min(1.0, 
                    self.metrics.stability_factor * 0.8 + coherence * 0.2
                ))
                
        except Exception as e:
            logger.error(f"Error updating from consciousness state: {e}")
    
    def update_breathing_phase(self, phase_value: float) -> None:
        """Update breathing phase for SCUP calculation"""
        try:
            with self._lock:
                self.metrics.breathing_phase = phase_value
                
        except Exception as e:
            logger.error(f"Error updating breathing phase: {e}")
    
    def get_current_scup(self) -> float:
        """Get current SCUP value"""
        return self.metrics.current_scup
    
    def get_emergency_level(self) -> SCUPEmergencyLevel:
        """Get current emergency level"""
        return self.metrics.emergency_level
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive SCUP system status"""
        try:
            with self._lock:
                avg_calc_time = np.mean(self.calculation_times) if self.calculation_times else 0.0
                
                return {
                    # Core SCUP metrics
                    "current_scup": self.metrics.current_scup,
                    "raw_scup": self.metrics.raw_scup,
                    "composite_scup": self.metrics.composite_scup,
                    "zone": self.metrics.zone,
                    "emergency_level": self.metrics.emergency_level.value,
                    
                    # Components
                    "alignment": self.metrics.alignment,
                    "entropy": self.metrics.entropy,
                    "pressure": self.metrics.pressure,
                    "stability_factor": self.metrics.stability_factor,
                    "breathing_phase": self.metrics.breathing_phase,
                    
                    # Thermal integration
                    "thermal_pressure": self.metrics.thermal_pressure,
                    "thermal_momentum": self.metrics.thermal_momentum,
                    "thermal_zone": self.metrics.thermal_zone,
                    
                    # Trends
                    "trend_direction": self.metrics.trend_direction,
                    "stability_trend": self.metrics.stability_trend,
                    "recovery_potential": self.metrics.recovery_potential,
                    
                    # System state
                    "emergency_recovery_active": self.emergency_recovery_active,
                    "update_frequency": self.metrics.update_frequency,
                    "running": self.running,
                    
                    # Performance
                    "average_calculation_time": avg_calc_time,
                    "total_updates": self.update_count,
                    "emergency_events_count": len(self.emergency_events),
                    "history_size": len(self.scup_buffer),
                    
                    # Integration
                    "orchestrator_callbacks": len(self.orchestrator_callbacks),
                    "emergency_callbacks": len(self.emergency_callbacks)
                }
                
        except Exception as e:
            logger.error(f"Error getting comprehensive status: {e}")
            return {"error": str(e)}
    
    def register_orchestrator_callback(self, callback: Callable) -> None:
        """Register callback for orchestrator integration"""
        self.orchestrator_callbacks.append(callback)
        logger.debug(f"Registered SCUP orchestrator callback: {callback.__name__}")
    
    def register_emergency_callback(self, callback: Callable) -> None:
        """Register callback for emergency conditions"""
        self.emergency_callbacks.append(callback)
        logger.debug(f"Registered SCUP emergency callback: {callback.__name__}")
    
    def force_emergency_recovery(self, reason: str = "manual_trigger") -> None:
        """Force emergency recovery (for testing or manual intervention)"""
        logger.warning(f"🚨 Forced SCUP emergency recovery: {reason}")
        self._trigger_emergency_recovery()
    
    def get_recent_history(self, seconds: int = 60) -> List[Dict[str, Any]]:
        """Get recent SCUP history"""
        try:
            cutoff_time = time.time() - seconds
            recent_history = [
                entry for entry in self.scup_buffer 
                if entry['timestamp'] >= cutoff_time
            ]
            return recent_history
        except Exception as e:
            logger.error(f"Error getting recent history: {e}")
            return []

# Global instance for singleton access
_enhanced_scup_system = None

def get_enhanced_scup_system() -> EnhancedSCUPSystem:
    """Get global EnhancedSCUPSystem instance"""
    global _enhanced_scup_system
    if _enhanced_scup_system is None:
        _enhanced_scup_system = EnhancedSCUPSystem()
    return _enhanced_scup_system

def start_scup_monitoring() -> EnhancedSCUPSystem:
    """Start SCUP monitoring system"""
    scup_system = get_enhanced_scup_system()
    scup_system.start_scup_monitoring()
    return scup_system

def stop_scup_monitoring() -> None:
    """Stop SCUP monitoring system"""
    global _enhanced_scup_system
    if _enhanced_scup_system:
        _enhanced_scup_system.stop_scup_monitoring()

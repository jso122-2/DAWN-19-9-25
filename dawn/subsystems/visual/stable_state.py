#!/usr/bin/env python3
"""
DAWN Stable State Detection and Recovery System
===============================================

Automatic detection and validation of DAWN's stable cognitive states.
Provides continuous monitoring, degradation detection, and automatic recovery.
"""

import time
import json
import threading
import uuid
import logging
import pickle
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
from collections import deque
from pathlib import Path
import weakref
import statistics

try:
    from .stable_state_core import (
        StabilityLevel, RecoveryAction, StabilityMetrics, 
        StableSnapshot, StabilityEvent, calculate_stability_score
    )
except ImportError:
    # Try absolute import if relative fails
    from stable_state_core import (
        StabilityLevel, RecoveryAction, StabilityMetrics, 
        StableSnapshot, StabilityEvent, calculate_stability_score
    )

logger = logging.getLogger(__name__)

class StableStateDetector:
    """
    Automatic detection and validation of DAWN's stable cognitive states.
    
    Continuously monitors system health, detects degradation, captures stable
    snapshots, and performs automatic recovery when needed.
    """
    
    def __init__(self, monitoring_interval: float = 1.0, 
                 snapshot_threshold: float = 0.85,
                 critical_threshold: float = 0.3):
        """
        Initialize the stable state detection system.
        
        Args:
            monitoring_interval: Seconds between stability checks
            snapshot_threshold: Stability score required for golden snapshots
            critical_threshold: Stability score triggering emergency recovery
        """
        self.detector_id = str(uuid.uuid4())
        self.creation_time = time.time()
        
        # Configuration
        self.monitoring_interval = monitoring_interval
        self.snapshot_threshold = snapshot_threshold
        self.critical_threshold = critical_threshold
        
        # Monitoring state
        self.running = False
        self.monitor_thread = None
        self.stability_history = deque(maxlen=1000)
        self.current_metrics = None
        
        # Module connections for monitoring
        self.monitored_modules = {}
        self.module_lock = threading.RLock()
        
        # Stable snapshots management
        self.snapshots = {}  # snapshot_id -> StableSnapshot
        self.snapshot_dir = Path("dawn_stable_states")
        self.snapshot_dir.mkdir(exist_ok=True)
        self.golden_snapshots = deque(maxlen=50)
        
        # Recovery mechanisms
        self.recovery_callbacks = {}
        self.rollback_in_progress = False
        
        # Event tracking
        self.stability_events = deque(maxlen=500)
        self.event_callbacks = []
        
        # Performance metrics
        self.metrics = {
            'stability_checks': 0,
            'snapshots_captured': 0,
            'recoveries_performed': 0,
            'rollbacks_executed': 0,
            'degradations_detected': 0,
            'uptime_seconds': 0
        }
        
        # Stability thresholds and weights
        self.stability_weights = {
            'entropy_stability': 0.20,
            'memory_coherence': 0.20,
            'sigil_cascade_health': 0.15,
            'recursive_depth_safe': 0.15,
            'symbolic_organ_synergy': 0.15,
            'unified_field_coherence': 0.15
        }
        
        # Safe operational limits
        self.safe_limits = {
            'max_entropy_variance': 0.1,
            'min_memory_coherence': 0.8,
            'max_cascade_depth': 5,
            'max_recursive_depth': 8,
            'min_organ_synergy': 0.6,
            'min_field_coherence': 0.7
        }
        
        logger.info(f"🔒 StableStateDetector initialized: {self.detector_id}")
        
    def start_monitoring(self):
        """Start continuous stability monitoring."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="stability_monitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("🔒 Stable state monitoring activated")
        
    def stop_monitoring(self):
        """Stop stability monitoring."""
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
        logger.info("🔒 Stable state monitoring deactivated")
        
    def register_module(self, module_name: str, module_instance: Any,
                       health_callback: Optional[Callable] = None) -> str:
        """
        Register a DAWN module for stability monitoring.
        
        Args:
            module_name: Name of the module
            module_instance: Instance of the module
            health_callback: Function to get module health metrics
            
        Returns:
            Registration ID
        """
        registration_id = f"{module_name}_{int(time.time())}"
        
        with self.module_lock:
            self.monitored_modules[registration_id] = {
                'name': module_name,
                'instance': weakref.ref(module_instance),
                'health_callback': health_callback,
                'registered_at': datetime.now(),
                'last_health_check': None,
                'health_history': deque(maxlen=100)
            }
            
        logger.info(f"🔗 Module registered: {module_name} -> {registration_id}")
        
        return registration_id
        
    def calculate_stability_score(self) -> StabilityMetrics:
        """Calculate comprehensive stability score for the DAWN system."""
        # Collect health data from all registered modules
        module_health = {}
        
        with self.module_lock:
            for reg_id, module_info in self.monitored_modules.items():
                module_instance = module_info['instance']()
                if module_instance is None:
                    continue
                    
                module_name = module_info['name']
                health_data = {}
                
                try:
                    # Try module-specific health callback first
                    if module_info['health_callback']:
                        health_data = module_info['health_callback'](module_instance)
                    else:
                        # Try standard health methods
                        if hasattr(module_instance, 'get_stability_metrics'):
                            health_data = module_instance.get_stability_metrics()
                        elif hasattr(module_instance, 'get_health_status'):
                            health_data = module_instance.get_health_status()
                        elif hasattr(module_instance, 'get_consciousness_state'):
                            health_data = module_instance.get_consciousness_state()
                            
                    module_info['last_health_check'] = datetime.now()
                    module_info['health_history'].append({
                        'timestamp': datetime.now(),
                        'health_data': health_data
                    })
                    
                    module_health[module_name] = health_data
                    
                except Exception as e:
                    logger.warning(f"Failed to get health data from {module_name}: {e}")
                    
        # Calculate using core function
        metrics = calculate_stability_score(module_health, self.safe_limits, self.stability_weights)
        
        # Calculate degradation rate
        metrics.degradation_rate = self._calculate_degradation_rate(metrics.overall_stability)
        
        # Calculate prediction horizon
        metrics.prediction_horizon = self._calculate_prediction_horizon(
            metrics.overall_stability, metrics.degradation_rate
        )
        
        self.metrics['stability_checks'] += 1
        
        return metrics
        
    def _calculate_degradation_rate(self, current_stability: float) -> float:
        """Calculate rate of stability degradation."""
        if len(self.stability_history) < 2:
            return 0.0
            
        # Get recent stability scores
        recent_scores = [m.overall_stability for m in list(self.stability_history)[-10:]]
        recent_scores.append(current_stability)
        
        if len(recent_scores) < 3:
            return 0.0
            
        # Calculate trend slope
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope  # Negative slope indicates degradation
        except:
            return 0.0
            
    def _calculate_prediction_horizon(self, current_stability: float, degradation_rate: float) -> float:
        """Calculate seconds until predicted failure."""
        if degradation_rate >= 0:
            return float('inf')  # No degradation or improving
            
        # Time to reach critical threshold
        stability_margin = current_stability - self.critical_threshold
        if stability_margin <= 0:
            return 0.0  # Already critical
            
        # Estimate time based on degradation rate
        time_to_critical = stability_margin / abs(degradation_rate)
        
        # Convert from monitoring intervals to seconds
        return time_to_critical * self.monitoring_interval
        
    def capture_stable_snapshot(self, metrics: StabilityMetrics, 
                               description: str = "") -> Optional[str]:
        """
        Capture a snapshot of the current stable state.
        
        Args:
            metrics: Current stability metrics
            description: Optional description of the snapshot
            
        Returns:
            Snapshot ID if successful, None otherwise
        """
        if metrics.overall_stability < self.snapshot_threshold:
            return None
            
        snapshot_id = f"stable_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Collect complete system state
        system_state = {}
        module_states = {}
        
        with self.module_lock:
            for reg_id, module_info in self.monitored_modules.items():
                module_instance = module_info['instance']()
                if module_instance is None:
                    continue
                    
                module_name = module_info['name']
                
                try:
                    # Get comprehensive module state
                    if hasattr(module_instance, 'get_full_state'):
                        module_states[module_name] = module_instance.get_full_state()
                    elif hasattr(module_instance, 'get_consciousness_state'):
                        module_states[module_name] = module_instance.get_consciousness_state()
                    elif hasattr(module_instance, '__dict__'):
                        # Fallback to instance variables (filtered)
                        state = {k: v for k, v in module_instance.__dict__.items() 
                                if not k.startswith('_') and not callable(v)}
                        module_states[module_name] = state
                        
                except Exception as e:
                    logger.warning(f"Failed to capture state from {module_name}: {e}")
                    
        # Create system state summary
        system_state = {
            'stability_metrics': asdict(metrics),
            'timestamp': datetime.now().isoformat(),
            'module_count': len(self.monitored_modules),
            'system_uptime': time.time() - self.creation_time
        }
        
        # Create configuration snapshot
        configuration = {
            'monitoring_interval': self.monitoring_interval,
            'snapshot_threshold': self.snapshot_threshold,
            'critical_threshold': self.critical_threshold,
            'stability_weights': self.stability_weights,
            'safe_limits': self.safe_limits
        }
        
        # Calculate state hash for integrity
        state_data = json.dumps({
            'system_state': system_state,
            'module_states': module_states,
            'configuration': configuration
        }, sort_keys=True, default=str)
        
        state_hash = hashlib.sha256(state_data.encode()).hexdigest()
        
        # Create snapshot
        snapshot = StableSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            stability_score=metrics.overall_stability,
            system_state=system_state,
            module_states=module_states,
            configuration=configuration,
            state_hash=state_hash,
            description=description
        )
        
        # Store snapshot
        self.snapshots[snapshot_id] = snapshot
        self.golden_snapshots.append(snapshot_id)
        
        # Save to disk
        snapshot_file = self.snapshot_dir / f"{snapshot_id}.json"
        try:
            with open(snapshot_file, 'w') as f:
                json.dump(asdict(snapshot), f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save snapshot {snapshot_id}: {e}")
            
        self.metrics['snapshots_captured'] += 1
        
        logger.info(f"📸 Stable snapshot captured: {snapshot_id}")
        logger.info(f"   📊 Stability score: {metrics.overall_stability:.3f}")
        
        return snapshot_id
        
    def detect_degradation(self, metrics: StabilityMetrics) -> Optional[StabilityEvent]:
        """
        Detect system degradation and determine recovery actions.
        
        Args:
            metrics: Current stability metrics
            
        Returns:
            Stability event if action is needed, None otherwise
        """
        # Check for immediate critical conditions
        if metrics.stability_level == StabilityLevel.CRITICAL:
            return self._create_stability_event(
                'critical_failure',
                metrics,
                RecoveryAction.EMERGENCY_STABILIZE
            )
            
        # Check for degradation trends
        if metrics.degradation_rate < -0.1:  # Rapid degradation
            if metrics.prediction_horizon < 30:  # Failure predicted soon
                return self._create_stability_event(
                    'degradation_critical',
                    metrics,
                    RecoveryAction.AUTO_ROLLBACK
                )
            elif metrics.prediction_horizon < 60:
                return self._create_stability_event(
                    'degradation_warning',
                    metrics,
                    RecoveryAction.SOFT_RESET
                )
                
        # Check for specific system failures
        if metrics.failing_systems:
            if len(metrics.failing_systems) >= 3:
                return self._create_stability_event(
                    'multiple_system_failure',
                    metrics,
                    RecoveryAction.AUTO_ROLLBACK
                )
            elif any('infinite_loops' in system for system in metrics.failing_systems):
                return self._create_stability_event(
                    'infinite_loop_detected',
                    metrics,
                    RecoveryAction.SELECTIVE_RESTART
                )
            else:
                return self._create_stability_event(
                    'system_failure',
                    metrics,
                    RecoveryAction.SOFT_RESET
                )
                
        # Check for unstable conditions
        if metrics.stability_level == StabilityLevel.UNSTABLE:
            return self._create_stability_event(
                'unstable_state',
                metrics,
                RecoveryAction.GRACEFUL_DEGRADATION
            )
            
        return None
        
    def _create_stability_event(self, event_type: str, metrics: StabilityMetrics,
                               recovery_action: RecoveryAction) -> StabilityEvent:
        """Create a stability event record."""
        event_id = str(uuid.uuid4())
        
        # Determine rollback target if needed
        rollback_target = None
        if recovery_action == RecoveryAction.AUTO_ROLLBACK and self.golden_snapshots:
            # Find most recent stable snapshot
            for snapshot_id in reversed(self.golden_snapshots):
                snapshot = self.snapshots.get(snapshot_id)
                if snapshot and snapshot.stability_score >= self.snapshot_threshold:
                    rollback_target = snapshot_id
                    break
                    
        event = StabilityEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            stability_score=metrics.overall_stability,
            failing_systems=metrics.failing_systems.copy(),
            degradation_rate=metrics.degradation_rate,
            recovery_action=recovery_action,
            rollback_target=rollback_target,
            details={
                'stability_level': metrics.stability_level.name,
                'warning_systems': metrics.warning_systems.copy(),
                'prediction_horizon': metrics.prediction_horizon
            }
        )
        
        self.stability_events.append(event)
        self.metrics['degradations_detected'] += 1
        
        return event
        
    def execute_recovery(self, event: StabilityEvent) -> bool:
        """
        Execute recovery action for a stability event.
        
        Args:
            event: Stability event requiring recovery
            
        Returns:
            True if recovery was successful
        """
        try:
            success = False
            
            if event.recovery_action == RecoveryAction.MONITOR:
                success = True  # Just continue monitoring
                
            elif event.recovery_action == RecoveryAction.SOFT_RESET:
                success = self._execute_soft_reset(event.failing_systems)
                
            elif event.recovery_action == RecoveryAction.SELECTIVE_RESTART:
                success = self._execute_selective_restart(event.failing_systems)
                
            # Update event with result
            event.success = success
            
            if success:
                self.metrics['recoveries_performed'] += 1
                logger.info(f"✅ Recovery successful: {event.recovery_action.value}")
            else:
                logger.error(f"❌ Recovery failed: {event.recovery_action.value}")
                
            return success
                
        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            event.success = False
            return False
            
    def _execute_soft_reset(self, failing_systems: List[str]) -> bool:
        """Execute soft reset of failing systems."""
        success = True
        
        with self.module_lock:
            for reg_id, module_info in self.monitored_modules.items():
                module_name = module_info['name']
                
                if module_name in failing_systems:
                    module_instance = module_info['instance']()
                    if module_instance is None:
                        continue
                        
                    try:
                        # Try various reset methods
                        if hasattr(module_instance, 'soft_reset'):
                            module_instance.soft_reset()
                        elif hasattr(module_instance, 'reset'):
                            module_instance.reset()
                        elif hasattr(module_instance, 'reset_state'):
                            module_instance.reset_state()
                        else:
                            logger.warning(f"No reset method found for {module_name}")
                            success = False
                            
                        logger.info(f"🔄 Soft reset: {module_name}")
                        
                    except Exception as e:
                        logger.error(f"Soft reset failed for {module_name}: {e}")
                        success = False
                        
        return success
        
    def _execute_selective_restart(self, failing_systems: List[str]) -> bool:
        """Execute selective restart of failing components."""
        success = True
        
        # Execute registered recovery callbacks
        for system in failing_systems:
            if system in self.recovery_callbacks:
                try:
                    callback = self.recovery_callbacks[system]
                    result = callback()
                    
                    if result:
                        logger.info(f"🔄 Selective restart successful: {system}")
                    else:
                        logger.error(f"🔄 Selective restart failed: {system}")
                        success = False
                        
                except Exception as e:
                    logger.error(f"Selective restart callback failed for {system}: {e}")
                    success = False
            else:
                logger.warning(f"No recovery callback registered for {system}")
                success = False
                
        return success

    def register_recovery_callback(self, system_name: str, callback: Callable) -> None:
        """Register a recovery callback for a specific system."""
        self.recovery_callbacks[system_name] = callback
        logger.info(f"🔧 Recovery callback registered: {system_name}")
        
    def get_stability_status(self) -> Dict[str, Any]:
        """Get current stability status and metrics."""
        return {
            "detector_id": self.detector_id,
            "running": self.running,
            "monitored_modules": len(self.monitored_modules),
            "golden_snapshots": len(self.golden_snapshots),
            "recent_events": len([e for e in self.stability_events if e.timestamp > datetime.now() - timedelta(hours=1)]),
            "rollback_in_progress": self.rollback_in_progress,
            "uptime_seconds": time.time() - self.creation_time,
            "metrics": dict(self.metrics)
        }
        
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        logger.info("🔒 Stability monitoring loop started")
        
        while self.running:
            try:
                # Calculate current stability
                metrics = self.calculate_stability_score()
                self.current_metrics = metrics
                self.stability_history.append(metrics)
                
                # Check for golden snapshot opportunity
                if metrics.overall_stability >= self.snapshot_threshold:
                    should_capture = True
                    
                    if self.golden_snapshots:
                        recent_snapshot_id = self.golden_snapshots[-1]
                        recent_snapshot = self.snapshots.get(recent_snapshot_id)
                        
                        if recent_snapshot:
                            time_since_last = (datetime.now() - recent_snapshot.timestamp).total_seconds()
                            score_improvement = metrics.overall_stability - recent_snapshot.stability_score
                            
                            should_capture = (score_improvement > 0.05) or (time_since_last > 300)
                            
                    if should_capture:
                        self.capture_stable_snapshot(metrics, "Automatic golden snapshot")
                        
                # Check for degradation
                stability_event = self.detect_degradation(metrics)
                
                if stability_event:
                    logger.warning(f"🚨 Stability issue: {stability_event.event_type}")
                    logger.warning(f"   📊 Score: {stability_event.stability_score:.3f}")
                    logger.warning(f"   🛠️ Recovery: {stability_event.recovery_action.value}")
                    
                    if stability_event.recovery_action != RecoveryAction.MONITOR:
                        self.execute_recovery(stability_event)
                        
                self.metrics["uptime_seconds"] = time.time() - self.creation_time
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)


# Global instance
_global_detector = None
_detector_lock = threading.Lock()

def get_stable_state_detector(auto_start: bool = True) -> StableStateDetector:
    """Get the global stable state detector instance."""
    global _global_detector
    
    with _detector_lock:
        if _global_detector is None:
            _global_detector = StableStateDetector()
            if auto_start:
                _global_detector.start_monitoring()
                
    return _global_detector

def register_module_for_stability(module_name: str, module_instance: Any,
                                 health_callback: Optional[Callable] = None) -> str:
    """Convenience function to register a module for stability monitoring."""
    detector = get_stable_state_detector()
    return detector.register_module(module_name, module_instance, health_callback)


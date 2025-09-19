#!/usr/bin/env python3
"""
🚨 Failure Mode Monitoring & Safeguard System
═════════════════════════════════════════════

Comprehensive failure mode detection and safeguard coordination for DAWN.
Monitors all documented failure modes and coordinates appropriate responses.

"Failure modes include rebloom stalls, over-rebloom floods, shimmer collapse,
entropy drift, tracer floods, false positives, and persistence drift."

Based on documentation: Fractal Memory/Failure Modes & Safeguards + Logs & Telemetry.rtf
"""

import logging
import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import uuid

logger = logging.getLogger(__name__)

class FailureMode(Enum):
    """All documented failure modes in DAWN"""
    REBLOOM_STALL = "rebloom_stall"           # Memory fails to rebloom despite access
    OVER_REBLOOM = "over_rebloom"             # Excess reblooming floods garden with shimmer
    SHIMMER_COLLAPSE = "shimmer_collapse"     # Shimmer decay accelerates too fast
    ENTROPY_DRIFT = "entropy_drift"           # Blooms with extreme entropy distort landscape
    TRACER_FLOOD = "tracer_flood"             # Too many tracers spawn simultaneously
    TRACER_FALSE_POSITIVE = "tracer_false_positive"  # Tracers over-report noise as signals
    TRACER_STARVATION = "tracer_starvation"   # Tracers fail to gather enough signals
    PERSISTENCE_DRIFT = "persistence_drift"   # Persistent tracers cling too long
    CACHE_THRASHING = "cache_thrashing"       # CARRIN cache in excessive turbulence
    MYCELIAL_STAGNATION = "mycelial_stagnation"  # Mycelial layer stops growing
    ASH_SOOT_IMBALANCE = "ash_soot_imbalance" # Toxic soot accumulation
    SCHEMA_FRAGMENTATION = "schema_fragmentation"  # Schema edges breaking down

class SafeguardLevel(Enum):
    """Levels of safeguard intervention"""
    MONITOR = "monitor"           # Just watch and log
    THROTTLE = "throttle"         # Reduce activity rates
    REDIRECT = "redirect"         # Route around problem
    EMERGENCY_STOP = "emergency_stop"  # Halt problematic subsystem
    SYSTEM_RESET = "system_reset" # Full subsystem restart

class FailureAlert(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class FailureDetection:
    """A detected failure mode instance"""
    failure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    failure_mode: FailureMode = FailureMode.REBLOOM_STALL
    severity: FailureAlert = FailureAlert.WARNING
    detection_time: float = field(default_factory=time.time)
    affected_systems: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    safeguards_triggered: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[float] = None
    
@dataclass
class SafeguardAction:
    """A safeguard action to take"""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    failure_mode: FailureMode = FailureMode.REBLOOM_STALL
    level: SafeguardLevel = SafeguardLevel.MONITOR
    target_system: str = ""
    action_function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    cooldown_seconds: float = 60.0
    last_triggered: Optional[float] = None

class FailureModeMonitor:
    """
    Central failure mode monitoring system that watches for all documented
    failure modes and coordinates appropriate safeguard responses.
    """
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 alert_retention_hours: int = 24):
        
        self.monitoring_interval = monitoring_interval
        self.alert_retention_hours = alert_retention_hours
        
        # Failure detection state
        self.active_failures: Dict[str, FailureDetection] = {}
        self.failure_history: deque = deque(maxlen=10000)
        self.system_metrics: Dict[str, Any] = {}
        
        # Safeguard system
        self.safeguard_actions: Dict[FailureMode, List[SafeguardAction]] = defaultdict(list)
        self.safeguard_stats = defaultdict(int)
        
        # Monitoring thresholds (from documentation)
        self.thresholds = {
            # Rebloom system thresholds
            'rebloom_stall_threshold': 10,  # Failed rebloom attempts before alert
            'over_rebloom_rate_limit': 5,   # Max reblooms per tick
            'juliet_density_max': 0.8,      # Max Juliet flower density
            
            # Shimmer system thresholds
            'shimmer_collapse_rate': 0.1,   # Max shimmer decay rate
            'ghost_trace_retention_min': 100,  # Min ghost traces to keep
            
            # Entropy thresholds
            'entropy_outlier_threshold': 2.0,  # Standard deviations from mean
            'entropy_normalization_trigger': 0.9,  # When to normalize
            
            # Tracer system thresholds
            'tracer_spawn_rate_limit': 20,  # Max tracers per tick
            'tracer_false_positive_rate': 0.3,  # Max false positive rate
            'tracer_signal_threshold': 0.1,  # Min signals required
            'persistence_max_ticks': 1000,  # Max tracer persistence
            
            # Cache system thresholds
            'cache_hit_rate_min': 0.6,     # Min acceptable hit rate
            'turbulence_duration_max': 30,  # Max seconds in turbulent state
            
            # Mycelial thresholds
            'growth_stagnation_ticks': 50,  # Ticks without growth = stagnation
            'nutrient_starvation_threshold': 0.1,  # Min nutrient level
            
            # Ash/Soot thresholds
            'soot_ash_ratio_max': 2.0,     # Max toxic ratio
            'soot_accumulation_max': 1000   # Max soot particles
        }
        
        # Statistics
        self.stats = {
            'total_failures_detected': 0,
            'failures_by_type': defaultdict(int),
            'safeguards_triggered': 0,
            'false_alarms': 0,
            'system_uptime': time.time(),
            'monitoring_cycles': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        # Initialize safeguards
        self._initialize_safeguards()
        
        logger.info(f"🚨 FailureModeMonitor initialized - monitoring interval: {monitoring_interval}s")
    
    def _initialize_safeguards(self):
        """Initialize all documented safeguard actions"""
        
        # Rebloom stall safeguards
        self.add_safeguard(
            FailureMode.REBLOOM_STALL,
            SafeguardLevel.REDIRECT,
            "rebloom_engine",
            self._handle_rebloom_stall,
            {"fallback_to_fractal": True, "lower_pressure": True}
        )
        
        # Over-rebloom safeguards
        self.add_safeguard(
            FailureMode.OVER_REBLOOM,
            SafeguardLevel.THROTTLE,
            "rebloom_engine",
            self._handle_over_rebloom,
            {"rate_limit": 3, "shi_monitoring": True}
        )
        
        # Shimmer collapse safeguards
        self.add_safeguard(
            FailureMode.SHIMMER_COLLAPSE,
            SafeguardLevel.REDIRECT,
            "shimmer_decay_engine",
            self._handle_shimmer_collapse,
            {"retain_ghost_traces": True, "nutrient_reallocation": True}
        )
        
        # Entropy drift safeguards
        self.add_safeguard(
            FailureMode.ENTROPY_DRIFT,
            SafeguardLevel.REDIRECT,
            "entropy_analyzer",
            self._handle_entropy_drift,
            {"normalize_outliers": True, "route_to_soot": True}
        )
        
        # Tracer flood safeguards
        self.add_safeguard(
            FailureMode.TRACER_FLOOD,
            SafeguardLevel.THROTTLE,
            "tracer_manager",
            self._handle_tracer_flood,
            {"rate_limit_spawns": True, "emergency_retirement": True}
        )
        
        # Add more safeguards for other failure modes...
        self._add_remaining_safeguards()
    
    def add_safeguard(self,
                     failure_mode: FailureMode,
                     level: SafeguardLevel,
                     target_system: str,
                     action_function: Callable,
                     parameters: Dict[str, Any],
                     cooldown: float = 60.0):
        """Add a safeguard action for a failure mode"""
        
        safeguard = SafeguardAction(
            failure_mode=failure_mode,
            level=level,
            target_system=target_system,
            action_function=action_function,
            parameters=parameters,
            cooldown_seconds=cooldown
        )
        
        self.safeguard_actions[failure_mode].append(safeguard)
        logger.debug(f"🚨 Added safeguard for {failure_mode.value}: {level.value} on {target_system}")
    
    def update_system_metrics(self, system_name: str, metrics: Dict[str, Any]):
        """Update metrics for a system to monitor"""
        with self._lock:
            self.system_metrics[system_name] = {
                **metrics,
                'last_update': time.time()
            }
    
    def detect_failure(self, 
                      failure_mode: FailureMode,
                      severity: FailureAlert,
                      affected_systems: List[str],
                      metrics: Dict[str, Any]) -> FailureDetection:
        """Manually report a detected failure"""
        
        with self._lock:
            detection = FailureDetection(
                failure_mode=failure_mode,
                severity=severity,
                affected_systems=affected_systems,
                metrics=metrics
            )
            
            self.active_failures[detection.failure_id] = detection
            self.failure_history.append(detection)
            self.stats['total_failures_detected'] += 1
            self.stats['failures_by_type'][failure_mode.value] += 1
            
            logger.warning(f"🚨 Failure detected: {failure_mode.value} - {severity.value}")
            
            # Trigger safeguards
            self._trigger_safeguards(detection)
            
            return detection
    
    def resolve_failure(self, failure_id: str, resolution_notes: str = ""):
        """Mark a failure as resolved"""
        with self._lock:
            if failure_id in self.active_failures:
                failure = self.active_failures[failure_id]
                failure.resolved = True
                failure.resolution_time = time.time()
                
                del self.active_failures[failure_id]
                
                logger.info(f"🚨 Failure resolved: {failure.failure_mode.value} - {resolution_notes}")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                start_time = time.time()
                
                # Check all failure modes
                self._check_rebloom_failures()
                self._check_shimmer_failures()
                self._check_entropy_failures()
                self._check_tracer_failures()
                self._check_cache_failures()
                self._check_mycelial_failures()
                self._check_ash_soot_failures()
                
                # Clean up old failures
                self._cleanup_old_failures()
                
                self.stats['monitoring_cycles'] += 1
                
                # Sleep for remainder of interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in failure monitoring: {e}")
                time.sleep(1.0)
    
    def _check_rebloom_failures(self):
        """Check for rebloom-related failures"""
        rebloom_metrics = self.system_metrics.get('rebloom_engine', {})
        
        # Check for rebloom stalls
        failed_attempts = rebloom_metrics.get('failed_rebloom_attempts', 0)
        if failed_attempts >= self.thresholds['rebloom_stall_threshold']:
            self.detect_failure(
                FailureMode.REBLOOM_STALL,
                FailureAlert.WARNING,
                ['rebloom_engine'],
                {'failed_attempts': failed_attempts}
            )
        
        # Check for over-rebloom
        rebloom_rate = rebloom_metrics.get('reblooms_per_tick', 0)
        if rebloom_rate > self.thresholds['over_rebloom_rate_limit']:
            self.detect_failure(
                FailureMode.OVER_REBLOOM,
                FailureAlert.CRITICAL,
                ['rebloom_engine', 'memory_system'],
                {'rebloom_rate': rebloom_rate}
            )
    
    def _check_shimmer_failures(self):
        """Check for shimmer-related failures"""
        shimmer_metrics = self.system_metrics.get('shimmer_decay_engine', {})
        
        # Check for shimmer collapse
        decay_rate = shimmer_metrics.get('average_decay_rate', 0)
        if decay_rate > self.thresholds['shimmer_collapse_rate']:
            self.detect_failure(
                FailureMode.SHIMMER_COLLAPSE,
                FailureAlert.WARNING,
                ['shimmer_decay_engine'],
                {'decay_rate': decay_rate}
            )
    
    def _check_entropy_failures(self):
        """Check for entropy-related failures"""
        entropy_metrics = self.system_metrics.get('entropy_analyzer', {})
        
        # Check for entropy drift
        entropy_outliers = entropy_metrics.get('outlier_count', 0)
        if entropy_outliers > 0:
            self.detect_failure(
                FailureMode.ENTROPY_DRIFT,
                FailureAlert.WARNING,
                ['entropy_analyzer', 'bloom_system'],
                {'outlier_count': entropy_outliers}
            )
    
    def _check_tracer_failures(self):
        """Check for tracer-related failures"""
        tracer_metrics = self.system_metrics.get('tracer_manager', {})
        
        # Check for tracer flood
        spawn_rate = tracer_metrics.get('tracers_spawned_per_tick', 0)
        if spawn_rate > self.thresholds['tracer_spawn_rate_limit']:
            self.detect_failure(
                FailureMode.TRACER_FLOOD,
                FailureAlert.CRITICAL,
                ['tracer_manager'],
                {'spawn_rate': spawn_rate}
            )
        
        # Check for false positives
        false_positive_rate = tracer_metrics.get('false_positive_rate', 0)
        if false_positive_rate > self.thresholds['tracer_false_positive_rate']:
            self.detect_failure(
                FailureMode.TRACER_FALSE_POSITIVE,
                FailureAlert.WARNING,
                ['tracer_manager'],
                {'false_positive_rate': false_positive_rate}
            )
    
    def _check_cache_failures(self):
        """Check for cache-related failures"""
        cache_metrics = self.system_metrics.get('carrin_cache', {})
        
        # Check for cache thrashing
        hit_rate = cache_metrics.get('hit_rate', 1.0)
        if hit_rate < self.thresholds['cache_hit_rate_min']:
            self.detect_failure(
                FailureMode.CACHE_THRASHING,
                FailureAlert.WARNING,
                ['carrin_cache'],
                {'hit_rate': hit_rate}
            )
    
    def _check_mycelial_failures(self):
        """Check for mycelial layer failures"""
        mycelial_metrics = self.system_metrics.get('mycelial_layer', {})
        
        # Check for growth stagnation
        ticks_without_growth = mycelial_metrics.get('ticks_without_growth', 0)
        if ticks_without_growth > self.thresholds['growth_stagnation_ticks']:
            self.detect_failure(
                FailureMode.MYCELIAL_STAGNATION,
                FailureAlert.WARNING,
                ['mycelial_layer'],
                {'stagnation_ticks': ticks_without_growth}
            )
    
    def _check_ash_soot_failures(self):
        """Check for ash/soot balance failures"""
        ash_soot_metrics = self.system_metrics.get('ash_soot_engine', {})
        
        # Check for soot imbalance
        soot_ash_ratio = ash_soot_metrics.get('soot_ash_ratio', 0)
        if soot_ash_ratio > self.thresholds['soot_ash_ratio_max']:
            self.detect_failure(
                FailureMode.ASH_SOOT_IMBALANCE,
                FailureAlert.CRITICAL,
                ['ash_soot_engine'],
                {'soot_ash_ratio': soot_ash_ratio}
            )
    
    def _trigger_safeguards(self, failure: FailureDetection):
        """Trigger appropriate safeguards for a failure"""
        safeguards = self.safeguard_actions.get(failure.failure_mode, [])
        
        for safeguard in safeguards:
            # Check cooldown
            if (safeguard.last_triggered and 
                time.time() - safeguard.last_triggered < safeguard.cooldown_seconds):
                continue
            
            try:
                # Execute safeguard action
                if safeguard.action_function:
                    result = safeguard.action_function(failure, safeguard.parameters)
                    failure.safeguards_triggered.append(safeguard.action_id)
                    safeguard.last_triggered = time.time()
                    self.safeguard_stats[safeguard.level.value] += 1
                    self.stats['safeguards_triggered'] += 1
                    
                    logger.info(f"🚨 Safeguard triggered: {safeguard.level.value} for {failure.failure_mode.value}")
                
            except Exception as e:
                logger.error(f"Error executing safeguard {safeguard.action_id}: {e}")
    
    def _handle_rebloom_stall(self, failure: FailureDetection, params: Dict[str, Any]) -> bool:
        """Handle rebloom stall safeguard"""
        logger.info("🚨 Activating rebloom stall safeguard - fallback to direct fractal retrieval")
        # Implementation would coordinate with rebloom engine
        return True
    
    def _handle_over_rebloom(self, failure: FailureDetection, params: Dict[str, Any]) -> bool:
        """Handle over-rebloom safeguard"""
        logger.info("🚨 Activating over-rebloom safeguard - capping rebloom rate")
        # Implementation would throttle rebloom engine
        return True
    
    def _handle_shimmer_collapse(self, failure: FailureDetection, params: Dict[str, Any]) -> bool:
        """Handle shimmer collapse safeguard"""
        logger.info("🚨 Activating shimmer collapse safeguard - retaining ghost traces")
        # Implementation would coordinate with shimmer decay engine
        return True
    
    def _handle_entropy_drift(self, failure: FailureDetection, params: Dict[str, Any]) -> bool:
        """Handle entropy drift safeguard"""
        logger.info("🚨 Activating entropy drift safeguard - normalizing outliers")
        # Implementation would coordinate with entropy analyzer
        return True
    
    def _handle_tracer_flood(self, failure: FailureDetection, params: Dict[str, Any]) -> bool:
        """Handle tracer flood safeguard"""
        logger.info("🚨 Activating tracer flood safeguard - rate limiting spawns")
        # Implementation would coordinate with tracer manager
        return True
    
    def _add_remaining_safeguards(self):
        """Add safeguards for remaining failure modes"""
        # Add safeguards for other failure modes as needed
        pass
    
    def _cleanup_old_failures(self):
        """Clean up old resolved failures from history"""
        cutoff_time = time.time() - (self.alert_retention_hours * 3600)
        
        # Remove old items from history
        while (self.failure_history and 
               self.failure_history[0].detection_time < cutoff_time):
            self.failure_history.popleft()
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        with self._lock:
            return {
                'active_failures': len(self.active_failures),
                'failure_breakdown': {
                    failure.failure_mode.value: {
                        'severity': failure.severity.value,
                        'affected_systems': failure.affected_systems,
                        'duration': time.time() - failure.detection_time
                    }
                    for failure in self.active_failures.values()
                },
                'statistics': self.stats,
                'safeguard_stats': dict(self.safeguard_stats),
                'system_uptime': time.time() - self.stats['system_uptime'],
                'monitoring_health': {
                    'cycles_completed': self.stats['monitoring_cycles'],
                    'last_metrics_update': {
                        system: metrics.get('last_update', 0)
                        for system, metrics in self.system_metrics.items()
                    }
                }
            }
    
    def shutdown(self):
        """Shutdown the failure mode monitor"""
        self._monitoring_active = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("🚨 FailureModeMonitor shutdown complete")


# Global failure mode monitor instance
_failure_monitor = None

def get_failure_monitor(config: Optional[Dict[str, Any]] = None) -> FailureModeMonitor:
    """Get the global failure mode monitor instance"""
    global _failure_monitor
    if _failure_monitor is None:
        config = config or {}
        _failure_monitor = FailureModeMonitor(
            monitoring_interval=config.get('monitoring_interval', 1.0),
            alert_retention_hours=config.get('alert_retention_hours', 24)
        )
    return _failure_monitor


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize failure monitor
    monitor = FailureModeMonitor()
    
    # Simulate some system metrics
    monitor.update_system_metrics('rebloom_engine', {
        'failed_rebloom_attempts': 5,
        'reblooms_per_tick': 2,
        'active_juliet_flowers': 50
    })
    
    monitor.update_system_metrics('tracer_manager', {
        'tracers_spawned_per_tick': 25,  # This will trigger tracer flood
        'false_positive_rate': 0.1,
        'active_tracers': 200
    })
    
    # Let it run for monitoring
    time.sleep(10)
    
    # Get health report
    health = monitor.get_system_health_report()
    print(f"System health: {json.dumps(health, indent=2)}")
    
    # Shutdown
    monitor.shutdown()

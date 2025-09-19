#!/usr/bin/env python3
"""
DAWN Symbolic Failure Detection and Monitoring
==============================================

Implementation of comprehensive symbolic failure detection and monitoring
for the DAWN sigil system. Detects sigil drift, broken house operations,
containment breaches, and other symbolic failures as documented.

Based on DAWN's documented failure detection architecture.
"""

import time
import logging
import json
import threading
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

from core.schema_anomaly_logger import log_anomaly, AnomalySeverity
from schema.registry import registry
from schema.sigil_glyph_codex import SigilHouse, sigil_glyph_codex
from schema.archetypal_house_operations import HOUSE_OPERATORS
from schema.tracer_house_alignment import tracer_house_alignment
from rhizome.propagation import emit_signal, SignalType
from utils.metrics_collector import metrics

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of symbolic failures"""
    SIGIL_DRIFT = "sigil_drift"                     # Sigil meaning drift over time
    BROKEN_HOUSE = "broken_house"                   # House operation failure
    CONTAINMENT_BREACH = "containment_breach"       # Ring containment failure
    TRACER_MISALIGNMENT = "tracer_misalignment"     # Tracer-house misalignment
    GLYPH_CORRUPTION = "glyph_corruption"           # Glyph symbol corruption
    RESONANCE_DECAY = "resonance_decay"             # Mythic resonance decay
    SYMBOLIC_INCOHERENCE = "symbolic_incoherence"   # Loss of symbolic meaning
    ARCHETYPAL_BREAKDOWN = "archetypal_breakdown"   # House archetype failure
    RING_OVERLOAD = "ring_overload"                 # System overload
    EMERGENCY_SEAL_TRIGGER = "emergency_seal_trigger" # Emergency containment

class FailureSeverity(Enum):
    """Severity levels for symbolic failures"""
    LOW = "low"           # Minor degradation
    MEDIUM = "medium"     # Noticeable impact
    HIGH = "high"         # Significant dysfunction
    CRITICAL = "critical" # System-threatening
    EMERGENCY = "emergency" # Immediate intervention required

@dataclass
class SymbolicFailure:
    """A detected symbolic failure"""
    failure_id: str
    failure_type: FailureType
    severity: FailureSeverity
    description: str
    affected_components: List[str] = field(default_factory=list)
    detection_timestamp: float = field(default_factory=time.time)
    symptoms: List[str] = field(default_factory=list)
    potential_causes: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None
    resolution_notes: Optional[str] = None
    
    def get_age_seconds(self) -> float:
        """Get age of failure in seconds"""
        return time.time() - self.detection_timestamp
    
    def is_stale(self, max_age_seconds: float = 3600) -> bool:
        """Check if failure is stale (older than max age)"""
        return self.get_age_seconds() > max_age_seconds

@dataclass
class DetectionMetrics:
    """Metrics for failure detection system"""
    total_failures_detected: int = 0
    failures_by_type: Dict[FailureType, int] = field(default_factory=lambda: defaultdict(int))
    failures_by_severity: Dict[FailureSeverity, int] = field(default_factory=lambda: defaultdict(int))
    false_positives: int = 0
    true_positives: int = 0
    detection_latency_ms: float = 0.0
    resolution_time_avg_seconds: float = 0.0
    
    def get_accuracy_rate(self) -> float:
        """Get detection accuracy rate"""
        total = self.true_positives + self.false_positives
        return (self.true_positives / total * 100) if total > 0 else 0.0

class SymbolicHealthMonitor:
    """Monitor for overall symbolic system health"""
    
    def __init__(self):
        self.health_scores: Dict[str, float] = {}
        self.health_history: deque = deque(maxlen=1000)
        self.baseline_metrics: Dict[str, float] = {}
        self.alert_thresholds: Dict[str, float] = {
            "overall_health": 0.6,
            "sigil_coherence": 0.5,
            "house_performance": 0.4,
            "ring_stability": 0.3,
            "tracer_alignment": 0.7
        }
    
    def update_health_score(self, component: str, score: float):
        """Update health score for a component"""
        self.health_scores[component] = max(0.0, min(1.0, score))
        
        # Log health change
        self.health_history.append({
            "timestamp": time.time(),
            "component": component,
            "score": score
        })
    
    def get_overall_health(self) -> float:
        """Calculate overall symbolic system health"""
        if not self.health_scores:
            return 1.0
        
        # Weighted health calculation
        weights = {
            "sigil_coherence": 0.25,
            "house_performance": 0.25,
            "ring_stability": 0.20,
            "tracer_alignment": 0.15,
            "glyph_integrity": 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for component, weight in weights.items():
            if component in self.health_scores:
                weighted_sum += self.health_scores[component] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 1.0
    
    def check_health_alerts(self) -> List[str]:
        """Check for health-based alerts"""
        alerts = []
        
        overall_health = self.get_overall_health()
        if overall_health < self.alert_thresholds["overall_health"]:
            alerts.append(f"Overall symbolic health degraded: {overall_health:.3f}")
        
        for component, score in self.health_scores.items():
            threshold = self.alert_thresholds.get(component)
            if threshold and score < threshold:
                alerts.append(f"{component} health below threshold: {score:.3f} < {threshold}")
        
        return alerts

class SymbolicFailureDetector:
    """
    Symbolic Failure Detection and Monitoring System
    
    Comprehensive monitoring and detection of symbolic failures in the DAWN
    sigil system. Implements pattern recognition, health monitoring, and
    automated failure detection as documented.
    """
    
    def __init__(self):
        self.active_failures: Dict[str, SymbolicFailure] = {}
        self.failure_history: deque = deque(maxlen=5000)
        self.detection_metrics = DetectionMetrics()
        self.health_monitor = SymbolicHealthMonitor()
        
        # Detection patterns and thresholds
        self.drift_thresholds = {
            "resonance_decay_rate": 0.1,
            "meaning_coherence_loss": 0.2,
            "archetypal_deviation": 0.15
        }
        
        # Monitoring threads
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.detection_executor = ThreadPoolExecutor(max_workers=4)
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[SymbolicFailure], None]] = []
        
        # Register with schema registry
        self._register()
        
        logger.info("🔍 Symbolic Failure Detection System initialized")
        logger.info(f"   Monitoring patterns: {len(self.drift_thresholds)} threshold types")
        logger.info(f"   Detection thread pool: 4 workers")
    
    def _register(self):
        """Register with schema registry"""
        registry.register(
            component_id="schema.symbolic_failure_detector",
            name="Symbolic Failure Detection System",
            component_type="MONITORING_SYSTEM",
            instance=self,
            capabilities=[
                "sigil_drift_detection",
                "house_operation_monitoring",
                "containment_breach_detection",
                "tracer_misalignment_detection",
                "symbolic_health_monitoring",
                "automated_alerting"
            ],
            version="1.0.0"
        )
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("🔍 Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("🔍 Symbolic failure monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("🔍 Symbolic failure monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Run detection checks
                self._run_detection_cycle()
                
                # Update health metrics
                self._update_health_metrics()
                
                # Check for stale failures
                self._cleanup_stale_failures()
                
                # Sleep between cycles
                time.sleep(2.0)  # 2-second monitoring cycle
                
            except Exception as e:
                logger.error(f"🔍 Error in monitoring loop: {e}")
                time.sleep(5.0)  # Longer sleep on error
    
    def _run_detection_cycle(self):
        """Run a complete detection cycle"""
        detection_tasks = [
            self._detect_sigil_drift,
            self._detect_broken_houses,
            self._detect_containment_breaches,
            self._detect_tracer_misalignments,
            self._detect_glyph_corruption,
            self._detect_resonance_decay
        ]
        
        # Submit detection tasks to thread pool
        futures = [self.detection_executor.submit(task) for task in detection_tasks]
        
        # Collect results
        for future in futures:
            try:
                failures = future.result(timeout=1.0)
                if failures:
                    for failure in failures:
                        self._register_failure(failure)
            except Exception as e:
                logger.error(f"🔍 Detection task failed: {e}")
    
    def _detect_sigil_drift(self) -> List[SymbolicFailure]:
        """Detect sigil meaning drift over time"""
        failures = []
        
        # Check glyph coherence
        codex_stats = sigil_glyph_codex.get_codex_stats()
        
        # Simulate drift detection (would use real metrics in production)
        drift_indicators = {
            "meaning_coherence": 0.85,  # Would be calculated from actual usage
            "symbolic_consistency": 0.78,
            "archetypal_alignment": 0.92
        }
        
        for indicator, value in drift_indicators.items():
            if value < self.drift_thresholds.get("meaning_coherence_loss", 0.8):
                failure = SymbolicFailure(
                    failure_id=f"drift_{indicator}_{int(time.time())}",
                    failure_type=FailureType.SIGIL_DRIFT,
                    severity=FailureSeverity.MEDIUM if value < 0.6 else FailureSeverity.LOW,
                    description=f"Sigil {indicator} drift detected",
                    symptoms=[f"{indicator} below threshold: {value:.3f}"],
                    potential_causes=["Extended usage without recalibration", "Symbolic context shift"],
                    recommended_actions=["Recalibrate glyph meanings", "Review symbolic context"],
                    metadata={"indicator": indicator, "value": value, "threshold": 0.8}
                )
                failures.append(failure)
        
        return failures
    
    def _detect_broken_houses(self) -> List[SymbolicFailure]:
        """Detect broken house operations"""
        failures = []
        
        for house, operator in HOUSE_OPERATORS.items():
            if not operator:
                continue
            
            # Check operator health
            avg_resonance = operator.get_average_resonance()
            operations_count = operator.operations_performed
            
            # Detect low resonance
            if avg_resonance < 0.4 and operations_count > 5:
                failure = SymbolicFailure(
                    failure_id=f"house_{house.value}_resonance_{int(time.time())}",
                    failure_type=FailureType.BROKEN_HOUSE,
                    severity=FailureSeverity.HIGH if avg_resonance < 0.2 else FailureSeverity.MEDIUM,
                    description=f"House {house.value} low mythic resonance",
                    affected_components=[f"house_{house.value}"],
                    symptoms=[f"Average resonance: {avg_resonance:.3f}"],
                    potential_causes=["Archetypal misalignment", "Operation parameter mismatch"],
                    recommended_actions=["Review house operations", "Recalibrate archetypal parameters"],
                    metadata={"house": house.value, "resonance": avg_resonance, "operations": operations_count}
                )
                failures.append(failure)
            
            # Detect operation failures (would track real failure rates)
            simulated_failure_rate = max(0.0, 0.1 - avg_resonance * 0.1)  # Higher failure with low resonance
            if simulated_failure_rate > 0.05:  # 5% failure rate threshold
                failure = SymbolicFailure(
                    failure_id=f"house_{house.value}_failures_{int(time.time())}",
                    failure_type=FailureType.ARCHETYPAL_BREAKDOWN,
                    severity=FailureSeverity.HIGH,
                    description=f"House {house.value} high operation failure rate",
                    affected_components=[f"house_{house.value}"],
                    symptoms=[f"Failure rate: {simulated_failure_rate:.1%}"],
                    potential_causes=["Archetypal corruption", "Resource exhaustion"],
                    recommended_actions=["Restart house operator", "Check archetypal integrity"],
                    metadata={"house": house.value, "failure_rate": simulated_failure_rate}
                )
                failures.append(failure)
        
        return failures
    
    def _detect_containment_breaches(self) -> List[SymbolicFailure]:
        """Detect ring containment breaches"""
        failures = []
        
        # Check enhanced sigil ring status (would import from enhanced_sigil_ring)
        # Simulating containment breach detection
        
        # Mock ring status for detection
        mock_ring_status = {
            "containment_breaches": 2,
            "emergency_seal_active": False,
            "ring_overloads": 1,
            "average_execution_time": 3.2
        }
        
        # Check for containment breaches
        if mock_ring_status["containment_breaches"] > 0:
            severity = FailureSeverity.CRITICAL if mock_ring_status["containment_breaches"] > 2 else FailureSeverity.HIGH
            
            failure = SymbolicFailure(
                failure_id=f"containment_breach_{int(time.time())}",
                failure_type=FailureType.CONTAINMENT_BREACH,
                severity=severity,
                description="Ring containment boundary breached",
                affected_components=["enhanced_sigil_ring"],
                symptoms=[f"{mock_ring_status['containment_breaches']} breaches detected"],
                potential_causes=["Dangerous glyph combinations", "Insufficient containment level"],
                recommended_actions=["Increase containment level", "Review glyph combinations"],
                metadata=mock_ring_status
            )
            failures.append(failure)
        
        # Check for ring overloads
        if mock_ring_status["ring_overloads"] > 0:
            failure = SymbolicFailure(
                failure_id=f"ring_overload_{int(time.time())}",
                failure_type=FailureType.RING_OVERLOAD,
                severity=FailureSeverity.MEDIUM,
                description="Sigil ring experiencing overload conditions",
                affected_components=["enhanced_sigil_ring"],
                symptoms=[f"{mock_ring_status['ring_overloads']} overloads"],
                potential_causes=["Excessive invocation rate", "Insufficient capacity"],
                recommended_actions=["Increase ring capacity", "Implement rate limiting"],
                metadata={"overloads": mock_ring_status["ring_overloads"]}
            )
            failures.append(failure)
        
        return failures
    
    def _detect_tracer_misalignments(self) -> List[SymbolicFailure]:
        """Detect tracer-house misalignments"""
        failures = []
        
        # Get alignment status
        alignment_status = tracer_house_alignment.get_alignment_status()
        
        # Check success rate
        success_rate = alignment_status.get("alignment_statistics", {}).get("success_rate", 100)
        
        if success_rate < 70:  # Less than 70% success rate
            failure = SymbolicFailure(
                failure_id=f"tracer_misalignment_{int(time.time())}",
                failure_type=FailureType.TRACER_MISALIGNMENT,
                severity=FailureSeverity.MEDIUM if success_rate > 50 else FailureSeverity.HIGH,
                description="Tracer-house alignment success rate degraded",
                affected_components=["tracer_house_alignment"],
                symptoms=[f"Success rate: {success_rate:.1f}%"],
                potential_causes=["House capacity constraints", "Tracer profile drift"],
                recommended_actions=["Optimize house capacities", "Recalibrate tracer profiles"],
                metadata={"success_rate": success_rate, "alignment_stats": alignment_status}
            )
            failures.append(failure)
        
        # Check for overloaded houses
        house_loads = alignment_status.get("house_loads", {})
        house_capacities = alignment_status.get("house_capacities", {})
        
        for house_name, load in house_loads.items():
            capacity = house_capacities.get(house_name, 10)
            utilization = load / capacity if capacity > 0 else 0
            
            if utilization > 0.9:  # Over 90% utilization
                failure = SymbolicFailure(
                    failure_id=f"house_overload_{house_name}_{int(time.time())}",
                    failure_type=FailureType.TRACER_MISALIGNMENT,
                    severity=FailureSeverity.MEDIUM,
                    description=f"House {house_name} overloaded",
                    affected_components=[f"house_{house_name}"],
                    symptoms=[f"Utilization: {utilization:.1%}"],
                    potential_causes=["Insufficient capacity", "Uneven tracer distribution"],
                    recommended_actions=[f"Increase {house_name} capacity", "Rebalance tracer alignments"],
                    metadata={"house": house_name, "load": load, "capacity": capacity, "utilization": utilization}
                )
                failures.append(failure)
        
        return failures
    
    def _detect_glyph_corruption(self) -> List[SymbolicFailure]:
        """Detect glyph symbol corruption"""
        failures = []
        
        # Check glyph codex integrity
        codex_stats = sigil_glyph_codex.get_codex_stats()
        
        # Simulate corruption detection
        corruption_indicators = {
            "symbol_integrity": 0.98,
            "meaning_consistency": 0.95,
            "layering_rules_intact": 0.97
        }
        
        for indicator, value in corruption_indicators.items():
            if value < 0.95:  # Less than 95% integrity
                severity = FailureSeverity.CRITICAL if value < 0.9 else FailureSeverity.HIGH
                
                failure = SymbolicFailure(
                    failure_id=f"glyph_corruption_{indicator}_{int(time.time())}",
                    failure_type=FailureType.GLYPH_CORRUPTION,
                    severity=severity,
                    description=f"Glyph {indicator} corruption detected",
                    affected_components=["sigil_glyph_codex"],
                    symptoms=[f"{indicator}: {value:.3f}"],
                    potential_causes=["Memory corruption", "Concurrent modification"],
                    recommended_actions=["Restore glyph codex", "Check memory integrity"],
                    metadata={"indicator": indicator, "value": value, "codex_stats": codex_stats}
                )
                failures.append(failure)
        
        return failures
    
    def _detect_resonance_decay(self) -> List[SymbolicFailure]:
        """Detect mythic resonance decay"""
        failures = []
        
        # Check overall mythic resonance across houses
        total_resonance = 0.0
        house_count = 0
        
        for house, operator in HOUSE_OPERATORS.items():
            if operator:
                avg_resonance = operator.get_average_resonance()
                total_resonance += avg_resonance
                house_count += 1
        
        overall_resonance = total_resonance / house_count if house_count > 0 else 1.0
        
        if overall_resonance < 0.5:  # Less than 50% overall resonance
            failure = SymbolicFailure(
                failure_id=f"resonance_decay_{int(time.time())}",
                failure_type=FailureType.RESONANCE_DECAY,
                severity=FailureSeverity.HIGH if overall_resonance < 0.3 else FailureSeverity.MEDIUM,
                description="System-wide mythic resonance decay",
                affected_components=["archetypal_house_operations"],
                symptoms=[f"Overall resonance: {overall_resonance:.3f}"],
                potential_causes=["Prolonged operation without recalibration", "Archetypal drift"],
                recommended_actions=["Recalibrate archetypal operations", "Review mythic parameters"],
                metadata={"overall_resonance": overall_resonance, "house_count": house_count}
            )
            failures.append(failure)
        
        return failures
    
    def _register_failure(self, failure: SymbolicFailure):
        """Register a detected failure"""
        # Check if similar failure already exists
        existing_failure = self._find_similar_failure(failure)
        
        if existing_failure:
            # Update existing failure
            existing_failure.symptoms.extend(failure.symptoms)
            existing_failure.metadata.update(failure.metadata)
            logger.debug(f"🔍 Updated existing failure: {existing_failure.failure_id}")
        else:
            # Register new failure
            self.active_failures[failure.failure_id] = failure
            self.failure_history.append(failure)
            
            # Update metrics
            self.detection_metrics.total_failures_detected += 1
            self.detection_metrics.failures_by_type[failure.failure_type] += 1
            self.detection_metrics.failures_by_severity[failure.severity] += 1
            
            # Log failure
            log_anomaly(
                f"SYMBOLIC_FAILURE_{failure.failure_type.value.upper()}",
                failure.description,
                AnomalySeverity.CRITICAL if failure.severity == FailureSeverity.EMERGENCY else AnomalySeverity.ERROR
            )
            
            # Emit failure signal
            emit_signal(
                SignalType.ENTROPY,
                "symbolic_failure_detector",
                {
                    "event": "failure_detected",
                    "failure_id": failure.failure_id,
                    "failure_type": failure.failure_type.value,
                    "severity": failure.severity.value,
                    "description": failure.description,
                    "affected_components": failure.affected_components
                }
            )
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(failure)
                except Exception as e:
                    logger.error(f"🔍 Alert callback failed: {e}")
            
            logger.warning(f"🔍 Detected symbolic failure: {failure.failure_type.value} - {failure.description}")
    
    def _find_similar_failure(self, failure: SymbolicFailure) -> Optional[SymbolicFailure]:
        """Find similar existing failure"""
        for existing_failure in self.active_failures.values():
            if (existing_failure.failure_type == failure.failure_type and
                existing_failure.affected_components == failure.affected_components and
                not existing_failure.resolved):
                return existing_failure
        return None
    
    def _update_health_metrics(self):
        """Update symbolic health metrics"""
        
        # Calculate sigil coherence
        codex_stats = sigil_glyph_codex.get_codex_stats()
        sigil_coherence = min(1.0, codex_stats["total_glyphs"] / 20.0)  # Normalize based on expected glyphs
        self.health_monitor.update_health_score("sigil_coherence", sigil_coherence)
        
        # Calculate house performance
        house_performance = 0.0
        active_houses = 0
        
        for house, operator in HOUSE_OPERATORS.items():
            if operator:
                house_health = operator.get_average_resonance()
                house_performance += house_health
                active_houses += 1
        
        if active_houses > 0:
            house_performance /= active_houses
            self.health_monitor.update_health_score("house_performance", house_performance)
        
        # Calculate ring stability (mock for now)
        ring_stability = 0.85  # Would calculate from actual ring metrics
        self.health_monitor.update_health_score("ring_stability", ring_stability)
        
        # Calculate tracer alignment health
        alignment_status = tracer_house_alignment.get_alignment_status()
        success_rate = alignment_status.get("alignment_statistics", {}).get("success_rate", 100)
        tracer_health = success_rate / 100.0
        self.health_monitor.update_health_score("tracer_alignment", tracer_health)
        
        # Calculate glyph integrity
        glyph_integrity = 0.98  # Would calculate from actual integrity checks
        self.health_monitor.update_health_score("glyph_integrity", glyph_integrity)
    
    def _cleanup_stale_failures(self):
        """Clean up stale failures"""
        stale_failures = [
            failure_id for failure_id, failure in self.active_failures.items()
            if failure.is_stale() and failure.resolved
        ]
        
        for failure_id in stale_failures:
            del self.active_failures[failure_id]
        
        if stale_failures:
            logger.debug(f"🔍 Cleaned up {len(stale_failures)} stale failures")
    
    def resolve_failure(self, failure_id: str, resolution_notes: str = "") -> bool:
        """Mark a failure as resolved"""
        if failure_id not in self.active_failures:
            return False
        
        failure = self.active_failures[failure_id]
        failure.resolved = True
        failure.resolution_timestamp = time.time()
        failure.resolution_notes = resolution_notes
        
        logger.info(f"🔍 Resolved failure: {failure_id} - {resolution_notes}")
        
        return True
    
    def get_active_failures(self, severity_filter: Optional[FailureSeverity] = None) -> List[SymbolicFailure]:
        """Get active failures, optionally filtered by severity"""
        failures = [f for f in self.active_failures.values() if not f.resolved]
        
        if severity_filter:
            failures = [f for f in failures if f.severity == severity_filter]
        
        return sorted(failures, key=lambda f: f.detection_timestamp, reverse=True)
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """Get comprehensive failure summary"""
        active_failures = self.get_active_failures()
        
        # Group by type and severity
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        
        for failure in active_failures:
            by_type[failure.failure_type.value] += 1
            by_severity[failure.severity.value] += 1
        
        # Get health alerts
        health_alerts = self.health_monitor.check_health_alerts()
        overall_health = self.health_monitor.get_overall_health()
        
        return {
            "active_failures": len(active_failures),
            "failures_by_type": dict(by_type),
            "failures_by_severity": dict(by_severity),
            "overall_health": overall_health,
            "health_alerts": health_alerts,
            "detection_metrics": {
                "total_detected": self.detection_metrics.total_failures_detected,
                "accuracy_rate": self.detection_metrics.get_accuracy_rate(),
                "detection_latency_ms": self.detection_metrics.detection_latency_ms
            },
            "monitoring_active": self.monitoring_active
        }
    
    def add_alert_callback(self, callback: Callable[[SymbolicFailure], None]):
        """Add alert callback for failure notifications"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[SymbolicFailure], None]):
        """Remove alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

# Global failure detector instance
symbolic_failure_detector = SymbolicFailureDetector()

# Export key functions for easy access
def start_failure_monitoring():
    """Start symbolic failure monitoring"""
    symbolic_failure_detector.start_monitoring()

def stop_failure_monitoring():
    """Stop symbolic failure monitoring"""
    symbolic_failure_detector.stop_monitoring()

def get_active_failures(severity_filter: Optional[FailureSeverity] = None) -> List[SymbolicFailure]:
    """Get active failures"""
    return symbolic_failure_detector.get_active_failures(severity_filter)

def get_failure_summary() -> Dict[str, Any]:
    """Get failure summary"""
    return symbolic_failure_detector.get_failure_summary()

def resolve_failure(failure_id: str, resolution_notes: str = "") -> bool:
    """Resolve a failure"""
    return symbolic_failure_detector.resolve_failure(failure_id, resolution_notes)

"""
Crow Tracer - Opportunistic Anomaly Scout

Fast-acting anomaly detector that surfaces unexpected signals, weak coherence
points, and volatile residues quickly without deep analysis. Modeled after
real crows - opportunistic, intelligent, thriving on scraps and anomalies.
"""

from typing import Dict, Any, List
from .base_tracer import BaseTracer, TracerType, TracerReport, AlertSeverity, TracerSpawnConditions
import logging

logger = logging.getLogger(__name__)


class CrowTracer(BaseTracer):
    """Fast anomaly detection tracer - the eyes of DAWN"""
    
    def __init__(self, tracer_id: str = None):
        super().__init__(tracer_id)
        self.anomalies_detected = 0
        self.false_positive_count = 0
        self.last_significant_find = None
        
    @property
    def tracer_type(self) -> TracerType:
        return TracerType.CROW
    
    @property
    def base_lifespan(self) -> int:
        return 3  # 2-5 ticks average - fast and ephemeral
    
    @property
    def base_nutrient_cost(self) -> float:
        return 0.1  # Very low cost - spawns freely
    
    @property
    def archetype_description(self) -> str:
        return "Opportunistic scout - spots anomalies fast (entropy spikes, drift slips)"
    
    def spawn_conditions_met(self, context: Dict[str, Any]) -> bool:
        """Spawn on entropy spikes, drift misalignment, pressure changes"""
        # DEBUG: Force spawning for testing
        tick_id = context.get('current_tick', 0) or context.get('tick_id', 0)
        
        # Force spawn every 3 ticks for demonstration
        if tick_id > 0 and (tick_id % 3) == 0:
            return True
            
        # Always spawn if any entropy value exists (very lenient for debugging)
        entropy = context.get('entropy', 0.0)
        if entropy > 0.0:  # Any entropy at all
            return True
            
        return False
    
    def observe(self, context: Dict[str, Any]) -> List[TracerReport]:
        """Scan for surface anomalies and outliers"""
        reports = []
        current_tick = context.get('tick_id', 0)
        
        # Check bloom anomalies
        self._check_bloom_anomalies(context, reports)
        
        # Check soot volatility risks
        self._check_soot_volatility(context, reports)
        
        # Check schema edge anomalies
        self._check_schema_anomalies(context, reports)
        
        # Check pressure fluctuations
        self._check_pressure_fluctuations(context, reports)
        
        # Update internal state
        if reports:
            self.anomalies_detected += len(reports)
            self.last_significant_find = current_tick
            
            # Log significant findings to telemetry
            self.log_to_telemetry('crow_anomalies_detected', {
                'anomaly_count': len(reports),
                'total_anomalies': self.anomalies_detected,
                'tick_id': current_tick,
                'report_types': [r.report_type for r in reports]
            })
            
            # Broadcast critical findings to consciousness
            critical_reports = [r for r in reports if r.severity == AlertSeverity.CRITICAL]
            if critical_reports:
                self.broadcast_to_consciousness('critical_anomalies_detected', {
                    'critical_count': len(critical_reports),
                    'anomaly_types': [r.report_type for r in critical_reports],
                    'immediate_attention_required': True
                })
            
        return reports
    
    def _check_bloom_anomalies(self, context: Dict[str, Any], reports: List[TracerReport]) -> None:
        """Check for anomalous bloom behavior"""
        blooms = context.get('active_blooms', [])
        
        for bloom in blooms:
            bloom_entropy = bloom.get('entropy', 0)
            bloom_intensity = bloom.get('intensity', 0)
            bloom_id = bloom.get('id', 'unknown')
            
            # High entropy blooms (unstable)
            if bloom_entropy > 0.8:
                reports.append(TracerReport(
                    tracer_id=self.tracer_id,
                    tracer_type=self.tracer_type,
                    tick_id=context.get('tick_id', 0),
                    timestamp=context.get('timestamp', 0),
                    severity=AlertSeverity.WARN,
                    report_type="bloom_anomaly",
                    metadata={
                        "anomaly_type": "entropy",
                        "bloom_id": bloom_id,
                        "entropy_level": bloom_entropy,
                        "risk": "unstable_bloom"
                    }
                ))
            
            # Intensity spikes (unexpected activation)
            if bloom_intensity > 0.9:
                reports.append(TracerReport(
                    tracer_id=self.tracer_id,
                    tracer_type=self.tracer_type,
                    tick_id=context.get('tick_id', 0),
                    timestamp=context.get('timestamp', 0),
                    severity=AlertSeverity.CRITICAL,
                    report_type="bloom_anomaly",
                    metadata={
                        "anomaly_type": "intensity_spike",
                        "bloom_id": bloom_id,
                        "intensity_level": bloom_intensity,
                        "risk": "overactivation"
                    }
                ))
    
    def _check_soot_volatility(self, context: Dict[str, Any], reports: List[TracerReport]) -> None:
        """Check for volatile soot fragments that risk reignition"""
        soot_fragments = context.get('soot_fragments', [])
        volatile_soot = [s for s in soot_fragments if s.get('volatility', 0) > 0.7]
        
        if len(volatile_soot) > 3:  # Threshold for concern
            reports.append(TracerReport(
                tracer_id=self.tracer_id,
                tracer_type=self.tracer_type,
                tick_id=context.get('tick_id', 0),
                timestamp=context.get('timestamp', 0),
                severity=AlertSeverity.CRITICAL,
                report_type="soot_volatility",
                metadata={
                    "anomaly_type": "volatile_soot",
                    "volatile_count": len(volatile_soot),
                    "total_soot": len(soot_fragments),
                    "risk": "reignition",
                    "volatile_fragments": [s.get('id', 'unknown') for s in volatile_soot[:5]]
                }
            ))
    
    def _check_schema_anomalies(self, context: Dict[str, Any], reports: List[TracerReport]) -> None:
        """Check for schema edge and coherence anomalies"""
        schema_edges = context.get('schema_edges', [])
        
        # Check for edges with sudden tension spikes
        high_tension_edges = [e for e in schema_edges if e.get('tension', 0) > 0.8]
        
        if high_tension_edges:
            for edge in high_tension_edges[:3]:  # Report up to 3 worst edges
                reports.append(TracerReport(
                    tracer_id=self.tracer_id,
                    tracer_type=self.tracer_type,
                    tick_id=context.get('tick_id', 0),
                    timestamp=context.get('timestamp', 0),
                    severity=AlertSeverity.WARN,
                    report_type="schema_anomaly",
                    metadata={
                        "anomaly_type": "edge_tension",
                        "edge_id": edge.get('id', 'unknown'),
                        "tension_level": edge.get('tension', 0),
                        "risk": "edge_rupture"
                    }
                ))
        
        # Check for coherence drops
        clusters = context.get('schema_clusters', [])
        low_coherence_clusters = [c for c in clusters if c.get('coherence', 1.0) < 0.4]
        
        if low_coherence_clusters:
            reports.append(TracerReport(
                tracer_id=self.tracer_id,
                tracer_type=self.tracer_type,
                tick_id=context.get('tick_id', 0),
                timestamp=context.get('timestamp', 0),
                severity=AlertSeverity.WARN,
                report_type="schema_anomaly",
                metadata={
                    "anomaly_type": "coherence_drop",
                    "affected_clusters": len(low_coherence_clusters),
                    "cluster_ids": [c.get('id', 'unknown') for c in low_coherence_clusters[:3]],
                    "risk": "schema_fragmentation"
                }
            ))
    
    def _check_pressure_fluctuations(self, context: Dict[str, Any], reports: List[TracerReport]) -> None:
        """Check for rapid pressure fluctuations"""
        pressure = context.get('pressure', 0.0)
        pressure_history = context.get('pressure_history', [])
        
        if len(pressure_history) >= 3:
            # Check for oscillations (rapid up/down pattern)
            recent_pressures = pressure_history[-3:] + [pressure]
            
            fluctuation_count = 0
            for i in range(len(recent_pressures) - 1):
                change = abs(recent_pressures[i+1] - recent_pressures[i])
                if change > 0.2:  # Significant change threshold
                    fluctuation_count += 1
            
            if fluctuation_count >= 2:  # Multiple rapid changes
                reports.append(TracerReport(
                    tracer_id=self.tracer_id,
                    tracer_type=self.tracer_type,
                    tick_id=context.get('tick_id', 0),
                    timestamp=context.get('timestamp', 0),
                    severity=AlertSeverity.WARN,
                    report_type="pressure_anomaly",
                    metadata={
                        "anomaly_type": "pressure_oscillation",
                        "fluctuation_count": fluctuation_count,
                        "current_pressure": pressure,
                        "pressure_history": recent_pressures,
                        "risk": "system_instability"
                    }
                ))
    
    def should_retire(self, context: Dict[str, Any]) -> bool:
        """Retire after lifespan or when no more anomalies detected"""
        current_tick = context.get('tick_id', 0)
        age = self.get_age(current_tick)
        
        # Age-based retirement
        if age >= self.base_lifespan:
            return True
            
        # Early retirement if no anomalies detected and running for a while
        if age >= 2 and self.anomalies_detected == 0:
            return True
            
        # Retire if haven't found anything significant recently
        if (age > 1 and self.last_significant_find is not None and 
            current_tick - self.last_significant_find > 2):
            return True
            
        return False

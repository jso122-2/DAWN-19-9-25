"""
Spider Tracer - Schema Tension Sensor

Lives in DAWN's schema webs, feeling for tension across connections. Detects
when coherence stretches too thin or drift pulls nodes apart, acting as
early-warning for structural stress. Web vibration detection specialist.
"""

from typing import Dict, Any, List
from .base_tracer import BaseTracer, TracerType, TracerReport, AlertSeverity, TracerSpawnConditions
import logging

logger = logging.getLogger(__name__)


class SpiderTracer(BaseTracer):
    """Schema edge tension monitoring tracer - the web tension sensor of DAWN"""
    
    def __init__(self, tracer_id: str = None):
        super().__init__(tracer_id)
        self.monitored_edges = set()
        self.tension_history = {}
        self.vibration_patterns = {}
        self.web_health_score = 1.0
        self.rupture_predictions = []
        
    @property
    def tracer_type(self) -> TracerType:
        return TracerType.SPIDER
    
    @property
    def base_lifespan(self) -> int:
        return 12  # 5-20 ticks average - structural monitoring
    
    @property
    def base_nutrient_cost(self) -> float:
        return 0.5  # Medium cost
    
    @property
    def archetype_description(self) -> str:
        return "Web tension sensor - detects schema strain & edge stress"
    
    def spawn_conditions_met(self, context: Dict[str, Any]) -> bool:
        """Spawn when schema edges show stress indicators"""
        # Check for high tension edges
        if TracerSpawnConditions.schema_tension(context, threshold=0.6):
            return True
            
        schema_edges = context.get('schema_edges', [])
        
        # Check for edge volatility (shimmer effects)
        volatile_edges = [e for e in schema_edges if e.get('volatility', 0) > 0.7]
        if len(volatile_edges) > 2:
            return True
            
        # Check for drift misalignment across clusters
        drift_variance = context.get('drift_variance', 0.0)
        if drift_variance > 0.4:
            return True
            
        # Check for coherence drops that might stress connections
        avg_coherence = context.get('avg_schema_coherence', 1.0)
        if avg_coherence < 0.5:
            return True
            
        # Check for rapid schema changes
        schema_change_rate = context.get('schema_change_rate', 0.0)
        if schema_change_rate > 0.6:
            return True
            
        return False
    
    def observe(self, context: Dict[str, Any]) -> List[TracerReport]:
        """Monitor edge tensions and detect strain patterns"""
        reports = []
        schema_edges = context.get('schema_edges', [])
        
        # Monitor individual edge tensions
        self._monitor_edge_tensions(schema_edges, context, reports)
        
        # Detect vibration patterns (oscillations)
        self._detect_vibration_patterns(schema_edges, context, reports)
        
        # Check for web-wide strain indicators
        self._assess_web_health(schema_edges, context, reports)
        
        # Predict potential ruptures
        self._predict_ruptures(schema_edges, context, reports)
        
        return reports
    
    def _monitor_edge_tensions(self, schema_edges: List[Dict[str, Any]], 
                             context: Dict[str, Any], reports: List[TracerReport]) -> None:
        """Monitor individual edge tensions"""
        for edge in schema_edges:
            edge_id = edge.get('id', f"{edge.get('source', 'unknown')}-{edge.get('target', 'unknown')}")
            current_tension = edge.get('tension', 0.0)
            current_entropy = edge.get('entropy', 0.0)
            
            # Track this edge
            self.monitored_edges.add(edge_id)
            
            # Update tension history
            if edge_id not in self.tension_history:
                self.tension_history[edge_id] = []
            self.tension_history[edge_id].append(current_tension)
            
            # Keep only recent history (sliding window)
            if len(self.tension_history[edge_id]) > 8:
                self.tension_history[edge_id].pop(0)
            
            # High tension alerts
            if current_tension > 0.7:
                severity = AlertSeverity.CRITICAL if current_tension > 0.9 else AlertSeverity.WARN
                reports.append(TracerReport(
                    tracer_id=self.tracer_id,
                    tracer_type=self.tracer_type,
                    tick_id=context.get('tick_id', 0),
                    timestamp=context.get('timestamp', 0),
                    severity=severity,
                    report_type="tension_alert",
                    metadata={
                        "edge_id": edge_id,
                        "tension": current_tension,
                        "entropy": current_entropy,
                        "trend": self._calculate_tension_trend(edge_id),
                        "risk_level": "critical" if current_tension > 0.9 else "high"
                    }
                ))
            
            # Sudden tension spikes
            if len(self.tension_history[edge_id]) >= 2:
                tension_change = current_tension - self.tension_history[edge_id][-2]
                if tension_change > 0.3:  # Significant sudden increase
                    reports.append(TracerReport(
                        tracer_id=self.tracer_id,
                        tracer_type=self.tracer_type,
                        tick_id=context.get('tick_id', 0),
                        timestamp=context.get('timestamp', 0),
                        severity=AlertSeverity.WARN,
                        report_type="tension_spike",
                        metadata={
                            "edge_id": edge_id,
                            "tension_change": tension_change,
                            "current_tension": current_tension,
                            "risk": "sudden_stress"
                        }
                    ))
    
    def _detect_vibration_patterns(self, schema_edges: List[Dict[str, Any]], 
                                 context: Dict[str, Any], reports: List[TracerReport]) -> None:
        """Detect oscillating tensions (web vibrations)"""
        for edge_id in self.monitored_edges:
            if len(self.tension_history.get(edge_id, [])) >= 4:
                tensions = self.tension_history[edge_id]
                
                # Detect oscillation pattern
                oscillation_strength = self._calculate_oscillation_strength(tensions)
                
                if oscillation_strength > 0.6:
                    # Record vibration pattern
                    self.vibration_patterns[edge_id] = {
                        'strength': oscillation_strength,
                        'frequency': self._calculate_oscillation_frequency(tensions),
                        'tick': context.get('tick_id', 0)
                    }
                    
                    reports.append(TracerReport(
                        tracer_id=self.tracer_id,
                        tracer_type=self.tracer_type,
                        tick_id=context.get('tick_id', 0),
                        timestamp=context.get('timestamp', 0),
                        severity=AlertSeverity.WARN,
                        report_type="tension_oscillation",
                        metadata={
                            "edge_id": edge_id,
                            "oscillation_strength": oscillation_strength,
                            "tension_history": tensions,
                            "risk": "edge_instability",
                            "damping_needed": oscillation_strength > 0.8
                        }
                    ))
    
    def _assess_web_health(self, schema_edges: List[Dict[str, Any]], 
                          context: Dict[str, Any], reports: List[TracerReport]) -> None:
        """Assess overall web health and structural integrity"""
        if not schema_edges:
            return
            
        # Calculate overall web health metrics
        avg_tension = sum(e.get('tension', 0) for e in schema_edges) / len(schema_edges)
        high_tension_ratio = len([e for e in schema_edges if e.get('tension', 0) > 0.6]) / len(schema_edges)
        unstable_edges = len([eid for eid in self.vibration_patterns 
                            if self.vibration_patterns[eid]['strength'] > 0.6])
        
        # Update web health score
        previous_health = self.web_health_score
        self.web_health_score = 1.0 - (avg_tension * 0.4 + high_tension_ratio * 0.4 + 
                                      min(unstable_edges / len(schema_edges), 1.0) * 0.2)
        
        health_change = self.web_health_score - previous_health
        
        # Report significant web health changes
        if abs(health_change) > 0.2 or self.web_health_score < 0.4:
            severity = AlertSeverity.CRITICAL if self.web_health_score < 0.3 else AlertSeverity.WARN
            
            reports.append(TracerReport(
                tracer_id=self.tracer_id,
                tracer_type=self.tracer_type,
                tick_id=context.get('tick_id', 0),
                timestamp=context.get('timestamp', 0),
                severity=severity,
                report_type="web_health_assessment",
                metadata={
                    "web_health_score": self.web_health_score,
                    "health_change": health_change,
                    "avg_tension": avg_tension,
                    "high_tension_ratio": high_tension_ratio,
                    "unstable_edges": unstable_edges,
                    "total_edges": len(schema_edges),
                    "assessment": self._get_health_assessment()
                }
            ))
    
    def _predict_ruptures(self, schema_edges: List[Dict[str, Any]], 
                         context: Dict[str, Any], reports: List[TracerReport]) -> None:
        """Predict potential edge ruptures based on tension trends"""
        rupture_candidates = []
        
        for edge in schema_edges:
            edge_id = edge.get('id', f"{edge.get('source', 'unknown')}-{edge.get('target', 'unknown')}")
            current_tension = edge.get('tension', 0.0)
            
            if edge_id in self.tension_history and len(self.tension_history[edge_id]) >= 3:
                tensions = self.tension_history[edge_id]
                
                # Calculate rupture risk based on multiple factors
                rupture_risk = self._calculate_rupture_risk(tensions, current_tension, edge)
                
                if rupture_risk > 0.7:
                    rupture_candidates.append({
                        'edge_id': edge_id,
                        'risk': rupture_risk,
                        'current_tension': current_tension,
                        'predicted_ticks': self._estimate_rupture_timing(tensions)
                    })
        
        # Report rupture predictions
        if rupture_candidates:
            # Sort by risk level
            rupture_candidates.sort(key=lambda x: x['risk'], reverse=True)
            
            reports.append(TracerReport(
                tracer_id=self.tracer_id,
                tracer_type=self.tracer_type,
                tick_id=context.get('tick_id', 0),
                timestamp=context.get('timestamp', 0),
                severity=AlertSeverity.CRITICAL,
                report_type="rupture_prediction",
                metadata={
                    "rupture_candidates": rupture_candidates[:5],  # Top 5 risks
                    "total_at_risk": len(rupture_candidates),
                    "immediate_attention_needed": len([r for r in rupture_candidates if r['risk'] > 0.9])
                }
            ))
            
            self.rupture_predictions = rupture_candidates
    
    def _calculate_tension_trend(self, edge_id: str) -> str:
        """Calculate if tension is increasing, decreasing, or stable"""
        if edge_id not in self.tension_history or len(self.tension_history[edge_id]) < 3:
            return "unknown"
            
        tensions = self.tension_history[edge_id]
        recent_avg = sum(tensions[-2:]) / 2
        older_avg = sum(tensions[:-2]) / max(1, len(tensions) - 2)
        
        diff = recent_avg - older_avg
        if diff > 0.15:
            return "increasing"
        elif diff < -0.15:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_oscillation_strength(self, tensions: List[float]) -> float:
        """Calculate the strength of oscillation in tension values"""
        if len(tensions) < 4:
            return 0.0
            
        # Calculate differences between consecutive values
        diffs = [tensions[i+1] - tensions[i] for i in range(len(tensions)-1)]
        
        # Count sign changes (direction reversals)
        sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
        
        # Calculate magnitude of oscillations
        avg_magnitude = sum(abs(d) for d in diffs) / len(diffs)
        
        # Oscillation strength combines frequency and magnitude
        max_sign_changes = len(diffs) - 1
        frequency_factor = sign_changes / max_sign_changes if max_sign_changes > 0 else 0
        
        return min(1.0, frequency_factor * 2 + avg_magnitude)
    
    def _calculate_oscillation_frequency(self, tensions: List[float]) -> float:
        """Calculate oscillation frequency (cycles per tick)"""
        if len(tensions) < 4:
            return 0.0
            
        diffs = [tensions[i+1] - tensions[i] for i in range(len(tensions)-1)]
        sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
        
        # Frequency = sign changes / 2 (since one cycle = 2 sign changes) / time window
        cycles = sign_changes / 2
        time_window = len(tensions) - 1
        
        return cycles / time_window if time_window > 0 else 0.0
    
    def _calculate_rupture_risk(self, tensions: List[float], current_tension: float, 
                              edge: Dict[str, Any]) -> float:
        """Calculate rupture risk for an edge"""
        risk_factors = []
        
        # High absolute tension
        risk_factors.append(min(1.0, current_tension / 0.95))
        
        # Increasing tension trend
        if len(tensions) >= 3:
            recent_slope = (tensions[-1] - tensions[-3]) / 2
            risk_factors.append(min(1.0, max(0.0, recent_slope * 5)))
        
        # Oscillation instability
        oscillation = self._calculate_oscillation_strength(tensions)
        risk_factors.append(oscillation)
        
        # Edge properties
        edge_entropy = edge.get('entropy', 0)
        risk_factors.append(edge_entropy)
        
        # Combine risk factors (weighted average)
        weights = [0.4, 0.3, 0.2, 0.1]
        total_risk = sum(factor * weight for factor, weight in zip(risk_factors, weights))
        
        return min(1.0, total_risk)
    
    def _estimate_rupture_timing(self, tensions: List[float]) -> int:
        """Estimate ticks until potential rupture"""
        if len(tensions) < 3:
            return 999  # Unknown
            
        # Simple linear extrapolation
        recent_slope = (tensions[-1] - tensions[-3]) / 2
        
        if recent_slope <= 0:
            return 999  # Not increasing
            
        current_tension = tensions[-1]
        rupture_threshold = 0.95
        
        ticks_to_rupture = (rupture_threshold - current_tension) / recent_slope
        
        return max(1, int(ticks_to_rupture))
    
    def _get_health_assessment(self) -> str:
        """Get textual health assessment"""
        if self.web_health_score > 0.8:
            return "excellent"
        elif self.web_health_score > 0.6:
            return "good"
        elif self.web_health_score > 0.4:
            return "concerning"
        elif self.web_health_score > 0.2:
            return "poor"
        else:
            return "critical"
    
    def should_retire(self, context: Dict[str, Any]) -> bool:
        """Retire when tensions resolve or lifespan exceeded"""
        current_tick = context.get('tick_id', 0)
        age = self.get_age(current_tick)
        
        # Retire after lifespan
        if age >= self.base_lifespan:
            return True
            
        # Early retirement if all monitored edges are stable
        if age > 5:
            schema_edges = context.get('schema_edges', [])
            high_tension_edges = [
                e for e in schema_edges 
                if e.get('id') in self.monitored_edges and e.get('tension', 0) > 0.6
            ]
            
            # No high tension edges and web health is good
            if not high_tension_edges and self.web_health_score > 0.7:
                return True
                
        # Early retirement if no edges to monitor
        if age > 2 and not self.monitored_edges:
            return True
            
        return False

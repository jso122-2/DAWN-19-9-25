"""
Whale Tracer - Deep Context Diver

DAWN's slow, heavy, thorough tracer that dives deep into schema space,
collecting wide-context patterns and surfacing structural insights that
lighter tracers cannot perceive. Provides ballast against over-reactivity.
"""

from typing import Dict, Any, List
import numpy as np
from .base_tracer import BaseTracer, TracerType, TracerReport, AlertSeverity, TracerSpawnConditions
import logging

logger = logging.getLogger(__name__)


class WhaleTracer(BaseTracer):
    """Deep context scanning and macro trend analysis tracer - the sonar of DAWN"""
    
    def __init__(self, tracer_id: str = None):
        super().__init__(tracer_id)
        self.scan_layers = ['ash', 'schema', 'mycelial', 'memory']
        self.context_data = {}
        self.trend_analysis = {}
        self.deep_scan_complete = False
        self.analysis_phase = "scanning"  # scanning -> analysis -> synthesis
        self.insights_generated = []
        
    @property
    def tracer_type(self) -> TracerType:
        return TracerType.WHALE
    
    @property
    def base_lifespan(self) -> int:
        return 30  # 10-50 ticks average - long-lived analysis
    
    @property
    def base_nutrient_cost(self) -> float:
        return 2.0  # High cost - rare spawning
    
    @property
    def archetype_description(self) -> str:
        return "Deep diver - surfaces macro trends, schema averages"
    
    def spawn_conditions_met(self, context: Dict[str, Any]) -> bool:
        """Spawn for sustained entropy, high drift variance, or forecast requests"""
        # Sustained entropy (not just spikes)
        if TracerSpawnConditions.sustained_entropy(context, threshold=0.6, window=5):
            return True
        
        # High drift variance across multiple ticks
        drift_history = context.get('drift_history', [])
        if len(drift_history) >= 3:
            drift_variance = np.var(drift_history[-3:]) if len(drift_history) >= 3 else 0
            if drift_variance > context.get('drift_variance_threshold', 0.3):
                return True
        
        # Explicit forecasting requests
        if context.get('forecast_request', False):
            return True
            
        # Complex system state requiring deep analysis
        complexity_indicators = [
            context.get('schema_complexity', 0),
            context.get('memory_fragmentation', 0),
            context.get('cross_layer_tensions', 0)
        ]
        avg_complexity = sum(complexity_indicators) / len(complexity_indicators)
        if avg_complexity > 0.7:
            return True
            
        # Multiple subsystem stress indicators
        stress_count = 0
        if context.get('entropy', 0) > 0.5:
            stress_count += 1
        if context.get('soot_ratio', 0) > 0.3:
            stress_count += 1
        if context.get('avg_schema_coherence', 1.0) < 0.6:
            stress_count += 1
        if context.get('memory_pressure', 0) > 0.6:
            stress_count += 1
            
        if stress_count >= 3:  # Multiple stress indicators
            return True
            
        return False
    
    def observe(self, context: Dict[str, Any]) -> List[TracerReport]:
        """Perform deep multi-layer scanning and analysis"""
        reports = []
        current_tick = context.get('tick_id', 0)
        age = self.get_age(current_tick)
        
        if self.analysis_phase == "scanning":
            # Deep scanning phase (first portion of lifetime)
            if age <= 8:
                self._perform_deep_scan(context)
            else:
                self.analysis_phase = "analysis"
                self.deep_scan_complete = True
                reports.append(self._create_scan_completion_report(context))
                
        elif self.analysis_phase == "analysis":
            # Analysis phase (middle portion)
            if age <= 20:
                analysis_results = self._analyze_trends(context)
                if analysis_results:
                    reports.append(TracerReport(
                        tracer_id=self.tracer_id,
                        tracer_type=self.tracer_type,
                        tick_id=current_tick,
                        timestamp=context.get('timestamp', 0),
                        severity=AlertSeverity.INFO,
                        report_type="context_scan",
                        metadata={
                            "scope": "multi_layer",
                            "tick_window": [self.spawn_tick, current_tick],
                            "findings": analysis_results
                        }
                    ))
            else:
                self.analysis_phase = "synthesis"
                
        elif self.analysis_phase == "synthesis":
            # Final synthesis phase
            synthesis = self._synthesize_insights(context)
            if synthesis:
                recommendations = self._generate_recommendations(synthesis)
                confidence = self._calculate_confidence_score()
                
                reports.append(TracerReport(
                    tracer_id=self.tracer_id,
                    tracer_type=self.tracer_type,
                    tick_id=current_tick,
                    timestamp=context.get('timestamp', 0),
                    severity=AlertSeverity.INFO,
                    report_type="deep_context_synthesis",
                    metadata={
                        "scope": "macro_trends",
                        "tick_window": [self.spawn_tick, current_tick],
                        "synthesis": synthesis,
                        "recommendations": recommendations,
                        "confidence": confidence
                    }
                ))
                self.insights_generated.append(synthesis)
                
                # Log deep insights to telemetry
                self.log_to_telemetry('whale_deep_synthesis', {
                    'analysis_phase': self.analysis_phase,
                    'confidence_score': confidence,
                    'recommendations_count': len(recommendations),
                    'synthesis_scope': 'macro_trends',
                    'tick_window_duration': current_tick - self.spawn_tick
                })
                
                # Broadcast high-confidence insights to consciousness
                if confidence > 0.7:
                    self.broadcast_to_consciousness('deep_insights_available', {
                        'confidence': confidence,
                        'recommendations': recommendations,
                        'synthesis_summary': {
                            'system_health': synthesis.get('system_health', {}),
                            'predictive_indicators': synthesis.get('predictive_indicators', {})
                        },
                        'requires_system_attention': any(
                            rec.get('priority') == 'critical' for rec in recommendations
                        )
                    })
        
        return reports
    
    def _perform_deep_scan(self, context: Dict[str, Any]) -> None:
        """Scan multiple memory layers for comprehensive data collection"""
        current_tick = context.get('tick_id', 0)
        
        # Ash layer analysis
        ash_fragments = context.get('ash_fragments', [])
        if ash_fragments:
            crystallization_levels = [f.get('crystallization', 0) for f in ash_fragments]
            pigments = [f.get('pigment', [0, 0, 0]) for f in ash_fragments if f.get('pigment')]
            
            self.context_data['ash_analysis'] = {
                'total_fragments': len(ash_fragments),
                'avg_crystallization': np.mean(crystallization_levels) if crystallization_levels else 0,
                'crystallization_variance': np.var(crystallization_levels) if len(crystallization_levels) > 1 else 0,
                'pigment_distribution': self._analyze_pigment_distribution(pigments),
                'nutrient_content': sum(f.get('nutrients', 0) for f in ash_fragments),
                'age_distribution': self._analyze_age_distribution(ash_fragments)
            }
        
        # Schema cluster analysis
        schema_clusters = context.get('schema_clusters', [])
        if schema_clusters:
            coherence_levels = [c.get('coherence', 0) for c in schema_clusters]
            connectivity_data = [c.get('connectivity', 0) for c in schema_clusters]
            
            self.context_data['schema_analysis'] = {
                'cluster_count': len(schema_clusters),
                'avg_coherence': np.mean(coherence_levels) if coherence_levels else 0,
                'coherence_variance': np.var(coherence_levels) if len(coherence_levels) > 1 else 0,
                'connectivity_matrix': self._build_connectivity_matrix(schema_clusters),
                'health_distribution': [c.get('health_score', 0) for c in schema_clusters],
                'avg_connectivity': np.mean(connectivity_data) if connectivity_data else 0,
                'fragmentation_score': self._calculate_fragmentation_score(schema_clusters)
            }
        
        # Mycelial flow analysis
        mycelial_flows = context.get('mycelial_flows', [])
        if mycelial_flows:
            throughput_data = [f.get('throughput', 0) for f in mycelial_flows]
            efficiency_data = [f.get('efficiency', 0) for f in mycelial_flows]
            
            self.context_data['mycelial_analysis'] = {
                'flow_count': len(mycelial_flows),
                'total_throughput': sum(throughput_data),
                'avg_throughput': np.mean(throughput_data) if throughput_data else 0,
                'flow_efficiency': np.mean(efficiency_data) if efficiency_data else 0,
                'efficiency_variance': np.var(efficiency_data) if len(efficiency_data) > 1 else 0,
                'bottlenecks': [f.get('id') for f in mycelial_flows if f.get('congestion', 0) > 0.7],
                'flow_balance': self._analyze_flow_balance(mycelial_flows)
            }
        
        # Memory layer analysis
        memory_metrics = context.get('memory_metrics', {})
        self.context_data['memory_analysis'] = {
            'utilization': memory_metrics.get('utilization', 0),
            'fragmentation': memory_metrics.get('fragmentation', 0),
            'access_patterns': memory_metrics.get('access_patterns', {}),
            'pressure_level': context.get('memory_pressure', 0),
            'bloom_activity': len(context.get('active_blooms', [])),
            'trace_density': memory_metrics.get('trace_density', 0)
        }
    
    def _analyze_trends(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends across the collected data"""
        trends = {}
        
        # Entropy trend analysis
        entropy_history = context.get('entropy_history', [])
        if len(entropy_history) >= 3:
            recent_entropy = entropy_history[-3:]
            trends['entropy_analysis'] = {
                'mean': float(np.mean(recent_entropy)),
                'trend': self._calculate_trend_direction(recent_entropy),
                'volatility': float(np.std(recent_entropy)),
                'acceleration': self._calculate_acceleration(recent_entropy)
            }
        
        # Drift pattern analysis
        drift_history = context.get('drift_history', [])
        if len(drift_history) >= 3:
            recent_drift = drift_history[-3:]
            trends['drift_analysis'] = {
                'magnitude': float(np.mean(recent_drift)),
                'direction': self._calculate_trend_direction(recent_drift),
                'stability': 'stable' if np.std(recent_drift) < 0.1 else 'volatile',
                'correlation_with_entropy': self._calculate_correlation(entropy_history, drift_history)
            }
        
        # Pressure trend analysis
        pressure_history = context.get('pressure_history', [])
        if len(pressure_history) >= 3:
            recent_pressure = pressure_history[-3:]
            trends['pressure_analysis'] = {
                'current_level': float(recent_pressure[-1]),
                'trend_direction': self._calculate_trend_direction(recent_pressure),
                'peak_pressure': float(max(recent_pressure)),
                'pressure_cycles': self._detect_pressure_cycles(pressure_history),
                'stability_index': self._calculate_stability_index(recent_pressure)
            }
        
        # Cross-layer correlation analysis
        if self.context_data:
            trends['cross_layer_correlations'] = self._analyze_cross_layer_correlations(context)
        
        return trends
    
    def _synthesize_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level insights from all collected data"""
        synthesis = {}
        
        # Overall system health assessment
        if 'schema_analysis' in self.context_data:
            health_scores = self.context_data['schema_analysis']['health_distribution']
            if health_scores:
                avg_health = np.mean(health_scores)
                synthesis['system_health'] = {
                    'overall_score': float(avg_health),
                    'assessment': self._get_health_assessment(avg_health),
                    'variability': float(np.std(health_scores)) if len(health_scores) > 1 else 0.0,
                    'risk_factors': self._identify_health_risk_factors()
                }
        
        # Resource allocation efficiency
        if 'mycelial_analysis' in self.context_data:
            mycelial_data = self.context_data['mycelial_analysis']
            synthesis['resource_efficiency'] = {
                'flow_efficiency': mycelial_data['flow_efficiency'],
                'throughput_capacity': mycelial_data['total_throughput'],
                'bottleneck_severity': self._assess_bottleneck_severity(mycelial_data['bottlenecks']),
                'allocation_balance': mycelial_data['flow_balance']
            }
        
        # Memory consolidation status
        if 'ash_analysis' in self.context_data:
            ash_data = self.context_data['ash_analysis']
            synthesis['memory_consolidation'] = {
                'crystallization_health': self._assess_crystallization_health(ash_data),
                'consolidation_rate': ash_data['avg_crystallization'],
                'memory_retention': self._estimate_memory_retention(ash_data),
                'pigment_balance': ash_data['pigment_distribution']
            }
        
        # Structural integrity assessment
        if 'schema_analysis' in self.context_data:
            schema_data = self.context_data['schema_analysis']
            synthesis['structural_integrity'] = {
                'coherence_stability': schema_data['avg_coherence'],
                'fragmentation_risk': schema_data['fragmentation_score'],
                'connectivity_health': schema_data['avg_connectivity'],
                'structural_stress': self._assess_structural_stress(schema_data)
            }
        
        # Predictive indicators
        synthesis['predictive_indicators'] = self._generate_predictive_indicators(context)
        
        return synthesis
    
    def _generate_recommendations(self, synthesis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on synthesis"""
        recommendations = []
        
        # System health recommendations
        if 'system_health' in synthesis:
            health_score = synthesis['system_health']['overall_score']
            if health_score < 0.4:
                recommendations.append({
                    "action": "deploy_beetle_tracers",
                    "reason": "critical_system_health",
                    "priority": "critical",
                    "target": "system_wide"
                })
            elif health_score < 0.6:
                recommendations.append({
                    "action": "increase_monitoring",
                    "reason": "declining_system_health",
                    "priority": "high",
                    "target": "health_monitoring"
                })
        
        # Resource efficiency recommendations
        if 'resource_efficiency' in synthesis:
            bottleneck_severity = synthesis['resource_efficiency']['bottleneck_severity']
            if bottleneck_severity == "severe":
                recommendations.append({
                    "action": "increase_bee_pollination",
                    "reason": "severe_resource_bottlenecks",
                    "priority": "high",
                    "target": "mycelial_flows"
                })
            elif bottleneck_severity == "moderate":
                recommendations.append({
                    "action": "optimize_resource_allocation",
                    "reason": "moderate_bottlenecks",
                    "priority": "medium",
                    "target": "flow_optimization"
                })
        
        # Memory consolidation recommendations
        if 'memory_consolidation' in synthesis:
            consolidation_health = synthesis['memory_consolidation']['crystallization_health']
            if consolidation_health == "poor":
                recommendations.append({
                    "action": "enhance_owl_archival",
                    "reason": "poor_memory_consolidation",
                    "priority": "medium",
                    "target": "memory_systems"
                })
        
        # Structural integrity recommendations
        if 'structural_integrity' in synthesis:
            fragmentation_risk = synthesis['structural_integrity']['fragmentation_risk']
            if fragmentation_risk > 0.7:
                recommendations.append({
                    "action": "deploy_spider_tracers",
                    "reason": "high_fragmentation_risk",
                    "priority": "high",
                    "target": "schema_edges"
                })
        
        return recommendations
    
    def _calculate_confidence_score(self) -> float:
        """Calculate confidence in the analysis based on data completeness"""
        data_completeness = len(self.context_data) / len(self.scan_layers)
        analysis_depth = len(self.trend_analysis) / 4  # Expected trend categories
        synthesis_breadth = len(self.insights_generated)
        
        confidence = (data_completeness * 0.4 + analysis_depth * 0.4 + 
                     min(synthesis_breadth / 3, 1.0) * 0.2)
        
        return min(1.0, confidence)
    
    def _create_scan_completion_report(self, context: Dict[str, Any]) -> TracerReport:
        """Create report when deep scan phase completes"""
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="deep_scan_complete",
            metadata={
                "layers_scanned": list(self.context_data.keys()),
                "scan_duration": context.get('tick_id', 0) - self.spawn_tick,
                "data_points_collected": sum(len(v) if isinstance(v, (list, dict)) else 1 
                                           for v in self.context_data.values()),
                "next_phase": "analysis"
            }
        )
    
    # Helper methods for analysis
    def _analyze_pigment_distribution(self, pigments: List[List[float]]) -> List[float]:
        """Analyze RGB pigment distribution"""
        if not pigments:
            return [0.0, 0.0, 0.0]
        
        r_total = sum(p[0] for p in pigments if len(p) > 0)
        g_total = sum(p[1] for p in pigments if len(p) > 1)
        b_total = sum(p[2] for p in pigments if len(p) > 2)
        
        total = r_total + g_total + b_total
        return [r_total/total, g_total/total, b_total/total] if total > 0 else [0.0, 0.0, 0.0]
    
    def _analyze_age_distribution(self, fragments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze age distribution of fragments"""
        ages = [f.get('age', 0) for f in fragments]
        if not ages:
            return {"mean": 0.0, "max": 0.0, "variance": 0.0}
        
        return {
            "mean": float(np.mean(ages)),
            "max": float(max(ages)),
            "variance": float(np.var(ages)) if len(ages) > 1 else 0.0
        }
    
    def _build_connectivity_matrix(self, clusters: List[Dict[str, Any]]) -> List[List[float]]:
        """Build connectivity matrix between clusters"""
        n = len(clusters)
        if n == 0:
            return []
        
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i, cluster in enumerate(clusters):
            connections = cluster.get('connections', [])
            for conn in connections:
                target_id = conn.get('target_cluster')
                strength = conn.get('strength', 0.0)
                
                for j, target_cluster in enumerate(clusters):
                    if target_cluster.get('id') == target_id:
                        matrix[i][j] = strength
                        break
        
        return matrix
    
    def _calculate_fragmentation_score(self, clusters: List[Dict[str, Any]]) -> float:
        """Calculate schema fragmentation score"""
        if len(clusters) <= 1:
            return 0.0
        
        total_connections = 0
        possible_connections = len(clusters) * (len(clusters) - 1)
        
        for cluster in clusters:
            total_connections += len(cluster.get('connections', []))
        
        connectivity_ratio = total_connections / possible_connections if possible_connections > 0 else 0
        return 1.0 - connectivity_ratio  # Higher fragmentation = lower connectivity
    
    def _analyze_flow_balance(self, flows: List[Dict[str, Any]]) -> float:
        """Analyze balance of mycelial flows"""
        if not flows:
            return 1.0
        
        throughputs = [f.get('throughput', 0) for f in flows]
        if not throughputs:
            return 1.0
        
        mean_throughput = np.mean(throughputs)
        if mean_throughput == 0:
            return 1.0
        
        variance = np.var(throughputs)
        coefficient_of_variation = np.sqrt(variance) / mean_throughput
        
        # Lower coefficient = better balance
        return max(0.0, 1.0 - coefficient_of_variation)
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values"""
        if len(values) < 2:
            return "unknown"
        
        slope = (values[-1] - values[0]) / (len(values) - 1)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_acceleration(self, values: List[float]) -> float:
        """Calculate acceleration (second derivative) of values"""
        if len(values) < 3:
            return 0.0
        
        # Simple second difference
        first_diff = values[-1] - values[-2]
        second_diff = values[-2] - values[-3]
        
        return first_diff - second_diff
    
    def _calculate_correlation(self, series1: List[float], series2: List[float]) -> float:
        """Calculate correlation between two series"""
        min_len = min(len(series1), len(series2))
        if min_len < 2:
            return 0.0
        
        s1 = series1[-min_len:]
        s2 = series2[-min_len:]
        
        return float(np.corrcoef(s1, s2)[0, 1]) if np.var(s1) > 0 and np.var(s2) > 0 else 0.0
    
    def _detect_pressure_cycles(self, pressure_history: List[float]) -> int:
        """Detect cyclical patterns in pressure"""
        if len(pressure_history) < 6:
            return 0
        
        peaks = 0
        for i in range(1, len(pressure_history) - 1):
            if (pressure_history[i] > pressure_history[i-1] and 
                pressure_history[i] > pressure_history[i+1]):
                peaks += 1
        
        return peaks
    
    def _calculate_stability_index(self, values: List[float]) -> float:
        """Calculate stability index (inverse of volatility)"""
        if len(values) < 2:
            return 1.0
        
        volatility = np.std(values)
        mean_val = np.mean(values)
        
        if mean_val == 0:
            return 1.0 if volatility == 0 else 0.0
        
        coefficient_of_variation = volatility / mean_val
        return max(0.0, 1.0 - coefficient_of_variation)
    
    def _analyze_cross_layer_correlations(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze correlations between different system layers"""
        correlations = {}
        
        # Example correlations (would be expanded based on available data)
        if 'ash_analysis' in self.context_data and 'schema_analysis' in self.context_data:
            ash_health = self.context_data['ash_analysis']['avg_crystallization']
            schema_health = self.context_data['schema_analysis']['avg_coherence']
            correlations['ash_schema_correlation'] = abs(ash_health - schema_health)  # Simplified
        
        return correlations
    
    def _get_health_assessment(self, score: float) -> str:
        """Convert health score to textual assessment"""
        if score > 0.8:
            return "excellent"
        elif score > 0.6:
            return "good"
        elif score > 0.4:
            return "concerning"
        elif score > 0.2:
            return "poor"
        else:
            return "critical"
    
    def _identify_health_risk_factors(self) -> List[str]:
        """Identify risk factors affecting system health"""
        risk_factors = []
        
        if 'schema_analysis' in self.context_data:
            if self.context_data['schema_analysis']['fragmentation_score'] > 0.6:
                risk_factors.append("high_fragmentation")
            if self.context_data['schema_analysis']['coherence_variance'] > 0.3:
                risk_factors.append("coherence_instability")
        
        if 'mycelial_analysis' in self.context_data:
            if len(self.context_data['mycelial_analysis']['bottlenecks']) > 3:
                risk_factors.append("resource_bottlenecks")
        
        return risk_factors
    
    def _assess_bottleneck_severity(self, bottlenecks: List[str]) -> str:
        """Assess severity of resource bottlenecks"""
        count = len(bottlenecks)
        if count == 0:
            return "none"
        elif count <= 2:
            return "mild"
        elif count <= 5:
            return "moderate"
        else:
            return "severe"
    
    def _assess_crystallization_health(self, ash_data: Dict[str, Any]) -> str:
        """Assess health of memory crystallization process"""
        avg_crystallization = ash_data.get('avg_crystallization', 0)
        variance = ash_data.get('crystallization_variance', 0)
        
        if avg_crystallization > 0.7 and variance < 0.2:
            return "excellent"
        elif avg_crystallization > 0.5 and variance < 0.3:
            return "good"
        elif avg_crystallization > 0.3:
            return "fair"
        else:
            return "poor"
    
    def _estimate_memory_retention(self, ash_data: Dict[str, Any]) -> float:
        """Estimate memory retention capacity"""
        crystallization = ash_data.get('avg_crystallization', 0)
        fragment_count = ash_data.get('total_fragments', 0)
        
        # Simple estimation based on crystallization quality and quantity
        quality_factor = crystallization
        quantity_factor = min(1.0, fragment_count / 100.0)  # Normalized to 100 fragments
        
        return (quality_factor * 0.7 + quantity_factor * 0.3)
    
    def _assess_structural_stress(self, schema_data: Dict[str, Any]) -> float:
        """Assess structural stress in schema"""
        coherence_variance = schema_data.get('coherence_variance', 0)
        fragmentation = schema_data.get('fragmentation_score', 0)
        avg_coherence = schema_data.get('avg_coherence', 1.0)
        
        # Stress increases with variance, fragmentation, and decreases with coherence
        stress = (coherence_variance * 0.4 + fragmentation * 0.4 + (1.0 - avg_coherence) * 0.2)
        
        return min(1.0, stress)
    
    def _generate_predictive_indicators(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictive indicators for future system behavior"""
        indicators = {}
        
        # Entropy trajectory prediction
        entropy_history = context.get('entropy_history', [])
        if len(entropy_history) >= 3:
            recent_trend = self._calculate_trend_direction(entropy_history[-3:])
            indicators['entropy_trajectory'] = {
                'direction': recent_trend,
                'confidence': 0.7 if len(entropy_history) >= 5 else 0.5
            }
        
        # System stability forecast
        if self.context_data:
            stability_factors = []
            
            if 'schema_analysis' in self.context_data:
                stability_factors.append(self.context_data['schema_analysis']['avg_coherence'])
            
            if 'mycelial_analysis' in self.context_data:
                stability_factors.append(self.context_data['mycelial_analysis']['flow_efficiency'])
            
            if stability_factors:
                avg_stability = np.mean(stability_factors)
                indicators['stability_forecast'] = {
                    'score': float(avg_stability),
                    'outlook': 'stable' if avg_stability > 0.6 else 'volatile'
                }
        
        return indicators
    
    def should_retire(self, context: Dict[str, Any]) -> bool:
        """Retire after completing deep analysis or when coherence stabilizes"""
        current_tick = context.get('tick_id', 0)
        age = self.get_age(current_tick)
        
        # Retire after lifespan
        if age >= self.base_lifespan:
            return True
        
        # Early retirement if synthesis phase is complete and insights generated
        if self.analysis_phase == "synthesis" and self.insights_generated:
            return True
        
        # Early retirement if coherence has stabilized (job done)
        if age > 15:
            recent_entropy = context.get('entropy_history', [])[-3:] if context.get('entropy_history') else []
            if recent_entropy and np.std(recent_entropy) < 0.1:
                # System is stable, deep analysis no longer needed
                return True
        
        # Emergency retirement if system state becomes too chaotic for analysis
        if age > 10:
            entropy = context.get('entropy', 0)
            if entropy > 0.9:  # Extreme chaos
                logger.warning(f"Whale {self.tracer_id} emergency retirement due to extreme system entropy")
                return True
        
        return False

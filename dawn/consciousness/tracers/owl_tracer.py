"""
Owl Tracer - Long-Memory Auditor & Archivist

Preserves and audits DAWN's long-term memory - dialogue, internal reflections,
and slow-forming context. Converts lived tick history into durable records,
feeding back summaries and corrections to stabilize schema over time.
"""

from typing import Dict, Any, List, Optional
from .base_tracer import BaseTracer, TracerType, TracerReport, AlertSeverity
import logging
import json

logger = logging.getLogger(__name__)


class OwlTracer(BaseTracer):
    """Long-memory auditor and archivist - the archivist of DAWN"""
    
    def __init__(self, tracer_id: str = None):
        super().__init__(tracer_id)
        self.audit_window_start = None
        self.audit_window_end = None
        self.dialogue_logs = []
        self.critical_decisions = []
        self.rebloom_events = []
        self.schema_drift_measurements = []
        self.residue_balance_history = []
        self.epoch_summary = {}
        self.archival_complete = False
        self.audit_phase = "collection"  # collection -> analysis -> archival
        
    @property
    def tracer_type(self) -> TracerType:
        return TracerType.OWL
    
    @property
    def base_lifespan(self) -> int:
        return 50  # Long-lived, epoch scale
    
    @property
    def base_nutrient_cost(self) -> float:
        return 1.5  # High archival overhead
    
    @property
    def archetype_description(self) -> str:
        return "Long-memory auditor - archives, audits, and summarizes epochs"
    
    def spawn_conditions_met(self, context: Dict[str, Any]) -> bool:
        """Spawn on schedule or when SHI variance exceeds threshold"""
        current_tick = context.get('tick_id', 0)
        
        # Scheduled spawning (epoch boundaries)
        if current_tick % 100 == 0:  # Every 100 ticks (epoch boundary)
            return True
        
        # SHI variance indicates coherence wobble
        shi_variance = context.get('shi_variance', 0.0)
        if shi_variance > context.get('shi_variance_threshold', 0.3):
            return True
        
        # Significant system events requiring archival
        significant_events = context.get('significant_events', [])
        if len(significant_events) > 5:
            return True
        
        # Explicit archival requests
        if context.get('archival_requested', False):
            return True
        
        # Memory pressure requiring consolidation
        memory_pressure = context.get('memory_pressure', 0.0)
        if memory_pressure > 0.8:
            return True
        
        # Long period without archival (failsafe)
        last_archival = context.get('last_archival_tick', 0)
        if current_tick - last_archival > 200:  # 200 ticks without archival
            return True
        
        return False
    
    def observe(self, context: Dict[str, Any]) -> List[TracerReport]:
        """Execute archival process: collect -> analyze -> archive"""
        reports = []
        current_tick = context.get('tick_id', 0)
        
        # Set audit window on first observation
        if self.audit_window_start is None:
            self.audit_window_start = max(0, current_tick - 50)  # Look back 50 ticks
            self.audit_window_end = current_tick
        
        if self.audit_phase == "collection":
            success = self._collection_phase(context)
            if success:
                self.audit_phase = "analysis"
                reports.append(self._create_collection_report(context))
                
        elif self.audit_phase == "analysis":
            analysis_results = self._analysis_phase(context)
            if analysis_results:
                self.audit_phase = "archival"
                reports.append(self._create_analysis_report(context, analysis_results))
                
        elif self.audit_phase == "archival":
            archival_results = self._archival_phase(context)
            if archival_results:
                self.archival_complete = True
                reports.append(self._create_archival_report(context, archival_results))
        
        return reports
    
    def _collection_phase(self, context: Dict[str, Any]) -> bool:
        """Phase 1: Collect data from audit window"""
        # Collect dialogue and reflection logs
        self._collect_dialogue_logs(context)
        
        # Collect critical decisions and their context
        self._collect_critical_decisions(context)
        
        # Collect rebloom events (Juliet events)
        self._collect_rebloom_events(context)
        
        # Collect schema drift measurements
        self._collect_schema_drift_data(context)
        
        # Collect residue balance history
        self._collect_residue_balance_data(context)
        
        return True  # Collection always succeeds
    
    def _analysis_phase(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Phase 2: Analyze collected data for patterns and insights"""
        analysis = {}
        
        # Analyze dialogue patterns
        if self.dialogue_logs:
            analysis['dialogue_analysis'] = self._analyze_dialogue_patterns()
        
        # Analyze decision quality and outcomes
        if self.critical_decisions:
            analysis['decision_analysis'] = self._analyze_decision_patterns()
        
        # Analyze rebloom patterns (memory access effects)
        if self.rebloom_events:
            analysis['rebloom_analysis'] = self._analyze_rebloom_patterns()
        
        # Analyze schema drift trends
        if self.schema_drift_measurements:
            analysis['drift_analysis'] = self._analyze_drift_patterns()
        
        # Analyze residue balance evolution
        if self.residue_balance_history:
            analysis['residue_analysis'] = self._analyze_residue_patterns()
        
        # Cross-correlation analysis
        analysis['correlations'] = self._analyze_cross_correlations()
        
        return analysis if analysis else None
    
    def _archival_phase(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Phase 3: Create durable archives and generate recommendations"""
        # Generate epoch summary
        self.epoch_summary = self._generate_epoch_summary(context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(context)
        
        # Create replay indices for critical paths
        replay_indices = self._create_replay_indices()
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics()
        
        archival_results = {
            'epoch_summary': self.epoch_summary,
            'recommendations': recommendations,
            'replay_indices': replay_indices,
            'stability_metrics': stability_metrics,
            'archive_integrity': self._verify_archive_integrity()
        }
        
        return archival_results
    
    def _collect_dialogue_logs(self, context: Dict[str, Any]) -> None:
        """Collect system dialogue and reflection logs"""
        # In a real implementation, this would pull from fractal memory logs
        dialogue_events = context.get('dialogue_events', [])
        
        for event in dialogue_events:
            if (self.audit_window_start <= event.get('tick', 0) <= self.audit_window_end):
                self.dialogue_logs.append({
                    'tick': event.get('tick'),
                    'type': event.get('type', 'dialogue'),
                    'content': event.get('content', ''),
                    'participants': event.get('participants', []),
                    'context': event.get('context', {}),
                    'significance': event.get('significance', 0.5)
                })
    
    def _collect_critical_decisions(self, context: Dict[str, Any]) -> None:
        """Collect critical decisions and their context"""
        decision_events = context.get('decision_events', [])
        
        for decision in decision_events:
            if (self.audit_window_start <= decision.get('tick', 0) <= self.audit_window_end):
                self.critical_decisions.append({
                    'tick': decision.get('tick'),
                    'decision_id': decision.get('id'),
                    'decision_type': decision.get('type'),
                    'pressure': decision.get('pressure', 0),
                    'entropy': decision.get('entropy', 0),
                    'outcome': decision.get('outcome'),
                    'consequences': decision.get('consequences', []),
                    'confidence': decision.get('confidence', 0.5)
                })
    
    def _collect_rebloom_events(self, context: Dict[str, Any]) -> None:
        """Collect Juliet rebloom events"""
        rebloom_events = context.get('rebloom_events', [])
        
        for event in rebloom_events:
            if (self.audit_window_start <= event.get('tick', 0) <= self.audit_window_end):
                self.rebloom_events.append({
                    'tick': event.get('tick'),
                    'memory_id': event.get('memory_id'),
                    'access_depth': event.get('depth', 1),
                    'access_type': event.get('access_type', 'read'),
                    'modification': event.get('modification', False),
                    'intensity': event.get('intensity', 0.5),
                    'cascading_effects': event.get('cascading_effects', [])
                })
    
    def _collect_schema_drift_data(self, context: Dict[str, Any]) -> None:
        """Collect schema drift measurements"""
        current_tick = context.get('tick_id', 0)
        
        # Collect recent drift measurements within audit window
        drift_history = context.get('drift_history', [])
        tick_start = max(0, len(drift_history) - (self.audit_window_end - self.audit_window_start))
        
        for i, drift_value in enumerate(drift_history[tick_start:]):
            tick = self.audit_window_start + i
            self.schema_drift_measurements.append({
                'tick': tick,
                'drift_magnitude': drift_value,
                'hot_edges': context.get('hot_edges', []),
                'coherence_drop': context.get('coherence_drops', {}).get(str(tick), 0)
            })
    
    def _collect_residue_balance_data(self, context: Dict[str, Any]) -> None:
        """Collect residue balance evolution data"""
        # Collect soot/ash ratio history
        soot_history = context.get('soot_ratio_history', [])
        ash_history = context.get('ash_ratio_history', [])
        
        min_len = min(len(soot_history), len(ash_history))
        window_size = self.audit_window_end - self.audit_window_start
        start_idx = max(0, min_len - window_size)
        
        for i in range(start_idx, min_len):
            tick = self.audit_window_start + (i - start_idx)
            self.residue_balance_history.append({
                'tick': tick,
                'soot_ratio': soot_history[i] if i < len(soot_history) else 0,
                'ash_ratio': ash_history[i] if i < len(ash_history) else 0,
                'balance_score': self._calculate_balance_score(
                    soot_history[i] if i < len(soot_history) else 0,
                    ash_history[i] if i < len(ash_history) else 0
                )
            })
    
    def _analyze_dialogue_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in dialogue and reflection logs"""
        if not self.dialogue_logs:
            return {}
        
        # Dialogue frequency analysis
        dialogue_frequency = len(self.dialogue_logs) / max(1, self.audit_window_end - self.audit_window_start)
        
        # Significance distribution
        significance_scores = [log.get('significance', 0.5) for log in self.dialogue_logs]
        avg_significance = sum(significance_scores) / len(significance_scores)
        
        # Participant analysis
        all_participants = []
        for log in self.dialogue_logs:
            all_participants.extend(log.get('participants', []))
        
        participant_frequency = {}
        for participant in all_participants:
            participant_frequency[participant] = participant_frequency.get(participant, 0) + 1
        
        return {
            'total_dialogues': len(self.dialogue_logs),
            'dialogue_frequency': dialogue_frequency,
            'avg_significance': avg_significance,
            'participant_distribution': participant_frequency,
            'dialogue_quality': 'high' if avg_significance > 0.7 else 'medium' if avg_significance > 0.4 else 'low'
        }
    
    def _analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analyze critical decision patterns and quality"""
        if not self.critical_decisions:
            return {}
        
        # Decision quality metrics
        confidence_scores = [d.get('confidence', 0.5) for d in self.critical_decisions]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Pressure/entropy correlation
        pressure_values = [d.get('pressure', 0) for d in self.critical_decisions]
        entropy_values = [d.get('entropy', 0) for d in self.critical_decisions]
        
        # Decision outcome analysis
        outcomes = [d.get('outcome') for d in self.critical_decisions if d.get('outcome')]
        positive_outcomes = len([o for o in outcomes if o == 'positive'])
        outcome_ratio = positive_outcomes / len(outcomes) if outcomes else 0.5
        
        return {
            'total_decisions': len(self.critical_decisions),
            'avg_confidence': avg_confidence,
            'avg_pressure_context': sum(pressure_values) / len(pressure_values) if pressure_values else 0,
            'avg_entropy_context': sum(entropy_values) / len(entropy_values) if entropy_values else 0,
            'positive_outcome_ratio': outcome_ratio,
            'decision_quality': self._assess_decision_quality(avg_confidence, outcome_ratio)
        }
    
    def _analyze_rebloom_patterns(self) -> Dict[str, Any]:
        """Analyze rebloom (memory access) patterns"""
        if not self.rebloom_events:
            return {}
        
        # Rebloom frequency and intensity
        total_reblooms = len(self.rebloom_events)
        avg_intensity = sum(e.get('intensity', 0.5) for e in self.rebloom_events) / total_reblooms
        
        # Memory access depth analysis
        depths = [e.get('access_depth', 1) for e in self.rebloom_events]
        avg_depth = sum(depths) / len(depths)
        deep_accesses = len([d for d in depths if d > 2])
        
        # Modification rate
        modifications = len([e for e in self.rebloom_events if e.get('modification', False)])
        modification_rate = modifications / total_reblooms
        
        # Cascading effects analysis
        total_cascades = sum(len(e.get('cascading_effects', [])) for e in self.rebloom_events)
        avg_cascades = total_cascades / total_reblooms
        
        return {
            'total_reblooms': total_reblooms,
            'avg_intensity': avg_intensity,
            'avg_access_depth': avg_depth,
            'deep_access_count': deep_accesses,
            'modification_rate': modification_rate,
            'avg_cascading_effects': avg_cascades,
            'rebloom_health': self._assess_rebloom_health(modification_rate, avg_cascades)
        }
    
    def _analyze_drift_patterns(self) -> Dict[str, Any]:
        """Analyze schema drift patterns"""
        if not self.schema_drift_measurements:
            return {}
        
        drift_values = [m.get('drift_magnitude', 0) for m in self.schema_drift_measurements]
        
        # Drift statistics
        avg_drift = sum(drift_values) / len(drift_values)
        max_drift = max(drift_values)
        drift_variance = sum((x - avg_drift) ** 2 for x in drift_values) / len(drift_values)
        
        # Trend analysis
        if len(drift_values) >= 3:
            recent_trend = (drift_values[-1] - drift_values[0]) / len(drift_values)
            trend_direction = 'increasing' if recent_trend > 0.1 else 'decreasing' if recent_trend < -0.1 else 'stable'
        else:
            trend_direction = 'unknown'
        
        # Hot edge analysis
        all_hot_edges = []
        for measurement in self.schema_drift_measurements:
            all_hot_edges.extend(measurement.get('hot_edges', []))
        
        hot_edge_frequency = {}
        for edge in all_hot_edges:
            hot_edge_frequency[edge] = hot_edge_frequency.get(edge, 0) + 1
        
        return {
            'avg_drift_magnitude': avg_drift,
            'max_drift': max_drift,
            'drift_variance': drift_variance,
            'trend_direction': trend_direction,
            'hot_edge_frequency': hot_edge_frequency,
            'drift_stability': 'stable' if drift_variance < 0.1 else 'volatile'
        }
    
    def _analyze_residue_patterns(self) -> Dict[str, Any]:
        """Analyze residue balance patterns"""
        if not self.residue_balance_history:
            return {}
        
        soot_ratios = [h.get('soot_ratio', 0) for h in self.residue_balance_history]
        ash_ratios = [h.get('ash_ratio', 0) for h in self.residue_balance_history]
        balance_scores = [h.get('balance_score', 0.5) for h in self.residue_balance_history]
        
        return {
            'avg_soot_ratio': sum(soot_ratios) / len(soot_ratios),
            'avg_ash_ratio': sum(ash_ratios) / len(ash_ratios),
            'avg_balance_score': sum(balance_scores) / len(balance_scores),
            'soot_trend': self._calculate_trend(soot_ratios),
            'ash_trend': self._calculate_trend(ash_ratios),
            'balance_health': 'good' if sum(balance_scores) / len(balance_scores) > 0.6 else 'poor'
        }
    
    def _analyze_cross_correlations(self) -> Dict[str, float]:
        """Analyze correlations between different system aspects"""
        correlations = {}
        
        # Correlation between decisions and drift
        if self.critical_decisions and self.schema_drift_measurements:
            decision_confidence = [d.get('confidence', 0.5) for d in self.critical_decisions]
            drift_magnitudes = [m.get('drift_magnitude', 0) for m in self.schema_drift_measurements]
            
            # Simple correlation approximation
            if len(decision_confidence) == len(drift_magnitudes):
                correlations['decision_drift_correlation'] = self._simple_correlation(decision_confidence, drift_magnitudes)
        
        # Correlation between reblooms and residue balance
        if self.rebloom_events and self.residue_balance_history:
            rebloom_intensities = [e.get('intensity', 0.5) for e in self.rebloom_events]
            balance_scores = [h.get('balance_score', 0.5) for h in self.residue_balance_history]
            
            if len(rebloom_intensities) == len(balance_scores):
                correlations['rebloom_balance_correlation'] = self._simple_correlation(rebloom_intensities, balance_scores)
        
        return correlations
    
    def _generate_epoch_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive epoch summary"""
        current_tick = context.get('tick_id', 0)
        
        summary = {
            'epoch_info': {
                'start_tick': self.audit_window_start,
                'end_tick': self.audit_window_end,
                'duration': self.audit_window_end - self.audit_window_start,
                'summary_generated_at': current_tick
            },
            'system_activity': {
                'dialogue_events': len(self.dialogue_logs),
                'critical_decisions': len(self.critical_decisions),
                'rebloom_events': len(self.rebloom_events),
                'drift_measurements': len(self.schema_drift_measurements)
            },
            'key_metrics': self._calculate_key_metrics(),
            'significant_events': self._identify_significant_events(),
            'health_assessment': self._generate_health_assessment(),
            'narrative_summary': self._generate_narrative_summary()
        }
        
        return summary
    
    def _generate_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Rebloom management recommendations
        if self.rebloom_events:
            rebloom_analysis = self._analyze_rebloom_patterns()
            if rebloom_analysis.get('modification_rate', 0) > 0.7:
                recommendations.append({
                    'action': 'cap_rebloom_rate',
                    'reason': 'high_modification_rate',
                    'priority': 'high',
                    'details': f"Modification rate: {rebloom_analysis['modification_rate']:.2f}"
                })
        
        # Schema drift recommendations
        if self.schema_drift_measurements:
            drift_analysis = self._analyze_drift_patterns()
            if drift_analysis.get('avg_drift_magnitude', 0) > 0.5:
                recommendations.append({
                    'action': 'deploy_spider_tracers',
                    'reason': 'high_schema_drift',
                    'priority': 'medium',
                    'details': f"Average drift: {drift_analysis['avg_drift_magnitude']:.2f}"
                })
        
        # Residue balance recommendations
        if self.residue_balance_history:
            residue_analysis = self._analyze_residue_patterns()
            if residue_analysis.get('avg_soot_ratio', 0) > 0.6:
                recommendations.append({
                    'action': 'increase_beetle_activity',
                    'reason': 'high_soot_accumulation',
                    'priority': 'medium',
                    'details': f"Soot ratio: {residue_analysis['avg_soot_ratio']:.2f}"
                })
        
        # Decision quality recommendations
        if self.critical_decisions:
            decision_analysis = self._analyze_decision_patterns()
            if decision_analysis.get('avg_confidence', 0.5) < 0.4:
                recommendations.append({
                    'action': 'enhance_decision_support',
                    'reason': 'low_decision_confidence',
                    'priority': 'high',
                    'details': f"Average confidence: {decision_analysis['avg_confidence']:.2f}"
                })
        
        return recommendations
    
    def _create_replay_indices(self) -> List[Dict[str, Any]]:
        """Create replay indices for critical decision paths"""
        replay_indices = []
        
        # Create indices for high-impact decisions
        for decision in self.critical_decisions:
            if decision.get('significance', 0.5) > 0.7:
                replay_indices.append({
                    'type': 'critical_decision',
                    'tick': decision.get('tick'),
                    'decision_id': decision.get('decision_id'),
                    'context_window': [
                        max(0, decision.get('tick', 0) - 5),
                        decision.get('tick', 0) + 5
                    ],
                    'reconstruction_keys': {
                        'pressure': decision.get('pressure'),
                        'entropy': decision.get('entropy'),
                        'decision_type': decision.get('decision_type')
                    }
                })
        
        # Create indices for significant rebloom cascades
        for rebloom in self.rebloom_events:
            cascades = rebloom.get('cascading_effects', [])
            if len(cascades) > 3:  # Significant cascade
                replay_indices.append({
                    'type': 'rebloom_cascade',
                    'tick': rebloom.get('tick'),
                    'memory_id': rebloom.get('memory_id'),
                    'cascade_count': len(cascades),
                    'reconstruction_keys': {
                        'access_depth': rebloom.get('access_depth'),
                        'intensity': rebloom.get('intensity'),
                        'cascading_effects': cascades
                    }
                })
        
        return replay_indices
    
    def _calculate_stability_metrics(self) -> Dict[str, float]:
        """Calculate overall stability metrics for the epoch"""
        metrics = {}
        
        # Decision stability
        if self.critical_decisions:
            confidence_scores = [d.get('confidence', 0.5) for d in self.critical_decisions]
            metrics['decision_stability'] = sum(confidence_scores) / len(confidence_scores)
        
        # Schema stability
        if self.schema_drift_measurements:
            drift_values = [m.get('drift_magnitude', 0) for m in self.schema_drift_measurements]
            drift_variance = sum((x - sum(drift_values)/len(drift_values)) ** 2 for x in drift_values) / len(drift_values)
            metrics['schema_stability'] = max(0.0, 1.0 - drift_variance)
        
        # Memory stability
        if self.rebloom_events:
            modification_rate = len([e for e in self.rebloom_events if e.get('modification', False)]) / len(self.rebloom_events)
            metrics['memory_stability'] = 1.0 - modification_rate
        
        # Overall stability
        if metrics:
            metrics['overall_stability'] = sum(metrics.values()) / len(metrics)
        
        return metrics
    
    def _verify_archive_integrity(self) -> Dict[str, Any]:
        """Verify integrity of archived data"""
        integrity = {
            'data_completeness': 0.0,
            'consistency_check': True,
            'corruption_detected': False
        }
        
        # Check data completeness
        expected_data_types = ['dialogue_logs', 'critical_decisions', 'rebloom_events', 'schema_drift_measurements', 'residue_balance_history']
        present_data_types = []
        
        if self.dialogue_logs:
            present_data_types.append('dialogue_logs')
        if self.critical_decisions:
            present_data_types.append('critical_decisions')
        if self.rebloom_events:
            present_data_types.append('rebloom_events')
        if self.schema_drift_measurements:
            present_data_types.append('schema_drift_measurements')
        if self.residue_balance_history:
            present_data_types.append('residue_balance_history')
        
        integrity['data_completeness'] = len(present_data_types) / len(expected_data_types)
        
        # Basic consistency checks
        # Check for temporal consistency
        all_ticks = []
        for log in self.dialogue_logs:
            all_ticks.append(log.get('tick', 0))
        for decision in self.critical_decisions:
            all_ticks.append(decision.get('tick', 0))
        
        if all_ticks:
            integrity['temporal_consistency'] = all(self.audit_window_start <= tick <= self.audit_window_end for tick in all_ticks)
        
        return integrity
    
    # Helper methods
    def _calculate_balance_score(self, soot_ratio: float, ash_ratio: float) -> float:
        """Calculate balance score from soot and ash ratios"""
        total_ratio = soot_ratio + ash_ratio
        if total_ratio == 0:
            return 0.5  # Neutral
        
        # Ideal balance is roughly 30% soot, 70% ash
        ideal_soot = 0.3
        ideal_ash = 0.7
        
        soot_deviation = abs((soot_ratio / total_ratio) - ideal_soot)
        ash_deviation = abs((ash_ratio / total_ratio) - ideal_ash)
        
        balance_score = 1.0 - (soot_deviation + ash_deviation) / 2.0
        return max(0.0, balance_score)
    
    def _assess_decision_quality(self, avg_confidence: float, outcome_ratio: float) -> str:
        """Assess overall decision quality"""
        combined_score = (avg_confidence + outcome_ratio) / 2.0
        
        if combined_score > 0.8:
            return "excellent"
        elif combined_score > 0.6:
            return "good"
        elif combined_score > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _assess_rebloom_health(self, modification_rate: float, avg_cascades: float) -> str:
        """Assess rebloom system health"""
        # Moderate modification and cascade rates are healthy
        if 0.2 <= modification_rate <= 0.5 and avg_cascades < 2.0:
            return "healthy"
        elif modification_rate > 0.7 or avg_cascades > 3.0:
            return "concerning"
        else:
            return "stable"
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "unknown"
        
        slope = (values[-1] - values[0]) / len(values)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _simple_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate simple correlation between two series"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)
        
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        correlation = (n * sum_xy - sum_x * sum_y) / denominator
        return max(-1.0, min(1.0, correlation))
    
    def _calculate_key_metrics(self) -> Dict[str, float]:
        """Calculate key metrics for the epoch"""
        metrics = {}
        
        if self.critical_decisions:
            confidence_scores = [d.get('confidence', 0.5) for d in self.critical_decisions]
            metrics['avg_decision_confidence'] = sum(confidence_scores) / len(confidence_scores)
        
        if self.rebloom_events:
            intensities = [e.get('intensity', 0.5) for e in self.rebloom_events]
            metrics['avg_rebloom_intensity'] = sum(intensities) / len(intensities)
        
        if self.schema_drift_measurements:
            drifts = [m.get('drift_magnitude', 0) for m in self.schema_drift_measurements]
            metrics['avg_schema_drift'] = sum(drifts) / len(drifts)
        
        return metrics
    
    def _identify_significant_events(self) -> List[Dict[str, Any]]:
        """Identify the most significant events in the epoch"""
        significant_events = []
        
        # High-significance dialogues
        for log in self.dialogue_logs:
            if log.get('significance', 0.5) > 0.8:
                significant_events.append({
                    'type': 'significant_dialogue',
                    'tick': log.get('tick'),
                    'significance': log.get('significance'),
                    'content_summary': log.get('content', '')[:100] + '...' if len(log.get('content', '')) > 100 else log.get('content', '')
                })
        
        # High-impact decisions
        for decision in self.critical_decisions:
            if decision.get('confidence', 0.5) > 0.8 or len(decision.get('consequences', [])) > 3:
                significant_events.append({
                    'type': 'critical_decision',
                    'tick': decision.get('tick'),
                    'decision_type': decision.get('decision_type'),
                    'impact_level': 'high' if len(decision.get('consequences', [])) > 3 else 'medium'
                })
        
        # Major rebloom cascades
        for rebloom in self.rebloom_events:
            if len(rebloom.get('cascading_effects', [])) > 5:
                significant_events.append({
                    'type': 'major_rebloom_cascade',
                    'tick': rebloom.get('tick'),
                    'memory_id': rebloom.get('memory_id'),
                    'cascade_size': len(rebloom.get('cascading_effects', []))
                })
        
        # Sort by significance/impact
        significant_events.sort(key=lambda x: x.get('significance', 0.5), reverse=True)
        
        return significant_events[:10]  # Top 10 most significant
    
    def _generate_health_assessment(self) -> Dict[str, str]:
        """Generate overall health assessment for the epoch"""
        assessment = {}
        
        # Decision health
        if self.critical_decisions:
            decision_analysis = self._analyze_decision_patterns()
            assessment['decision_health'] = decision_analysis.get('decision_quality', 'unknown')
        
        # Memory health
        if self.rebloom_events:
            rebloom_analysis = self._analyze_rebloom_patterns()
            assessment['memory_health'] = rebloom_analysis.get('rebloom_health', 'unknown')
        
        # Schema health
        if self.schema_drift_measurements:
            drift_analysis = self._analyze_drift_patterns()
            assessment['schema_health'] = drift_analysis.get('drift_stability', 'unknown')
        
        # Residue health
        if self.residue_balance_history:
            residue_analysis = self._analyze_residue_patterns()
            assessment['residue_health'] = residue_analysis.get('balance_health', 'unknown')
        
        return assessment
    
    def _generate_narrative_summary(self) -> str:
        """Generate human-readable narrative summary"""
        summary_parts = []
        
        summary_parts.append(f"Epoch {self.audit_window_start}-{self.audit_window_end} summary:")
        
        if self.dialogue_logs:
            summary_parts.append(f"Recorded {len(self.dialogue_logs)} dialogue events")
        
        if self.critical_decisions:
            summary_parts.append(f"Tracked {len(self.critical_decisions)} critical decisions")
        
        if self.rebloom_events:
            summary_parts.append(f"Observed {len(self.rebloom_events)} memory rebloom events")
        
        # Add key insights
        if self.schema_drift_measurements:
            drift_analysis = self._analyze_drift_patterns()
            drift_stability = drift_analysis.get('drift_stability', 'unknown')
            summary_parts.append(f"Schema drift was {drift_stability}")
        
        if self.residue_balance_history:
            residue_analysis = self._analyze_residue_patterns()
            balance_health = residue_analysis.get('balance_health', 'unknown')
            summary_parts.append(f"Residue balance was {balance_health}")
        
        return " | ".join(summary_parts)
    
    def _create_collection_report(self, context: Dict[str, Any]) -> TracerReport:
        """Create report for data collection phase"""
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="data_collection_complete",
            metadata={
                "audit_window": [self.audit_window_start, self.audit_window_end],
                "data_collected": {
                    "dialogue_logs": len(self.dialogue_logs),
                    "critical_decisions": len(self.critical_decisions),
                    "rebloom_events": len(self.rebloom_events),
                    "drift_measurements": len(self.schema_drift_measurements),
                    "residue_history": len(self.residue_balance_history)
                }
            }
        )
    
    def _create_analysis_report(self, context: Dict[str, Any], analysis_results: Dict[str, Any]) -> TracerReport:
        """Create report for analysis phase"""
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="analysis_complete",
            metadata={
                "analysis_results": analysis_results,
                "insights_generated": len(analysis_results),
                "correlations_found": len(analysis_results.get('correlations', {}))
            }
        )
    
    def _create_archival_report(self, context: Dict[str, Any], archival_results: Dict[str, Any]) -> TracerReport:
        """Create final archival completion report"""
        return TracerReport(
            tracer_id=self.tracer_id,
            tracer_type=self.tracer_type,
            tick_id=context.get('tick_id', 0),
            timestamp=context.get('timestamp', 0),
            severity=AlertSeverity.INFO,
            report_type="long_memory_audit",
            metadata={
                "tick_window": [self.audit_window_start, self.audit_window_end],
                "sources": ["dialogue_log", "decision_events", "rebloom_events", "drift_measurements", "residue_balance"],
                "summary": self.epoch_summary,
                "recommendations": archival_results['recommendations'],
                "archive_integrity": archival_results['archive_integrity'],
                "stability_metrics": archival_results['stability_metrics']
            }
        )
    
    def should_retire(self, context: Dict[str, Any]) -> bool:
        """Retire when archival is complete or lifespan exceeded"""
        current_tick = context.get('tick_id', 0)
        age = self.get_age(current_tick)
        
        # Retire when archival is complete
        if self.archival_complete:
            return True
        
        # Retire after lifespan
        if age >= self.base_lifespan:
            return True
        
        # Early retirement if no significant data collected
        if (age > 10 and not self.dialogue_logs and not self.critical_decisions and 
            not self.rebloom_events and not self.schema_drift_measurements):
            return True
        
        return False

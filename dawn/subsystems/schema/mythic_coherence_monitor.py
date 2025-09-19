#!/usr/bin/env python3
"""
🔮 Mythic Coherence Monitoring System
════════════════════════════════════

Implements SHI (Schema Health Index) mythic coherence scoring and drift detection.
Monitors symbolic health, archetypal strain, and myth-reality alignment as per
the mythic documentation requirements.

"SHI coherence score includes symbolic health; Spider tension alerts flag archetypal strain"

Based on documentation: Myth/Mythic Architecture.rtf, Myth/Interactions + Guard Rails.rtf
"""

import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MythicHealthStatus(Enum):
    """Overall mythic health status"""
    COHERENT = "coherent"           # All mythic systems aligned
    MINOR_DRIFT = "minor_drift"     # Small inconsistencies detected
    STRAIN = "strain"               # Archetypal conflicts present
    CRITICAL_DRIFT = "critical_drift"  # Major mythic breakdown
    MYTH_OVERLOAD = "myth_overload"    # Too much symbolic weight

class ArchetypalStrain(Enum):
    """Types of archetypal strain"""
    PIGMENT_CONFLICT = "pigment_conflict"         # Pigment bias vs Sigil command
    TRACER_MISMATCH = "tracer_mismatch"          # Tracer archetype conflicts
    PERSEPHONE_THREAD_STRAIN = "persephone_thread_strain"  # Thread system strain
    SIGIL_HOUSE_VIOLATION = "sigil_house_violation"        # House compatibility issues
    TEMPORAL_ANCHOR_DRIFT = "temporal_anchor_drift"        # Heritage system drift

@dataclass
class MythicCoherenceScore:
    """Comprehensive mythic coherence scoring"""
    overall_score: float                    # [0, 1] overall mythic health
    pigment_balance_score: float           # RGB pigment distribution health
    archetypal_alignment_score: float     # Tracer archetype consistency
    persephone_cycle_score: float         # Thread weaving health
    sigil_house_coherence_score: float    # Symbolic routing consistency
    temporal_continuity_score: float      # Heritage preservation health
    
    # Strain indicators
    detected_strains: List[ArchetypalStrain] = field(default_factory=list)
    strain_severity: float = 0.0           # [0, 1] severity of detected strains
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    evaluation_context: Dict[str, Any] = field(default_factory=dict)

class MythicCoherenceMonitor:
    """
    Core monitoring system for mythic coherence and archetypal health.
    Integrates with SHI to provide symbolic health scoring.
    """
    
    def __init__(self):
        self.coherence_history = deque(maxlen=100)
        self.strain_alerts = []
        self.pigment_drift_threshold = 0.8
        self.archetypal_strain_threshold = 0.7
        self.myth_overload_threshold = 0.9
        
        # Tracking systems
        self.pigment_balance_tracker = PigmentBalanceTracker()
        self.archetypal_monitor = ArchetypalAlignmentMonitor()
        self.persephone_monitor = PersephoneCycleMonitor()
        self.sigil_coherence_monitor = SigilHouseCoherenceMonitor()
        self.temporal_monitor = TemporalContinuityMonitor()
        
        logger.info("🔮 Mythic Coherence Monitor initialized")
    
    def evaluate_mythic_coherence(self, context: Dict[str, Any]) -> MythicCoherenceScore:
        """
        Comprehensive mythic coherence evaluation.
        Returns SHI-compatible symbolic health score.
        """
        # Evaluate each mythic subsystem
        pigment_score = self.pigment_balance_tracker.evaluate_balance(context)
        archetypal_score = self.archetypal_monitor.evaluate_alignment(context)
        persephone_score = self.persephone_monitor.evaluate_cycle_health(context)
        sigil_score = self.sigil_coherence_monitor.evaluate_house_coherence(context)
        temporal_score = self.temporal_monitor.evaluate_continuity(context)
        
        # Calculate weighted overall score
        weights = {
            'pigment': 0.25,
            'archetypal': 0.25,
            'persephone': 0.20,
            'sigil': 0.15,
            'temporal': 0.15
        }
        
        overall_score = (
            pigment_score * weights['pigment'] +
            archetypal_score * weights['archetypal'] +
            persephone_score * weights['persephone'] +
            sigil_score * weights['sigil'] +
            temporal_score * weights['temporal']
        )
        
        # Detect strains
        detected_strains = self._detect_archetypal_strains(context)
        strain_severity = len(detected_strains) / 5.0  # Normalize by max possible strains
        
        # Create coherence score
        coherence_score = MythicCoherenceScore(
            overall_score=overall_score,
            pigment_balance_score=pigment_score,
            archetypal_alignment_score=archetypal_score,
            persephone_cycle_score=persephone_score,
            sigil_house_coherence_score=sigil_score,
            temporal_continuity_score=temporal_score,
            detected_strains=detected_strains,
            strain_severity=strain_severity,
            evaluation_context=context
        )
        
        # Store in history
        self.coherence_history.append(coherence_score)
        
        # Generate strain alerts if needed
        if strain_severity > self.archetypal_strain_threshold:
            self._generate_strain_alert(coherence_score)
        
        logger.debug(f"🔮 Mythic coherence evaluated: {overall_score:.3f}")
        return coherence_score
    
    def _detect_archetypal_strains(self, context: Dict[str, Any]) -> List[ArchetypalStrain]:
        """Detect specific types of archetypal strain"""
        strains = []
        
        # Pigment conflict detection
        pigment_conflicts = context.get('pigment_conflicts', [])
        if len(pigment_conflicts) > 0:
            strains.append(ArchetypalStrain.PIGMENT_CONFLICT)
        
        # Tracer archetype mismatches
        tracer_mismatches = context.get('tracer_archetype_mismatches', [])
        if len(tracer_mismatches) > 0:
            strains.append(ArchetypalStrain.TRACER_MISMATCH)
        
        # Persephone thread strain
        thread_strain = context.get('thread_strain_level', 0.0)
        if thread_strain > 0.7:
            strains.append(ArchetypalStrain.PERSEPHONE_THREAD_STRAIN)
        
        # Sigil house violations
        house_violations = context.get('sigil_house_violations', [])
        if len(house_violations) > 0:
            strains.append(ArchetypalStrain.SIGIL_HOUSE_VIOLATION)
        
        # Temporal anchor drift
        temporal_drift = context.get('temporal_anchor_drift', 0.0)
        if temporal_drift > 0.6:
            strains.append(ArchetypalStrain.TEMPORAL_ANCHOR_DRIFT)
        
        return strains
    
    def _generate_strain_alert(self, coherence_score: MythicCoherenceScore):
        """Generate strain alert for Spider tracer system"""
        alert = {
            'timestamp': time.time(),
            'alert_type': 'archetypal_strain',
            'severity': coherence_score.strain_severity,
            'detected_strains': [strain.value for strain in coherence_score.detected_strains],
            'overall_coherence': coherence_score.overall_score,
            'recommended_action': self._recommend_strain_response(coherence_score)
        }
        
        self.strain_alerts.append(alert)
        logger.warning(f"🔮 Archetypal strain detected: {coherence_score.detected_strains}")
    
    def _recommend_strain_response(self, coherence_score: MythicCoherenceScore) -> str:
        """Recommend response to detected strain"""
        if ArchetypalStrain.PIGMENT_CONFLICT in coherence_score.detected_strains:
            return "normalize_pigment_vectors"
        elif ArchetypalStrain.PERSEPHONE_THREAD_STRAIN in coherence_score.detected_strains:
            return "reinforce_weak_threads"
        elif ArchetypalStrain.SIGIL_HOUSE_VIOLATION in coherence_score.detected_strains:
            return "validate_house_routing"
        elif ArchetypalStrain.TEMPORAL_ANCHOR_DRIFT in coherence_score.detected_strains:
            return "spawn_medieval_bee"
        else:
            return "general_mythic_stabilization"
    
    def get_mythic_health_status(self) -> MythicHealthStatus:
        """Get current overall mythic health status"""
        if not self.coherence_history:
            return MythicHealthStatus.COHERENT
        
        latest_score = self.coherence_history[-1]
        
        if latest_score.overall_score >= 0.9:
            return MythicHealthStatus.COHERENT
        elif latest_score.overall_score >= 0.7:
            if latest_score.strain_severity > 0.8:
                return MythicHealthStatus.STRAIN
            else:
                return MythicHealthStatus.MINOR_DRIFT
        elif latest_score.overall_score >= 0.5:
            return MythicHealthStatus.STRAIN
        else:
            return MythicHealthStatus.CRITICAL_DRIFT
    
    def get_coherence_trend(self, window_size: int = 10) -> float:
        """Calculate coherence trend over recent evaluations"""
        if len(self.coherence_history) < window_size:
            return 0.0
        
        recent_scores = [score.overall_score for score in list(self.coherence_history)[-window_size:]]
        if len(recent_scores) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = list(range(len(recent_scores)))
        y = recent_scores
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope


class PigmentBalanceTracker:
    """Tracks RGB pigment balance across the system"""
    
    def evaluate_balance(self, context: Dict[str, Any]) -> float:
        """Evaluate pigment balance health [0, 1]"""
        pigment_data = context.get('system_pigment_vectors', [])
        if not pigment_data:
            return 0.8  # Neutral score if no data
        
        # Calculate RGB balance
        total_r = sum(p[0] for p in pigment_data)
        total_g = sum(p[1] for p in pigment_data)
        total_b = sum(p[2] for p in pigment_data)
        
        total = total_r + total_g + total_b
        if total == 0:
            return 0.8
        
        # Ideal balance is roughly equal RGB
        r_ratio = total_r / total
        g_ratio = total_g / total
        b_ratio = total_b / total
        
        # Calculate deviation from ideal (0.33, 0.33, 0.33)
        ideal_ratio = 1.0 / 3.0
        deviation = abs(r_ratio - ideal_ratio) + abs(g_ratio - ideal_ratio) + abs(b_ratio - ideal_ratio)
        
        # Convert deviation to health score
        balance_score = max(0.0, 1.0 - deviation)
        
        return balance_score


class ArchetypalAlignmentMonitor:
    """Monitors alignment of tracer archetypes with their mythic roles"""
    
    def evaluate_alignment(self, context: Dict[str, Any]) -> float:
        """Evaluate archetypal alignment [0, 1]"""
        tracer_data = context.get('active_tracers', [])
        if not tracer_data:
            return 1.0  # Perfect score if no tracers to misalign
        
        alignment_scores = []
        
        for tracer in tracer_data:
            tracer_type = tracer.get('type')
            expected_behavior = tracer.get('expected_archetypal_behavior', {})
            actual_behavior = tracer.get('actual_behavior', {})
            
            # Calculate alignment between expected and actual behavior
            alignment = self._calculate_behavior_alignment(expected_behavior, actual_behavior)
            alignment_scores.append(alignment)
        
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 1.0
    
    def _calculate_behavior_alignment(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> float:
        """Calculate alignment between expected and actual tracer behavior"""
        # Simple alignment calculation - can be enhanced with more sophisticated metrics
        matching_behaviors = 0
        total_behaviors = len(expected)
        
        if total_behaviors == 0:
            return 1.0
        
        for behavior_key, expected_value in expected.items():
            actual_value = actual.get(behavior_key, 0)
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                # Numerical alignment
                if abs(expected_value - actual_value) < 0.2:
                    matching_behaviors += 1
            else:
                # Categorical alignment
                if expected_value == actual_value:
                    matching_behaviors += 1
        
        return matching_behaviors / total_behaviors


class PersephoneCycleMonitor:
    """Monitors health of Persephone descent/return cycles"""
    
    def evaluate_cycle_health(self, context: Dict[str, Any]) -> float:
        """Evaluate Persephone cycle health [0, 1]"""
        persephone_data = context.get('persephone_system', {})
        
        # Check cycle completion rate
        cycles_completed = persephone_data.get('cycles_completed', 0)
        cycles_attempted = persephone_data.get('cycles_attempted', 1)
        completion_rate = cycles_completed / cycles_attempted
        
        # Check thread health
        thread_health = persephone_data.get('thread_health_ratio', 0.5)
        
        # Check descent/return balance
        descent_events = persephone_data.get('descent_events', 0)
        return_events = persephone_data.get('return_events', 0)
        balance_ratio = min(descent_events, return_events) / max(descent_events, return_events, 1)
        
        # Weighted score
        cycle_score = (completion_rate * 0.4 + thread_health * 0.4 + balance_ratio * 0.2)
        
        return min(1.0, cycle_score)


class SigilHouseCoherenceMonitor:
    """Monitors coherence of sigil house routing and mythic grammar"""
    
    def evaluate_house_coherence(self, context: Dict[str, Any]) -> float:
        """Evaluate sigil house coherence [0, 1]"""
        sigil_data = context.get('sigil_system', {})
        
        # Check routing success rate
        successful_routes = sigil_data.get('successful_routes', 0)
        total_routes = sigil_data.get('total_routes', 1)
        routing_success = successful_routes / total_routes
        
        # Check house compatibility violations
        violations = sigil_data.get('compatibility_violations', 0)
        violation_penalty = min(0.5, violations * 0.1)
        
        # Check mythic grammar consistency
        grammar_consistency = sigil_data.get('grammar_consistency_score', 0.8)
        
        # Weighted score
        coherence_score = routing_success * 0.5 + grammar_consistency * 0.3 + (1.0 - violation_penalty) * 0.2
        
        return min(1.0, max(0.0, coherence_score))


class TemporalContinuityMonitor:
    """Monitors temporal continuity and heritage preservation"""
    
    def evaluate_continuity(self, context: Dict[str, Any]) -> float:
        """Evaluate temporal continuity [0, 1]"""
        temporal_data = context.get('temporal_system', {})
        
        # Check heritage preservation success
        heritage_preserved = temporal_data.get('heritage_preservation_events', 0)
        heritage_attempts = temporal_data.get('heritage_preservation_attempts', 1)
        preservation_rate = heritage_preserved / heritage_attempts
        
        # Check cross-epochal link health
        active_links = temporal_data.get('active_epochal_links', 0)
        expected_links = temporal_data.get('expected_epochal_links', 1)
        link_health = min(1.0, active_links / expected_links)
        
        # Check amnesia prevention score
        amnesia_prevention = temporal_data.get('amnesia_prevention_score', 0.5)
        
        # Weighted score
        continuity_score = preservation_rate * 0.4 + link_health * 0.3 + amnesia_prevention * 0.3
        
        return min(1.0, continuity_score)


# Global monitor instance
_global_mythic_monitor = None

def get_mythic_coherence_monitor() -> MythicCoherenceMonitor:
    """Get global mythic coherence monitor instance"""
    global _global_mythic_monitor
    if _global_mythic_monitor is None:
        _global_mythic_monitor = MythicCoherenceMonitor()
    return _global_mythic_monitor
